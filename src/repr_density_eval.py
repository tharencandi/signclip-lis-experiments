"""
Given an embedding directory and model checkpoint, this script
uses the Sign Density Ratio (SDR) introduced in the SignCL paper,

which combines average inter-gloss distance and intra-gloss distance
to quantify the density of representations for each gloss.



The SDR for a specific gloss ($G_i$) is calculated as:

$$ \text{SDR}(G_i) = \frac{\text{Intra-Gloss Distance}}{\text{Inter-Gloss Distance}} = \frac{\text{avg. } D(G_i)}{ \text{Mean}_{j \neq i} (D(G_i, G_j)) } $$

Where:

*   $D(G_i, G_j)$: Represents the **Inter-Gloss Distance** between two glosses, $G_i$ and $G_j$.
*   $\text{avg. } D(G_i)$: Represents the average distance of $G_i$ to all other glosses.
*   $\text{Intra-Gloss Distance } D(G_i)$: Evaluates the average distance *within* a single gloss $G_i$.



### Distance Formulas

The distances are calculated as follows:

**Inter-Gloss Distance:**

$$ D(G_i, G_j) = \frac{1}{|G_i||G_j|} \sum_{x \in G_i, y \in G_j} d(x, y) \quad (2) $$

**Intra-Gloss Distance:**

$$ D(G_i) = \frac{1}{|G_i|(|G_i| - 1)} \sum_{x, y \in G_i, x \neq y} d(x, y) \quad (3) $$

Where:

*   $|G_i|$ and $|G_j|$: The number of instances within glosses $G_i$ and $G_j$, respectively.
*   $d(x, y)$: The distance measure between the embeddings of instances $x$ and $y$ (e.g., Euclidean distance).


The **average Sign Density Ratio (SDR)** across all glosses is calculated as
 $\text{SDR} = \text{Mean}(\text{SDR}(G_i))$. 
 
 This provides a comprehensive evaluation of the overall representation density of the dataset.

 python src/repr_density_eval.py \
    --embedding_dir dataset/embeddings/a3lis_normalised \
    --run_correlation \
    --sdr_on test \
    --n_bins 5 \
    --corr_plot runs/eval_results/sdr_bins.png
 
python src/repr_density_eval.py \
    --embedding_dir dataset/embeddings/a3lis_normalised \
    --run_sample_efficiency \
    --sdr_pct 0.20 \
    --k_shots 1 2 5 7 \
    --eff_plot runs/eval_results/sdr_efficiency.png

python src/repr_density_eval.py \
    --embedding_dir dataset/embeddings/a3lis_default_v2\
    --run_interventions \
    --intervention_dirs dataset/embeddings/a3lis_finetune \
    --intervention_labels "ft1" \
    --sdr_on test --sdr_pct 0.20


"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import wilcoxon, rankdata, pearsonr, spearmanr
from src.embedding_utils import load_a3lis_embeddings


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/path objects to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _resolve_output_path(user_path: Optional[Path], default_name: str, out_dir: Path) -> Path:
    """Use user-provided path when available, else place output under out_dir."""
    if user_path is not None:
        return user_path
    return out_dir / default_name



def compute_sdr(embeddings: np.ndarray, labels: List[str]) -> Tuple[Dict[str, float], float]:
    """
    Compute Sign Density Ratio (SDR) for each gloss and the average SDR.

    Uses a vectorised cosine-distance matrix (1 - normalised dot product) to
    avoid per-pair Python overhead.

    Args:
        embeddings: Array of shape (N, D)
        labels: List of labels corresponding to each embedding
    Returns:
        Tuple of (sdr_per_gloss, average_sdr)
    """
    # Precompute full pairwise cosine-distance matrix once: O(N^2 D)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
    normed = embeddings / norms
    # cosine distance = 1 - cosine similarity; clip to [0,2] for numerical safety
    dist_matrix = np.clip(1.0 - normed @ normed.T, 0.0, 2.0)

    label_arr = np.array(labels)
    unique_labels = np.unique(label_arr)
    sdr_per_gloss: Dict[str, float] = {}

    for gloss in unique_labels:
        mask = label_arr == gloss
        idx = np.where(mask)[0]
        other = np.where(~mask)[0]

        if len(idx) < 2:
            sdr_per_gloss[gloss] = float('nan')
            continue
        if len(other) == 0:
            sdr_per_gloss[gloss] = float('nan')
            continue

        # Intra: upper-triangle of the gloss×gloss sub-matrix
        intra_block = dist_matrix[np.ix_(idx, idx)]
        i_upper, j_upper = np.triu_indices(len(idx), k=1)
        avg_intra = intra_block[i_upper, j_upper].mean()

        # Inter: full gloss×other sub-matrix
        avg_inter = dist_matrix[np.ix_(idx, other)].mean()

        sdr_per_gloss[gloss] = avg_intra / avg_inter if avg_inter > 0 else float('inf')

    average_sdr = float(np.nanmean(list(sdr_per_gloss.values())))
    return sdr_per_gloss, average_sdr


def summarise_sdr(sdr_per_gloss: Dict[str, float], label: str = '') -> None:
    """Print summary statistics for an SDR distribution."""
    values = np.array([v for v in sdr_per_gloss.values() if not np.isnan(v)])
    tag = f" [{label}]" if label else ""
    if len(values) == 0:
        print(f"\nSDR summary{tag}  (n=0 glosses)")
        print("  No valid SDR values available for summary.")
        return
    print(f"\nSDR summary{tag}  (n={len(values)} glosses)")
    print(f"  mean ± std : {values.mean():.4f} ± {values.std():.4f}")
    print(f"  median     : {np.median(values):.4f}")
    q1, q3 = np.percentile(values, [25, 75])
    print(f"  IQR        : [{q1:.4f}, {q3:.4f}]")
    print(f"  min / max  : {values.min():.4f} / {values.max():.4f}")


def wilcoxon_compare(
    sdr_a: Dict[str, float],
    sdr_b: Dict[str, float],
    label_a: str = 'A',
    label_b: str = 'B',
) -> None:
    """Wilcoxon signed-rank test on paired per-gloss SDR values."""
    common = sorted(k for k in sdr_a if k in sdr_b
                    and not np.isnan(sdr_a[k]) and not np.isnan(sdr_b[k]))
    if len(common) < 10:
        print(f"\nWilcoxon: only {len(common)} paired glosses — skipping.")
        return

    a_vals = np.array([sdr_a[k] for k in common])
    b_vals = np.array([sdr_b[k] for k in common])
    diffs = b_vals - a_vals
    nonzero_diffs = diffs[diffs != 0]

    if len(nonzero_diffs) == 0:
        print(f"\nWilcoxon signed-rank test: {label_a} vs {label_b}  (n={len(common)} paired glosses)")
        print("  statistic  : 0.0000")
        print("  p-value    : 1")
        print("  effect size (rank-biserial r): 0.0000")
        print("  direction  : no difference; all paired SDR values are identical")
        return

    stat, p = wilcoxon(a_vals, b_vals)

    # Rank-biserial correlation as effect size
    ranks = rankdata(np.abs(nonzero_diffs))
    r_pos = ranks[nonzero_diffs > 0].sum()
    r_neg = ranks[nonzero_diffs < 0].sum()
    n = len(ranks)
    rbc = (r_pos - r_neg) / (n * (n + 1) / 2)

    print(f"\nWilcoxon signed-rank test: {label_a} vs {label_b}  (n={len(common)} paired glosses)")
    print(f"  statistic  : {stat:.4f}")
    print(f"  p-value    : {p:.4g}")
    print(f"  effect size (rank-biserial r): {rbc:.4f}")
    direction = f"{label_b} lower SDR" if rbc < 0 else f"{label_a} lower SDR"
    print(f"  direction  : {direction}  (lower SDR = better-separated)")


def top_k_analysis(
    sdr_per_gloss: Dict[str, float],
    k: int = 10,
    label: str = '',
) -> None:
    """Print the K best-separated and K worst-separated glosses."""
    valid = {g: v for g, v in sdr_per_gloss.items() if not np.isnan(v) and not np.isinf(v)}
    sorted_glosses = sorted(valid.items(), key=lambda x: x[1])
    tag = f" [{label}]" if label else ""
    print(f"\nTop-{k} best-separated (lowest SDR){tag}:")
    for gloss, sdr in sorted_glosses[:k]:
        print(f"  {gloss:<30s} SDR={sdr:.4f}")
    print(f"\nTop-{k} worst-separated (highest SDR){tag}:")
    for gloss, sdr in sorted_glosses[-k:][::-1]:
        print(f"  {gloss:<30s} SDR={sdr:.4f}")


def plot_sdr_distribution(
    sdr_dicts: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
) -> None:
    """Violin + strip plot of SDR distributions. Always saves to a file."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend — works in terminals
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    if output_path is None:
        output_path = Path(f"sdr_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    fig, ax = plt.subplots(figsize=(max(4, 2 * len(sdr_dicts)), 5))

    labels = list(sdr_dicts.keys())
    data = [
        [v for v in sdr_dicts[lbl].values() if not np.isnan(v) and not np.isinf(v)]
        for lbl in labels
    ]

    ax.violinplot(data, positions=range(len(labels)), showmedians=True)
    for i, d in enumerate(data):
        ax.scatter(
            np.random.default_rng(0).uniform(-0.15, 0.15, len(d)) + i,
            d, alpha=0.4, s=12, zorder=3,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('SDR (lower = better separated)')
    ax.set_title('Per-gloss Sign Density Ratio distribution')
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, label='SDR=1 (intra=inter)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 1: SDR as a predictor of class difficulty
# ---------------------------------------------------------------------------

def train_linear_probe(
    train_embeddings: np.ndarray,
    train_labels: List[str],
    val_embeddings: np.ndarray,
    val_labels: List[str],
    seed: int = 42,
) -> object:
    """
    Train a LogisticRegression linear probe on L2-normalised embeddings.
    Mirrors the approach in few_shot.py for consistency.
    """
    from sklearn.linear_model import LogisticRegression

    def _norm(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms == 0, 1.0, norms)

    X_train = _norm(train_embeddings)
    X_val   = _norm(val_embeddings)

    print("Training linear probe (LogisticRegression, max_iter=1000) ...")
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X_train, train_labels)

    val_acc = (clf.predict(X_val) == np.array(val_labels)).mean()
    print(f"  Val R@1: {val_acc:.4f}")
    return clf


def compute_per_class_accuracy(
    clf,
    test_embeddings: np.ndarray,
    test_labels: List[str],
) -> Dict[str, float]:
    """Returns per-class top-1 accuracy (R@1) on the test set."""
    norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    X_test = test_embeddings / np.where(norms == 0, 1.0, norms)

    known_classes = set(clf.classes_)
    mask = np.array([l in known_classes for l in test_labels])
    if not mask.all():
        print(f"  Warning: {(~mask).sum()} test samples skipped "
              "(label unseen during probe training).")
    X_test         = X_test[mask]
    filtered_labels = [l for l, m in zip(test_labels, mask) if m]

    preds = clf.predict(X_test)

    per_class: Dict[str, float] = {}
    for cls in set(filtered_labels):
        idx = np.array([i for i, l in enumerate(filtered_labels) if l == cls])
        per_class[cls] = float((preds[idx] == cls).mean())
    return per_class


def sdr_accuracy_correlation(
    sdr_per_gloss: Dict[str, float],
    per_class_acc: Dict[str, float],
) -> Dict:
    """
    Compute Pearson and Spearman correlations between per-class SDR and R@1.
    Prints a formatted report and returns a results dict.
    """
    common = sorted(
        k for k in sdr_per_gloss
        if k in per_class_acc
        and not np.isnan(sdr_per_gloss[k])
        and not np.isinf(sdr_per_gloss[k])
    )

    if len(common) < 5:
        print(f"Only {len(common)} aligned classes — cannot compute correlation.")
        return {}

    sdr_vals = np.array([sdr_per_gloss[k] for k in common])
    acc_vals = np.array([per_class_acc[k]  for k in common])

    pr_res = pearsonr(sdr_vals, acc_vals)
    sr_res = spearmanr(sdr_vals, acc_vals)
    p_r = float(pr_res.statistic)  # type: ignore[union-attr]
    p_p = float(pr_res.pvalue)     # type: ignore[union-attr]
    s_r = float(sr_res.statistic)  # type: ignore[union-attr]
    s_p = float(sr_res.pvalue)     # type: ignore[union-attr]

    print(f"\n{'='*60}")
    print(f"Experiment 1: SDR vs Per-Class Accuracy Correlation")
    print(f"{'='*60}")
    print(f"  Aligned classes : {len(common)}")
    print(f"  SDR  mean±std   : {sdr_vals.mean():.4f} ± {sdr_vals.std():.4f}")
    print(f"  Acc  mean±std   : {acc_vals.mean():.4f} ± {acc_vals.std():.4f}")
    print(f"\n  Pearson   r = {p_r:+.4f},  p = {p_p:.4g}")
    print(f"  Spearman  r = {s_r:+.4f},  p = {s_p:.4g}")
    sig  = "SIGNIFICANT (p < 0.05)" if min(p_p, s_p) < 0.05 else "not significant"
    dirn = "negative — higher SDR → lower accuracy" if p_r < 0 else "positive"
    print(f"\n  Result: {sig}, {dirn}")
    print(f"{'='*60}\n")

    return {
        'n_classes': len(common),
        'pearson_r':  float(p_r), 'pearson_p':  float(p_p),
        'spearman_r': float(s_r), 'spearman_p': float(s_p),
        'class_names': common,
        'sdr_vals': sdr_vals.tolist(),
        'acc_vals': acc_vals.tolist(),
    }


def plot_sdr_bins(
    sdr_per_gloss: Dict[str, float],
    per_class_acc: Dict[str, float],
    n_bins: int = 5,
    output_path: Optional[Path] = None,
) -> None:
    """
    Bar chart of mean per-class R@1 across equal-frequency SDR bins.
    Experiment 1 visual: proves monotonic accuracy drop from least- to most-dense bin.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping bin plot.")
        return

    common = sorted(
        k for k in sdr_per_gloss
        if k in per_class_acc
        and not np.isnan(sdr_per_gloss[k])
        and not np.isinf(sdr_per_gloss[k])
    )
    if not common:
        print("No aligned classes — skipping bin plot.")
        return

    sdr_vals = np.array([sdr_per_gloss[k] for k in common])
    acc_vals = np.array([per_class_acc[k]  for k in common])

    # Equal-frequency bins via percentile edges
    bin_edges = np.percentile(sdr_vals, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-9  # include the maximum value

    bin_labels, bin_accs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (sdr_vals >= bin_edges[i]) & (sdr_vals < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_labels.append(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})")
        bin_accs.append(acc_vals[mask].mean())
        bin_counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(bin_labels)), 4))
    bars = ax.bar(range(len(bin_labels)), bin_accs,
                  color='steelblue', edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(
        [f"{lbl}\n(n={cnt})" for lbl, cnt in zip(bin_labels, bin_counts)],
        rotation=20, ha='right', fontsize=9,
    )
    ax.set_xlabel('SDR bin  (low = well-separated  →  high = dense/overlapping)')
    ax.set_ylabel('Mean per-class R@1 accuracy')
    ax.set_title(
        'Exp 1: Per-class accuracy by SDR bin\n'
        '(monotonic drop expected if SDR predicts class difficulty)'
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=9,
        )
    plt.tight_layout()

    if output_path is None:
        output_path = Path(f"sdr_bins_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_path, dpi=150)
    print(f"SDR bin plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: SDR as a predictor of sample efficiency
# ---------------------------------------------------------------------------

def stratify_classes_by_sdr(
    sdr_per_gloss: Dict[str, float],
    pct: float = 0.20,
) -> Tuple[List[str], List[str]]:
    """
    Return the bottom-`pct` (best-separated, low SDR) and top-`pct`
    (most-dense, high SDR) class groups.
    """
    valid = sorted(
        [(g, v) for g, v in sdr_per_gloss.items()
         if not np.isnan(v) and not np.isinf(v)],
        key=lambda x: x[1],
    )
    cutoff = max(1, int(round(len(valid) * pct)))
    low_classes  = [g for g, _ in valid[:cutoff]]
    high_classes = [g for g, _ in valid[-cutoff:]]
    return low_classes, high_classes


def run_knn_few_shot(
    train_embeddings: np.ndarray,
    train_labels: List[str],
    test_embeddings: np.ndarray,
    test_labels: List[str],
    class_subset: List[str],
    k_shot: int,
    seed: int = 42,
) -> float:
    """
    K-NN few-shot evaluation restricted to `class_subset` at `k_shot` examples/class.

    Mirrors few_shot.py: KNeighborsClassifier(n_neighbors=k_shot, metric='cosine')
    on L2-normalised embeddings.

    Returns:
        R@1 accuracy over test samples belonging to class_subset.
    """
    from collections import defaultdict as _dd
    from sklearn.neighbors import KNeighborsClassifier

    subset = set(class_subset)
    rng = np.random.default_rng(seed)

    # filter train to subset, then sample k_shot per class
    train_mask = np.array([l in subset for l in train_labels])
    X_tr_all = train_embeddings[train_mask]
    y_tr_all = [l for l, m in zip(train_labels, train_mask) if m]

    class_idx: Dict[str, List[int]] = _dd(list)
    for i, l in enumerate(y_tr_all):
        class_idx[l].append(i)

    selected: List[int] = []
    for cls, idxs in class_idx.items():
        k = min(k_shot, len(idxs))
        chosen = rng.choice(len(idxs), size=k, replace=False)
        selected.extend([idxs[j] for j in chosen])
    selected.sort()

    X_tr = X_tr_all[selected]
    y_tr = [y_tr_all[i] for i in selected]

    # filter test to subset
    test_mask = np.array([l in subset for l in test_labels])
    X_te = test_embeddings[test_mask]
    y_te = [l for l, m in zip(test_labels, test_mask) if m]

    if len(X_tr) == 0 or len(X_te) == 0:
        return float('nan')

    def _norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(n == 0, 1.0, n)

    X_tr, X_te = _norm(X_tr), _norm(X_te)

    # n_neighbors = k_shot (mirrors few_shot.py shot-limited mode)
    clf = KNeighborsClassifier(n_neighbors=min(k_shot, len(X_tr)), metric='cosine')
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    return float((np.array(preds) == np.array(y_te)).mean())


def plot_sample_efficiency(
    results: Dict[str, Dict[int, float]],
    k_shots: List[int],
    group_sdrs: Dict[str, Tuple[float, float]],
    output_path: Optional[Path] = None,
) -> None:
    """
    Line plot of R@1 vs. shots for low-SDR and high-SDR groups.
    Experiment 2 visual: low-SDR saturates quickly, high-SDR needs more shots.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping sample-efficiency plot.")
        return

    colours = {'low_sdr': 'steelblue', 'high_sdr': 'tomato'}
    nice    = {'low_sdr': 'Low SDR (best-separated)', 'high_sdr': 'High SDR (most-dense)'}

    fig, ax = plt.subplots(figsize=(7, 4))
    for group, colour in colours.items():
        if group not in results:
            continue
        ys = [results[group].get(k, float('nan')) for k in k_shots]
        ms, ss = group_sdrs.get(group, (0.0, 0.0))
        label = f"{nice[group]}  (SDR {ms:.3f}\u00b1{ss:.3f})"
        ax.plot(k_shots, ys, marker='o', color=colour, label=label, linewidth=2)
        for x, y in zip(k_shots, ys):
            if not np.isnan(y):
                ax.annotate(f'{y:.3f}', (x, y),
                            textcoords='offset points', xytext=(0, 7),
                            ha='center', fontsize=8)

    ax.set_xlabel('Number of shots (K)')
    ax.set_ylabel('R@1 accuracy')
    ax.set_xticks(k_shots)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        'Exp 2: Sample efficiency by SDR group\n'
        '(low-SDR saturates fast; high-SDR needs more shots)'
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    if output_path is None:
        output_path = Path(f"sdr_sample_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_path, dpi=150)
    print(f"Sample-efficiency plot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3: SDR guiding targeted training interventions
# ---------------------------------------------------------------------------

def compute_delta_sdr(
    sdr_before: Dict[str, float],
    sdr_after: Dict[str, float],
) -> Dict[str, float]:
    """
    Per-class ΔSDR = SDR_after − SDR_before.
    Negative values mean density reduction (improvement).
    Only includes classes valid in both dicts.
    """
    return {
        k: sdr_after[k] - sdr_before[k]
        for k in sdr_before
        if k in sdr_after
        and not np.isnan(sdr_before[k]) and not np.isinf(sdr_before[k])
        and not np.isnan(sdr_after[k])  and not np.isinf(sdr_after[k])
    }


def analyse_intervention(
    sdr_baseline: Dict[str, float],
    delta_sdr: Dict[str, float],
    delta_r1: Dict[str, float],
    label: str,
    pct: float = 0.20,
) -> Dict:
    """
    Stratify per-class ΔSDR and ΔR@1 by initial SDR tier and test whether
    high-initial-SDR classes benefit disproportionately (Mann-Whitney U).

    H1: high-SDR classes get more negative ΔSDR (larger density reduction).
    """
    from scipy.stats import mannwhitneyu

    low_cls, high_cls = stratify_classes_by_sdr(sdr_baseline, pct=pct)
    low_set, high_set = set(low_cls), set(high_cls)

    low_dsdr  = [delta_sdr[k] for k in delta_sdr if k in low_set]
    high_dsdr = [delta_sdr[k] for k in delta_sdr if k in high_set]
    low_dr1   = [delta_r1.get(k, float('nan')) for k in delta_sdr if k in low_set]
    high_dr1  = [delta_r1.get(k, float('nan')) for k in delta_sdr if k in high_set]
    low_dr1   = [v for v in low_dr1  if not np.isnan(v)]
    high_dr1  = [v for v in high_dr1 if not np.isnan(v)]

    r1_str_low  = f",  ΔR@1 {np.mean(low_dr1):+.4f}±{np.std(low_dr1):.4f}" if low_dr1  else ""
    r1_str_high = f",  ΔR@1 {np.mean(high_dr1):+.4f}±{np.std(high_dr1):.4f}" if high_dr1 else ""
    print(f"\n  [{label}]")
    print(f"    Low-SDR  tier: ΔSDR {np.mean(low_dsdr):+.4f}±{np.std(low_dsdr):.4f}  "
          f"(n={len(low_dsdr)}){r1_str_low}")
    print(f"    High-SDR tier: ΔSDR {np.mean(high_dsdr):+.4f}±{np.std(high_dsdr):.4f}  "
          f"(n={len(high_dsdr)}){r1_str_high}")

    mw_stat, mw_p = None, None
    if len(low_dsdr) >= 3 and len(high_dsdr) >= 3:
        mw = mannwhitneyu(high_dsdr, low_dsdr, alternative='less')
        mw_stat, mw_p = float(mw.statistic), float(mw.pvalue)  # type: ignore[union-attr]
        sig = "SIGNIFICANT (p<0.05)" if mw_p < 0.05 else "not significant"
        print(f"    Mann-Whitney (H1: high-SDR gets larger SDR reduction): "
              f"stat={mw_stat:.2f}, p={mw_p:.4g} — {sig}")

    return {
        'label': label,
        'n_low': len(low_dsdr), 'n_high': len(high_dsdr),
        'low_mean_dsdr':  float(np.mean(low_dsdr)),
        'high_mean_dsdr': float(np.mean(high_dsdr)),
        'low_mean_dr1':   float(np.mean(low_dr1))  if low_dr1  else None,
        'high_mean_dr1':  float(np.mean(high_dr1)) if high_dr1 else None,
        'mw_stat': mw_stat, 'mw_p': mw_p,
    }


def plot_delta_sdr_scatter(
    sdr_baseline: Dict[str, float],
    delta_sdrs: Dict[str, Dict[str, float]],
    pct: float = 0.20,
    output_path: Optional[Path] = None,
) -> None:
    """
    Scatter plots of initial SDR (x) vs. ΔSDR (y) for each intervention model.
    Points are coloured by initial SDR tier. A trend line shows the corrective bias.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not available — skipping scatter plot.")
        return

    models = list(delta_sdrs.keys())
    ncols  = min(len(models), 3)
    nrows  = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False)

    low_cls, high_cls = stratify_classes_by_sdr(sdr_baseline, pct=pct)
    low_set, high_set = set(low_cls), set(high_cls)

    for idx, (lbl, dsdr) in enumerate(delta_sdrs.items()):
        ax = axes[idx // ncols][idx % ncols]
        common = [k for k in dsdr if k in sdr_baseline
                  and not np.isnan(sdr_baseline[k]) and not np.isinf(sdr_baseline[k])]
        xs = np.array([sdr_baseline[k] for k in common])
        ys = np.array([dsdr[k]         for k in common])
        cs = ['steelblue' if k in low_set else
              'tomato'    if k in high_set else 'grey'
              for k in common]
        ax.scatter(xs, ys, c=cs, alpha=0.65, s=30, edgecolors='none')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Initial SDR (baseline)')
        ax.set_ylabel('ΔSDR  (negative = improvement)')
        ax.set_title(lbl)
        if len(xs) >= 3:
            z = np.polyfit(xs, ys, 1)
            xfit = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xfit, np.polyval(z, xfit), color='black', linewidth=1.2, alpha=0.6)

    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_elements = [
        Patch(facecolor='steelblue', label=f'Low-SDR tier ({pct*100:.0f}%)'),
        Patch(facecolor='tomato',    label=f'High-SDR tier ({pct*100:.0f}%)'),
        Patch(facecolor='grey',      label='Mid-SDR'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Exp 3: Initial SDR vs. ΔSDR by intervention', fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if output_path is None:
        output_path = Path(f"sdr_delta_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Delta-SDR scatter saved to {output_path}")
    plt.close(fig)


def plot_delta_sdr_tiers(
    tier_results: List[Dict],
    output_path: Optional[Path] = None,
) -> None:
    """
    Grouped bar chart of mean ΔSDR (and optionally ΔR@1) per initial SDR tier,
    grouped by intervention model.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping tier bar chart.")
        return

    labels    = [r['label']         for r in tier_results]
    low_dsdr  = [r['low_mean_dsdr'] for r in tier_results]
    high_dsdr = [r['high_mean_dsdr']for r in tier_results]
    has_r1    = any(r['low_mean_dr1'] is not None for r in tier_results)

    nrows = 2 if has_r1 else 1
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(max(6, 2 * len(labels)), 4 * nrows),
                             squeeze=False)

    x, w = np.arange(len(labels)), 0.35

    ax = axes[0][0]
    bl = ax.bar(x - w/2, low_dsdr,  w, label='Low-SDR tier',
                color='steelblue', edgecolor='black', alpha=0.85)
    bh = ax.bar(x + w/2, high_dsdr, w, label='High-SDR tier',
                color='tomato',    edgecolor='black', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Mean ΔSDR  (negative = improvement)')
    ax.set_title('Exp 3: Mean ΔSDR by initial SDR tier\n'
                 '(high-SDR tier expected to have larger reduction)')
    ax.legend()
    for bar in list(bl) + list(bh):
        offset = 0.002 if bar.get_height() >= 0 else -0.012
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + offset,
                f'{bar.get_height():+.3f}', ha='center', va='bottom', fontsize=8)

    if has_r1:
        low_dr1  = [r['low_mean_dr1']  if r['low_mean_dr1']  is not None else 0.0 for r in tier_results]
        high_dr1 = [r['high_mean_dr1'] if r['high_mean_dr1'] is not None else 0.0 for r in tier_results]
        ax2 = axes[1][0]
        ax2.bar(x - w/2, low_dr1,  w, label='Low-SDR tier',
                color='steelblue', edgecolor='black', alpha=0.85)
        ax2.bar(x + w/2, high_dr1, w, label='High-SDR tier',
                color='tomato',    edgecolor='black', alpha=0.85)
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15, ha='right')
        ax2.set_ylabel('Mean ΔR@1')
        ax2.set_title('Mean ΔR@1 by initial SDR tier')
        ax2.legend()

    plt.tight_layout()
    if output_path is None:
        output_path = Path(f"sdr_delta_tiers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_path, dpi=150)
    print(f"Tier analysis plot saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate representation density using Sign Density Ratio (SDR)"
    )
    parser.add_argument(
        '--embedding_dir', type=Path, required=True,
        help='Embedding dir for the primary model (e.g. dataset/embeddings/a3lis_normalised_70_10_20)'
    )
    parser.add_argument(
        '--compare_dir', type=Path, default=None,
        help='Optional second embedding dir for comparison (e.g. fine-tuned model)'
    )
    parser.add_argument(
        '--split', type=str, default='test',
        help='Split to evaluate on (default: test)'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='Number of top/bottom glosses to show (default: 10)'
    )
    parser.add_argument(
        '--label_a', type=str, default='default',
        help='Display name for primary model'
    )
    parser.add_argument(
        '--label_b', type=str, default='finetuned',
        help='Display name for comparison model'
    )
    parser.add_argument(
        '--plot', type=Path, default=None,
        help='Path to save distribution plot (always saved; if omitted, uses auto-generated timestamped .png)'
    )

    # --- Experiment 1: SDR as predictor of class difficulty ---
    parser.add_argument(
        '--run_correlation', action='store_true', default=False,
        help='Run Experiment 1: correlate per-class SDR with linear-probe accuracy'
    )
    parser.add_argument(
        '--sdr_on', type=str, default='test', choices=['test', 'all'],
        help='Compute SDR on the test split only ("test") or the full dataset ("all") '
             '(default: test — measures tangling in the unseen signer space)'
    )
    parser.add_argument(
        '--n_bins', type=int, default=5,
        help='Number of equal-frequency SDR bins for the bar chart (default: 5)'
    )
    parser.add_argument(
        '--corr_plot', type=Path, default=None,
        help='Path to save the SDR-bin bar chart; auto-generated if omitted'
    )

    # --- Experiment 2: SDR as predictor of sample efficiency ---
    parser.add_argument(
        '--run_sample_efficiency', action='store_true', default=False,
        help='Run Experiment 2: stratified K-NN few-shot across SDR groups'
    )
    parser.add_argument(
        '--sdr_pct', type=float, default=0.20,
        help='Fraction of classes for low/high SDR groups (default: 0.20 = top/bottom 20%%)'
    )
    parser.add_argument(
        '--k_shots', type=int, nargs='+', default=[1, 2, 5, 7],
        help='Shot counts to evaluate (default: 1 2 5 7; max is 7 per class in train)'
    )
    parser.add_argument(
        '--eff_plot', type=Path, default=None,
        help='Path to save the sample-efficiency line plot; auto-generated if omitted'
    )

    # --- Experiment 3: SDR guiding targeted training interventions ---
    parser.add_argument(
        '--run_interventions', action='store_true', default=False,
        help='Run Experiment 3: per-class ΔSDR and ΔR@1 by SDR tier across interventions'
    )
    parser.add_argument(
        '--intervention_dirs', type=Path, nargs='+', default=None,
        help='One or more fine-tuned embedding dirs to compare against --embedding_dir (baseline)'
    )
    parser.add_argument(
        '--intervention_labels', type=str, nargs='+', default=None,
        help='Display names for each intervention dir (must match count of --intervention_dirs)'
    )
    parser.add_argument(
        '--delta_scatter', type=Path, default=None,
        help='Path to save initial-SDR vs. ΔSDR scatter plot; auto-generated if omitted'
    )
    parser.add_argument(
        '--delta_tiers_plot', type=Path, default=None,
        help='Path to save the ΔSDR-by-tier grouped bar chart; auto-generated if omitted'
    )

    args = parser.parse_args()

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = Path('runs/eval_results')
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {eval_dir}")

    print(f"Loading {args.split} embeddings from {args.embedding_dir} ...")
    emb_a, labels_a, _, _ = load_a3lis_embeddings(
        args.embedding_dir, split=args.split, label_language='english'
    )
    print("computing SDR for primary model ...")
    sdr_a, avg_a = compute_sdr(emb_a, labels_a)
    print("Summary for primary model ...")
    summarise_sdr(sdr_a, label=args.label_a)
    print(f"Average SDR [{args.label_a}]: {avg_a:.4f}")
    top_k_analysis(sdr_a, k=args.top_k, label=args.label_a)

    sdr_dicts = {args.label_a: sdr_a}

    if args.compare_dir is not None:
        print(f"\nLoading {args.split} embeddings from {args.compare_dir} ...")
        emb_b, labels_b, _, _ = load_a3lis_embeddings(
            args.compare_dir, split=args.split, label_language='english'
        )
        print("computing SDR for comparison model ...")
        sdr_b, avg_b = compute_sdr(emb_b, labels_b)
        print("Summary for comparison model ...")
        summarise_sdr(sdr_b, label=args.label_b)
        print(f"Average SDR [{args.label_b}]: {avg_b:.4f}")
        print("Comparison metrics ...")
        top_k_analysis(sdr_b, k=args.top_k, label=args.label_b)
        wilcoxon_compare(sdr_a, sdr_b, label_a=args.label_a, label_b=args.label_b)
        sdr_dicts[args.label_b] = sdr_b

    plot_sdr_distribution(
        sdr_dicts,
        output_path=_resolve_output_path(
            args.plot, f"sdr_distribution_{run_ts}.png", eval_dir
        ),
    )

    # --- Experiment 1 ---
    if args.run_correlation:
        print("\n" + "="*60)
        print("Experiment 1: SDR as a Predictor of Class Difficulty")
        print("="*60)

        print("\nLoading train embeddings ...")
        emb_train, labels_train, _, _ = load_a3lis_embeddings(
            args.embedding_dir, split='train', label_language='english'
        )
        print("Loading val embeddings ...")
        emb_val, labels_val, _, _ = load_a3lis_embeddings(
            args.embedding_dir, split='val', label_language='english'
        )
        print("Loading test embeddings ...")
        emb_test, labels_test, _, _ = load_a3lis_embeddings(
            args.embedding_dir, split='test', label_language='english'
        )

        if args.sdr_on == 'test':
            print("\nComputing SDR on test split only ...")
            sdr_corr, _ = compute_sdr(emb_test, labels_test)
        else:
            print("\nComputing SDR on full dataset (train + val + test) ...")
            all_emb    = np.concatenate([emb_train, emb_val, emb_test], axis=0)
            all_labels = labels_train + labels_val + labels_test
            sdr_corr, _ = compute_sdr(all_emb, all_labels)

        clf = train_linear_probe(emb_train, labels_train, emb_val, labels_val)

        print("\nComputing per-class accuracy on test split ...")
        per_class_acc = compute_per_class_accuracy(clf, emb_test, labels_test)

        corr_results = sdr_accuracy_correlation(sdr_corr, per_class_acc)

        plot_sdr_bins(
            sdr_corr, per_class_acc,
            n_bins=args.n_bins,
            output_path=_resolve_output_path(
                args.corr_plot, f"sdr_bins_{run_ts}.png", eval_dir
            ),
        )

        if corr_results:
            out_json = eval_dir / f"sdr_correlation_{run_ts}.json"
            with open(out_json, 'w') as f:
                json.dump(_to_jsonable(corr_results), f, indent=2)
            print(f"Correlation results saved to {out_json}")

    # --- Experiment 2 ---
    if args.run_sample_efficiency:
        print("\n" + "="*60)
        print("Experiment 2: SDR as a Predictor of Sample Efficiency")
        print("="*60)

        print("\nLoading embeddings ...")
        emb_tr2, lab_tr2, _, _ = load_a3lis_embeddings(
            args.embedding_dir, split='train', label_language='english'
        )
        emb_va2, lab_va2, _, _ = load_a3lis_embeddings(
            args.embedding_dir, split='val',   label_language='english'
        )
        emb_te2, lab_te2, _, _ = load_a3lis_embeddings(
            args.embedding_dir, split='test',  label_language='english'
        )

        if args.sdr_on == 'test':
            print("Computing SDR on test split only ...")
            sdr_eff, _ = compute_sdr(emb_te2, lab_te2)
        else:
            print("Computing SDR on full dataset ...")
            sdr_eff, _ = compute_sdr(
                np.concatenate([emb_tr2, emb_va2, emb_te2], axis=0),
                lab_tr2 + lab_va2 + lab_te2,
            )

        low_cls, high_cls = stratify_classes_by_sdr(sdr_eff, pct=args.sdr_pct)

        low_sdrs  = [sdr_eff[c] for c in low_cls  if c in sdr_eff]
        high_sdrs = [sdr_eff[c] for c in high_cls if c in sdr_eff]
        print(f"\nStratification ({args.sdr_pct*100:.0f}% each group):")
        print(f"  Low-SDR  : {len(low_cls):>3} classes,  "
              f"SDR {np.mean(low_sdrs):.4f} \u00b1 {np.std(low_sdrs):.4f}")
        print(f"  High-SDR : {len(high_cls):>3} classes,  "
              f"SDR {np.mean(high_sdrs):.4f} \u00b1 {np.std(high_sdrs):.4f}")

        k_shots_sorted = sorted(set(args.k_shots))
        results_eff: Dict[str, Dict[int, float]] = {'low_sdr': {}, 'high_sdr': {}}

        print(f"\n{'K':>4}  {'Low-SDR R@1':>14}  {'High-SDR R@1':>14}")
        print("-" * 40)
        for k in k_shots_sorted:
            r1_low  = run_knn_few_shot(
                emb_tr2, lab_tr2, emb_te2, lab_te2, low_cls,  k_shot=k
            )
            r1_high = run_knn_few_shot(
                emb_tr2, lab_tr2, emb_te2, lab_te2, high_cls, k_shot=k
            )
            results_eff['low_sdr'][k]  = r1_low
            results_eff['high_sdr'][k] = r1_high
            print(f"{k:>4}  {r1_low:>14.4f}  {r1_high:>14.4f}")

        group_sdrs = {
            'low_sdr':  (float(np.mean(low_sdrs)),  float(np.std(low_sdrs))),
            'high_sdr': (float(np.mean(high_sdrs)), float(np.std(high_sdrs))),
        }
        plot_sample_efficiency(
            results_eff,
            k_shots_sorted,
            group_sdrs,
            output_path=_resolve_output_path(
                args.eff_plot, f"sdr_sample_efficiency_{run_ts}.png", eval_dir
            ),
        )

        out_json = eval_dir / f"sdr_sample_efficiency_{run_ts}.json"
        with open(out_json, 'w') as f:
            json.dump(_to_jsonable({
                'sdr_pct': args.sdr_pct,
                'k_shots': k_shots_sorted,
                'low_classes': low_cls, 'high_classes': high_cls,
                'group_sdrs': group_sdrs,
                'results': {
                    g: {str(k): v for k, v in d.items()}
                    for g, d in results_eff.items()
                },
            }), f, indent=2)
        print(f"Sample-efficiency results saved to {out_json}")

    # --- Experiment 3 ---
    if args.run_interventions:
        print("\n" + "="*60)
        print("Experiment 3: SDR Guiding Targeted Training Interventions")
        print("="*60)

        if not args.intervention_dirs:
            print("  Error: --intervention_dirs is required for Experiment 3.")
        else:
            int_dirs   = args.intervention_dirs
            int_labels = list(args.intervention_labels) if args.intervention_labels else [
                f'model_{i+1}' for i in range(len(int_dirs))
            ]
            if len(int_labels) != len(int_dirs):
                print("  Warning: label count != dir count; using auto-generated names.")
                int_labels = [f'model_{i+1}' for i in range(len(int_dirs))]

            # --- baseline ---
            print("\nLoading baseline embeddings ...")
            emb_bl_tr, lab_bl_tr, _, _ = load_a3lis_embeddings(
                args.embedding_dir, split='train', label_language='english')
            emb_bl_va, lab_bl_va, _, _ = load_a3lis_embeddings(
                args.embedding_dir, split='val',   label_language='english')
            emb_bl_te, lab_bl_te, _, _ = load_a3lis_embeddings(
                args.embedding_dir, split='test',  label_language='english')

            if args.sdr_on == 'test':
                sdr_bl, _ = compute_sdr(emb_bl_te, lab_bl_te)
            else:
                sdr_bl, _ = compute_sdr(
                    np.concatenate([emb_bl_tr, emb_bl_va, emb_bl_te], axis=0),
                    lab_bl_tr + lab_bl_va + lab_bl_te,
                )
            print("Training baseline linear probe ...")
            clf_bl  = train_linear_probe(emb_bl_tr, lab_bl_tr, emb_bl_va, lab_bl_va)
            acc_bl  = compute_per_class_accuracy(clf_bl, emb_bl_te, lab_bl_te)

            all_delta_sdrs: Dict[str, Dict[str, float]] = {}
            tier_results_list: List[Dict] = []

            print(f"\n{'='*60}")
            print("Per-tier \u0394SDR and \u0394R@1")
            print(f"{'='*60}")

            for idir, ilabel in zip(int_dirs, int_labels):
                print(f"\nLoading [{ilabel}] from {idir} ...")
                emb_it_tr, lab_it_tr, _, _ = load_a3lis_embeddings(
                    idir, split='train', label_language='english')
                emb_it_va, lab_it_va, _, _ = load_a3lis_embeddings(
                    idir, split='val',   label_language='english')
                emb_it_te, lab_it_te, _, _ = load_a3lis_embeddings(
                    idir, split='test',  label_language='english')

                if args.sdr_on == 'test':
                    sdr_it, _ = compute_sdr(emb_it_te, lab_it_te)
                else:
                    sdr_it, _ = compute_sdr(
                        np.concatenate([emb_it_tr, emb_it_va, emb_it_te], axis=0),
                        lab_it_tr + lab_it_va + lab_it_te,
                    )

                print(f"Training [{ilabel}] linear probe ...")
                clf_it = train_linear_probe(emb_it_tr, lab_it_tr, emb_it_va, lab_it_va)
                acc_it = compute_per_class_accuracy(clf_it, emb_it_te, lab_it_te)

                dsdr = compute_delta_sdr(sdr_bl, sdr_it)
                dr1  = {
                    k: acc_it.get(k, float('nan')) - acc_bl.get(k, float('nan'))
                    for k in dsdr
                }
                all_delta_sdrs[ilabel] = dsdr
                tier_results_list.append(
                    analyse_intervention(sdr_bl, dsdr, dr1,
                                        label=ilabel, pct=args.sdr_pct)
                )

            plot_delta_sdr_scatter(
                sdr_bl, all_delta_sdrs,
                pct=args.sdr_pct,
                output_path=_resolve_output_path(
                    args.delta_scatter, f"sdr_delta_scatter_{run_ts}.png", eval_dir
                ),
            )
            plot_delta_sdr_tiers(
                tier_results_list,
                output_path=_resolve_output_path(
                    args.delta_tiers_plot, f"sdr_delta_tiers_{run_ts}.png", eval_dir
                ),
            )

            out_json = eval_dir / f"sdr_interventions_{run_ts}.json"
            with open(out_json, 'w') as f:
                json.dump(_to_jsonable({
                    'baseline': str(args.embedding_dir),
                    'interventions': [str(d) for d in int_dirs],
                    'labels': int_labels,
                    'sdr_pct': args.sdr_pct,
                    'tier_results': tier_results_list,
                    'delta_sdrs': {
                        lbl: dict(dsdr) for lbl, dsdr in all_delta_sdrs.items()
                    },
                }), f, indent=2)
            print(f"\nIntervention results saved to {out_json}")

if __name__ == "__main__":
    main()