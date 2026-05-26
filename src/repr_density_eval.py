"""
given an embeddings dir and model checkpoint, we
use the sign density ratio (SDR) introduced in SignCL paper, 

which is a metric that combines the average inter-gloss distance, 
and intra-gloss distance to quantify the density of rperesetnations for each gloss



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

"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import rankdata
from src.embedding_utils import load_a3lis_embeddings



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
        help='Path to save distribution plot (omit to display interactively)'
    )
    args = parser.parse_args()

    print(f"Loading {args.split} embeddings from {args.embedding_dir} ...")
    emb_a, labels_a, _, _ = load_a3lis_embeddings(
        args.embedding_dir, split=args.split, label_language='english'
    )
    print("computing SDR for primary model ...")
    sdr_a, avg_a = compute_sdr(emb_a, labels_a)
    print("Summary for primary model ...")
    summarise_sdr(sdr_a, label=args.label_a)
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
        print("Comparison metrics ...")
        top_k_analysis(sdr_b, k=args.top_k, label=args.label_b)
        wilcoxon_compare(sdr_a, sdr_b, label_a=args.label_a, label_b=args.label_b)
        sdr_dicts[args.label_b] = sdr_b

    plot_sdr_distribution(sdr_dicts, output_path=args.plot)

if __name__ == "__main__":
    main()