"""
Zero-shot evaluation using precomputed embeddings for both poses and text labels.

This script loads precomputed pose embeddings and generates text embeddings on-the-fly
using SignCLIP. It works directly from embedding files without needing the original
.pose files.

Usage:
    # A3LIS dataset with raw glosses (no template)
    python src/zero_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
        --split test \
        --label_language english
    
    # A3LIS with language tag prompt (paper standard: "<en> <lis> {label}")
    python src/zero_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
        --split test \
        --label_language english \
        --prompt_type en_lis
    
    # A3LIS with macro categories instead of micro labels
    python src/zero_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2 \
        --split test \
        --use_categories \
        --prompt_type en_lis
    
    # A3LIS with custom template
    python src/zero_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2 \
        --split test \
        --label_language italian \
        --text_template "Italian sign language: {}"
    
    # Legacy format with precomputed text embeddings
    python src/zero_shot.py \
        --pose_embeddings_dir dataset/embeddings/ \
        --text_embeddings dataset/embeddings/text_embeddings_default_en_lis_{}.npy \
        --text_metadata dataset/embeddings/text_metadata_default_en_lis_{}.json \
        --legacy_format

Prompt Types:
    raw          - {} (no template, raw gloss matching)
    it_lis       - <it> <lis> {} (Italian language tag)
    en_lis       - <en> <lis> {} (paper standard for multilingual models)
"""

import argparse
import sys
import json
import csv
from pathlib import Path
import numpy as np
import statistics
from typing import Optional
from tqdm import tqdm
from datetime import datetime

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Add project root to path for demo_sign import
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.demo_sign import embed_text
from src.embedding_utils import PROMPT_TEMPLATES


def compute_cmc_curve(ranks, num_classes):
    """
    Compute CMC values where x-axis is rank (1..num_classes) and y-axis is accuracy.

    Args:
        ranks: List of zero-indexed ranks for each sample
        num_classes: Number of candidate text classes

    Returns:
        Tuple (ranks_1_indexed, cmc_values)
    """
    if num_classes <= 0:
        return [], []

    ranks_arr = np.array(ranks, dtype=np.int32)
    x_ranks = np.arange(1, num_classes + 1)
    cmc_values = []

    for k in x_ranks:
        # For rank-k retrieval, count samples with rank <= k (convert to 0-index: <= k-1)
        cmc_values.append(float(np.mean(ranks_arr <= (k - 1))))

    return x_ranks.tolist(), cmc_values


def assign_accuracy_tier(recall_at_1):
    """
    Map class R@1 to explicit tiers used for structural analysis.
    """
    if recall_at_1 > 0.30:
        return 'High-Performers (>30% R@1)'
    if recall_at_1 >= 0.01:
        return 'Traces (1% - 29% R@1)'
    return 'Zero-Floor (0% R@1)'


def assign_retrieval_neighborhood(recall_at_1, recall_at_5, recall_at_10):
    """
    Group classes into retrieval neighborhoods:
    - Hit Neighborhood: class has at least one R@1 success
    - Near-Miss Neighborhood: no R@1 success, but succeeds at R@5 or R@10
    - Catastrophic Failure Neighborhood: no success up to R@10
    """
    if recall_at_1 > 0.0:
        return 'Hit Neighborhood (R@1 Success)'
    if recall_at_5 > 0.0 or recall_at_10 > 0.0:
        return 'Near-Miss Neighborhood (R@5/R@10 Success)'
    return 'Catastrophic Failure Neighborhood (0% through R@10)'


def create_class_eval_plots(output_dir, split, prompt_name, label_str, class_metrics, cmc_ranks, cmc_values):
    """
    Create and save class-eval diagnostic figures:
    1) CMC curve (Rank vs Accuracy)
    2) Violin plots of class Median Rank grouped by R@1 tiers

    Returns:
        Dict with saved plot paths (or empty dict if plotting is unavailable)
    """
    if plt is None:
        print("\nWARNING: matplotlib is not installed; skipping class-eval plots.")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    # Build deterministic filename stem shared by both plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_stem = f"zero_shot_{split}_{prompt_name}_{label_str}_{timestamp}"

    # Plot 1: CMC curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cmc_ranks, cmc_values, linewidth=2.5, color='#0b84a5')
    ax.set_title('Cumulative Match Characteristic (CMC) Curve')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Accuracy (R@k)')
    ax.set_xlim(1, max(cmc_ranks) if cmc_ranks else 1)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)

    cmc_file = output_dir / f"{file_stem}_cmc_curve.png"
    fig.tight_layout()
    fig.savefig(str(cmc_file), dpi=180)
    plt.close(fig)
    saved_paths['cmc_curve'] = str(cmc_file)

    # Plot 2: Violin plot by retrieval neighborhoods
    neighborhood_order = [
        'Hit Neighborhood (R@1 Success)',
        'Near-Miss Neighborhood (R@5/R@10 Success)',
        'Catastrophic Failure Neighborhood (0% through R@10)'
    ]
    neighborhood_to_median_ranks = {name: [] for name in neighborhood_order}

    for cls_data in class_metrics.values():
        neighborhood = assign_retrieval_neighborhood(
            cls_data['recall@1'],
            cls_data['recall@5'],
            cls_data['recall@10']
        )
        neighborhood_to_median_ranks[neighborhood].append(cls_data['median_rank'])

    violin_data = [neighborhood_to_median_ranks[name] for name in neighborhood_order]

    fig, ax = plt.subplots(figsize=(11, 6))
    vp = ax.violinplot(violin_data, showmeans=True, showmedians=True, widths=0.8)

    # Keep line styling explicit; body styling is left default for compatibility.
    vp['cmeans'].set_color('#1b1b1b')
    vp['cmedians'].set_color('#d62728')
    vp['cbars'].set_color('#1b1b1b')
    vp['cmaxes'].set_color('#1b1b1b')
    vp['cmins'].set_color('#1b1b1b')

    ax.set_title('Median Rank Distribution by Retrieval Neighborhood')
    ax.set_xlabel('Class Neighborhood')
    ax.set_ylabel('Median Rank')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(neighborhood_order, rotation=8)
    ax.grid(axis='y', alpha=0.25)

    violin_file = output_dir / f"{file_stem}_median_rank_tiers_violin.png"
    fig.tight_layout()
    fig.savefig(str(violin_file), dpi=180)
    plt.close(fig)
    saved_paths['tier_violin'] = str(violin_file)

    return saved_paths


def load_a3lis_embeddings(embedding_dir: Path, split: str = 'test', label_language: str = 'english', use_categories: bool = False):
    """
    Load A3LIS precomputed pose embeddings using embeddings_metadata.json.
    
    Args:
        embedding_dir: Directory containing .npy embedding files and metadata
        split: 'train', 'val', 'test', or 'all' (all splits combined)
        label_language: 'italian' or 'english' for label selection
        use_categories: If True, use macro categories instead of micro labels
    
    Returns:
        Tuple of (embeddings_array, labels_list, filenames_list, all_labels, categories, signers)
        all_labels is the set of unique labels for text embedding generation
    """
    # Load metadata
    metadata_path = embedding_dir / 'embeddings_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    embeddings = []
    labels = []
    filenames = []
    all_labels = set()
    categories = []
    signers = []
    
    # Process each embedding in metadata
    for item in tqdm(metadata['embeddings'], desc=f"Loading {split} embeddings"):
        # Filter by split ('all' loads everything)
        if split != 'all' and item['split'] != split:
            continue
        
        # Get label based on use_categories flag
        if use_categories:
            # Use macro category (should always exist)
            label = item.get('category', item['label_italian'])
        elif label_language == 'italian':
            label = item['label_italian']
        else:  # english
            # Use first English label
            label = item['labels_english'][0] if item['labels_english'] else item['label_italian']
        
        # Load embedding
        emb_path = embedding_dir / item['embedding_file']
        if not emb_path.exists():
            continue
        
        emb = np.load(emb_path)
        if emb.ndim > 1:
            emb = emb.squeeze()
        
        embeddings.append(emb)
        labels.append(label)
        filenames.append(item['embedding_file'])
        all_labels.add(label)
        categories.append(item.get('category', 'unknown'))
        signers.append(item.get('signer', 'unknown'))
    
    embeddings_array = np.array(embeddings) if embeddings else np.array([])
    
    return embeddings_array, labels, filenames, sorted(all_labels), categories, signers


def load_text_embeddings(text_embeddings_path: str, metadata_path: str):
    """
    Load precomputed text embeddings and their metadata.
    
    Args:
        text_embeddings_path: Path to text_embeddings_*.npy file
        metadata_path: Path to text_metadata_*.json file
    
    Returns:
        Tuple of (embeddings, labels, metadata_dict)
    """
    # Load embeddings
    text_embeddings = np.load(text_embeddings_path)
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load labels (from metadata or separate .txt file)
    if 'labels' in metadata:
        labels = metadata['labels']
    else:
        # Try to load from companion .txt file
        labels_path = text_embeddings_path.replace('.npy', '.txt').replace('text_embeddings', 'text_labels')
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
    
    assert len(labels) == text_embeddings.shape[0], \
        f"Mismatch: {len(labels)} labels but {text_embeddings.shape[0]} embeddings"
    
    return text_embeddings, labels, metadata


def parse_embedding_filename(filename: str, label_type: str = 'micro'):
    """
    Parse embedding filename: [video_id]_[macro]_[micro]_[split]_[start]_[end].npy
    
    Example: 01_animals_bear_test_04748_04782.npy
    """
    import re
    pattern = r'(.+?)_(.+?)_(.+?)_(train|test)_(\d+)_(\d+)\.npy$'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    video_id, macro, micro, split, start, end = match.groups()
    return {
        'filename': filename,
        'video_id': video_id,
        'macro_label': macro,
        'micro_label': micro,
        'label': micro if label_type == 'micro' else macro,
        'split': split,
        'start_frame': int(start),
        'end_frame': int(end),
    }


def load_pose_embeddings_for_split(embedding_dir: Path, split: str = 'test', label_type: str = 'micro'):
    """
    Load precomputed pose embeddings for a specific split by parsing embedding filenames.
    
    Args:
        embedding_dir: Directory containing .npy embedding files
        split: 'train' or 'test'
        label_type: 'micro' or 'macro' for label granularity
    
    Returns:
        Tuple of (embeddings_array, labels_list, filenames_list)
    """
    embeddings = []
    labels = []
    filenames = []
    
    # Find all .npy files in the directory
    all_npy_files = sorted(embedding_dir.glob('*.npy'))
    
    for npy_file in tqdm(all_npy_files, desc=f"Loading {split} embeddings"):
        # Skip text embedding files
        if 'text_embeddings' in npy_file.name:
            continue
        
        # Parse filename
        meta = parse_embedding_filename(npy_file.name, label_type)
        if meta is None:
            continue
        
        # Filter by split
        if meta['split'] != split:
            continue
        
        # Load embedding and squeeze to 1D if needed
        emb = np.load(npy_file)
        if emb.ndim > 1:
            emb = emb.squeeze()
        
        embeddings.append(emb)
        labels.append(meta['label'])
        filenames.append(meta['filename'])
    
    embeddings_array = np.array(embeddings) if embeddings else np.array([])
    
    return embeddings_array, labels, filenames


def evaluate_zero_shot(
    pose_embeddings_dir: str,
    text_embeddings_path: Optional[str] = None,
    text_metadata_path: Optional[str] = None,
    label_language: str = 'english',
    split: str = 'test',
    model_name: str = 'default',
    text_template: Optional[str] = None,
    prompt_type: Optional[str] = None,
    legacy_format: bool = False,
    label_type: str = 'micro',
    use_categories: bool = False,
    output_dir: Optional[str] = None,
    class_eval: bool = False
):
    """
    Perform zero-shot evaluation using precomputed embeddings.
    
    Args:
        pose_embeddings_dir: Directory containing precomputed pose embeddings
        text_embeddings_path: Path to text embeddings .npy file (legacy mode)
        text_metadata_path: Path to text metadata .json file (legacy mode)
        label_language: 'italian' or 'english' for A3LIS labels
        split: Which split to evaluate ('train' or 'test')
        model_name: SignCLIP model to use for text embedding generation
        text_template: Custom template for text (overrides prompt_type if both specified)
        prompt_type: Predefined prompt template ('raw', 'it_lis', 'en_lis')
        legacy_format: Use legacy filename parsing instead of metadata
        label_type: 'micro' or 'macro' for legacy format
        use_categories: Use macro categories instead of micro labels (A3LIS only)
        class_eval: Print per-class retrieval metrics table
    """
    print(f"\n{'='*60}")
    print(f"Zero-Shot Evaluation (Precomputed Embeddings)")
    print(f"{'='*60}\n")
    
    embedding_dir = Path(pose_embeddings_dir)
    
    if legacy_format:
        # Legacy mode: parse filenames and use precomputed text embeddings
        print("Using legacy format (filename parsing)")
        
        if not text_embeddings_path or not text_metadata_path:
            print("ERROR: Legacy format requires --text_embeddings and --text_metadata")
            return None
        
        # Load text embeddings
        print(f"Loading text embeddings from {text_embeddings_path}...")
        text_embeddings, text_labels, text_metadata = load_text_embeddings(
            text_embeddings_path, text_metadata_path
        )
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        print(f"  Text labels: {len(text_labels)}")
        
        # Load pose embeddings
        print(f"\nLoading pose embeddings from {pose_embeddings_dir}...")
        pose_embeddings, pose_labels, filenames = load_pose_embeddings_for_split(
            embedding_dir, split, label_type
        )
        pose_categories = ['unknown'] * len(pose_labels)
        pose_signers = ['unknown'] * len(pose_labels)
    
    else:
        # A3LIS mode: use metadata.json
        print(f"Using A3LIS format (metadata.json)")
        if use_categories:
            print(f"  Using macro categories")
        else:
            print(f"  Label language: {label_language}")
        
        # Load pose embeddings from A3LIS format
        print(f"\nLoading pose embeddings from {pose_embeddings_dir}...")
        pose_embeddings, pose_labels, filenames, unique_labels, pose_categories, pose_signers = load_a3lis_embeddings(
            embedding_dir, split, label_language, use_categories
        )
        
        # Resolve which template to use
        # Priority: custom text_template > prompt_type > raw (no template)
        template_to_use = None
        if text_template:
            template_to_use = text_template
            template_name = "custom"
        elif prompt_type and prompt_type in PROMPT_TEMPLATES:
            template_to_use = PROMPT_TEMPLATES[prompt_type]
            template_name = prompt_type
        else:
            # No template - use raw labels
            template_to_use = None
            template_name = "raw"
        
        # Generate text embeddings for unique labels
        print(f"\nGenerating text embeddings for {len(unique_labels)} unique labels...")
        if template_to_use:
            text_inputs = [template_to_use.format(label) for label in unique_labels]
            print(f"  Prompt type: {template_name}")
            print(f"  Template: '{template_to_use}'")
            print(f"  Example: '{text_inputs[0]}'")
        else:
            text_inputs = unique_labels
            print(f"  Prompt type: raw (no template)")
            print(f"  Example: '{text_inputs[0]}'")
        
        text_embeddings = embed_text(text_inputs, model_name=model_name)
        text_labels = unique_labels
        
        print(f"  Text embeddings shape: {text_embeddings.shape}")
    
    print(f"  Pose embeddings shape: {pose_embeddings.shape}")
    print(f"  Loaded: {len(pose_labels)} samples")
    print(f"  Unique classes: {len(set(pose_labels))}\n")
    
    # Check if we have any samples
    if len(pose_labels) == 0:
        print(f"\n{'='*60}")
        print(f"ERROR: No {split} samples found!")
        print(f"{'='*60}")
        print(f"\nPossible issues:")
        print(f"  1. No .npy embedding files with '{split}' in filename")
        print(f"  2. Embeddings not computed yet - run precompute_embeddings.py first")
        print(f"  3. Wrong directory - check --pose_embeddings_dir path")
        print(f"\nExpected filename format:")
        print(f"  [video_id]_[macro]_[micro]_[split]_[start]_[end].npy")
        print(f"  Example: 01_animals_bear_test_04748_04782.npy")
        return None
    
    # Normalize embeddings (L2 normalization for cosine similarity)
    print("Normalizing embeddings...")
    pose_embeddings_norm = pose_embeddings / np.linalg.norm(pose_embeddings, axis=1, keepdims=True)
    text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Compute similarity matrix: (num_test, num_classes)
    print("Computing similarities...")
    similarities = np.matmul(pose_embeddings_norm, text_embeddings_norm.T)
    
    # Rank classes by similarity for each test sample
    ranked_indices = np.argsort(-similarities, axis=1)  # descending order
    
    # Evaluate metrics
    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    ranks = []

    category_stats = {}
    signer_stats = {}
    class_stats = {}
    
    print("Evaluating predictions...")
    for i, gold_label in enumerate(tqdm(pose_labels, desc="Ranking")):
        
        cat = pose_categories[i] if not legacy_format else "unknown"
        signer = pose_signers[i] if not legacy_format else "unknown"

        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'hit_1': 0, 'hit_5': 0, 'hit_10': 0, 'ranks': []}
        if signer not in signer_stats:
            signer_stats[signer] = {'total': 0, 'hit_1': 0, 'hit_5': 0, 'hit_10': 0, 'ranks': []}
        if class_eval and gold_label not in class_stats:
            class_stats[gold_label] = {'total': 0, 'hit_1': 0, 'hit_5': 0, 'hit_10': 0, 'ranks': []}
        
        category_stats[cat]['total'] += 1
        signer_stats[signer]['total'] += 1
        if class_eval:
            class_stats[gold_label]['total'] += 1

        ranked_labels = [text_labels[idx] for idx in ranked_indices[i]]
        
        if gold_label in ranked_labels[:1]:
            hit_1 += 1
            category_stats[cat]['hit_1'] += 1
            signer_stats[signer]['hit_1'] += 1
            if class_eval:
                class_stats[gold_label]['hit_1'] += 1
        if gold_label in ranked_labels[:5]:
            hit_5 += 1
            category_stats[cat]['hit_5'] += 1
            signer_stats[signer]['hit_5'] += 1
            if class_eval:
                class_stats[gold_label]['hit_5'] += 1
        if gold_label in ranked_labels[:10]:
            hit_10 += 1
            category_stats[cat]['hit_10'] += 1
            signer_stats[signer]['hit_10'] += 1
            if class_eval:
                class_stats[gold_label]['hit_10'] += 1
            
        if gold_label in ranked_labels:
            rank = ranked_labels.index(gold_label)
            ranks.append(rank)
            category_stats[cat]['ranks'].append(rank)
            signer_stats[signer]['ranks'].append(rank)
            if class_eval:
                class_stats[gold_label]['ranks'].append(rank)
        else:
            # Gold label not in text labels
            ranks.append(len(text_labels))
            category_stats[cat]['ranks'].append(len(text_labels))
            signer_stats[signer]['ranks'].append(len(text_labels))
            if class_eval:
                class_stats[gold_label]['ranks'].append(len(text_labels))
    
    # Calculate metrics
    num_test = len(pose_labels)
    recall_1 = hit_1 / num_test
    recall_5 = hit_5 / num_test
    recall_10 = hit_10 / num_test
    median_rank = statistics.median(ranks) + 1  # +1 for 1-indexed rank
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Split: {split}")
    print(f"Test samples: {num_test}")
    print(f"Text classes: {len(text_labels)}")
    if not legacy_format:
        # Show prompt info for A3LIS mode
        if text_template:
            print(f"Prompt: custom '{text_template}'")
        elif prompt_type:
            print(f"Prompt: {prompt_type} '{PROMPT_TEMPLATES.get(prompt_type, 'unknown')}'")
        else:
            print(f"Prompt: raw (no template)")
    print(f"\nRetrieval Metrics:")
    print(f"  R@1↑:         {recall_1:>7.2%}  ({hit_1:>5}/{num_test})")
    print(f"  R@5↑:         {recall_5:>7.2%}  ({hit_5:>5}/{num_test})")
    print(f"  R@10↑:        {recall_10:>7.2%}  ({hit_10:>5}/{num_test})")
    print(f"  MedianR↓:     {median_rank:>7.1f}")
    print(f"\nAccuracy:")
    print(f"  Top-1:        {recall_1:>7.2%}  ({hit_1:>5}/{num_test})")

    if not legacy_format and not use_categories:
        print(f"\n{'='*75}")
        print(f"Metrics by Category (Predicting exact words)")
        print(f"{'='*75}")
        print(f"  {'Category':<18} | {'Total':<5} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'MedianR':<7}")
        print(f"  {'-'*18}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
        
        for cat, stats in sorted(category_stats.items()):
            tot = stats['total']
            r1 = stats['hit_1'] / tot
            r5 = stats['hit_5'] / tot
            r10 = stats['hit_10'] / tot
            med_r = statistics.median(stats['ranks']) + 1 if stats['ranks'] else 0.0
            
            print(f"  {cat:<18} | {tot:<5} | {r1:>6.1%} | {r5:>6.1%} | {r10:>6.1%} | {med_r:>7.1f}")

    if split == 'all' and not legacy_format:
        print(f"\n{'='*75}")
        print(f"Metrics by Signer (signer variability)")
        print(f"{'='*75}")
        print(f"  {'Signer':<10} | {'Total':<5} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'MedianR':<7}")
        print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        signer_r1_values = []
        for sgn, stats in sorted(signer_stats.items()):
            tot = stats['total']
            r1 = stats['hit_1'] / tot
            r5 = stats['hit_5'] / tot
            r10 = stats['hit_10'] / tot
            med_r = statistics.median(stats['ranks']) + 1 if stats['ranks'] else 0.0
            signer_r1_values.append(r1)
            print(f"  {sgn:<10} | {tot:<5} | {r1:>6.1%} | {r5:>6.1%} | {r10:>6.1%} | {med_r:>7.1f}")

        if len(signer_r1_values) > 1:
            import statistics as _st
            print(f"  {'─'*10}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")
            print(f"  {'Mean':<10} | {'':5} | {_st.mean(signer_r1_values):>6.1%} |")
            print(f"  {'Std':<10} | {'':5} | {_st.stdev(signer_r1_values):>6.1%} |")
            print(f"  {'Min':<10} | {'':5} | {min(signer_r1_values):>6.1%} |")
            print(f"  {'Max':<10} | {'':5} | {max(signer_r1_values):>6.1%} |")

    if class_eval:
        print(f"\n{'='*90}")
        print(f"Metrics by Class Label")
        print(f"{'='*90}")
        print(f"  {'Class':<35} | {'Total':<5} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'MedianR':<7}")
        print(f"  {'-'*35}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for cls, stats in sorted(class_stats.items()):
            tot = stats['total']
            r1 = stats['hit_1'] / tot
            r5 = stats['hit_5'] / tot
            r10 = stats['hit_10'] / tot
            med_r = statistics.median(stats['ranks']) + 1 if stats['ranks'] else 0.0
            cls_display = cls[:35]
            print(f"  {cls_display:<35} | {tot:<5} | {r1:>6.1%} | {r5:>6.1%} | {r10:>6.1%} | {med_r:>7.1f}")

        # Additional structure analysis requested for class-level behavior
        class_metrics = {
            cls: {
                'total': stats['total'],
                'recall@1': stats['hit_1'] / stats['total'],
                'recall@5': stats['hit_5'] / stats['total'],
                'recall@10': stats['hit_10'] / stats['total'],
                'median_rank': (statistics.median(stats['ranks']) + 1) if stats['ranks'] else 0.0,
            }
            for cls, stats in sorted(class_stats.items())
        }

        # CMC: rank-wise retrieval accuracy profile
        cmc_ranks, cmc_values = compute_cmc_curve(ranks, len(text_labels))

        # Split classes into retrieval neighborhoods and summarize
        neighborhood_summary = {
            'Hit Neighborhood (R@1 Success)': {'count': 0, 'classes': []},
            'Near-Miss Neighborhood (R@5/R@10 Success)': {'count': 0, 'classes': []},
            'Catastrophic Failure Neighborhood (0% through R@10)': {'count': 0, 'classes': []},
        }
        class_clii = {}

        for cls, m in class_metrics.items():
            neighborhood = assign_retrieval_neighborhood(m['recall@1'], m['recall@5'], m['recall@10'])
            neighborhood_summary[neighborhood]['count'] += 1
            neighborhood_summary[neighborhood]['classes'].append(cls)

            # Cross-Lingual Iconicity Index (CLII) = 1 / median_rank
            median_rank = max(float(m['median_rank']), 1.0)
            class_clii[cls] = 1.0 / median_rank

        # Rank classes by R@1 and CLII for inspection
        top_r1_classes = sorted(
            class_metrics.items(), key=lambda x: (x[1]['recall@1'], x[1]['recall@5']), reverse=True
        )[:10]
        top_clii_classes = sorted(class_clii.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"\n{'='*90}")
        print("Class-Level Structural Analysis")
        print(f"{'='*90}")
        print("Retrieval neighborhood counts:")
        for neighborhood_name, neighborhood_data in neighborhood_summary.items():
            print(f"  {neighborhood_name}: {neighborhood_data['count']} classes")

        if cmc_values:
            print("\nCMC checkpoints:")
            checkpoints = [1, 5, 10, max(1, len(text_labels) // 2), len(text_labels)]
            for ck in sorted(set(checkpoints)):
                idx = min(max(ck - 1, 0), len(cmc_values) - 1)
                print(f"  Rank-{ck:<4}: {cmc_values[idx]:.2%}")

        print("\nTop classes by R@1 (candidate cross-lingual iconicity winners):")
        for cls, m in top_r1_classes:
            print(
                f"  {cls:<35} R@1={m['recall@1']:.1%} "
                f"MedianR={m['median_rank']:.1f} CLII={class_clii[cls]:.4f}"
            )

        print("\nTop classes by CLII (CLII = 1 / Median Rank):")
        for cls, clii in top_clii_classes:
            print(f"  {cls:<35} CLII={clii:.4f}")

        # Optional plot export (saved alongside result CSV files)
        prompt_str_for_plot = prompt_type if prompt_type else ('custom' if text_template else 'raw')
        label_str_for_plot = 'categories' if use_categories else label_language
        plot_dir = output_dir if output_dir else 'runs/zero_shot'
        plot_paths = create_class_eval_plots(
            output_dir=plot_dir,
            split=split,
            prompt_name=prompt_str_for_plot,
            label_str=label_str_for_plot,
            class_metrics=class_metrics,
            cmc_ranks=cmc_ranks,
            cmc_values=cmc_values
        )

        if plot_paths:
            print("\nSaved class-eval plots:")
            for plot_name, plot_path in plot_paths.items():
                print(f"  {plot_name}: {plot_path}")

    print(f"{'='*75}\n")
    
    # Show some example predictions
    print("Example predictions (first 5):")
    for i in range(min(5, num_test)):
        top5_predictions = [text_labels[idx] for idx in ranked_indices[i][:5]]
        top5_scores = [similarities[i][idx] for idx in ranked_indices[i][:5]]
        
        match_symbol = "✓" if pose_labels[i] in top5_predictions else "✗"
        print(f"\n{match_symbol} Sample: {filenames[i]}")
        print(f"  Gold: {pose_labels[i]}")
        print(f"  Top-5 predictions:")
        for j, (pred, score) in enumerate(zip(top5_predictions, top5_scores), 1):
            marker = "***" if pred == pose_labels[i] else "   "
            print(f"    {j}. {pred:<30} (sim: {score:.4f}) {marker}")
    
    # Infer dataset name from embeddings directory
    dataset_name = 'unknown'
    if 'a3lis' in pose_embeddings_dir.lower():
        dataset_name = 'A3LIS'
    elif 'signit' in pose_embeddings_dir.lower():
        dataset_name = 'SignIT'
    
    results = {
        'recall@1': recall_1,
        'recall@5': recall_5,
        'recall@10': recall_10,
        'median_rank': median_rank,
        'num_test': num_test,
        'num_classes': len(text_labels),
        'prompt_type': prompt_type if prompt_type else 'raw',
        'text_template': text_template,
        'split': split,
        'label_language': label_language,
        'use_categories': use_categories,
        'model_name': model_name,
        'pose_embeddings_dir': pose_embeddings_dir,
        'class_eval': class_eval,
        'timestamp': datetime.now().isoformat()
    }

    if class_eval:
        results['class_metrics'] = {
            cls: {
                'total': stats['total'],
                'recall@1': stats['hit_1'] / stats['total'],
                'recall@5': stats['hit_5'] / stats['total'],
                'recall@10': stats['hit_10'] / stats['total'],
                'median_rank': (statistics.median(stats['ranks']) + 1) if stats['ranks'] else 0.0,
                'clii': 1.0 / max((statistics.median(stats['ranks']) + 1) if stats['ranks'] else 1.0, 1.0),
                'tier': assign_accuracy_tier(stats['hit_1'] / stats['total']),
                'retrieval_neighborhood': assign_retrieval_neighborhood(
                    stats['hit_1'] / stats['total'],
                    stats['hit_5'] / stats['total'],
                    stats['hit_10'] / stats['total']
                )
            }
            for cls, stats in sorted(class_stats.items())
        }

        # Persist CMC data and global CLII summary for downstream analysis
        cmc_ranks, cmc_values = compute_cmc_curve(ranks, len(text_labels))
        clii_values = [v['clii'] for v in results['class_metrics'].values()]

        results['cmc_curve'] = {
            'ranks': cmc_ranks,
            'accuracies': cmc_values
        }
        results['clii_summary'] = {
            'definition': 'CLII = 1 / Median Rank of Zero-Shot Match',
            'mean': float(np.mean(clii_values)) if clii_values else 0.0,
            'median': float(np.median(clii_values)) if clii_values else 0.0,
            'min': float(np.min(clii_values)) if clii_values else 0.0,
            'max': float(np.max(clii_values)) if clii_values else 0.0,
        }
    
    # Save results to CSV if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create descriptive filename
        prompt_str = prompt_type if prompt_type else 'custom' if text_template else 'raw'
        label_str = 'categories' if use_categories else label_language
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"zero_shot_{split}_{prompt_str}_{label_str}_{timestamp}.csv"
        
        results_file = output_path / filename
        scalar_results = {
            key: value for key, value in results.items()
            if key not in {'class_metrics', 'cmc_curve', 'clii_summary'}
        }
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(scalar_results.keys()))
            writer.writeheader()
            writer.writerow(scalar_results)
        
        print(f"\nResults saved to {results_file}")

        if class_eval and 'class_metrics' in results:
            class_metrics_file = output_path / f"zero_shot_{split}_{prompt_str}_{label_str}_{timestamp}_class_metrics.csv"
            with open(class_metrics_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'class_label', 'total', 'recall@1', 'recall@5', 'recall@10',
                        'median_rank', 'clii', 'tier', 'retrieval_neighborhood'
                    ]
                )
                writer.writeheader()
                for class_label, metrics in results['class_metrics'].items():
                    writer.writerow({'class_label': class_label, **metrics})
            print(f"Class metrics saved to {class_metrics_file}")

        if class_eval and 'cmc_curve' in results:
            cmc_file = output_path / f"zero_shot_{split}_{prompt_str}_{label_str}_{timestamp}_cmc_curve.csv"
            with open(cmc_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['rank', 'accuracy'])
                writer.writeheader()
                for rank, accuracy_value in zip(results['cmc_curve']['ranks'], results['cmc_curve']['accuracies']):
                    writer.writerow({'rank': rank, 'accuracy': accuracy_value})
            print(f"CMC data saved to {cmc_file}")

        if class_eval and 'clii_summary' in results:
            clii_file = output_path / f"zero_shot_{split}_{prompt_str}_{label_str}_{timestamp}_clii_summary.csv"
            with open(clii_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['definition', 'mean', 'median', 'min', 'max']
                )
                writer.writeheader()
                writer.writerow(results['clii_summary'])
            print(f"CLII summary saved to {clii_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation using precomputed embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Raw glosses (no template)
  python src/zero_shot.py \\
      --pose_embeddings_dir dataset/embeddings/a3lis_normalised \\
      --split test --label_language english

  # With language tag (paper standard)
  python src/zero_shot.py \\
      --pose_embeddings_dir dataset/embeddings/a3lis_normalised \\
      --split test --label_language english \\
      --prompt_type en_lis

  # Custom template
  python src/zero_shot.py \\
      --pose_embeddings_dir dataset/embeddings/a3lis_normalised \\
      --split test --label_language english \\
      --text_template "Italian sign language: {}"

Available prompt types:
  raw          - {} (no template, raw gloss)
  it_lis       - <it> <lis> {} (Italian language tag)
  en_lis       - <en> <lis> {} (paper standard)
"""
    )
    parser.add_argument('--pose_embeddings_dir', type=str, required=True,
                        help='Directory containing precomputed pose .npy embeddings')
    parser.add_argument('--text_embeddings', type=str,
                        help='Path to precomputed text embeddings .npy file (legacy mode)')
    parser.add_argument('--text_metadata', type=str,
                        help='Path to text metadata .json file (legacy mode)')
    parser.add_argument('--label_language', type=str, default='english',
                        choices=['italian', 'english'],
                        help='Language for A3LIS labels (default: english)')
    parser.add_argument('--use_categories', action='store_true',
                        help='Use macro categories instead of micro labels (A3LIS only)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to evaluate. Use all for full-dataset evaluation with signer variability breakdown.')
    parser.add_argument('--model_name', type=str, default='default',
                        choices=['default', 'asl_citizen', 'asl_finetune', 'suisse', 'a3lis_finetune'],
                        help='SignCLIP model for text embedding generation')
    parser.add_argument('--prompt_type', type=str,
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help='Predefined prompt template (see examples below)')
    parser.add_argument('--text_template', type=str,
                        help='Custom template string with {} placeholder (overrides --prompt_type)')
    parser.add_argument('--legacy_format', action='store_true',
                        help='Use legacy filename parsing format')
    parser.add_argument('--label_type', type=str, default='micro',
                        choices=['micro', 'macro'],
                        help='Label granularity (legacy format only)')
    parser.add_argument('--output_dir', type=str, default='runs/zero_shot',
                        help='Output directory for results CSV (default: runs/zero_shot)')
    parser.add_argument('--class_eval', action='store_true',
                        help='Print per-class retrieval metrics and save per-class metrics to CSV')
    
    args = parser.parse_args()
    
    results = evaluate_zero_shot(
        pose_embeddings_dir=args.pose_embeddings_dir,
        text_embeddings_path=args.text_embeddings,
        text_metadata_path=args.text_metadata,
        label_language=args.label_language,
        split=args.split,
        model_name=args.model_name,
        text_template=args.text_template,
        prompt_type=args.prompt_type,
        legacy_format=args.legacy_format,
        label_type=args.label_type,
        use_categories=args.use_categories,
        output_dir=args.output_dir,
        class_eval=args.class_eval
    )


if __name__ == '__main__':
    main()
