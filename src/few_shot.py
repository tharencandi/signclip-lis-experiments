"""
Few-shot evaluation for sign language recognition using SignCLIP.

Implements three few-shot approaches:
1. K-Nearest Neighbors (KNN): Nonparametric classification using K=num_classes
2. Linear Probe: Logistic regression trained on frozen embeddings
3. SVM (Advanced): Support Vector Machine with RBF kernel for non-linear classification

For A3LIS dataset:
- Standard mode: Uses predefined 70/30 split (7 train signers, 3 test signers)
- LOSO mode: 10-fold Leave-One-Signer-Out cross-validation

Usage:
    # Standard 70/30 split
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
        --method knn \
        --label_language english

    # LOSO cross-validation (10 folds, for fair comparison with Smart Head)
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
        --method knn \
        --label_language english \
        --loso
    
    # LOSO with specific fold
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
        --method svm \
        --loso \
        --fold 0
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import statistics
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Add project root to path for signclip imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_a3lis_embeddings(embedding_dir: Path, split: str, label_language: str = 'english', use_categories: bool = False):
    """
    Load A3LIS precomputed pose embeddings using embeddings_metadata.json.
    
    Args:
        embedding_dir: Directory containing .npy embedding files and metadata
        split: 'train' or 'test'
        label_language: 'italian' or 'english' for label selection
        use_categories: If True, use macro categories instead of micro labels
    
    Returns:
        Tuple of (embeddings_array, labels_list, filenames_list)
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
    
    # Process each embedding in metadata
    for item in tqdm(metadata['embeddings'], desc=f"Loading {split} embeddings"):
        # Filter by split
        if item['split'] != split:
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
    
    embeddings_array = np.array(embeddings) if embeddings else np.array([])
    
    return embeddings_array, labels, filenames


def load_all_embeddings_with_signers(embedding_dir: Path, label_language: str = 'english', use_categories: bool = False):
    """
    Load ALL A3LIS embeddings (train + test) with signer information for LOSO.
    
    Args:
        embedding_dir: Directory containing .npy embedding files and metadata
        label_language: 'italian' or 'english' for label selection
        use_categories: If True, use macro categories instead of micro labels
    
    Returns:
        Tuple of (embeddings_array, labels_list, signers_list, filenames_list)
    """
    metadata_path = embedding_dir / 'embeddings_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    embeddings = []
    labels = []
    signers = []
    filenames = []
    
    for item in tqdm(metadata['embeddings'], desc="Loading all embeddings"):
        # Get label
        if use_categories:
            label = item.get('category', item['label_italian'])
        elif label_language == 'italian':
            label = item['label_italian']
        else:
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
        signers.append(item['signer'])
        filenames.append(item['embedding_file'])
    
    embeddings_array = np.array(embeddings) if embeddings else np.array([])
    
    return embeddings_array, labels, signers, filenames


def evaluate_few_shot(
    pose_embeddings_dir: str,
    method: str = 'knn',
    label_language: str = 'english',
    seed: int = 42,
    use_categories: bool = False
):
    """
    Perform few-shot evaluation using KNN or Linear Probe.
    
    For A3LIS: Uses signer-independent split with ~7 training examples per class.
    
    Args:
        pose_embeddings_dir: Directory containing precomputed pose embeddings
        method: 'knn', 'linear_probe', or 'svm'
        label_language: 'italian' or 'english' for A3LIS labels
        seed: Random seed for reproducibility
        use_categories: Use macro categories instead of micro labels
    """
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Few-Shot Evaluation - {method.upper()}")
    print(f"{'='*60}\n")
    
    embedding_dir = Path(pose_embeddings_dir)
    
    # Load train embeddings
    print("Loading training set...")
    train_embeddings, train_labels, train_files = load_a3lis_embeddings(
        embedding_dir, 'train', label_language, use_categories
    )
    
    # Load test embeddings
    print("\nLoading test set...")
    test_embeddings, test_labels, test_files = load_a3lis_embeddings(
        embedding_dir, 'test', label_language, use_categories
    )
    
    # Get unique classes
    unique_train_classes = sorted(set(train_labels))
    unique_test_classes = sorted(set(test_labels))
    common_classes = sorted(set(unique_train_classes) & set(unique_test_classes))
    
    print(f"\nDataset Statistics:")
    print(f"  Label type: {'Categories (macro)' if use_categories else 'Signs (micro)'}")
    print(f"  Train samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")
    print(f"  Train classes: {len(unique_train_classes)}")
    print(f"  Test classes: {len(unique_test_classes)}")
    print(f"  Common classes: {len(common_classes)}")
    
    # Calculate examples per class
    from collections import Counter
    train_class_counts = Counter(train_labels)
    avg_examples_per_class = np.mean(list(train_class_counts.values()))
    print(f"  Avg examples per class (train): {avg_examples_per_class:.1f}")
    
    if len(common_classes) < len(unique_test_classes):
        print(f"\nWarning: {len(unique_test_classes) - len(common_classes)} test classes not in training set")
        print("Filtering test set to common classes...")
        # Filter test set to only common classes
        mask = np.array([label in common_classes for label in test_labels])
        test_embeddings = test_embeddings[mask]
        test_labels = [test_labels[i] for i in range(len(test_labels)) if mask[i]]
        test_files = [test_files[i] for i in range(len(test_files)) if mask[i]]
        print(f"  Filtered test samples: {len(test_labels)}")
    
    # Normalize embeddings for cosine similarity
    print("\nNormalizing embeddings...")
    train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    test_embeddings_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    
    # Train and evaluate based on method
    if method == 'knn':
        print(f"\nTraining KNN classifier (K={len(common_classes)})...")
        # K = number of classes (paper standard)
        clf = KNeighborsClassifier(n_neighbors=len(common_classes), metric='cosine')
        clf.fit(train_embeddings_norm, train_labels)
        
    elif method == 'linear_probe':
        print("\nTraining Logistic Regression (default scikit-learn settings)...")
        # Default scikit-learn LogisticRegression settings
        clf = LogisticRegression(random_state=seed, max_iter=1000)
        clf.fit(train_embeddings_norm, train_labels)
    
    elif method == 'svm':
        print("\nTraining SVM with RBF kernel (advanced non-linear classifier)...")
        # SVM with RBF kernel for non-linear decision boundaries
        # Using probability=True to enable predict_proba for ranking
        clf = SVC(kernel='rbf', random_state=seed, probability=True, max_iter=1000)
        clf.fit(train_embeddings_norm, train_labels)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'knn', 'linear_probe', or 'svm'")
    
    # Predict on test set
    print("\nEvaluating on test set...")
    predictions = clf.predict(test_embeddings_norm)
    
    # If method supports probability, get confidence scores
    if hasattr(clf, 'predict_proba'):
        probabilities = clf.predict_proba(test_embeddings_norm)
    else:
        # For KNN with cosine, get distances for ranking
        distances, indices = clf.kneighbors(test_embeddings_norm)
        # Use negative distance as score (closer = higher score)
        probabilities = None
    
    # Calculate metrics
    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    ranks = []
    
    # For ranking, we need class probabilities or distances
    class_labels = clf.classes_
    
    for i, gold_label in enumerate(tqdm(test_labels, desc="Evaluating")):
        pred_label = predictions[i]
        
        # Top-1 accuracy
        if pred_label == gold_label:
            hit_1 += 1
        
        # For top-5 and top-10, we need to rank all classes
        if probabilities is not None:
            # Sort by probability (descending)
            sorted_indices = np.argsort(-probabilities[i])
            ranked_labels = [class_labels[idx] for idx in sorted_indices]
        else:
            # For KNN without proba, we'll use a different approach
            # Compute similarity to all training examples and aggregate by class
            similarities = np.dot(train_embeddings_norm, test_embeddings_norm[i])
            class_scores = defaultdict(list)
            for j, label in enumerate(train_labels):
                class_scores[label].append(similarities[j])
            # Average similarity per class
            avg_scores = [(label, np.mean(scores)) for label, scores in class_scores.items()]
            avg_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_labels = [label for label, score in avg_scores]
        
        # Check top-5 and top-10
        if gold_label in ranked_labels[:5]:
            hit_5 += 1
        if gold_label in ranked_labels[:10]:
            hit_10 += 1
        
        # Get rank
        if gold_label in ranked_labels:
            rank = ranked_labels.index(gold_label)
            ranks.append(rank)
        else:
            ranks.append(len(ranked_labels))
    
    # Calculate final metrics
    num_test = len(test_labels)
    accuracy = hit_1 / num_test
    recall_5 = hit_5 / num_test
    recall_10 = hit_10 / num_test
    median_rank = statistics.median(ranks) + 1 if ranks else float('inf')
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results - {method.upper()}")
    print(f"{'='*60}")
    print(f"Method: {method}")
    print(f"Train samples: {len(train_labels)}")
    print(f"Test samples: {num_test}")
    print(f"Classes: {len(common_classes)}")
    print(f"Avg examples per class: {avg_examples_per_class:.1f}")
    if method == 'knn':
        print(f"K (neighbors): {len(common_classes)}")
    elif method == 'svm':
        print(f"Kernel: RBF (Radial Basis Function)")
    print(f"Seed: {seed}")
    print(f"\nRetrieval Metrics:")
    print(f"  R@1↑:             {accuracy:>7.2%}  ({hit_1:>5}/{num_test})")
    print(f"  R@5↑:             {recall_5:>7.2%}  ({hit_5:>5}/{num_test})")
    print(f"  R@10↑:            {recall_10:>7.2%}  ({hit_10:>5}/{num_test})")
    print(f"  MedianR↓:         {median_rank:>7.1f}")
    print(f"\nAccuracy:")
    print(f"  Top-1:            {accuracy:>7.2%}  ({hit_1:>5}/{num_test})")
    print(f"{'='*60}\n")
    
    # Show some example predictions
    print("Example predictions (first 5):")
    for i in range(min(5, num_test)):
        if probabilities is not None:
            sorted_indices = np.argsort(-probabilities[i])
            top5_labels = [class_labels[idx] for idx in sorted_indices[:5]]
            top5_scores = [probabilities[i][idx] for idx in sorted_indices[:5]]
        else:
            # Use similarity-based ranking
            similarities = np.dot(train_embeddings_norm, test_embeddings_norm[i])
            class_scores = defaultdict(list)
            for j, label in enumerate(train_labels):
                class_scores[label].append(similarities[j])
            avg_scores = [(label, np.mean(scores)) for label, scores in class_scores.items()]
            avg_scores.sort(key=lambda x: x[1], reverse=True)
            top5_labels = [label for label, score in avg_scores[:5]]
            top5_scores = [score for label, score in avg_scores[:5]]
        
        match_symbol = "✓" if predictions[i] == test_labels[i] else "✗"
        print(f"\n{match_symbol} Sample: {test_files[i]}")
        print(f"  Gold: {test_labels[i]}")
        print(f"  Predicted: {predictions[i]}")
        print(f"  Top-5:")
        for j, (label, score) in enumerate(zip(top5_labels, top5_scores), 1):
            marker = "***" if label == test_labels[i] else "   "
            print(f"    {j}. {label:<30} (score: {score:.4f}) {marker}")
    
    return {
        'method': method,
        'accuracy': accuracy,
        'recall@5': recall_5,
        'recall@10': recall_10,
        'median_rank': median_rank,
        'num_train': len(train_labels),
        'num_test': num_test,
        'num_classes': len(common_classes),
        'avg_examples_per_class': avg_examples_per_class
    }


def evaluate_single_fold(
    fold: int,
    train_embeddings: np.ndarray,
    train_labels: list,
    test_embeddings: np.ndarray,
    test_labels: list,
    test_signer: str,
    method: str,
    seed: int
):
    """Evaluate a single LOSO fold."""
    
    unique_classes = sorted(set(train_labels))
    num_classes = len(unique_classes)
    
    # Normalize embeddings
    train_embeddings_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    test_embeddings_norm = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    
    # Train classifier
    if method == 'knn':
        clf = KNeighborsClassifier(n_neighbors=num_classes, metric='cosine')
        clf.fit(train_embeddings_norm, train_labels)
    elif method == 'linear_probe':
        clf = LogisticRegression(random_state=seed, max_iter=1000)
        clf.fit(train_embeddings_norm, train_labels)
    elif method == 'svm':
        clf = SVC(kernel='rbf', random_state=seed, probability=True, max_iter=1000)
        clf.fit(train_embeddings_norm, train_labels)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Predict
    predictions = clf.predict(test_embeddings_norm)
    
    # Get ranked predictions
    if hasattr(clf, 'predict_proba'):
        probabilities = clf.predict_proba(test_embeddings_norm)
        class_labels = clf.classes_
    else:
        probabilities = None
        class_labels = clf.classes_
    
    # Calculate metrics
    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    ranks = []
    
    for i, gold_label in enumerate(test_labels):
        if predictions[i] == gold_label:
            hit_1 += 1
        
        # Get ranked predictions
        if probabilities is not None:
            sorted_indices = np.argsort(-probabilities[i])
            ranked_labels = [class_labels[idx] for idx in sorted_indices]
        else:
            # KNN: aggregate similarities by class
            similarities = np.dot(train_embeddings_norm, test_embeddings_norm[i])
            class_scores = defaultdict(list)
            for j, label in enumerate(train_labels):
                class_scores[label].append(similarities[j])
            avg_scores = [(label, np.mean(scores)) for label, scores in class_scores.items()]
            avg_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_labels = [label for label, score in avg_scores]
        
        if gold_label in ranked_labels[:5]:
            hit_5 += 1
        if gold_label in ranked_labels[:10]:
            hit_10 += 1
        
        if gold_label in ranked_labels:
            rank = ranked_labels.index(gold_label)
            ranks.append(rank)
        else:
            ranks.append(len(ranked_labels))
    
    num_test = len(test_labels)
    r_at_1 = hit_1 / num_test
    r_at_5 = hit_5 / num_test
    r_at_10 = hit_10 / num_test
    median_rank = statistics.median(ranks) + 1 if ranks else float('inf')
    
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1} Results (Test Signer: {test_signer})")
    print(f"{'='*60}")
    print(f"  Train samples: {len(train_labels)}")
    print(f"  Test samples:  {num_test}")
    print(f"  Classes:       {num_classes}")
    print(f"\n  R@1↑:        {r_at_1:>7.2%}  ({hit_1:>5}/{num_test})")
    print(f"  R@5↑:        {r_at_5:>7.2%}  ({hit_5:>5}/{num_test})")
    print(f"  R@10↑:       {r_at_10:>7.2%}  ({hit_10:>5}/{num_test})")
    print(f"  MedianR↓:    {median_rank:>7.1f}")
    print(f"{'='*60}\n")
    
    return {
        'fold': fold,
        'test_signer': test_signer,
        'r@1': r_at_1,
        'r@5': r_at_5,
        'r@10': r_at_10,
        'median_rank': median_rank,
        'hit_1': hit_1,
        'hit_5': hit_5,
        'hit_10': hit_10,
        'num_test': num_test,
        'num_train': len(train_labels),
        'num_classes': num_classes
    }


def run_loso_cross_validation(
    pose_embeddings_dir: str,
    method: str,
    label_language: str,
    use_categories: bool,
    seed: int,
    fold: int = None,
    output_dir: str = None
):
    """
    Run Leave-One-Signer-Out cross-validation for fair comparison with Smart Head.
    
    Args:
        pose_embeddings_dir: Directory containing precomputed embeddings
        method: 'knn', 'linear_probe', or 'svm'
        label_language: 'italian' or 'english'
        use_categories: Use macro categories instead of micro labels
        seed: Random seed
        fold: Specific fold to run (0-9), or None for all folds
        output_dir: Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"LOSO Cross-Validation - {method.upper()}")
    print(f"{'='*60}\n")
    
    embedding_dir = Path(pose_embeddings_dir)
    
    # Load all data with signer information
    embeddings, labels, signers, filenames = load_all_embeddings_with_signers(
        embedding_dir, label_language, use_categories
    )
    
    unique_signers = sorted(set(signers))
    print(f"Total samples: {len(embeddings)}")
    print(f"Unique signers: {unique_signers}")
    print(f"Number of folds: {len(unique_signers)}")
    
    if len(unique_signers) != 10:
        print(f"WARNING: Expected 10 signers for A3LIS, found {len(unique_signers)}")
    
    # Run LOSO cross-validation
    all_results = []
    
    for fold_idx, test_signer in enumerate(unique_signers):
        # Skip if only running specific fold
        if fold is not None and fold_idx != fold:
            continue
        
        # Split by signer
        train_mask = np.array([s != test_signer for s in signers])
        test_mask = np.array([s == test_signer for s in signers])
        
        train_embeddings = embeddings[train_mask]
        train_labels = [labels[i] for i in range(len(labels)) if train_mask[i]]
        
        test_embeddings = embeddings[test_mask]
        test_labels = [labels[i] for i in range(len(labels)) if test_mask[i]]
        
        # Evaluate this fold
        fold_result = evaluate_single_fold(
            fold_idx, train_embeddings, train_labels,
            test_embeddings, test_labels, test_signer,
            method, seed
        )
        
        all_results.append(fold_result)
    
    # Aggregate results
    if len(all_results) == len(unique_signers):
        print(f"\n{'='*60}")
        print(f"LOSO Cross-Validation Results (All 10 Folds)")
        print(f"{'='*60}\n")
        
        # Per-fold results
        for result in all_results:
            print(f"Fold {result['fold']+1:>2} ({result['test_signer']:>4}): "
                  f"R@1={result['r@1']:.4f}, R@5={result['r@5']:.4f}, "
                  f"R@10={result['r@10']:.4f}, MedianR={result['median_rank']:.1f}")
        
        # Average results
        avg_r1 = np.mean([r['r@1'] for r in all_results])
        avg_r5 = np.mean([r['r@5'] for r in all_results])
        avg_r10 = np.mean([r['r@10'] for r in all_results])
        avg_median_rank = np.mean([r['median_rank'] for r in all_results])
        
        std_r1 = np.std([r['r@1'] for r in all_results])
        std_r5 = np.std([r['r@5'] for r in all_results])
        std_r10 = np.std([r['r@10'] for r in all_results])
        std_median_rank = np.std([r['median_rank'] for r in all_results])
        
        print(f"\n{'='*60}")
        print(f"Average Across All Folds")
        print(f"{'='*60}")
        print(f"  R@1↑:        {avg_r1:.4f} ± {std_r1:.4f}")
        print(f"  R@5↑:        {avg_r5:.4f} ± {std_r5:.4f}")
        print(f"  R@10↑:       {avg_r10:.4f} ± {std_r10:.4f}")
        print(f"  MedianR↓:    {avg_median_rank:.2f} ± {std_median_rank:.2f}")
        print(f"{'='*60}\n")
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / f'loso_{method}_results.json'
            with open(results_file, 'w') as f:
                json.dump({
                    'method': method,
                    'pose_embeddings_dir': pose_embeddings_dir,
                    'all_folds': all_results,
                    'average': {
                        'r@1': {'mean': avg_r1, 'std': std_r1},
                        'r@5': {'mean': avg_r5, 'std': std_r5},
                        'r@10': {'mean': avg_r10, 'std': std_r10},
                        'median_rank': {'mean': avg_median_rank, 'std': std_median_rank}
                    },
                    'hyperparameters': {
                        'method': method,
                        'label_language': label_language,
                        'use_categories': use_categories,
                        'seed': seed
                    }
                }, f, indent=2)
            
            print(f"Results saved to {results_file}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot sign language recognition evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 70/30 split
  python src/few_shot.py \\
      --pose_embeddings_dir dataset/embeddings/a3lis_normalised \\
      --method knn \\
      --label_language english

  # LOSO cross-validation (10 folds, fair comparison with Smart Head)
  python src/few_shot.py \\
      --pose_embeddings_dir dataset/embeddings/a3lis_normalised \\
      --method knn \\
      --loso

  # LOSO with specific fold
  python src/few_shot.py \\
      --pose_embeddings_dir dataset/embeddings/a3lis_normalised \\
      --method svm \\
      --loso \\
      --fold 0

Methods:
  knn          - K-Nearest Neighbors (K=num_classes, cosine similarity)
  linear_probe - Logistic Regression (default scikit-learn settings)
  svm          - Support Vector Machine with RBF kernel (advanced non-linear)
"""
    )
    parser.add_argument('--pose_embeddings_dir', type=str, required=True,
                        help='Directory containing precomputed pose .npy embeddings')
    parser.add_argument('--method', type=str, required=True,
                        choices=['knn', 'linear_probe', 'svm'],
                        help='Few-shot method to use')
    parser.add_argument('--label_language', type=str, default='english',
                        choices=['italian', 'english'],
                        help='Language for A3LIS labels (default: english)')
    parser.add_argument('--use_categories', action='store_true',
                        help='Use macro categories instead of micro labels')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # LOSO options
    parser.add_argument('--loso', action='store_true',
                        help='Use Leave-One-Signer-Out cross-validation (10 folds)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Run specific fold (0-9) in LOSO mode, or None for all folds')
    parser.add_argument('--output_dir', type=str, default='runs/few_shot',
                        help='Output directory for LOSO results')
    
    args = parser.parse_args()
    
    if args.loso:
        # Run LOSO cross-validation
        run_loso_cross_validation(
            pose_embeddings_dir=args.pose_embeddings_dir,
            method=args.method,
            label_language=args.label_language,
            use_categories=args.use_categories,
            seed=args.seed,
            fold=args.fold,
            output_dir=args.output_dir
        )
    else:
        # Run standard 70/30 evaluation
        evaluate_few_shot(
            pose_embeddings_dir=args.pose_embeddings_dir,
            method=args.method,
            label_language=args.label_language,
            seed=args.seed,
            use_categories=args.use_categories
        )


if __name__ == '__main__':
    main()
