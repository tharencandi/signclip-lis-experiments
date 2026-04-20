"""
Few-shot evaluation for sign language recognition using SignCLIP.

Implements three few-shot approaches:
1. K-Nearest Neighbors (KNN): Nonparametric classification using K=num_classes
2. Linear Probe: Logistic regression trained on frozen embeddings
3. SVM (Advanced): Support Vector Machine with RBF kernel for non-linear classification

For A3LIS dataset:
- Uses signer-independent split (7 train signers, 3 test signers)
- Each class has ~7 examples in training (one per train signer)
- "Few-shot" means limited examples per class (signer-based constraint)

Usage:
    # KNN (paper standard, K=num_classes)
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
        --method knn \
        --label_language english

    # Linear Probe (logistic regression)
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
        --method linear_probe \
        --label_language english
    
    # SVM with RBF kernel (advanced)
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
        --method svm \
        --label_language english
    
    # Use macro categories instead of micro labels
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
        --method svm \
        --use_categories
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


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot sign language recognition evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # KNN with K=num_classes (paper standard)
  python src/few_shot.py \\
      --pose_embeddings_dir dataset/embeddings/a3lis_normalised \\
      --method knn \\
      --label_language english

  # Logistic Regression (Linear Probe)
  python src/few_shot.py \\
      --pose_embeddings_dir dataset/embeddings/a3lis_normalised \\
      --method linear_probe \\
      --label_language english
  
  # SVM with RBF kernel (advanced)
  python src/few_shot.py \\
      --pose_embeddings_dir dataset/embeddings/a3lis_normalised \\
      --method svm \\
      --label_language english

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
    
    args = parser.parse_args()
    
    evaluate_few_shot(
        pose_embeddings_dir=args.pose_embeddings_dir,
        method=args.method,
        label_language=args.label_language,
        seed=args.seed,
        use_categories=args.use_categories
    )


if __name__ == '__main__':
    main()
