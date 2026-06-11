"""
Few-shot evaluation for sign language recognition using SignCLIP.

Implements three few-shot approaches:
1. K-Nearest Neighbors (KNN): Nonparametric classification using K=num_classes
2. Linear Probe: Logistic regression trained on frozen embeddings
3. SVM (Advanced): Support Vector Machine with RBF kernel for non-linear classification
4. MLP (Standard): Scikit-learn MLP classifier with SmartHead-like hidden widths

For A3LIS dataset:
- Standard mode: Uses predefined 70/30 split (7 train signers, 3 test signers)
- LOSO mode: 10-fold Leave-One-Signer-Out cross-validation

Usage:
    # Standard 70/30 split
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2 \
        --method knn \
        --label_language english

    # LOSO cross-validation (10 folds, for fair comparison with Smart Head)
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2 \
        --method knn \
        --label_language english \
        --loso
    
    # LOSO with specific fold
    python src/few_shot.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2 \
        --method svm \
        --loso \
        --fold 0
"""

import argparse
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import statistics
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
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
    categories = []
    
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
        categories.append(item.get('category', 'unknown'))
    
    embeddings_array = np.array(embeddings) if embeddings else np.array([])
    
    return embeddings_array, labels, filenames, categories


def load_all_embeddings_with_signers(embedding_dir: Path, label_language: str = 'english', use_categories: bool = False,
                                     exclude_splits: Optional[list] = None):
    """
    Load ALL A3LIS embeddings (train + test) with signer information for LOSO.
    
    Args:
        embedding_dir: Directory containing .npy embedding files and metadata
        label_language: 'italian' or 'english' for label selection
        use_categories: If True, use macro categories instead of micro labels
        exclude_splits: Optional list of split values to exclude (e.g. ['val', 'unknown']).
                        Use this to prevent the val signer from appearing in any LOSO
                        training fold, keeping evaluation signer-independent.
    
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
        # Exclude items from specified splits (e.g. val, unknown)
        if exclude_splits and item.get('split', 'unknown') in exclude_splits:
            continue

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
    use_categories: bool = False,
    train_split: str = 'train',
    eval_split: str = 'test',
    num_shots: Optional[int] = None,
    number_shot: int = 7,
    output_dir: Optional[str] = None,
    max_iter: int = 100,
    class_eval: bool = False,
    extra_metrics: bool = False,
):
    """
    Perform few-shot evaluation using KNN, linear probe, SVM, MLP, or prototypical.
    
    For A3LIS: Uses signer-independent split with ~7 training examples per class.
    
    Args:
        pose_embeddings_dir: Directory containing precomputed pose embeddings
        method: 'knn', 'linear_probe', 'svm', 'mlp', or 'prototypical'
        label_language: 'italian' or 'english' for A3LIS labels
        seed: Random seed for reproducibility
        use_categories: Use macro categories instead of micro labels
        train_split: 'train' (7 train signers only) or 'train_val' (adds val signer mrla).
                     Use 'train_val' only when NOT comparing against the fine-tuned model,
                     because fine-tuning used the val signer for checkpoint selection.
        eval_split: 'val' (for hyperparameter selection) or 'test' (for final reporting).
                    Always use 'val' when tuning k, method, or other hyperparameters;
                    switch to 'test' only for final numbers cited in the paper.
        num_shots: If set, limit training examples to this many per class (e.g. 1 or 5
                   for 1-shot / 5-shot evaluation). Default: use all available examples.
        number_shot: Support examples per class for prototypical method (default: 7, max: 7).
        output_dir: Directory to save results CSV. If None, results are not saved.
        class_eval: Print per-class retrieval metrics and save per-class metrics to CSV."""
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Few-Shot Evaluation - {method.upper()}")
    print(f"{'='*60}\n")
    
    embedding_dir = Path(pose_embeddings_dir)
    
    # Load train embeddings (optionally combined with val)
    if train_split == 'train_val':
        print("Loading training set (train + val signers)...")
        train_emb_t, train_lab_t, train_files_t, train_cat_t = load_a3lis_embeddings(
            embedding_dir, 'train', label_language, use_categories
        )
        print("Loading val set (adding to training)...")
        train_emb_v, train_lab_v, train_files_v, train_cat_v = load_a3lis_embeddings(
            embedding_dir, 'val', label_language, use_categories
        )
        train_embeddings = np.concatenate([train_emb_t, train_emb_v], axis=0)
        train_labels = train_lab_t + train_lab_v
        train_files = train_files_t + train_files_v
        train_categories = train_cat_t + train_cat_v
    else:
        print("Loading training set...")
        train_embeddings, train_labels, train_files, train_categories = load_a3lis_embeddings(
            embedding_dir, 'train', label_language, use_categories
        )

    # For prototypical, num_shots directly sets the number of support samples per class.
    if method == 'prototypical' and num_shots is not None and num_shots > 0:
        number_shot = num_shots

    # Shot limiting: keep at most num_shots examples per class (sklearn methods only)
    if method != 'prototypical' and num_shots is not None and num_shots > 0:
        class_indices = defaultdict(list)
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)
        rng = np.random.default_rng(seed)
        selected = []
        for label in sorted(class_indices.keys()):
            idxs = class_indices[label]
            k = min(num_shots, len(idxs))
            chosen = rng.choice(len(idxs), size=k, replace=False)
            selected.extend([idxs[j] for j in chosen])
        selected = sorted(selected)
        train_embeddings = train_embeddings[selected]
        train_labels = [train_labels[i] for i in selected]
        train_files = [train_files[i] for i in selected]
        print(f"  Shot-limited to {num_shots} example(s)/class → {len(train_labels)} train samples total")

    # Load eval embeddings (val for hyperparam selection, test for final results)
    print(f"\nLoading {eval_split} set...")
    test_embeddings, test_labels, test_files, test_categories = load_a3lis_embeddings(
        embedding_dir, eval_split, label_language, use_categories
    )
    
    # Get unique classes
    unique_train_classes = sorted(set(train_labels))
    unique_test_classes = sorted(set(test_labels))
    common_classes = sorted(set(unique_train_classes) & set(unique_test_classes))
    
    print(f"\nDataset Statistics:")
    print(f"  Label type: {'Categories (macro)' if use_categories else 'Signs (micro)'}")
    print(f"  Train samples: {len(train_labels)}")
    print(f"  {eval_split.capitalize()} samples: {len(test_labels)}")
    print(f"  Train classes: {len(unique_train_classes)}")
    print(f"  {eval_split.capitalize()} classes: {len(unique_test_classes)}")
    print(f"  Common classes: {len(common_classes)}")
    
    # Calculate examples per class
    from collections import Counter
    train_class_counts = Counter(train_labels)
    avg_examples_per_class = np.mean(list(train_class_counts.values()))
    print(f"  Avg examples per class (train): {avg_examples_per_class:.1f}")
    
    if method != 'prototypical' and len(common_classes) < len(unique_test_classes):
        print(f"\nWarning: {len(unique_test_classes) - len(common_classes)} test classes not in training set")
        print("Filtering test set to common classes...")
        # Filter test set to only common classes
        mask = np.array([label in common_classes for label in test_labels])
        test_embeddings = test_embeddings[mask]
        test_labels = [test_labels[i] for i in range(len(test_labels)) if mask[i]]
        test_files = [test_files[i] for i in range(len(test_files)) if mask[i]]
        test_categories = [test_categories[i] for i in range(len(test_categories)) if mask[i]]
        print(f"  Filtered test samples: {len(test_labels)}")
    elif method == 'prototypical' and len(common_classes) < len(unique_test_classes):
        print(
            f"\nOpen-world prototypical evaluation: keeping all test classes "
            f"(including {len(unique_test_classes) - len(common_classes)} unseen-in-support classes)."
        )
    
    # Train and evaluate based on method
    clf = None
    k = None
    mlp_label_encoder = None
    class_labels = None
    probabilities = None
    predictions = None
    if method == 'knn':
        # K = num_shots when shot-limited (K=1 for 1-shot, K=5 for 5-shot).
        # K = num_classes when using the full training set (paper standard).
        k = num_shots if num_shots is not None else len(common_classes)
        k = min(k, len(train_labels))  # guard against edge cases
      
        #use default metric - euclidean distance for KNN
        clf = Pipeline(
            steps = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))]
        )
        clf.fit(train_embeddings, train_labels)


        
    elif method == 'linear_probe':
        print("\nTraining Logistic Regression (default scikit-learn settings)...")
        # Default scikit-learn LogisticRegression settings
        clf = LogisticRegression(verbose= True,random_state=seed, max_iter=max_iter)
        clf.fit(train_embeddings, train_labels)
    
    elif method == 'svm':
        print("\nTraining SVM with RBF kernel (advanced non-linear classifier)...")
        # SVM with RBF kernel for non-linear decision boundaries
        # Using probability=True to enable predict_proba for ranking
        clf = SVC(kernel='rbf', random_state=seed, probability=True, max_iter=max_iter)
        clf.fit(train_embeddings, train_labels)

    elif method == 'mlp':
        input_dim = train_embeddings.shape[1]
        hidden_sizes = (input_dim * 2,)
        print(
            f"\nTraining MLP classifier (hidden layers: {hidden_sizes}, "
            f"internal validation split: disabled)..."
        )
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation='relu',
            solver='adam',
            early_stopping=False,
            n_iter_no_change=10,
            max_iter=max_iter,
            random_state=seed,
        )
        # Keep integer targets for MLP compatibility across sklearn versions.
        mlp_label_encoder = LabelEncoder()
        train_labels_encoded = mlp_label_encoder.fit_transform(train_labels)
        clf.fit(train_embeddings, train_labels_encoded)

    elif method == 'prototypical':
        max_support = max(train_class_counts.values())
        if number_shot > max_support:
            print(f"  number_shot={number_shot} exceeds max examples per class ({max_support}); capping to {max_support}.")
            number_shot = max_support

        print(
            f"\nTraining Prototypical classifier (class-average supports, number_shot={number_shot}, "
            "no scaling, euclidean scoring)..."
        )

        class_indices = defaultdict(list)
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)

        valid_classes = sorted(class_indices.keys())

        if not valid_classes:
            raise ValueError(
                f"No classes have enough support samples for number_shot={number_shot}."
            )

        rng = np.random.default_rng(seed)
        prototype_labels = []
        prototype_vectors = []
        for label in valid_classes:
            sampled_indices = rng.choice(class_indices[label], size=min(number_shot, len(class_indices[label])), replace=False)
            prototype = np.mean(train_embeddings[sampled_indices], axis=0)
            prototype_labels.append(label)
            prototype_vectors.append(prototype)

        prototype_matrix = np.vstack(prototype_vectors)
        class_labels = np.array(prototype_labels, dtype=object)

        # Open-world scoring: Euclidean distance query-to-prototype (lower is better)
        distances = np.linalg.norm(test_embeddings[:, np.newaxis, :] - prototype_matrix[np.newaxis, :, :], axis=2)
        probabilities = distances
        sorted_indices = np.argsort(probabilities, axis=1)
        predictions = np.array([class_labels[row[0]] for row in sorted_indices], dtype=object)

        print(f"  Built prototypes: {len(class_labels)} classes")

    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'knn', 'linear_probe', 'svm', 'mlp', or 'prototypical'"
        )
    
    # Predict on test set
    print("\nEvaluating on test set...")
    if method != 'prototypical':
        if clf is None:
            raise RuntimeError("Classifier was not initialized for non-prototypical method.")
        predictions = clf.predict(test_embeddings)
        if method == 'mlp' and mlp_label_encoder is not None:
            predictions = mlp_label_encoder.inverse_transform(predictions)

        class_labels = clf.classes_
        if method == 'mlp' and mlp_label_encoder is not None:
            class_labels = mlp_label_encoder.inverse_transform(class_labels)

        # If method supports probability, get confidence scores
        if method == 'svm':
            # Use raw geometric distances for SVM ranking.
            probabilities = getattr(clf, 'decision_function')(test_embeddings)
        elif method in {'knn', 'linear_probe', 'mlp'}:
            probabilities = clf.predict_proba(test_embeddings)
 

    # Calculate metrics
    if predictions is None:
        raise RuntimeError("Predictions were not computed.")

    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    ranks = []
    
    category_stats = {}
    class_stats = {}
    # For ranking, we need class probabilities or distances
    if class_labels is None:
        raise RuntimeError("Class labels were not computed.")
    
    for i, gold_label in enumerate(tqdm(test_labels, desc="Evaluating")):
        pred_label = predictions[i]
        cat = test_categories[i]

        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'hit_1': 0, 'hit_5': 0, 'hit_10': 0, 'ranks': []}
        category_stats[cat]['total'] += 1
        if class_eval and gold_label not in class_stats:
            class_stats[gold_label] = {'total': 0, 'hit_1': 0, 'hit_5': 0, 'hit_10': 0, 'ranks': []}
        if class_eval:
            class_stats[gold_label]['total'] += 1
        
        # Top-1 accuracy
        if pred_label == gold_label:
            hit_1 += 1
            category_stats[cat]['hit_1'] += 1
            if class_eval:
                class_stats[gold_label]['hit_1'] += 1
        
        # For top-5 and top-10, we need to rank all classes
        if probabilities is not None:
            if method == 'prototypical':
                # For distance scores, smaller is better.
                sorted_indices = np.argsort(probabilities[i])
            else:
                # For classifier scores/probabilities, larger is better.
                sorted_indices = np.argsort(-probabilities[i])
            ranked_labels = [class_labels[idx] for idx in sorted_indices]
        else:
            # For KNN without proba, we'll use a different approach
            # Compute similarity to all training examples and aggregate by class
            similarities = np.dot(train_embeddings, test_embeddings[i])
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
            category_stats[cat]['hit_5'] += 1
            if class_eval:
                class_stats[gold_label]['hit_5'] += 1
        if gold_label in ranked_labels[:10]:
            hit_10 += 1
            category_stats[cat]['hit_10'] += 1
            if class_eval:
                class_stats[gold_label]['hit_10'] += 1
        
        # Get rank
        if gold_label in ranked_labels:
            rank = ranked_labels.index(gold_label)
            ranks.append(rank)
            category_stats[cat]['ranks'].append(rank)
            if class_eval:
                class_stats[gold_label]['ranks'].append(rank)
        else:
            ranks.append(len(ranked_labels))
            category_stats[cat]['ranks'].append(len(ranked_labels))
            if class_eval:
                class_stats[gold_label]['ranks'].append(len(ranked_labels))
    
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
    print(f"Eval split: {eval_split}")
    print(f"Train samples: {len(train_labels)}")
    print(f"Eval samples: {num_test}")
    print(f"Classes: {len(class_labels) if class_labels is not None else len(common_classes)}")
    print(f"Avg examples per class: {avg_examples_per_class:.1f}")
    if method == 'knn' and k is not None:
        print(f"K (neighbors): {k}")
    elif method == 'svm':
        print(f"Kernel: RBF (Radial Basis Function)")
    elif method == 'prototypical':
        print(f"number_shot (prototype supports): {number_shot}")
        print(f"Prototype classes: {len(class_labels)}")
        print("Scoring: Euclidean distance to class prototypes (lower is better)")
    print(f"Seed: {seed}")
    print(f"\nRetrieval Metrics:")
    print(f"  R@1↑:             {accuracy:>7.2%}  ({hit_1:>5}/{num_test})")
    print(f"  R@5↑:             {recall_5:>7.2%}  ({hit_5:>5}/{num_test})")
    print(f"  R@10↑:            {recall_10:>7.2%}  ({hit_10:>5}/{num_test})")
    print(f"  MedianR↓:         {median_rank:>7.1f}")
    print(f"\nAccuracy:")
    print(f"  Top-1:            {accuracy:>7.2%}  ({hit_1:>5}/{num_test})")

    extra_metrics_results = None
    if extra_metrics:
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            test_labels, predictions, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            test_labels, predictions, average='weighted', zero_division=0
        )
        extra_metrics_results = {
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
        }

    if not use_categories:
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
            cls_display = str(cls)[:35]
            print(f"  {cls_display:<35} | {tot:<5} | {r1:>6.1%} | {r5:>6.1%} | {r10:>6.1%} | {med_r:>7.1f}")

    if extra_metrics and extra_metrics_results is not None:
        print(f"\nClassification Metrics (Top-1 Predictions):")
        print(f"  Accuracy↑:        {accuracy:>7.2%}")
        print(f"  Precision (class-mean/macro):   {extra_metrics_results['precision_macro']:>7.4f}")
        print(f"  Recall (class-mean/macro):      {extra_metrics_results['recall_macro']:>7.4f}")
        print(f"  F1 (class-mean/macro):          {extra_metrics_results['f1_macro']:>7.4f}")
        print(f"  Precision (weighted):{extra_metrics_results['precision_weighted']:>7.4f}")
        print(f"  Recall (weighted):   {extra_metrics_results['recall_weighted']:>7.4f}")
        print(f"  F1 (weighted):       {extra_metrics_results['f1_weighted']:>7.4f}")

    print(f"{'='*60}\n")
    
    # Show some example predictions
    print("Example predictions (first 5):")
    for i in range(min(5, num_test)):
        if probabilities is not None:
            if method == 'prototypical':
                sorted_indices = np.argsort(probabilities[i])
            else:
                sorted_indices = np.argsort(-probabilities[i])
            top5_labels = [class_labels[idx] for idx in sorted_indices[:5]]
            top5_scores = [probabilities[i][idx] for idx in sorted_indices[:5]]
        else:
            # Use similarity-based ranking
            similarities = np.dot(train_embeddings, test_embeddings[i])
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
            if method == 'prototypical':
                print(f"    {j}. {label:<30} (distance: {score:.4f}) {marker}")
            else:
                print(f"    {j}. {label:<30} (score: {score:.4f}) {marker}")
    
    results = {
        'method': method,
        'recall@1': accuracy,
        'accuracy': accuracy,
        'recall@5': recall_5,
        'recall@10': recall_10,
        'median_rank': median_rank,
        'num_train': len(train_labels),
        'num_test': num_test,
        'num_classes': len(class_labels) if class_labels is not None else len(common_classes),
        'avg_examples_per_class': avg_examples_per_class,
        'num_shots': num_shots,
        'number_shot': number_shot,
        'eval_split': eval_split,
        'train_split': train_split,
        'pose_embeddings_dir': str(pose_embeddings_dir),
        'class_eval': class_eval,
        'extra_metrics': extra_metrics,
    }

    if extra_metrics and extra_metrics_results is not None:
        results.update(extra_metrics_results)

    if class_eval:
        results['class_metrics'] = {
            cls: {
                'total': stats['total'],
                'recall@1': stats['hit_1'] / stats['total'],
                'recall@5': stats['hit_5'] / stats['total'],
                'recall@10': stats['hit_10'] / stats['total'],
                'median_rank': (statistics.median(stats['ranks']) + 1) if stats['ranks'] else 0.0,
            }
            for cls, stats in sorted(class_stats.items())
        }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if method == 'prototypical':
            shots_str = f"{number_shot}shot"
        else:
            shots_str = f"{num_shots}shot" if num_shots else "allshot"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"few_shot_{method}_{shots_str}_{train_split}_{eval_split}_{timestamp}.csv"
        results_file = output_path / filename
        scalar_results = {
            key: value for key, value in results.items()
            if key != 'class_metrics'
        }
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(scalar_results.keys()))
            writer.writeheader()
            writer.writerow(scalar_results)
        print(f"\nResults saved to {results_file}")

        if class_eval and 'class_metrics' in results:
            class_results_file = output_path / f"few_shot_{method}_{shots_str}_{train_split}_{eval_split}_{timestamp}_class_metrics.csv"
            with open(class_results_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['class_label', 'total', 'recall@1', 'recall@5', 'recall@10', 'median_rank']
                )
                writer.writeheader()
                for class_label, metrics in results['class_metrics'].items():
                    writer.writerow({'class_label': class_label, **metrics})
            print(f"Class metrics saved to {class_results_file}")

    return results


def evaluate_single_fold(
    fold: int,
    train_embeddings: np.ndarray,
    train_labels: list,
    test_embeddings: np.ndarray,
    test_labels: list,
    test_signer: str,
    method: str,
    seed: int,
    max_iter: int = 100,
    extra_metrics: bool = False,
):
    """Evaluate a single LOSO fold."""
    
    unique_classes = sorted(set(train_labels))
    num_classes = len(unique_classes)
    
    # Train classifier
    clf = None
    class_labels = None
    probabilities = None
    predictions = None
    if method == 'knn':
        clf = Pipeline(
            steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=num_classes))]
        )
        clf.fit(train_embeddings, train_labels)
    elif method == 'linear_probe':
        clf = LogisticRegression(random_state=seed, max_iter=max_iter)
        clf.fit(train_embeddings, train_labels)
    elif method == 'svm':
        clf = SVC(kernel='rbf', random_state=seed, probability=True, max_iter=max_iter)
        clf.fit(train_embeddings, train_labels)
    elif method == 'prototypical':
        # LOSO prototypical: use ALL support samples from all non-held-out signers.
        class_indices = defaultdict(list)
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)

        prototype_labels = sorted(class_indices.keys())
        prototype_vectors = []
        for label in prototype_labels:
            idxs = class_indices[label]
            prototype_vectors.append(np.mean(train_embeddings[idxs], axis=0))

        prototype_matrix = np.vstack(prototype_vectors)
        class_labels = np.array(prototype_labels, dtype=object)
        probabilities = np.linalg.norm(
            test_embeddings[:, np.newaxis, :] - prototype_matrix[np.newaxis, :, :], axis=2
        )
        sorted_indices = np.argsort(probabilities, axis=1)
        predictions = np.array([class_labels[row[0]] for row in sorted_indices], dtype=object)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'knn', 'linear_probe', 'svm', or 'prototypical'")
    
    # Predict
    if method != 'prototypical':
        if clf is None:
            raise RuntimeError("Classifier was not initialized for non-prototypical LOSO method.")
        predictions = clf.predict(test_embeddings)

        # Get ranked predictions
        if hasattr(clf, 'predict_proba'):
            probabilities = clf.predict_proba(test_embeddings)
            class_labels = clf.classes_
        else:
            probabilities = None
            class_labels = clf.classes_

    if predictions is None:
        raise RuntimeError("Predictions were not computed in LOSO evaluation.")
    if class_labels is None:
        raise RuntimeError("Class labels were not computed in LOSO evaluation.")
    
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
            if method == 'prototypical':
                sorted_indices = np.argsort(probabilities[i])
            else:
                sorted_indices = np.argsort(-probabilities[i])
            ranked_labels = [class_labels[idx] for idx in sorted_indices]
        else:
            # KNN: aggregate similarities by class
            similarities = np.dot(train_embeddings, test_embeddings[i])
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

    extra_metrics_results = None
    if extra_metrics:
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            test_labels, predictions, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            test_labels, predictions, average='weighted', zero_division=0
        )
        extra_metrics_results = {
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
        }
    
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
    if extra_metrics and extra_metrics_results is not None:
        print(f"  Precision (class-mean/macro):   {extra_metrics_results['precision_macro']:.4f}")
        print(f"  Recall (class-mean/macro):      {extra_metrics_results['recall_macro']:.4f}")
        print(f"  F1 (class-mean/macro):          {extra_metrics_results['f1_macro']:.4f}")
        print(f"  Precision (weighted):{extra_metrics_results['precision_weighted']:.4f}")
        print(f"  Recall (weighted):   {extra_metrics_results['recall_weighted']:.4f}")
        print(f"  F1 (weighted):       {extra_metrics_results['f1_weighted']:.4f}")
    print(f"{'='*60}\n")

    fold_results = {
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

    if extra_metrics and extra_metrics_results is not None:
        fold_results.update(extra_metrics_results)

    return fold_results


def run_loso_cross_validation(
    pose_embeddings_dir: str,
    method: str,
    label_language: str,
    use_categories: bool,
    seed: int,
    fold: Optional[int] = None,
    output_dir: Optional[str] = None,
    max_iter: int = 100,
    extra_metrics: bool = False,
):
    """
    Run Leave-One-Signer-Out cross-validation for fair comparison with Smart Head.
    
    Args:
        pose_embeddings_dir: Directory containing precomputed embeddings
        method: 'knn', 'linear_probe', 'svm', or 'prototypical'
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
    
    # Load all data with signer information, excluding only unknown split.
    # This includes the val signer in LOSO folds.
    embeddings, labels, signers, filenames = load_all_embeddings_with_signers(
        embedding_dir, label_language, use_categories,
        exclude_splits=['unknown']
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
            method, seed, max_iter, extra_metrics
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

        if extra_metrics:
            avg_precision_macro = np.mean([r['precision_macro'] for r in all_results])
            avg_recall_macro = np.mean([r['recall_macro'] for r in all_results])
            avg_f1_macro = np.mean([r['f1_macro'] for r in all_results])
            avg_precision_weighted = np.mean([r['precision_weighted'] for r in all_results])
            avg_recall_weighted = np.mean([r['recall_weighted'] for r in all_results])
            avg_f1_weighted = np.mean([r['f1_weighted'] for r in all_results])

            std_precision_macro = np.std([r['precision_macro'] for r in all_results])
            std_recall_macro = np.std([r['recall_macro'] for r in all_results])
            std_f1_macro = np.std([r['f1_macro'] for r in all_results])
            std_precision_weighted = np.std([r['precision_weighted'] for r in all_results])
            std_recall_weighted = np.std([r['recall_weighted'] for r in all_results])
            std_f1_weighted = np.std([r['f1_weighted'] for r in all_results])
        
        print(f"\n{'='*60}")
        print(f"Average Across All Folds")
        print(f"{'='*60}")
        print(f"  R@1↑:        {avg_r1:.4f} ± {std_r1:.4f}")
        print(f"  R@5↑:        {avg_r5:.4f} ± {std_r5:.4f}")
        print(f"  R@10↑:       {avg_r10:.4f} ± {std_r10:.4f}")
        print(f"  MedianR↓:    {avg_median_rank:.2f} ± {std_median_rank:.2f}")
        if extra_metrics:
            print(f"  Precision (class-mean/macro):   {avg_precision_macro:.4f} ± {std_precision_macro:.4f}")
            print(f"  Recall (class-mean/macro):      {avg_recall_macro:.4f} ± {std_recall_macro:.4f}")
            print(f"  F1 (class-mean/macro):          {avg_f1_macro:.4f} ± {std_f1_macro:.4f}")
            print(f"  Precision (weighted):{avg_precision_weighted:.4f} ± {std_precision_weighted:.4f}")
            print(f"  Recall (weighted):   {avg_recall_weighted:.4f} ± {std_recall_weighted:.4f}")
            print(f"  F1 (weighted):       {avg_f1_weighted:.4f} ± {std_f1_weighted:.4f}")
        print(f"{'='*60}\n")
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / f'loso_{method}_results.csv'
            fold_fieldnames = [
                'fold', 'test_signer', 'r@1', 'r@5', 'r@10', 'median_rank',
                'hit_1', 'hit_5', 'hit_10', 'num_test', 'num_train', 'num_classes'
            ]
            if extra_metrics:
                fold_fieldnames.extend([
                    'precision_macro', 'recall_macro', 'f1_macro',
                    'precision_weighted', 'recall_weighted', 'f1_weighted'
                ])
            with open(results_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fold_fieldnames
                )
                writer.writeheader()
                for row in all_results:
                    writer.writerow(row)

            summary_file = output_path / f'loso_{method}_summary.csv'
            summary_fieldnames = [
                'method', 'pose_embeddings_dir', 'label_language', 'use_categories', 'seed',
                'r@1_mean', 'r@1_std', 'r@5_mean', 'r@5_std',
                'r@10_mean', 'r@10_std', 'median_rank_mean', 'median_rank_std'
            ]
            if extra_metrics:
                summary_fieldnames.extend([
                    'precision_macro_mean', 'precision_macro_std',
                    'recall_macro_mean', 'recall_macro_std',
                    'f1_macro_mean', 'f1_macro_std',
                    'precision_weighted_mean', 'precision_weighted_std',
                    'recall_weighted_mean', 'recall_weighted_std',
                    'f1_weighted_mean', 'f1_weighted_std'
                ])

            with open(summary_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=summary_fieldnames
                )
                writer.writeheader()
                summary_row = {
                    'method': method,
                    'pose_embeddings_dir': pose_embeddings_dir,
                    'label_language': label_language,
                    'use_categories': use_categories,
                    'seed': seed,
                    'r@1_mean': avg_r1,
                    'r@1_std': std_r1,
                    'r@5_mean': avg_r5,
                    'r@5_std': std_r5,
                    'r@10_mean': avg_r10,
                    'r@10_std': std_r10,
                    'median_rank_mean': avg_median_rank,
                    'median_rank_std': std_median_rank,
                }
                if extra_metrics:
                    summary_row.update({
                        'precision_macro_mean': avg_precision_macro,
                        'precision_macro_std': std_precision_macro,
                        'recall_macro_mean': avg_recall_macro,
                        'recall_macro_std': std_recall_macro,
                        'f1_macro_mean': avg_f1_macro,
                        'f1_macro_std': std_f1_macro,
                        'precision_weighted_mean': avg_precision_weighted,
                        'precision_weighted_std': std_precision_weighted,
                        'recall_weighted_mean': avg_recall_weighted,
                        'recall_weighted_std': std_recall_weighted,
                        'f1_weighted_mean': avg_f1_weighted,
                        'f1_weighted_std': std_f1_weighted,
                    })
                writer.writerow(summary_row)
            
            print(f"Results saved to {results_file}")
            print(f"Summary saved to {summary_file}\n")
    
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
    knn          - K-Nearest Neighbors (K=num_classes, StandardScaler + euclidean)
  linear_probe - Logistic Regression (default scikit-learn settings)
  svm          - Support Vector Machine with RBF kernel (advanced non-linear)
    mlp          - Scikit-learn MLP with SmartHead-like hidden widths
    prototypical - Class-average prototypes with Euclidean-distance ranking (no scaling)
"""
    )
    parser.add_argument('--pose_embeddings_dir', type=str, required=True,
                        help='Directory containing precomputed pose .npy embeddings')
    parser.add_argument('--method', type=str, required=True,
                                                choices=['knn', 'linear_probe', 'svm', 'mlp', 'prototypical'],
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
                        help='Output directory for results CSV (LOSO and standard mode)')
    parser.add_argument('--num_shots', type=int, default=None,
                        help='Limit training examples per class (e.g. 1 or 5 for shot-limited '
                             'evaluation). Default: use all available training examples. '
                             'Ignored in LOSO mode.')
    parser.add_argument('--number_shot', type=int, default=7,
                        help='Support samples per class for prototypical method (default: 7, max: 7).')
    parser.add_argument('--train_split', type=str, default='train',
                        choices=['train', 'train_val'],
                        help=(
                            'Training data for few-shot classifiers (default: train). '
                            'Use train_val to include val signers (mrla) in training, '
                            'which is valid when NOT comparing against the fine-tuned model '
                            '(fine-tuning used mrla for checkpoint selection). '
                            'Ignored in LOSO mode (all signers are used).'
                        ))
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        dest='eval_split',
                        help=(
                            'Evaluation split (default: test). Use val to tune '
                            'hyperparameters (k, method) without touching the test set. '
                            'Only report numbers from test in the paper. '
                            'Ignored in LOSO mode.'
                        ))
    parser.add_argument('--max_iter', type=int, default=100, dest='max_iter',
                        help='Maximum iterations for linear_probe, svm, and mlp (default: 100).')
    parser.add_argument('--class_eval', action='store_true',
                        help='Print per-class retrieval metrics and save per-class metrics to CSV (standard mode only).')
    parser.add_argument('--extra_metrics', action='store_true',
                        help='Compute and print classification metrics from top-1 predictions: precision/recall/F1 (macro and weighted).')
        
    args = parser.parse_args()

    if args.loso:
        if args.method == 'mlp':
            raise ValueError("method='mlp' is currently supported only in standard split mode (without --loso).")
        if args.class_eval:
            print("Warning: --class_eval is currently ignored in --loso mode.")
        # Run LOSO cross-validation
        run_loso_cross_validation(
            pose_embeddings_dir=args.pose_embeddings_dir,
            method=args.method,
            label_language=args.label_language,
            use_categories=args.use_categories,
            seed=args.seed,
            fold=args.fold,
            output_dir=args.output_dir,
            max_iter=args.max_iter,
            extra_metrics=args.extra_metrics,
        )
    else:
        # Run standard signer-independent evaluation
        evaluate_few_shot(
            pose_embeddings_dir=args.pose_embeddings_dir,
            method=args.method,
            label_language=args.label_language,
            seed=args.seed,
            use_categories=args.use_categories,
            train_split=args.train_split,
            eval_split=args.eval_split,
            num_shots=args.num_shots,
            number_shot=args.number_shot,
            output_dir=args.output_dir,
            max_iter=args.max_iter,
            class_eval=args.class_eval,
            extra_metrics=args.extra_metrics,
        )


if __name__ == '__main__':
    main()
