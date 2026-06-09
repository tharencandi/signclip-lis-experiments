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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import statistics
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from src.demo_sign import embed_text
from src.embedding_utils import EN_LIS_PROMPT_TEMPLATE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Add project root to path for signclip imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


AI_HEAD_PROMPT_TEMPLATE = EN_LIS_PROMPT_TEMPLATE
AI_HEAD_TEXT_BATCH_SIZE = 16


def load_precomputed_text_embeddings_from_dir(text_embeddings_dir: str, model_name: str) -> Dict[str, np.ndarray]:
    """Load precomputed text embeddings from a directory and return label->embedding map."""
    dir_path = Path(text_embeddings_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Text embeddings directory not found: {text_embeddings_dir}")

    embedding_files = sorted(dir_path.glob('text_embeddings*.npy'))
    if not embedding_files:
        raise FileNotFoundError(
            f"No text embedding files found in {text_embeddings_dir}. Expected files like text_embeddings*.npy"
        )

    preferred = [p for p in embedding_files if model_name in p.name]
    embedding_file = preferred[0] if preferred else embedding_files[0]

    metadata_candidate = embedding_file.with_name(embedding_file.name.replace('text_embeddings', 'text_metadata')).with_suffix('.json')
    labels_candidate = embedding_file.with_name(embedding_file.name.replace('text_embeddings', 'text_labels')).with_suffix('.txt')

    text_embeddings = np.load(embedding_file)
    if text_embeddings.ndim == 1:
        text_embeddings = text_embeddings.reshape(1, -1)

    labels = None
    if metadata_candidate.exists():
        with open(metadata_candidate, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        labels = metadata.get('labels')

    if labels is None and labels_candidate.exists():
        with open(labels_candidate, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]

    if labels is None:
        all_label_files = sorted(dir_path.glob('text_labels*.txt'))
        if all_label_files:
            with open(all_label_files[0], 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]

    if labels is None:
        raise FileNotFoundError(
            f"Could not find labels for {embedding_file.name}. Expected text_metadata*.json with 'labels' "
            "or text_labels*.txt in the same directory."
        )

    if len(labels) != text_embeddings.shape[0]:
        raise ValueError(
            f"Mismatch in precomputed text artifacts: {len(labels)} labels but "
            f"{text_embeddings.shape[0]} embeddings in {embedding_file.name}"
        )

    label_to_embedding = {str(label): text_embeddings[i] for i, label in enumerate(labels)}
    print(f"Loaded precomputed text embeddings: {len(label_to_embedding)} labels from {embedding_file}")
    return label_to_embedding


def resolve_ai_head_precomputed_map(
    target_labels: list,
    class_prompt_labels: dict,
    loaded_label_to_embedding: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Resolve current dataset class labels to precomputed text anchor embeddings."""
    resolved = {}
    missing = []

    for label in sorted(set(target_labels)):
        prompt_label = str(class_prompt_labels.get(label, str(label)))
        candidates = [
            prompt_label,
            prompt_label.strip(),
            AI_HEAD_PROMPT_TEMPLATE.format(prompt_label),
            str(label),
            str(label).strip(),
            AI_HEAD_PROMPT_TEMPLATE.format(str(label)),
        ]

        matched = None
        for key in candidates:
            if key in loaded_label_to_embedding:
                matched = loaded_label_to_embedding[key]
                break

        if matched is None:
            missing.append((label, prompt_label))
        else:
            resolved[label] = matched

    if missing:
        preview = ', '.join([f"{cls}->{prompt}" for cls, prompt in missing[:5]])
        raise ValueError(
            "Missing precomputed text embeddings for some classes. "
            f"Examples: {preview}. Ensure precomputed labels align with AI-Head prompts."
        )

    return resolved


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


def l2_normalize_rows(array: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D array row-wise with zero-norm protection."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


def load_class_prompt_labels(embedding_dir: Path, label_language: str = 'english', use_categories: bool = False):
    """Map current class labels to English prompt anchors for AI-Head."""
    metadata_path = embedding_dir / 'embeddings_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    prompt_map = {}
    for item in metadata['embeddings']:
        if use_categories:
            current_label = item.get('category', item['label_italian'])
            prompt_label = item.get('category', item['labels_english'][0] if item['labels_english'] else item['label_italian'])
        elif label_language == 'italian':
            current_label = item['label_italian']
            prompt_label = item['labels_english'][0] if item['labels_english'] else item['label_italian']
        else:
            current_label = item['labels_english'][0] if item['labels_english'] else item['label_italian']
            prompt_label = current_label

        if current_label not in prompt_map:
            prompt_map[current_label] = prompt_label

    return prompt_map


def build_ai_head_weights(
    train_embeddings: np.ndarray,
    train_labels: list,
    class_prompt_labels: dict,
    number_shot: int,
    seed: int,
    model_name: str,
    model_checkpoint_path: Optional[str] = None,
    use_all_support: bool = False,
    text_batch_size: int = AI_HEAD_TEXT_BATCH_SIZE,
    precomputed_text_embeddings: Optional[dict] = None,
    text_alpha: float = 0.2,
):
    """Blend visual prototypes toward text anchors in the shared raw embedding space.

    Both pose and text embeddings live in the same joint SignCLIP space, so Euclidean
    distance is the right metric (no L2 normalization).
    text_alpha=0 → pure visual prototype (identical to prototypical baseline).
    text_alpha=1 → pure text anchor.
    Returns (class_labels, class_centers, text_proto_dists, skipped_classes).
    """
    class_indices = defaultdict(list)
    for i, label in enumerate(train_labels):
        class_indices[label].append(i)

    if use_all_support:
        valid_classes = sorted(class_indices.keys())
        skipped_classes = []
    else:
        valid_classes = sorted([label for label, idxs in class_indices.items() if len(idxs) >= number_shot])
        skipped_classes = sorted([label for label, idxs in class_indices.items() if len(idxs) < number_shot])

    if not valid_classes:
        raise ValueError("No classes have enough support samples to construct AI-Head weights.")

    rng = np.random.default_rng(seed)
    prototype_labels = []
    prototype_vectors = []
    for label in valid_classes:
        idxs = class_indices[label]
        if use_all_support:
            support_indices = idxs
        else:
            support_indices = rng.choice(idxs, size=number_shot, replace=False)
        prototype = np.mean(train_embeddings[support_indices], axis=0)
        prototype_labels.append(label)
        prototype_vectors.append(prototype)

    prototype_matrix = np.vstack(prototype_vectors)
    if precomputed_text_embeddings is not None:
        text_embeddings = np.vstack([precomputed_text_embeddings[label] for label in prototype_labels])
    else:
        print("precomputing text embeddings for AI-Head class anchors...")
        prompt_texts = [
            AI_HEAD_PROMPT_TEMPLATE.format(class_prompt_labels.get(label, str(label)))
            for label in prototype_labels
        ]
        # Batch text embedding extraction to reduce peak memory usage.
        text_chunks = []
        for i in range(0, len(prompt_texts), text_batch_size):
            chunk = prompt_texts[i:i + text_batch_size]
            print(f"processing chunk {i // text_batch_size + 1} / {(len(prompt_texts) + text_batch_size - 1) // text_batch_size}")
            text_chunks.append(
                embed_text(
                    chunk,
                    model_name=model_name#,
                    #checkpoint_path=model_checkpoint_path,
                )
            )
        text_embeddings = np.vstack(text_chunks)

    # Raw embedding space (no normalization) — Euclidean is the right metric here.
    # Blend class center from visual prototype toward text anchor by text_alpha.
    class_centers = (1.0 - text_alpha) * prototype_matrix + text_alpha * text_embeddings

    # Diagnostic: per-class distance between text anchor and visual prototype in the joint space.
    text_proto_dists = np.linalg.norm(text_embeddings - prototype_matrix, axis=1)  # (C,)

    return np.array(prototype_labels, dtype=object), class_centers, text_proto_dists, skipped_classes


def evaluate_few_shot(
    pose_embeddings_dir: str,
    method: str = 'knn',
    model_name: str = 'default',
    model_checkpoint_path: Optional[str] = None,
    label_language: str = 'english',
    seed: int = 42,
    use_categories: bool = False,
    train_split: str = 'train',
    eval_split: str = 'test',
    num_shots: Optional[int] = None,
    number_shot: int = 7,
    text_embeddings_dir: Optional[str] = None,
    text_alpha: float = 0.2,
    output_dir: Optional[str] = None,
    max_iter: int = 100,
):
    """
    Perform few-shot evaluation using KNN, linear probe, SVM, MLP, Prototypical, or AI-Head.
    
    For A3LIS: Uses signer-independent split with ~7 training examples per class.
    
    Args:
        pose_embeddings_dir: Directory containing precomputed pose embeddings
        method: 'knn', 'linear_probe', 'svm', 'mlp', 'prototypical', or 'ai_head'
        model_name: Base SignCLIP model used for AI-Head text anchors
        model_checkpoint_path: Optional checkpoint overlaid on model_name for AI-Head text anchors
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
        output_dir: Directory to save results JSON. If None, results are not saved.
    """
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

    # For prototypical / ai_head, num_shots directly sets the number of support samples per class.
    if method in {'prototypical', 'ai_head'} and num_shots is not None and num_shots > 0:
        number_shot = num_shots

    # Shot limiting: keep at most num_shots examples per class (sklearn methods only)
    if method not in {'prototypical', 'ai_head'} and num_shots is not None and num_shots > 0:
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
        #k = 1 #overide
        k = 7
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
        if number_shot < 1 or number_shot > 7:
            raise ValueError("number_shot must be in [1, 7] for A3LIS signer split.")

        print(
            f"\nTraining Prototypical classifier (class-average supports, number_shot={number_shot}, "
            "no scaling, euclidean scoring)..."
        )

        class_indices = defaultdict(list)
        for i, label in enumerate(train_labels):
            class_indices[label].append(i)

        valid_classes = sorted([label for label, idxs in class_indices.items() if len(idxs) >= number_shot])
        skipped_classes = sorted([label for label, idxs in class_indices.items() if len(idxs) < number_shot])

        if not valid_classes:
            raise ValueError(
                f"No classes have enough support samples for number_shot={number_shot}."
            )
        if skipped_classes:
            print(
                f"Warning: {len(skipped_classes)} classes have < {number_shot} support samples and "
                "will be excluded from prototype construction."
            )

        rng = np.random.default_rng(seed)
        prototype_labels = []
        prototype_vectors = []
        for label in valid_classes:
            sampled_indices = rng.choice(class_indices[label], size=number_shot, replace=False)
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

    elif method == 'ai_head':
        if number_shot < 1 or number_shot > 7:
            raise ValueError("number_shot must be in [1, 7] for A3LIS signer split.")

        print(
            f"\nTraining AI-Head (number_shot={number_shot}, model={model_name}, "
            f"checkpoint={'provided' if model_checkpoint_path else 'none'})..."
        )
        print("loading class prompt labels for AI-Head...")
        class_prompt_labels = load_class_prompt_labels(embedding_dir, label_language, use_categories)
        precomputed_text_embeddings = None
        if text_embeddings_dir:
            loaded_label_to_embedding = load_precomputed_text_embeddings_from_dir(text_embeddings_dir, model_name)
            precomputed_text_embeddings = resolve_ai_head_precomputed_map(
                target_labels=train_labels,
                class_prompt_labels=class_prompt_labels,
                loaded_label_to_embedding=loaded_label_to_embedding,
            )
            print(f"Using precomputed text embeddings from: {text_embeddings_dir}")
        print("building AI-Head weights...")
        class_labels, class_centers, text_proto_dists, skipped_classes = build_ai_head_weights(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            class_prompt_labels=class_prompt_labels,
            number_shot=number_shot,
            seed=seed,
            model_name=model_name,
            model_checkpoint_path=model_checkpoint_path,
            use_all_support=False,
            precomputed_text_embeddings=precomputed_text_embeddings,
            text_alpha=text_alpha,
        )

        if skipped_classes:
            print(
                f"Warning: {len(skipped_classes)} classes have < {number_shot} support samples and "
                "will be excluded from AI-Head construction."
            )
        # Euclidean distance to blended class centers in the raw joint embedding space.
        # Same distance metric and normalization regime as the prototypical baseline.
        print(f"scoring with AI-Head: Euclidean to blended class centers (text_alpha={text_alpha:.2f})...")
        distances = np.linalg.norm(
            test_embeddings[:, np.newaxis, :] - class_centers[np.newaxis, :, :], axis=2
        )  # (N, C)
        probabilities = -distances  # negate: higher score = closer center
        sorted_indices = np.argsort(-probabilities, axis=1)
        predictions = np.array([class_labels[row[0]] for row in sorted_indices], dtype=object)

        print(f"  Built AI-Head: {len(class_labels)} classes, text_alpha={text_alpha:.2f}")
        print(f"  Text-proto dist (mean={float(np.mean(text_proto_dists)):.4f}, "
              f"min={float(np.min(text_proto_dists)):.4f}, max={float(np.max(text_proto_dists)):.4f})")

    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'knn', 'linear_probe', 'svm', 'mlp', 'prototypical', or 'ai_head'"
        )
    
    # Predict on test set
    print("\nEvaluating on test set...")
    if method not in {'prototypical', 'ai_head'}:
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
    # For ranking, we need class probabilities or distances
    if class_labels is None:
        raise RuntimeError("Class labels were not computed.")
    
    for i, gold_label in enumerate(tqdm(test_labels, desc="Evaluating")):
        pred_label = predictions[i]
        cat = test_categories[i]

        if cat not in category_stats:
            category_stats[cat] = {'total': 0, 'hit_1': 0, 'hit_5': 0, 'hit_10': 0, 'ranks': []}
        category_stats[cat]['total'] += 1
        
        # Top-1 accuracy
        if pred_label == gold_label:
            hit_1 += 1
            category_stats[cat]['hit_1'] += 1
        
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
        if gold_label in ranked_labels[:10]:
            hit_10 += 1
            category_stats[cat]['hit_10'] += 1
        
        # Get rank
        if gold_label in ranked_labels:
            rank = ranked_labels.index(gold_label)
            ranks.append(rank)
            category_stats[cat]['ranks'].append(rank)
        else:
            ranks.append(len(ranked_labels))
            category_stats[cat]['ranks'].append(len(ranked_labels))
    
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
    elif method == 'ai_head':
        print(f"number_shot (support samples): {number_shot}")
        print(f"AI-Head classes: {len(class_labels)}")
        print(f"Text model: {model_name}")
        print(f"text_alpha (text blend weight): {text_alpha:.2f}")
        if model_checkpoint_path:
            print(f"Checkpoint: {model_checkpoint_path}")
        print("Scoring: Euclidean to blended class center in raw joint embedding space")
    print(f"Seed: {seed}")
    print(f"\nRetrieval Metrics:")
    print(f"  R@1↑:             {accuracy:>7.2%}  ({hit_1:>5}/{num_test})")
    print(f"  R@5↑:             {recall_5:>7.2%}  ({hit_5:>5}/{num_test})")
    print(f"  R@10↑:            {recall_10:>7.2%}  ({hit_10:>5}/{num_test})")
    print(f"  MedianR↓:         {median_rank:>7.1f}")
    print(f"\nAccuracy:")
    print(f"  Top-1:            {accuracy:>7.2%}  ({hit_1:>5}/{num_test})")

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
            elif method == 'ai_head':
                print(f"    {j}. {label:<30} (logit: {score:.4f}) {marker}")
            else:
                print(f"    {j}. {label:<30} (score: {score:.4f}) {marker}")
    
    results = {
        'method': method,
        'model_name': model_name,
        'model_checkpoint_path': model_checkpoint_path,
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
        'text_embeddings_dir': text_embeddings_dir,
        'text_alpha': text_alpha if method == 'ai_head' else None,
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if method in {'prototypical', 'ai_head'}:
            shots_str = f"{number_shot}shot"
        else:
            shots_str = f"{num_shots}shot" if num_shots else "allshot"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"few_shot_{method}_{shots_str}_{train_split}_{eval_split}_{timestamp}.json"
        results_file = output_path / filename
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_file}")

    return results


def evaluate_single_fold(
    fold: int,
    train_embeddings: np.ndarray,
    train_labels: list,
    test_embeddings: np.ndarray,
    test_labels: list,
    test_signer: str,
    method: str,
    embedding_dir: Path,
    label_language: str,
    use_categories: bool,
    model_name: str,
    model_checkpoint_path: Optional[str],
    class_prompt_labels: Optional[dict],
    precomputed_text_embeddings: Optional[dict],
    seed: int,
    max_iter: int = 100,
    text_alpha: float = 0.2,
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
    elif method == 'ai_head':
        local_prompt_labels = class_prompt_labels
        if local_prompt_labels is None:
            local_prompt_labels = load_class_prompt_labels(embedding_dir, label_language, use_categories)
        class_labels, class_centers, _, _ = build_ai_head_weights(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            class_prompt_labels=local_prompt_labels,
            number_shot=1,
            seed=seed,
            model_name=model_name,
            model_checkpoint_path=model_checkpoint_path,
            use_all_support=True,
            precomputed_text_embeddings=precomputed_text_embeddings,
            text_alpha=text_alpha,
        )
        distances = np.linalg.norm(
            test_embeddings[:, np.newaxis, :] - class_centers[np.newaxis, :, :], axis=2
        )
        probabilities = -distances
        sorted_indices = np.argsort(-probabilities, axis=1)
        predictions = np.array([class_labels[row[0]] for row in sorted_indices], dtype=object)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'knn', 'linear_probe', 'svm', 'prototypical', or 'ai_head'")
    
    # Predict
    if method not in {'prototypical', 'ai_head'}:
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
            elif method == 'ai_head':
                sorted_indices = np.argsort(-probabilities[i])
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
    model_name: str,
    model_checkpoint_path: Optional[str],
    label_language: str,
    use_categories: bool,
    seed: int,
    text_embeddings_dir: Optional[str] = None,
    text_alpha: float = 0.2,
    fold: Optional[int] = None,
    output_dir: Optional[str] = None,
    max_iter: int = 100
):
    """
    Run Leave-One-Signer-Out cross-validation for fair comparison with Smart Head.
    
    Args:
        pose_embeddings_dir: Directory containing precomputed embeddings
        method: 'knn', 'linear_probe', 'svm', 'prototypical', or 'ai_head'
        model_name: Base SignCLIP model used for AI-Head text anchors
        model_checkpoint_path: Optional checkpoint for AI-Head text anchors
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

    # Build text anchors once for AI-Head and reuse across folds to avoid repeated heavy embedding calls.
    ai_head_prompt_labels = None
    ai_head_text_embedding_map = None
    if method == 'ai_head':
        ai_head_prompt_labels = load_class_prompt_labels(embedding_dir, label_language, use_categories)
        all_labels = sorted(set(labels))
        if text_embeddings_dir:
            loaded_label_to_embedding = load_precomputed_text_embeddings_from_dir(text_embeddings_dir, model_name)
            ai_head_text_embedding_map = resolve_ai_head_precomputed_map(
                target_labels=all_labels,
                class_prompt_labels=ai_head_prompt_labels,
                loaded_label_to_embedding=loaded_label_to_embedding,
            )
            print(f"Using precomputed text embeddings from: {text_embeddings_dir}")
        else:
            prompt_texts = [
                AI_HEAD_PROMPT_TEMPLATE.format(ai_head_prompt_labels.get(label, str(label)))
                for label in all_labels
            ]
            text_chunks = []
            for i in range(0, len(prompt_texts), AI_HEAD_TEXT_BATCH_SIZE):
                chunk = prompt_texts[i:i + AI_HEAD_TEXT_BATCH_SIZE]
                text_chunks.append(
                    embed_text(
                        chunk,
                        model_name=model_name,
                        checkpoint_path=model_checkpoint_path,
                    )
                )
            text_embs = np.vstack(text_chunks)
            ai_head_text_embedding_map = {label: emb for label, emb in zip(all_labels, text_embs)}
    
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
            method, embedding_dir, label_language, use_categories, model_name, model_checkpoint_path,
            ai_head_prompt_labels, ai_head_text_embedding_map, seed, max_iter, text_alpha
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
                        'model_name': model_name,
                        'model_checkpoint_path': model_checkpoint_path,
                        'text_embeddings_dir': text_embeddings_dir,
                        'label_language': label_language,
                        'use_categories': use_categories,
                        'seed': seed,
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
    knn          - K-Nearest Neighbors (K=num_classes, StandardScaler + euclidean)
  linear_probe - Logistic Regression (default scikit-learn settings)
  svm          - Support Vector Machine with RBF kernel (advanced non-linear)
    mlp          - Scikit-learn MLP with SmartHead-like hidden widths
    prototypical - Class-average prototypes with Euclidean-distance ranking (no scaling)
    ai_head      - Adaptive Iconicity Head with text-prototype fusion and per-class alpha
"""
    )
    parser.add_argument('--pose_embeddings_dir', type=str, required=True,
                        help='Directory containing precomputed pose .npy embeddings')
    parser.add_argument('--method', type=str, required=True,
                        choices=['knn', 'linear_probe', 'svm', 'mlp', 'prototypical', 'ai_head'],
                        help='Few-shot method to use')
    parser.add_argument('--model_name', type=str, default='default',
                        choices=['default', 'asl_citizen', 'asl_finetune', 'suisse', 'a3lis_finetune'],
                        help='Base SignCLIP model used for AI-Head text anchors (default: default)')
    parser.add_argument('--model_checkpoint_path', type=str, default=None,
                        help='Optional checkpoint path overlaid on --model_name for AI-Head text anchors')
    parser.add_argument('--text_embeddings_dir', type=str, default=None,
                        help='Optional directory with precomputed text embeddings (text_embeddings*.npy + labels metadata). Used by ai_head.')
    parser.add_argument('--ai_head_alpha', type=float, default=0.2,
                        help='AI-Head text blend weight: 0.0=pure prototype, 1.0=pure text (default: 0.2). Only used with --method ai_head.')
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
                        help='Output directory for results JSON (LOSO and standard mode)')
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
        
    args = parser.parse_args()

    if args.loso:
        if args.method == 'mlp':
            raise ValueError("method='mlp' is currently supported only in standard split mode (without --loso).")
        # Run LOSO cross-validation
        run_loso_cross_validation(
            pose_embeddings_dir=args.pose_embeddings_dir,
            method=args.method,
            model_name=args.model_name,
            model_checkpoint_path=args.model_checkpoint_path,
            label_language=args.label_language,
            use_categories=args.use_categories,
            seed=args.seed,
            text_embeddings_dir=args.text_embeddings_dir,
            text_alpha=args.ai_head_alpha,
            fold=args.fold,
            output_dir=args.output_dir,
            max_iter=args.max_iter,
        )
    else:
        # Run standard signer-independent evaluation
        evaluate_few_shot(
            pose_embeddings_dir=args.pose_embeddings_dir,
            method=args.method,
            model_name=args.model_name,
            model_checkpoint_path=args.model_checkpoint_path,
            label_language=args.label_language,
            seed=args.seed,
            use_categories=args.use_categories,
            train_split=args.train_split,
            eval_split=args.eval_split,
            num_shots=args.num_shots,
            number_shot=args.number_shot,
            text_embeddings_dir=args.text_embeddings_dir,
            text_alpha=args.ai_head_alpha,
            output_dir=args.output_dir,
            max_iter=args.max_iter,
        )


if __name__ == '__main__':
    main()
