"""
Shared utilities for loading A3LIS precomputed embeddings.
Used by zero_shot.py, few_shot.py, and repr_density_eval.py.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def load_a3lis_embeddings(
    embedding_dir: Path,
    split: str = 'test',
    label_language: str = 'english',
    use_categories: bool = False,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """
    Load A3LIS precomputed pose embeddings using embeddings_metadata.json.

    Args:
        embedding_dir: Directory containing .npy embedding files and metadata
        split: Split to load, e.g. 'train', 'val', 'test'
        label_language: 'italian' or 'english' for label selection
        use_categories: If True, use macro categories instead of micro labels

    Returns:
        Tuple of (embeddings_array, labels, filenames, all_labels)
        all_labels is sorted list of unique labels (useful for text embedding generation)
    """
    metadata_path = embedding_dir / 'embeddings_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    embeddings = []
    labels = []
    filenames = []
    all_labels = set()

    for item in tqdm(metadata['embeddings'], desc=f"Loading {split} embeddings"):
        if item['split'] != split:
            continue

        if use_categories:
            label = item.get('category', item['label_italian'])
        elif label_language == 'italian':
            label = item['label_italian']
        else:
            label = item['labels_english'][0] if item['labels_english'] else item['label_italian']

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

    embeddings_array = np.array(embeddings) if embeddings else np.array([])
    return embeddings_array, labels, filenames, sorted(all_labels)


def load_all_embeddings_with_signers(
    embedding_dir: Path,
    label_language: str = 'english',
    use_categories: bool = False,
    exclude_splits: list = None,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """
    Load ALL A3LIS embeddings with signer information for LOSO cross-validation.

    Args:
        embedding_dir: Directory containing .npy embedding files and metadata
        label_language: 'italian' or 'english' for label selection
        use_categories: If True, use macro categories instead of micro labels
        exclude_splits: Split values to exclude (e.g. ['val', 'unknown']) to
                        prevent the val signer from appearing in any LOSO fold.

    Returns:
        Tuple of (embeddings_array, labels, signers, filenames)
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
        if exclude_splits and item.get('split', 'unknown') in exclude_splits:
            continue

        if use_categories:
            label = item.get('category', item['label_italian'])
        elif label_language == 'italian':
            label = item['label_italian']
        else:
            label = item['labels_english'][0] if item['labels_english'] else item['label_italian']

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
