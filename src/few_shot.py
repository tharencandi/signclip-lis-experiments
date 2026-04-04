"""
Few-shot evaluation for sign language recognition using SignCLIP.
Uses prototype-based matching: average similarity to k examples per class.
Usage: python few_shot.py --data_dir /path/to/poses --embedding_dir /path/to/embeddings --k_shot 5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import statistics
from collections import defaultdict
from tqdm import tqdm

# Add project root to path for signclip imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pose_dataset import PoseDataset


def evaluate_few_shot(data_dir: str, embedding_dir: str, k_shot: int, 
                      label_type: str = 'micro', seed: int = 42):
    """
    Perform few-shot evaluation using prototype-based matching.
    
    Args:
        data_dir: Directory containing .pose files
        embedding_dir: Directory containing precomputed .npy embeddings
        k_shot: Number of support examples per class
        label_type: 'micro' or 'macro' label granularity
        seed: Random seed for sampling support set
    """
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = PoseDataset(data_dir, label_type=label_type)
    
    # Get train and test splits
    train_data = dataset.filter_split('train').sample_k_shot(k=k_shot, seed=seed)
    test_data = dataset.filter_split('test')
    
    # Filter test set to only include classes in training set (closed-set evaluation)
    common_classes = set(train_data.classes) & set(test_data.classes)
    train_data = train_data.filter_classes(list(common_classes))
    test_data = test_data.filter_classes(list(common_classes))
    
    print(f"\n{k_shot}-shot Evaluation Setup:")
    print(f"  Train: {len(train_data)} samples, {train_data.num_classes} classes")
    print(f"  Test:  {len(test_data)} samples, {test_data.num_classes} classes")
    
    # Load embeddings
    embedding_path = Path(embedding_dir)
    
    # Load train embeddings grouped by class
    print(f"\nLoading train embeddings...")
    train_embeddings_grouped = defaultdict(list)
    
    for idx in tqdm(range(len(train_data)), desc="Loading train"):
        sample = train_data.samples[idx]
        emb_file = embedding_path / sample['filename'].replace('.pose', '.npy')
        
        if not emb_file.exists():
            print(f"\nWarning: Missing embedding for {sample['filename']}")
            continue
        
        emb = np.load(emb_file)
        if emb.ndim > 1:
            emb = emb.squeeze()
        
        train_embeddings_grouped[sample['label']].append(emb)
    
    # Convert to numpy arrays
    for label in train_embeddings_grouped:
        train_embeddings_grouped[label] = np.array(train_embeddings_grouped[label])
    
    # Load test embeddings
    print(f"\nLoading test embeddings...")
    test_embeddings = []
    test_labels = []
    
    for idx in tqdm(range(len(test_data)), desc="Loading test"):
        sample = test_data.samples[idx]
        emb_file = embedding_path / sample['filename'].replace('.pose', '.npy')
        
        if not emb_file.exists():
            print(f"\nWarning: Missing embedding for {sample['filename']}")
            continue
        
        emb = np.load(emb_file)
        if emb.ndim > 1:
            emb = emb.squeeze()
        
        test_embeddings.append(emb)
        test_labels.append(sample['label'])
    
    test_embeddings = np.array(test_embeddings)
    
    # Evaluate
    print(f"\nEvaluating...")
    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    ranks = []
    
    for i, gold_label in enumerate(tqdm(test_labels, desc="Testing")):
        test_emb = test_embeddings[i]
        scores_per_class = []
        
        # Compute average similarity to each class's support set
        for class_label, class_embeddings in train_embeddings_grouped.items():
            # Compute similarities to all support examples
            similarities = np.dot(class_embeddings, test_emb)
            # Average similarity
            avg_similarity = np.mean(similarities)
            scores_per_class.append((class_label, avg_similarity))
        
        # Sort by score (descending)
        scores_per_class.sort(key=lambda x: x[1], reverse=True)
        ranked_labels = [label for label, score in scores_per_class]
        
        # Check hits
        if gold_label in ranked_labels[:1]:
            hit_1 += 1
        if gold_label in ranked_labels[:5]:
            hit_5 += 1
        if gold_label in ranked_labels[:10]:
            hit_10 += 1
        
        if gold_label in ranked_labels:
            rank = ranked_labels.index(gold_label)
            ranks.append(rank)
    
    # Calculate metrics
    num_test = len(test_labels)
    recall_1 = hit_1 / num_test
    recall_5 = hit_5 / num_test
    recall_10 = hit_10 / num_test
    median_rank = statistics.median(ranks) + 1 if ranks else float('inf')
    
    # Print results
    print(f"\n{'='*50}")
    print(f"{k_shot}-Shot Evaluation Results")
    print(f"{'='*50}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {num_test}")
    print(f"Classes: {len(common_classes)}")
    print(f"Seed: {seed}")
    print(f"\nMetrics:")
    print(f"  Recall@1:  {recall_1:.4f} ({hit_1}/{num_test})")
    print(f"  Recall@5:  {recall_5:.4f} ({hit_5}/{num_test})")
    print(f"  Recall@10: {recall_10:.4f} ({hit_10}/{num_test})")
    print(f"  Median Rank: {median_rank:.1f}")
    print(f"{'='*50}\n")
    
    return {
        'recall@1': recall_1,
        'recall@5': recall_5,
        'recall@10': recall_10,
        'median_rank': median_rank,
        'num_test': num_test,
        'num_classes': len(common_classes),
        'k_shot': k_shot
    }


def main():
    parser = argparse.ArgumentParser(description="Few-shot sign language recognition evaluation")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .pose files')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Directory containing precomputed embeddings')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support examples per class')
    parser.add_argument('--label_type', type=str, default='micro',
                        choices=['micro', 'macro'],
                        help='Label granularity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for support set sampling')
    
    args = parser.parse_args()
    
    evaluate_few_shot(
        data_dir=args.data_dir,
        embedding_dir=args.embedding_dir,
        k_shot=args.k_shot,
        label_type=args.label_type,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
