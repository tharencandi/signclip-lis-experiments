"""
Zero-shot evaluation for sign language recognition using SignCLIP.
Usage: python zero_shot.py --data_dir /path/to/poses --embedding_dir /path/to/embeddings
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import statistics
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from pose_dataset import PoseDataset
from demo_sign import embed_text


def evaluate_zero_shot(data_dir: str, embedding_dir: str, model_name: str = 'default', 
                       label_type: str = 'micro', text_template: str = None):
    """
    Perform zero-shot evaluation: match test pose embeddings to text embeddings of classes.
    
    Args:
        data_dir: Directory containing .pose files
        embedding_dir: Directory containing precomputed .npy embeddings
        model_name: SignCLIP model to use
        label_type: 'micro' or 'macro' label granularity
        text_template: Template for text (e.g., "{}" or "<en> <ase> {}")
    """
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = PoseDataset(data_dir, label_type=label_type)
    test_data = dataset.filter_split('test')
    
    print(f"Test set: {len(test_data)} samples, {test_data.num_classes} classes")
    
    # Get unique class labels
    unique_labels = test_data.get_unique_labels()
    print(f"\nClasses: {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''}")
    
    # Compute text embeddings for all classes
    print(f"\nComputing text embeddings for {len(unique_labels)} classes...")
    if text_template is None:
        text_labels = unique_labels
    else:
        text_labels = [text_template.format(label) for label in unique_labels]
    
    text_embeddings = embed_text(text_labels, model_name=model_name)  # shape: (num_classes, dim)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Load pose embeddings
    embedding_path = Path(embedding_dir)
    print(f"\nLoading pose embeddings from {embedding_dir}...")
    
    pose_embeddings = []
    labels = []
    
    for idx in tqdm(range(len(test_data)), desc="Loading embeddings"):
        sample = test_data.samples[idx]
        emb_file = embedding_path / sample['filename'].replace('.pose', '.npy')
        
        if not emb_file.exists():
            print(f"\nWarning: Missing embedding for {sample['filename']}")
            continue
        
        # Load embedding and squeeze to 1D if needed
        emb = np.load(emb_file)
        if emb.ndim > 1:
            emb = emb.squeeze()
        
        pose_embeddings.append(emb)
        labels.append(sample['label'])
    
    pose_embeddings = np.array(pose_embeddings)  # shape: (num_test, dim)
    print(f"Pose embeddings shape: {pose_embeddings.shape}")
    
    # Compute similarity scores: (num_test, num_classes)
    print("\nComputing similarities...")
    similarities = np.matmul(pose_embeddings, text_embeddings.T)
    
    # Rank classes by similarity for each test sample
    ranked_indices = np.argsort(-similarities, axis=1)  # descending order
    
    # Evaluate metrics
    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    ranks = []
    
    for i, gold_label in enumerate(labels):
        ranked_labels = [unique_labels[idx] for idx in ranked_indices[i]]
        
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
    num_test = len(labels)
    recall_1 = hit_1 / num_test
    recall_5 = hit_5 / num_test
    recall_10 = hit_10 / num_test
    median_rank = statistics.median(ranks) + 1 if ranks else float('inf')
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Zero-Shot Evaluation Results")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Test samples: {num_test}")
    print(f"Classes: {len(unique_labels)}")
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
        'num_classes': len(unique_labels)
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot sign language recognition evaluation")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .pose files')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Directory containing precomputed embeddings')
    parser.add_argument('--model_name', type=str, default='default',
                        choices=['default', 'asl_citizen', 'asl_finetune', 'suisse'],
                        help='SignCLIP model to use')
    parser.add_argument('--label_type', type=str, default='micro',
                        choices=['micro', 'macro'],
                        help='Label granularity')
    parser.add_argument('--text_template', type=str, default=None,
                        help='Template for text labels (e.g., "<en> <ase> {}")')
    
    args = parser.parse_args()
    
    evaluate_zero_shot(
        data_dir=args.data_dir,
        embedding_dir=args.embedding_dir,
        model_name=args.model_name,
        label_type=args.label_type,
        text_template=args.text_template
    )


if __name__ == '__main__':
    main()