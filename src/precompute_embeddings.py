"""
Precompute and save pose embeddings for faster experimentation.
Usage: python precompute_embeddings.py --data_dir /path/to/poses --output_dir /path/to/embeddings
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to path for signclip imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pose_dataset import PoseDataset
from demo_sign import embed_pose


def precompute_embeddings(data_dir: str, output_dir: str, model_name: str = 'default', label_type: str = 'micro'):
    """
    Precompute embeddings for all poses in dataset.
    
    Args:
        data_dir: Directory containing .pose files
        output_dir: Directory to save .npy embeddings
        model_name: SignCLIP model to use (default, asl_finetune, etc.)
        label_type: 'micro' or 'macro' label granularity
    """
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = PoseDataset(data_dir, label_type=label_type)
    print(f"Found {len(dataset)} pose files")
    print(f"Classes: {dataset.num_classes}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute embeddings
    print(f"\nComputing embeddings using model '{model_name}'...")
    for idx in tqdm(range(len(dataset)), desc="Processing poses"):
        pose, label, meta = dataset[idx]
        
        # Check if embedding already exists
        out_file = output_path / meta['filename'].replace('.pose', '.npy')
        if out_file.exists():
            continue
        
        # Compute embedding
        try:
            embedding = embed_pose(pose, model_name=model_name)
            
            # Save embedding
            np.save(out_file, embedding)
        except Exception as e:
            print(f"\nError processing {meta['filename']}: {e}")
            continue
    
    print(f"\n✓ Embeddings saved to {output_dir}")
    
    # Print summary by split
    train_samples = dataset.filter_split('train')
    test_samples = dataset.filter_split('test')
    print(f"\nSummary:")
    print(f"  Train: {len(train_samples)} samples, {train_samples.num_classes} classes")
    print(f"  Test:  {len(test_samples)} samples, {test_samples.num_classes} classes")


def main():
    parser = argparse.ArgumentParser(description="Precompute pose embeddings for SignCLIP evaluation")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .pose files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save embeddings')
    parser.add_argument('--model_name', type=str, default='default', 
                        choices=['default', 'asl_citizen', 'asl_finetune', 'suisse'],
                        help='SignCLIP model to use')
    parser.add_argument('--label_type', type=str, default='micro', 
                        choices=['micro', 'macro'],
                        help='Label granularity')
    
    args = parser.parse_args()
    
    precompute_embeddings(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_type=args.label_type
    )


if __name__ == '__main__':
    main()
