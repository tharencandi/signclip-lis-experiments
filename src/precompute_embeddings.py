"""
Precompute pose embeddings.

Default behavior uses the canonical SignCLIP preprocessing exactly once via
demo_sign.embed_pose -> preprocess_pose.

Optional first-frame anonymization can be enabled before embed_pose.

Usage:
    # Recommended: canonical preprocessing only (single pass)
    python src/precompute_embeddings.py \
        --dataset_root dataset/A3LIS_dataset_poses \
        --output_dir dataset/embeddings/a3lis_canonical \
        --model_name default

    # Optional: first-frame anonymization before canonical preprocessing
    python src/precompute_embeddings.py \
        --dataset_root dataset/A3LIS_dataset_poses \
        --output_dir dataset/embeddings/a3lis_anonymized \
        --model_name default \
        --anonymize
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from demo_sign import embed_pose


def load_and_normalise_pose(pose_path: Path, anonymize: bool = False) -> Optional[np.ndarray]:
    """
    Load a .pose file and optionally apply first-frame anonymization.
    
    Args:
        pose_path: Path to .pose file
        anonymize: Apply first-frame anonymization before embed_pose().
    
    Returns:
        pose (normalised or raw) as numpy array, or None if failed
    """
    try:
        from pose_format import Pose
        from pose_format.numpy import NumPyPoseBody
        
        # Load pose
        with open(pose_path, 'rb') as f:
            pose = Pose.read(f.read(), NumPyPoseBody)
        
        if anonymize and pose.body.data.shape[0] > 1:
            try:
                data = pose.body.data
                mean_pose = np.mean(data, axis=0, keepdims=True)
                first_frame = data[0:1]
                pose.body.data = data - first_frame + mean_pose
            except Exception:
                pass
        
        return pose
        
    except Exception as e:
        print(f"\nError loading/normalizing {pose_path.name}: {e}")
        return None


def precompute_normalised_embeddings(
    pose_dir: str,
    output_dir: str,
    model_name: str = 'default',
    checkpoint_path: Optional[str] = None,
    anonymize: bool = False,
    dataset_root: Optional[str] = None
):
    """
    Compute pose embeddings with canonical preprocessing and optional anonymization.
    
    Args:
        pose_dir: Directory containing original .pose files (legacy mode)
        output_dir: Directory to save embeddings
        model_name: SignCLIP model to use
        checkpoint_path: Optional checkpoint override (used with model_name=a3lis_finetune)
        anonymize: Apply first-frame anonymization before embed_pose().
        dataset_root: Root directory for A3LIS dataset (uses data_loader.py)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which mode to use
    use_dataloader = dataset_root is not None
    
    if use_dataloader:
        # A3LIS dataset mode - use a3lis package
        print("Using A3LIS dataset package")
        
        # Import the a3lis package
        try:
            # Add dataset directory to path if needed
            dataset_parent = Path(dataset_root).parent
            if str(dataset_parent) not in sys.path:
                sys.path.insert(0, str(dataset_parent))
            
            # Import from the package
            from A3LIS_dataset_poses.a3lis import get_dataset, get_split_info
            
            # Load all data (train + test)
            dataset = get_dataset(use_categories=True)
            
            if not dataset:
                print(f"No data loaded from A3LIS dataset")
                return
            
            # Show split information
            split_info = get_split_info()
            if split_info:
                print(f"  Split strategy: {split_info['strategy']}")
                print(f"  Train: {split_info.get('train_count', '?')} samples from {len(split_info['train_signers'])} signers")
                print(f"  Test:  {split_info.get('test_count', '?')} samples from {len(split_info['test_signers'])} signers")
                if split_info.get('val_signers'):
                    print(f"  Val:   {split_info.get('val_count', '?')} samples from {len(split_info['val_signers'])} signers")
                print(f"  Total: {split_info.get('total_count', '?')} samples")
            else:
                print(f"  Found {len(dataset)} pose files (no split info)")
            
            pose_items = dataset
            
        except ImportError as e:
            print(f"ERROR: Could not import A3LIS dataset package: {e}")
            print(f"Make sure the dataset is at {dataset_root}")
            return
        
    else:
        # Legacy mode - scan pose directory
        print("Using legacy mode - scanning pose directory")
        pose_path = Path(pose_dir)
        pose_files = sorted(pose_path.glob('*.pose'))
        if not pose_files:
            print(f"No .pose files found in {pose_dir}")
            return
        
        print(f"Found {len(pose_files)} pose files")
        # Convert to dict format for uniform processing
        pose_items = [{
            'file_path': str(f),
            'signer': 'unknown',
            'label_italian': f.stem,
            'labels_english': ['unknown']
        } for f in pose_files]
    
    print(f"Anonymization (first-frame): {'ENABLED' if anonymize else 'DISABLED'}")
    print("Canonical preprocess_pose() inside embed_pose() is always applied once.")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_name}\n")
    if checkpoint_path:
        print(f"Checkpoint override: {checkpoint_path}\n")
    
    # Process each pose file
    success_count = 0
    skip_count = 0
    error_count = 0
    metadata_list = []
    
    for item in tqdm(pose_items, desc="Processing poses"):
        pose_file = Path(item['file_path'])
        
        # Create output filename from original pose filename
        output_name = pose_file.stem + '.npy'
        embedding_file = output_path / output_name
        
        # Check if embedding already exists
        if embedding_file.exists():
            skip_count += 1
            continue
        
        # Load pose (with or without normalization)
        pose = load_and_normalise_pose(pose_file, anonymize)
        if pose is None:
            error_count += 1
            continue
        
        try:
            # Compute embedding from pose
            embedding = embed_pose(
                pose,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
            )
            
            # Save embedding
            np.save(embedding_file, embedding)
            success_count += 1
            
            # Store metadata
            meta_item = {
                'file_path': str(pose_file),
                'embedding_file': output_name,
                'signer': item.get('signer', 'unknown'),
                'label_italian': item.get('label_italian', ''),
                'labels_english': item.get('labels_english', []),
                'split': item.get('split', 'unknown')
            }
            
            # Add category if available
            if 'category' in item:
                meta_item['category'] = item['category']
            
            metadata_list.append(meta_item)
            
        except Exception as e:
            print(f"\nError computing embedding for {pose_file.name}: {e}")
            error_count += 1
            continue
    
    # Save metadata
    if metadata_list:
        # Compute split statistics
        from collections import Counter
        split_counts = Counter([m['split'] for m in metadata_list])
        category_counts = Counter([m.get('category', 'no_category') for m in metadata_list])
        
        metadata_file = output_path / 'embeddings_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': model_name,
                'checkpoint_path': checkpoint_path,
                'preprocess': {
                    'canonical_preprocess_pose': True,
                    'anonymize_first_frame': anonymize,
                    'note': 'Canonical preprocess_pose() is applied in embed_pose().'
                },
                'dataset': {
                    'total_embeddings': len(metadata_list),
                    'split_counts': dict(split_counts),
                    'category_counts': dict(category_counts)
                },
                'embeddings': metadata_list
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Metadata saved to {metadata_file}")
        print(f"  Split breakdown: {dict(split_counts)}")
        if category_counts:
            print(f"  Category breakdown: {dict(sorted(category_counts.items()))}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped (already exist): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(pose_items)}")
    print(f"{'='*60}")
    
    if success_count > 0 or skip_count > 0:
        print(f"\n✓ Embeddings saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute pose embeddings (canonical preprocessing by default)"
    )
    
    # Add mutually exclusive group for dataset source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--dataset_root',
        type=str,
        help='Root directory for A3LIS dataset (uses data_loader.py)'
    )
    source_group.add_argument(
        '--pose_dir',
        type=str,
        help='Directory containing original .pose files (legacy mode)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save embeddings'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='default',
        choices=['default', 'asl_citizen', 'asl_finetune', 'suisse', 'a3lis_finetune'],
        help='SignCLIP model to use (a3lis_finetune loads from runs/signclip_a3lis_finetune/best_checkpoint.pt)'
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Optional checkpoint override for model loading. '
             'Most useful with --model_name a3lis_finetune.'
    )
    
    parser.add_argument(
        '--anonymize',
        action='store_true',
        help='Apply first-frame anonymization before canonical preprocess_pose().'
    )
    
    args = parser.parse_args()
    
    precompute_normalised_embeddings(
        pose_dir=args.pose_dir or '',
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        anonymize=args.anonymize,
        dataset_root=args.dataset_root
    )


if __name__ == '__main__':
    main()
