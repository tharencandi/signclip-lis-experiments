"""
Precompute pose embeddings with optional SignCLIP-style normalisation.

Normalisation steps (when --normalize is enabled):
1. Shoulder-based normalisation: D_shoulders = 1, mid-point = (0, 0)
2. E6: Remove redundant keypoints (keep body, hands, face)
3. E6.2: Anonymisation - subtract first frame, add mean pose (motion-relative)

Note: E6.1 distribution standardization requires global dataset statistics which we don't have.
Per-sample normalisation destroys discriminative features (accuracy drops from 50% to 20%).

Usage:
    # A3LIS dataset WITH normalisation  
    python src/precompute_embeddings.py \
        --dataset_root dataset/A3LIS_dataset_poses \
        --output_dir dataset/embeddings/a3lis_normalised \
        --normalize \
        --model_name default
    
    # A3LIS dataset WITHOUT normalisation (raw embeddings)
    python src/precompute_normalised_embeddings.py \
        --dataset_root dataset/A3LIS_dataset_poses \
        --output_dir dataset/embeddings/a3lis_raw \
        --no-normalize \
        --model_name default
    
    # Legacy mode with normalisation
    python src/precompute_normalised_embeddings.py \
        --pose_dir dataset/poses \
        --output_dir dataset/embeddings/normalised \
        --normalize \
        --model_name default
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


def load_and_normalise_pose(pose_path: Path, normalize: bool = False,
                            remove_redundant: bool = True, 
                            anonymize: bool = False) -> Optional[np.ndarray]:
    """
    Load a .pose file and optionally apply SignCLIP normalization.
    
    Args:
        pose_path: Path to .pose file
        normalize: Apply normalization (if False, load raw pose)
        remove_redundant: Remove redundant keypoints (E6) - only if normalize=True
        anonymize: Apply first-frame anonymization (E6.2) - only if normalize=True
    
    Returns:
        pose (normalised or raw) as numpy array, or None if failed
    """
    try:
        from pose_format import Pose
        from pose_format.numpy import NumPyPoseBody
        
        # Load pose
        with open(pose_path, 'rb') as f:
            pose = Pose.read(f.read(), NumPyPoseBody)
        
        # Apply normalization only if enabled
        if normalize:
            # E6: Remove redundant keypoints (keep body, hands, face - filter out extras)
            # This step is optional and depends on the pose format
            if remove_redundant:
                try:
                    # Get component names from header
                    components = [c.name for c in pose.header.components]
                    
                    # Keep essential components for sign language
                    keep_components = []
                    for component in ['POSE_LANDMARKS', 'LEFT_HAND_LANDMARKS', 'RIGHT_HAND_LANDMARKS', 
                                    'FACE_LANDMARKS', 'pose_keypoints_2d', 'hand_left_keypoints_2d',
                                    'hand_right_keypoints_2d', 'face_keypoints_2d']:
                        if component in components:
                            keep_components.append(component)
                    
                    # Only filter if we found components to keep
                    if keep_components and len(keep_components) < len(components):
                        pose = pose.get_components(keep_components)
                except Exception as e:
                    # If component filtering fails, continue without it
                    pass
            
            # Step 1: Regular normalization (shoulder-based)
            # This sets D_shoulders = 1, mid-point = (0, 0)
            pose.normalize()
            
            # E6.2: Anonymization - remove first frame appearance, add mean pose
            # This makes motion relative while preserving dynamics
            if anonymize and pose.body.data.shape[0] > 1:  # Need at least 2 frames
                try:
                    # Get pose data: shape is (frames, people, points, dims)
                    data = pose.body.data
                    
                    # Compute mean pose across all frames for this video
                    mean_pose = np.mean(data, axis=0, keepdims=True)  # (1, people, points, dims)
                    
                    # Subtract first frame, then add mean pose
                    first_frame = data[0:1]  # (1, people, points, dims)
                    pose.body.data = data - first_frame + mean_pose
                    
                except Exception as e:
                    # If anonymization fails, continue without it
                    pass
            
            # Note: E6.1 distribution standardization requires global mean/std across entire dataset
            # Applying per-sample normalization destroys discriminative features
            # We skip it here since we don't have the global statistics
        
        return pose
        
    except Exception as e:
        print(f"\nError loading/normalizing {pose_path.name}: {e}")
        return None


def precompute_normalised_embeddings(
    pose_dir: str,
    output_dir: str,
    model_name: str = 'default',
    normalize: bool = False,
    remove_redundant: bool = True,
    anonymize: bool = False,
    dataset_root: Optional[str] = None,
    signit_root: Optional[str] = None,
):
    """
    Compute pose embeddings with optional normalization.
    
    Args:
        pose_dir: Directory containing original .pose files (legacy mode)
        output_dir: Directory to save embeddings
        model_name: SignCLIP model to use
        normalize: Apply normalization (if False, use raw poses)
        remove_redundant: Remove redundant keypoints (E6) - only if normalize=True
        anonymize: Apply first-frame anonymization (E6.2) - only if normalize=True
        dataset_root: Root directory for A3LIS dataset
        signit_root: Root directory for SignIT dataset (uses signit package)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which mode to use
    if dataset_root is not None:
        # A3LIS dataset mode
        print("Using A3LIS dataset package")
        try:
            dataset_parent = Path(dataset_root).parent
            if str(dataset_parent) not in sys.path:
                sys.path.insert(0, str(dataset_parent))
            from A3LIS_dataset_poses.a3lis import get_dataset, get_split_info
            pose_items = get_dataset(use_categories=True)
            if not pose_items:
                print(f"No data loaded from A3LIS dataset")
                return
            split_info = get_split_info()
            if split_info:
                print(f"  Split strategy: {split_info['strategy']}")
                print(f"  Train: {split_info.get('train_count', '?')} samples from {len(split_info['train_signers'])} signers")
                print(f"  Test:  {split_info.get('test_count', '?')} samples from {len(split_info['test_signers'])} signers")
                if split_info.get('val_signers'):
                    print(f"  Val:   {split_info.get('val_count', '?')} samples from {len(split_info['val_signers'])} signers")
                print(f"  Total: {split_info.get('total_count', '?')} samples")
            else:
                print(f"  Found {len(pose_items)} pose files (no split info)")
        except ImportError as e:
            print(f"ERROR: Could not import A3LIS dataset package: {e}")
            print(f"Make sure the dataset is at {dataset_root}")
            return

    elif signit_root is not None:
        # SignIT dataset mode
        print("Using SignIT dataset package")
        try:
            signit_path = Path(signit_root)
            if str(signit_path) not in sys.path:
                sys.path.insert(0, str(signit_path))
            from signit import get_dataset
            pose_items = get_dataset(use_categories=True)
            if not pose_items:
                print(f"No data loaded from SignIT dataset")
                return
            from collections import Counter as _Counter
            split_counts = _Counter(item['split'] for item in pose_items)
            print(f"  Found {len(pose_items)} pose files: {dict(split_counts)}")
        except ImportError as e:
            print(f"ERROR: Could not import SignIT package: {e}")
            print(f"Make sure {signit_root}/signit/__init__.py exists")
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
    
    if normalize:
        print(f"Normalization steps:")
        print(f"  - Shoulder-based: D_shoulders=1, mid-point=(0,0)")
        if remove_redundant:
            print(f"  - E6: Remove redundant keypoints")
        if anonymize:
            print(f"  - E6.2: Anonymization (first-frame relative + mean pose)")
    else:
        print(f"Normalization: DISABLED (using raw poses)")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_name}\n")
    
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
        pose = load_and_normalise_pose(pose_file, normalize, remove_redundant, anonymize)
        if pose is None:
            error_count += 1
            continue
        
        try:
            # Compute embedding from pose
            embedding = embed_pose(pose, model_name=model_name)
            
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
                'normalization': {
                    'enabled': normalize,
                    'shoulder_based': normalize,
                    'remove_redundant': normalize and remove_redundant,
                    'anonymize': normalize and anonymize
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
        if normalize:
            print(f"\n✓ Normalised embeddings saved to {output_dir}")
        else:
            print(f"\n✓ Raw embeddings saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute pose embeddings with optional SignCLIP-style normalization"
    )
    
    # Add mutually exclusive group for dataset source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--dataset_root',
        type=str,
        help='Root directory for A3LIS dataset'
    )
    source_group.add_argument(
        '--signit_root',
        type=str,
        help='Root directory for SignIT dataset (must contain signit/ package)'
    )
    source_group.add_argument(
        '--pose_dir',
        type=str,
        help='Directory containing .pose files (legacy mode, labels from filenames)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save normalised embeddings'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='default',
        choices=['default', 'asl_citizen', 'asl_finetune', 'suisse', 'a3lis_finetune'],
        help='SignCLIP model to use (a3lis_finetune loads from runs/signclip_a3lis_finetune/best_checkpoint.pt)'
    )
    
    # Normalization control
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Apply pose normalization (default: True)'
    )
    normalize_group.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip normalization, use raw poses'
    )
    
    # Fine-grained normalization control (only applies when --normalize is used)
    parser.add_argument(
        '--no-remove-redundant',
        action='store_true',
        help='Skip removing redundant keypoints (E6) - only applies when normalizing'
    )
    
    parser.add_argument(
        '--no-anonymize',
        action='store_true',
        help='Skip first-frame anonymization (E6.2) - only applies when normalizing'
    )
    
    args = parser.parse_args()
    
    # Determine normalize flag
    normalize = not args.no_normalize
    
    precompute_normalised_embeddings(
        pose_dir=args.pose_dir or '',
        output_dir=args.output_dir,
        model_name=args.model_name,
        normalize=normalize,
        remove_redundant=not args.no_remove_redundant,
        anonymize=not args.no_anonymize,
        dataset_root=args.dataset_root,
        signit_root=args.signit_root,
    )


if __name__ == '__main__':
    main()
