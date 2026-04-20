#!/usr/bin/env python
"""
Setup script for local SignCLIP configuration.

This script:
1. Creates necessary directories
2. Copies and updates config with local paths
3. Creates dummy dataset files
4. Creates symlink to pretrained model

Usage:
    python setup_local_config.py
"""

from pathlib import Path
import re

def setup_local_config():
    """Setup local SignCLIP configuration for pose/text embedding."""
    
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")
    
    # 1. Create directories
    print("\n[1/4] Creating directories...")
    runs_dir = project_root / "runs" / "signclip_local"
    runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created {runs_dir}")
    
    projects_dir = project_root / "projects" / "retri" / "signclip_v1_1"
    projects_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created {projects_dir}")
    
    # 2. Copy and update config file
    print("\n[2/4] Creating config file...")
    src_config = project_root / "configs" / "signclip_v1_1" / "baseline_temporal.yaml"
    dst_config = projects_dir / "baseline_temporal.yaml"
    
    if not src_config.exists():
        print(f"  ✗ Source config not found: {src_config}")
        return False
    
    # Read config and replace paths
    config_content = src_config.read_text()
    
    # Replace all absolute paths with local paths
    config_content = re.sub(
        r'/home/zifjia/.*?/(test|val|train)\.txt',
        str(runs_dir / r'dummy_\1.txt'),
        config_content
    )
    config_content = re.sub(
        r'/shares/.*?/SperadTheSign\.csv',
        str(runs_dir / 'dummy_metadata.csv'),
        config_content
    )
    config_content = re.sub(
        r'vfeat_dir: /shares/.*',
        f'vfeat_dir: {runs_dir}',
        config_content
    )
    config_content = re.sub(
        r'save_dir: runs/retri_v1_1/baseline_temporal',
        f'save_dir: {runs_dir}',
        config_content
    )
    config_content = re.sub(
        r'save_path: runs/retri_v1_1/baseline_temporal',
        f'save_path: {runs_dir}',
        config_content
    )
    config_content = re.sub(
        r'tensorboard_logdir: run',
        f'tensorboard_logdir: {runs_dir}',
        config_content
    )
    
    dst_config.write_text(config_content)
    print(f"  ✓ Created {dst_config}")
    
    # 3. Create dummy dataset files
    print("\n[3/4] Creating dummy dataset files...")
    for split in ['train', 'val', 'test']:
        dummy_file = runs_dir / f"dummy_{split}.txt"
        dummy_file.touch(exist_ok=True)
        print(f"  ✓ Created {dummy_file}")
    
    # Create dummy metadata CSV
    metadata_file = runs_dir / "dummy_metadata.csv"
    metadata_file.touch(exist_ok=True)
    print(f"  ✓ Created {metadata_file}")
    
    # 4. Create symlink to checkpoint
    print("\n[4/4] Setting up model checkpoint...")
    model_file = project_root / "pretrained_models" / "baseline_temporal_checkpoint_best.pt"
    checkpoint_link = runs_dir / "checkpoint_best.pt"
    
    if not model_file.exists():
        print(f"  ⚠ Model not found: {model_file}")
        print(f"    Download it and place it in pretrained_models/")
        return False
    
    # Create symlink
    if checkpoint_link.exists():
        checkpoint_link.unlink()
    checkpoint_link.symlink_to(model_file)
    
    size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Model linked: {model_file} ({size_mb:.1f} MB)")
    print(f"  ✓ Symlink created: {checkpoint_link}")
    
    # Success summary
    print("\n" + "="*60)
    print("✓ Setup complete!")
    print("="*60)
    print("\nYou can now use SignCLIP for embedding:")
    print("\n  from src.demo_sign import embed_pose, embed_text")
    print("  embedding = embed_pose(pose, model_name='default')")
    print("  text_emb = embed_text('hello', model_name='default')")
    print("\nOr run precompute scripts:")
    print("\n  python src/precompute_embeddings.py text \\")
    print("    --labels_path dataset/labels/ \\")
    print("    --output_dir dataset/embeddings/ \\")
    print("    --template '<en> <lis> {}'")
    print("="*60)
    
    return True


if __name__ == '__main__':
    success = setup_local_config()
    exit(0 if success else 1)
