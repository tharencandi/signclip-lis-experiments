#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly in src/
Run this from the project root: python src/test_imports.py
"""

import sys
from pathlib import Path

# Add project root to path (same as other scripts)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Testing imports...")
print(f"Python path includes: {project_root}")

# Test importing from signclip package
try:
    from signclip.models import MMPTModel
    print("✓ signclip.models.MMPTModel")
except ImportError as e:
    print(f"✗ signclip.models.MMPTModel: {e}")

# Test importing src modules
try:
    from pose_dataset import PoseDataset
    print("✓ pose_dataset.PoseDataset")
except ImportError as e:
    print(f"✗ pose_dataset.PoseDataset: {e}")

try:
    from demo_sign import embed_pose, embed_text
    print("✓ demo_sign.embed_pose, embed_text")
except ImportError as e:
    print(f"✗ demo_sign.embed_pose, embed_text: {e}")

# Test other dependencies
try:
    from pose_format import Pose
    print("✓ pose_format.Pose")
except ImportError as e:
    print(f"✗ pose_format.Pose: {e}")

try:
    import torch
    print(f"✓ torch (version {torch.__version__})")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    import numpy as np
    print(f"✓ numpy (version {np.__version__})")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    from tqdm import tqdm
    print("✓ tqdm")
except ImportError as e:
    print(f"✗ tqdm: {e}")

print("\nAll imports tested!")
