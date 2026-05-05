"""
Canonical path constants for the A3LIS-147 dataset.

All paths are relative to the project root (the directory that contains
`signclip/`, `dataset/`, `src/`, etc.).  Import these instead of
repeating string literals across scripts.

Used by:
  - signclip/tasks/a3lis_finetune.py
  - src/eval_finetune.py
"""

# Root directory of all A3LIS pose files
POSES_ROOT = "dataset/A3LIS_dataset_poses/a3lis_poses"

# Sign dictionary CSV (Italian → English label mapping)
CSV_PATH = "sign_dictionary.csv"

# 70/10/20 signer-independent split definition
SPLIT_CONFIG_PATH = "dataset/A3LIS_dataset_poses/train_test_val_split.json"

# Pre-tokenized English label cache (speeds up dataset init)
PRETOKENIZED_LABELS_PATH = "dataset/A3LIS_dataset_poses/a3lis_labels_tokenized.pkl"

# Default fine-tuning checkpoint produced by scripts/run_finetune.py
DEFAULT_CHECKPOINT_PATH = "runs/signclip_a3lis_finetune/checkpoint_best.pt"
