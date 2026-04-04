# SignCLIP Evaluation Tools

Python package for zero-shot and few-shot evaluation of sign language recognition using SignCLIP.

## Structure

```
src/
├── __init__.py                  # Package exports
├── pose_dataset.py              # Dataset class for loading pose files
├── demo_sign.py                 # SignCLIP embedding utilities
├── precompute_embeddings.py     # Script to compute and save embeddings
├── zero_shot.py                 # Script for zero-shot evaluation
├── few_shot.py                  # Script for few-shot evaluation
├── test_imports.py              # Test script to verify imports work
└── README.md                    # This file
```

**Note:** All scripts automatically add the project root to `sys.path`, so they can import from the `signclip` package without manual path configuration.

## Usage

All scripts can be imported as a package or run directly from the command line.

### As a Package

```python
import sys
sys.path.insert(0, 'path/to/signclip-lis-experiments/src')

from pose_dataset import PoseDataset
from demo_sign import embed_pose, embed_text

# Use the classes and functions
dataset = PoseDataset('/path/to/poses', label_type='micro')
embeddings = embed_pose(pose, model_name='default')
```

### As Command Line Scripts

**1. Precompute Embeddings**

```bash
python src/precompute_embeddings.py --data_dir /path/to/poses --output_dir /path/to/embeddings
```

**2. Zero-Shot Evaluation**

```bash
python src/zero_shot.py --data_dir /path/to/poses --embedding_dir /path/to/embeddings
```

**3. Few-Shot Evaluation**

```bash
python src/few_shot.py --data_dir /path/to/poses --embedding_dir /path/to/embeddings --k_shot 5
```

## Verify Imports

Test that all dependencies and imports work correctly:

```bash
python src/test_imports.py
```

This will verify:
- `signclip` package is accessible
- `pose_format`, `torch`, `numpy`, etc. are installed
- All src modules can be imported

## Workflow

### 1. Precompute Embeddings (Recommended)

First, compute embeddings for all poses to enable fast experimentation:

```bash
python precompute_embeddings.py \
    --data_dir /path/to/pose/files \
    --output_dir /path/to/embeddings/default_model \
    --model_name default \
    --label_type micro
```

**Arguments:**
- `--data_dir`: Directory containing `.pose` files
- `--output_dir`: Where to save `.npy` embedding files
- `--model_name`: SignCLIP model (`default`, `asl_finetune`, `asl_citizen`, `suisse`)
- `--label_type`: Use `micro` (fine-grained) or `macro` (coarse) labels

**Note:** Run this once per model. Embeddings are reused across all experiments.

### 2. Zero-Shot Evaluation

Match test poses directly to text class labels:

```bash
python zero_shot.py \
    --data_dir /path/to/pose/files \
    --embedding_dir /path/to/embeddings/default_model \
    --model_name default \
    --label_type micro
```

**Optional:** Add text templates (e.g., for ASL):
```bash
python zero_shot.py \
    --data_dir /path/to/pose/files \
    --embedding_dir /path/to/embeddings/default_model \
    --model_name default \
    --text_template "<en> <ase> {}"
```

### 3. Few-Shot Evaluation

Evaluate using k support examples per class:

```bash
python few_shot.py \
    --data_dir /path/to/pose/files \
    --embedding_dir /path/to/embeddings/default_model \
    --k_shot 5 \
    --label_type micro \
    --seed 42
```

**Run multiple experiments:**
```bash
# Compare different shot sizes
for k in 1 5 10 100; do
    python few_shot.py \
        --data_dir /path/to/poses \
        --embedding_dir /path/to/embeddings \
        --k_shot $k
done
```

## File Naming Convention

Pose files must follow this naming pattern:
```
[video_id]_[macro_label]_[micro_label]_[train|test]_[start_frame]_[end_frame].pose
```

**Example:**
```
01_animals_parrot_test_04783_05094.pose
```

This parses to:
- Video ID: `01`
- Macro label: `animals`
- Micro label: `parrot`
- Split: `test`
- Frames: 4783-5094

## Metrics

All scripts report:
- **Recall@1**: Top-1 accuracy
- **Recall@5**: Top-5 accuracy
- **Recall@10**: Top-10 accuracy
- **Median Rank**: Median position of correct class

## Example Output

```
==================================================
Zero-Shot Evaluation Results
==================================================
Model: default
Test samples: 1523
Classes: 187

Metrics:
  Recall@1:  0.4235 (645/1523)
  Recall@5:  0.7214 (1099/1523)
  Recall@10: 0.8213 (1251/1523)
  Median Rank: 3.0
==================================================
```

## Tips

1. **Precompute once, evaluate many times**: Precomputing embeddings makes experiments much faster
2. **Compare models**: Precompute embeddings for different models separately
3. **Experiment with text templates**: Try different prompt formats for zero-shot
4. **Use consistent seeds**: Set `--seed` for reproducible few-shot results
