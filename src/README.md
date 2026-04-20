# SignCLIP Evaluation Tools

Python package for zero-shot and few-shot evaluation of sign language recognition using SignCLIP.

## Structure

```
src/
├── __init__.py                              # Package exports
├── demo_sign.py                             # SignCLIP embedding utilities
├── precompute_normalised_embeddings.py      # Compute embeddings (with/without normalization)
├── zero_shot_precomputed.py                 # Zero-shot with prompt engineering
├── few_shot.py                              # Few-shot (KNN, Linear Probe, SVM)
├── visualize_embeddings.py                  # Visualize embeddings (no SignCLIP needed!)
└── README.md                                # This file
```

**Note:** All scripts automatically add the project root to `sys.path`, so they can import from the `signclip` package without manual path configuration.

## Dataset Setup

### A3LIS Italian Sign Language Dataset

The A3LIS dataset contains 1,490 pose files for 147 Italian Sign Language signs, organized with signer-independent train/test split.

**Dataset structure:**
```
dataset/A3LIS_dataset_poses/
├── a3lis/
│   ├── __init__.py
│   └── data_loader.py                        # Dataset loading logic
├── train_test_split.json                     # Signer-based split (7 train, 3 test)
├── sign_dictionary_with_categories.csv       # Sign labels + 6 macro categories
└── poses/
    ├── [signer]_[label_italian].pose         # Pose files (NOT in repo)
    └── ...
```

**⚠️ Important: Pose files are NOT included in the repository due to size constraints.**

### How to Set Up the Dataset

1. **Obtain the pose files**: Contact the dataset maintainer or extract poses from A3LIS videos
2. **Place pose files** in `dataset/A3LIS_dataset_poses/poses/` with naming format:
   ```
   [signer]_[label_italian].pose
   ```
   Example: `fal_acqua.pose`, `mrla_ciao.pose`

3. **Verify structure**:
   ```bash
   ls dataset/A3LIS_dataset_poses/poses/*.pose | wc -l
   # Should show: 1490 pose files
   ```

4. **Required files** (already in repo):
   - `train_test_split.json` - Defines 70/30 signer split
   - `sign_dictionary_with_categories.csv` - Maps Italian→English labels + 6 categories
   - `a3lis/data_loader.py` - Loading logic

### Dataset Statistics

- **Total signs**: 147 unique signs
- **Total samples**: 1,490 pose files
- **Signers**: 10 total (7 train: fal, fef, fsf, mdp, mdq, mic, mmr; 3 test: mrla, mrlb, msf)
- **Split**: 70/30 signer-independent (train: ~1,043 samples, test: ~447 samples)
- **Categories**: 6 macro categories (Ente Pubblico-39, Stazione-35, Istruzione-30, Ospedale-19, Vita Comune-16, Autostrada-8)

### Alternative: Use Your Own Dataset

To use a different dataset, create a similar structure:

1. Create a package directory with `data_loader.py` implementing `get_dataset()` function
2. Create a CSV mapping labels to categories (optional)
3. Create a train/test split JSON (optional)
4. Modify `--dataset_root` to point to your dataset directory

## Quick Start (A3LIS Dataset)

### 1. Precompute Embeddings

```bash
# WITH normalization (SignCLIP-style shoulder-based, E6, E6.2)
python src/precompute_normalised_embeddings.py \
    --dataset_root dataset/A3LIS_dataset_poses \
    --output_dir dataset/embeddings/a3lis_normalised \
    --normalize \
    --model_name default

# WITHOUT normalization (raw poses)
python src/precompute_normalised_embeddings.py \
    --dataset_root dataset/A3LIS_dataset_poses \
    --output_dir dataset/embeddings/a3lis_raw \
    --no-normalize \
    --model_name default
```

### 2. Zero-Shot Evaluation

```bash
# Raw glosses (no template)
python src/zero_shot_precomputed.py \
    --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
    --split test \
    --label_language english

# With language tag (paper standard: "<en> <lis> {label}")
python src/zero_shot_precomputed.py \
    --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
    --split test \
    --label_language english \
    --prompt_type en_lis

# Macro categories (6 categories: Ente Pubblico, Stazione, Istruzione, Ospedale, Vita Comune, Autostrada)
python src/zero_shot_precomputed.py \
    --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
    --split test \
    --use_categories \
    --prompt_type en_lis
```

### 3. Few-Shot Evaluation

```bash
# K-Nearest Neighbors (paper standard)
python src/few_shot.py \
    --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
    --method knn \
    --label_language english

# Linear Probe (Logistic Regression)
python src/few_shot.py \
    --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
    --method linear_probe \
    --label_language english

# SVM with RBF kernel (advanced non-linear)
python src/few_shot.py \
    --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
    --method svm \
    --use_categories
```

## Model Configuration

SignCLIP requires a custom YAML config file that points to the downloaded model weights.

**Available models:**
- `default` - SignCLIP v1.1 baseline
- `asl_citizen` - ASL Citizen finetuned
- `asl_finetune` - ASL finetuned
- `suisse` - Swiss Sign Language finetuned

**Pre-configured for Colab:**
The repo includes a Colab-ready config at [`configs/signclip_v1_1/baseline_temporal_colab.yaml`](../configs/signclip_v1_1/baseline_temporal_colab.yaml) with paths set for `/content/signclip-lis-experiments/`.

**How it works:**
1. Notebook downloads weights to `runs/signclip_embed/checkpoint_best.pt`
2. Copies `configs/signclip_v1_1/baseline_temporal_colab.yaml` → `projects/retri/signclip_v1_1/baseline_temporal.yaml`
3. Creates dummy dataset files referenced in the config
4. Model loads automatically when you call `embed_pose()` or `embed_text()`

**Config structure:**
```
runs/signclip_embed/
  ├── checkpoint_best.pt              # Model weights
  └── dummy_*.txt                     # Placeholder files
  
projects/retri/signclip_v1_1/
  └── baseline_temporal.yaml          # Config copied from repo
  
configs/signclip_v1_1/
  └── baseline_temporal_colab.yaml    # Source config in repo
```

The config defines:
- Dataset processors (PoseProcessor, TextProcessor)
- Model architecture (MMFusionSeparate with BERT encoder)
- Checkpoint location (restore_file: checkpoint_best.pt)
- Training parameters (batch size, fp16, etc.)

**Note:** The Colab notebook handles this setup automatically — no manual config needed!

## Workflow

### 0. Inspect Dataset

Before running evaluations, check your dataset statistics:

```bash
python inspect_dataset.py \
    --data_dir /path/to/pose/files \
    --label_type micro
```

**Output:**
```
Total samples: 2345
Total classes: 215

Train split:
  Samples: 1523
  Classes: 187

Test split:
  Samples: 822
  Classes: 187

Class overlap (train ∩ test): 187 classes
```

This helps you understand:
- How many samples are in train vs test
- Whether you have class overlap (needed for few-shot) or not (for true zero-shot)
- How balanced your dataset is

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

All scripts report standard retrieval metrics:
- **R@1↑**: Recall@1 (Top-1 accuracy, higher is better)
- **R@5↑**: Recall@5 (correct answer in top 5, higher is better)
- **R@10↑**: Recall@10 (correct answer in top 10, higher is better)
- **MedianR↓**: Median Rank (median position of correct answer, lower is better)

## Example Output

```
============================================================
Results
============================================================
Split: test
Test samples: 447
Text classes: 147
Prompt: en_lis '<en> <lis> {}'

Retrieval Metrics:
  R@1↑:          45.23%  (  202/  447)
  R@5↑:          78.45%  (  351/  447)
  R@10↑:         89.04%  (  398/  447)
  MedianR↓:        3.2

Accuracy:
  Top-1:         45.23%  (  202/  447)
============================================================
```

## Prompt Engineering (Zero-Shot)

Available prompt templates for zero-shot evaluation:

- **`raw`**: `{}` - No template, raw gloss matching
- **`it_lis`**: `<it> <lis> {}` - Italian language tag
- **`en_lis`**: `<en> <lis> {}` - English language tag (paper standard for multilingual models)

Usage:
```bash
python src/zero_shot_precomputed.py \
    --pose_embeddings_dir dataset/embeddings/a3lis_normalised \
    --split test \
    --label_language english \
    --prompt_type en_lis
```

## Few-Shot Methods

Three methods implemented:

1. **KNN** (K=num_classes): Nonparametric classification using cosine similarity, K set to number of classes (paper standard)
2. **Linear Probe**: Logistic regression trained on frozen embeddings (default scikit-learn settings)
3. **SVM**: Support Vector Machine with RBF kernel for non-linear classification (advanced)

For A3LIS: Uses signer-independent split (~7 training examples per class, not random k-shot sampling).

## Normalization Options

SignCLIP-style normalization (when `--normalize` is used):

1. **Shoulder-based**: D_shoulders = 1, mid-point = (0, 0)
2. **E6**: Remove redundant keypoints (keep body, hands, face)
3. **E6.2**: Anonymization - subtract first frame, add mean pose (motion-relative)

Fine-grained control:
```bash
# Full normalization (default)
python src/precompute_normalised_embeddings.py \
    --dataset_root dataset/A3LIS_dataset_poses \
    --output_dir dataset/embeddings/a3lis_normalised \
    --normalize

# Skip E6 keypoint removal
python src/precompute_normalised_embeddings.py \
    --dataset_root dataset/A3LIS_dataset_poses \
    --output_dir dataset/embeddings/a3lis_normalised \
    --normalize \
    --no-remove-redundant

# Skip E6.2 anonymization
python src/precompute_normalised_embeddings.py \
    --dataset_root dataset/A3LIS_dataset_poses \
    --output_dir dataset/embeddings/a3lis_normalised \
    --normalize \
    --no-anonymize
```

## Tips

1. **Precompute once, evaluate many times**: Precomputing embeddings makes experiments much faster
2. **Compare models**: Precompute embeddings for different models separately
3. **Experiment with text templates**: Try different prompt formats for zero-shot
4. **Use consistent seeds**: Set `--seed` for reproducible few-shot results

## Working with Embeddings (No SignCLIP Needed!)

Once embeddings are precomputed as `.npy` files, you can do all kinds of analysis **without loading the SignCLIP model or heavy dependencies**. Just numpy, scikit-learn, and matplotlib!

### Visualize Embeddings

Create t-SNE, PCA plots, and similarity matrices:

```bash
python visualize_embeddings.py \
    --embedding_dir /path/to/embeddings/default_model \
    --output_dir ./plots \
    --label_type micro
```

**Outputs:**
- `tsne_visualization.png` - t-SNE plot colored by class and train/test split
- `pca_visualization.png` - PCA plot with explained variance
- `similarity_matrix.png` - Cosine similarity heatmap
- `class_statistics.json` - Per-class intra-class similarity stats
- `nearest_neighbors.json` - k-NN for each sample

**Optional arguments:**
```bash
python visualize_embeddings.py \
    --embedding_dir /path/to/embeddings \
    --output_dir ./plots \
    --label_type micro \
    --tsne_perplexity 30 \
    --k_neighbors 5 \
    --max_samples_sim 100  # Limit samples for similarity matrix
```

### Custom Analysis with Embeddings

You can load embeddings directly in your own Python scripts:

```python
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
embedding_dir = Path('/path/to/embeddings')
embeddings = []
labels = []

for npy_file in embedding_dir.glob("*.npy"):
    emb = np.load(npy_file).squeeze()
    embeddings.append(emb)
    
    # Parse label from filename
    # Format: [video_id]_[macro]_[micro]_[split]_[start]_[end].npy
    parts = npy_file.stem.split('_')
    label = parts[2]  # micro label
    labels.append(label)

embeddings = np.array(embeddings)

# Compute cosine similarity
sim_matrix = cosine_similarity(embeddings)

# Find nearest neighbors
query_idx = 0
similarities = sim_matrix[query_idx]
top_5_idx = np.argsort(similarities)[::-1][1:6]  # Exclude self

print(f"Top 5 similar to {labels[query_idx]}:")
for idx in top_5_idx:
    print(f"  {labels[idx]}: {similarities[idx]:.3f}")
```

**What you can do:**
- ✅ Cosine similarity search
- ✅ t-SNE / UMAP / PCA visualization
- ✅ Clustering (K-means, DBSCAN, etc.)
- ✅ Nearest neighbor classification
- ✅ Confusion matrix analysis
- ✅ Retrieve-by-example queries
- ✅ Cross-model comparison

**Dependencies for visualization:**
```bash
pip install numpy scikit-learn matplotlib seaborn
# Optional for UMAP:
pip install umap-learn
```

### Working with Text Embeddings

Text embeddings are saved in a different format for easy reuse:

```
text_embeddings_default_<en>_<lis>_.npy    # Numpy array (num_labels, embedding_dim)
text_labels_default_<en>_<lis>_.txt        # One label per line
```

**Load and use text embeddings:**

```python
import numpy as np

# Load text embeddings
text_embeddings = np.load('text_embeddings_default_<en>_<lis>_.npy')
with open('text_labels_default_<en>_<lis>_.txt', 'r') as f:
    text_labels = [line.strip() for line in f]

print(f"Text embeddings: {text_embeddings.shape}")
# (215, 512) - 215 class labels, 512-dim embeddings

# Use for zero-shot classification
pose_embedding = np.load('some_pose.npy').squeeze()
similarities = np.matmul(pose_embedding, text_embeddings.T)
top_k = np.argsort(similarities)[::-1][:5]

print("Top 5 predictions:")
for rank, idx in enumerate(top_k, 1):
    print(f"{rank}. {text_labels[idx]}: {similarities[idx]:.3f}")
```

**Recompute text embeddings for different templates:**

```bash
# For Italian Sign Language (LIS)
python precompute_embeddings.py text \
    --labels_path dataset/labels/ \
    --output_dir embeddings/ \
    --template "<en> <lis> {}"

# For American Sign Language (ASL)
python precompute_embeddings.py text \
    --labels_path dataset/labels/ \
    --output_dir embeddings/ \
    --template "<en> <asl> {}" \
    --model_name asl_finetune
```

This lets you:
- ✅ Precompute text embeddings once, reuse for all zero-shot evaluations
- ✅ Compare different text templates without re-embedding poses
- ✅ Analyze semantic space of different sign languages
- ✅ Build custom retrieval systems with cached embeddings
