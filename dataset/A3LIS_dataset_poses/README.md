# A3LIS Dataset Package

Italian Sign Language (LIS) pose dataset with train/test splits and semantic categories.

## Package Structure

```
dataset/A3LIS_dataset_poses/
├── a3lis/                          # Python package
│   └── __init__.py                 # Main loader code
├── a3lis_poses/                    # Pose data organized by signer
│   ├── fal/                        # Signer subdirectories
│   ├── fef/
│   ├── fsf/
│   ├── mdp/
│   ├── mdq/
│   ├── mic/
│   ├── mmr/
│   ├── mrla/
│   ├── mrlb/
│   └── msf/
├── sign_dictionary.csv             # Italian to English translations
├── sign_dictionary_with_categories.csv  # With macro categories
├── train_test_split.json           # Split configuration
├── data_loader.py                  # Legacy loader (deprecated)
└── example_usage.py                # Usage examples
```

## Dataset Statistics

- **Total samples**: 1,490 pose files
- **Signers**: 10 (fal, fef, fsf, mdp, mdq, mic, mmr, mrla, mrlb, msf)
- **Signs**: 147 unique Italian signs
- **Train/Test split**: 70/30 (signer-independent)
  - Train: 1,043 samples (7 signers)
  - Test: 447 samples (3 signers)

## Categories

| Category | Signs | Samples | Description |
|----------|-------|---------|-------------|
| Ente Pubblico | 25 | 250 | Public institutions (bank, municipality, etc.) |
| Vita Comune | 29 | 290 | Common life (home, food, days of week, etc.) |
| Istruzione | 20 | 200 | Education (school, teacher, student, etc.) |
| Ospedale | 14 | 140 | Hospital (doctor, nurse, medicine, etc.) |
| Stazione | 12 | 120 | Railway station (train, ticket, platform, etc.) |
| Autostrada | 3 | 30 | Highway (toll, traffic, road, etc.) |

## Usage

### Import the package

```python
from dataset.A3LIS_dataset_poses.a3lis import get_dataset, get_split_info
```

### Load all data

```python
all_data = get_dataset()
# Returns: List of 1,490 items with metadata
```

### Load by split

```python
train_data = get_dataset(split_filter='train')  # 1,043 samples
test_data = get_dataset(split_filter='test')    # 447 samples
```

### Get split information

```python
split_info = get_split_info()
print(f"Train: {split_info['train_count']} samples")
print(f"Test: {split_info['test_count']} samples")
print(f"Train signers: {split_info['train_signers']}")
print(f"Test signers: {split_info['test_signers']}")
```

### Sample item structure

```python
{
    'file_path': '/path/to/a3lis_poses/fal/fal_casa.pose',
    'signer': 'fal',
    'label_italian': 'casa',
    'labels_english': ['house', 'home'],
    'split': 'train',
    'category': 'Vita Comune'
}
```

### Filter by category

```python
from collections import defaultdict

all_data = get_dataset()
by_category = defaultdict(list)

for item in all_data:
    if 'category' in item:
        by_category[item['category']].append(item)

# Access specific category
hospital_signs = by_category['Ospedale']  # 140 samples
```

## Split Configuration

Edit `train_test_split.json` to customize the split:

```json
{
  "split_strategy": "signer_independent",
  "train_signers": ["fal", "fef", "fsf", "mdp", "mdq", "mic", "mmr"],
  "test_signers": ["mrla", "mrlb", "msf"]
}
```

### Alternative split strategies

1. **Signer-independent** (current): Train and test on different signers
2. **Random split**: Randomly assign 70% of each signer's samples to train
3. **Leave-one-out**: Use each signer as test set once (10-fold CV)
4. **Sign-independent**: Train on some signs, test on completely different signs

## Integration with Scripts

### Precompute embeddings

```bash
python src/precompute_normalized_embeddings.py \
    --dataset_root dataset/A3LIS_dataset_poses \
    --output_dir dataset/embeddings/a3lis_normalized \
    --model_name default
```

The output metadata will include split and category information:

```json
{
  "model_name": "default",
  "dataset": {
    "split_counts": {"train": 1043, "test": 447},
    "category_counts": {...}
  },
  "embeddings": [
    {
      "embedding_file": "fal_casa.npy",
      "signer": "fal",
      "label_italian": "casa",
      "labels_english": ["house", "home"],
      "split": "train",
      "category": "Vita Comune"
    },
    ...
  ]
}
```

## Example Script

Run the provided example:

```bash
python dataset/A3LIS_dataset_poses/example_usage.py
```

## Notes

- The legacy `data_loader.py` is deprecated but kept for backward compatibility
- Place names (ancona, roma, etc.), colors, and special signs are excluded from main categories
- Some signs have multiple English translations separated by semicolons
- The signer-independent split ensures the model generalizes to unseen signers
