# A3LIS Dataset Loader
# Loads Italian Sign Language (LIS) pose data with train/test splits

import pandas as pd
import os
import json
from glob import glob
from pathlib import Path

# --- CONFIG ---
dataset_path = "a3lis_poses"  # Updated to scan subdirectories
csv_path = "sign_dictionary.csv"
csv_with_categories_path = "sign_dictionary_with_categories.csv"
split_config_path = "train_test_split.json"
# --------------

def load_split_config():
    """Load train/test split configuration."""
    if not os.path.exists(split_config_path):
        print(f"Warning: {split_config_path} not found. All samples will be marked as 'unknown' split.")
        return None
    
    with open(split_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def load_dictionary(use_categories=True):
    """
    Loads the CSV and creates lookup dictionaries.
    
    Args:
        use_categories: If True, try to load CSV with categories first
    
    Returns:
        Tuple of (italian_to_english, italian_to_category)
    """
    # Try to load CSV with categories first
    csv_file = csv_path
    has_categories = False
    
    if use_categories and os.path.exists(csv_with_categories_path):
        csv_file = csv_with_categories_path
        has_categories = True
    elif not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}!")
    
    df = pd.read_csv(csv_file)
    
    # Create italian -> english lookup
    it_to_en = {}
    for index, row in df.iterrows():
        italian = row['label_italian']
        english_raw = str(row['label_english'])
        # Handle multiple labels split by semicolon
        english_list = [word.strip() for word in english_raw.split(';')]
        it_to_en[italian] = english_list
    
    # Create italian -> category lookup if available
    it_to_category = {}
    if has_categories and 'category' in df.columns:
        for index, row in df.iterrows():
            italian = row['label_italian']
            category = str(row['category']).strip()
            # Skip if category has multiple options (marked with /)
            if '/' not in category and category not in ['nan', '', 'PLACE_NAME', 'COLOR', 'OTHER']:
                it_to_category[italian] = category
    
    return it_to_en, it_to_category

def get_dataset(use_categories=True, split_filter=None):
    """
    Scans the pose folders and links every file to its labels and metadata.
    
    Args:
        use_categories: If True, load category information from CSV
        split_filter: Optional filter - 'train', 'test', or None for all
    
    Returns:
        List of items: [{'file_path': ..., 'signer': ..., 'split': ..., ...}, ...]
    """
    # 1. Load the dictionaries
    it_to_en, it_to_category = load_dictionary(use_categories=use_categories)
    
    # 2. Load split configuration
    split_config = load_split_config()
    train_signers = set(split_config['train_signers']) if split_config else set()
    test_signers = set(split_config['test_signers']) if split_config else set()
    
    # 3. Find all .pose files recursively
    files = sorted(glob(os.path.join(dataset_path, "**", "*.pose"), recursive=True))
    
    if not files:
        print(f"Warning: No .pose files found in {dataset_path}")
        return []
    
    dataset = []
    
    for f in files:
        # Extract signer and sign from filename
        filename = os.path.basename(f)
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        
        # Logic: signer_word -> extract both
        if len(parts) >= 2:
            signer = parts[0]
            label_it = "_".join(parts[1:])
            
            # Determine split based on signer
            if signer in train_signers:
                split = 'train'
            elif signer in test_signers:
                split = 'test'
            else:
                split = 'unknown'
            
            # Apply split filter if specified
            if split_filter and split != split_filter:
                continue
            
            # Get English labels
            english_labels = it_to_en.get(label_it, ["UNKNOWN"])
            
            # Get category if available
            category = it_to_category.get(label_it, None)
            
            # Build item
            item = {
                "file_path": f,
                "signer": signer,
                "label_italian": label_it,
                "labels_english": english_labels,
                "split": split
            }
            
            if category:
                item["category"] = category
            
            dataset.append(item)
    
    return dataset

""" HOW TO LOAD THE DATA USING THIS

from data_loader import get_dataset

# Load all data (train + test)
all_data = get_dataset()
print(f"Total samples: {len(all_data)}")
print(all_data[0])
# Output: {'file_path': 'a3lis_poses/fal/fal_casa.pose', 
#          'signer': 'fal', 
#          'label_italian': 'casa', 
#          'labels_english': ['house', 'home'],
#          'category': 'Vita Comune',
#          'split': 'train'}

# Load only training data
train_data = get_dataset(split_filter='train')
print(f"Training samples: {len(train_data)}")

# Load only test data  
test_data = get_dataset(split_filter='test')
print(f"Test samples: {len(test_data)}")

# Load without categories
data_no_cat = get_dataset(use_categories=False)

# Group by category
from collections import defaultdict
by_category = defaultdict(list)
for item in all_data:
    if 'category' in item:
        by_category[item['category']].append(item)

for cat, items in sorted(by_category.items()):
    print(f"{cat}: {len(items)} signs")

"""