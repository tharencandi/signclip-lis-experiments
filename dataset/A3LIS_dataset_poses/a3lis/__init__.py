"""
A3LIS Dataset Loader Package
Italian Sign Language (LIS) pose dataset with train/test splits and categories
"""

import pandas as pd
import os
import json
from glob import glob
from pathlib import Path

# Get package directory (absolute path)
_PACKAGE_DIR = Path(__file__).parent.parent.resolve()

# --- CONFIG (absolute paths to dataset files) ---
DATASET_PATH = (_PACKAGE_DIR / "a3lis_poses").resolve()
CSV_PATH = (_PACKAGE_DIR / "sign_dictionary_fixed.csv").resolve()
CSV_WITH_CATEGORIES_PATH = (_PACKAGE_DIR / "sign_dictionary_fixed.csv").resolve()
CSV_PATH_FALLBACK = (_PACKAGE_DIR / "sign_dictionary.csv").resolve()
CSV_WITH_CATEGORIES_PATH_FALLBACK = (_PACKAGE_DIR / "sign_dictionary_fitted.csv").resolve()
SPLIT_CONFIG_PATH = (_PACKAGE_DIR / "train_test_val_split.json").resolve()
# ------------------------------------------------

def load_split_config():
    """Load train/test split configuration."""
    if not SPLIT_CONFIG_PATH.exists():
        print(f"Warning: {SPLIT_CONFIG_PATH} not found. All samples will be marked as 'unknown' split.")
        return None
    
    with open(SPLIT_CONFIG_PATH, 'r', encoding='utf-8') as f:
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
    csv_file = CSV_PATH
    has_categories = False
    
    if use_categories and CSV_WITH_CATEGORIES_PATH.exists():
        csv_file = CSV_WITH_CATEGORIES_PATH
        has_categories = True
    elif use_categories and CSV_WITH_CATEGORIES_PATH_FALLBACK.exists():
        csv_file = CSV_WITH_CATEGORIES_PATH_FALLBACK
        has_categories = True
    elif CSV_PATH.exists():
        csv_file = CSV_PATH
    elif CSV_PATH_FALLBACK.exists():
        csv_file = CSV_PATH_FALLBACK
    elif not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dictionary CSVs: {CSV_PATH} or {CSV_PATH_FALLBACK}!"
        )
    
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
    val_signers   = set(split_config['val_signers'])   if split_config and 'val_signers' in split_config else set()
    test_signers  = set(split_config['test_signers'])  if split_config else set()
    
    # 3. Find all .pose files recursively
    files = sorted(DATASET_PATH.glob("**/*.pose"))
    
    if not files:
        print(f"Warning: No .pose files found in {DATASET_PATH}")
        return []
    
    dataset = []
    
    for f in files:
        # Extract signer and sign from filename
        filename = f.name
        name_no_ext = f.stem
        parts = name_no_ext.split('_')
        
        # Logic: signer_word -> extract both
        if len(parts) >= 2:
            signer = parts[0]
            label_it = "_".join(parts[1:])
            
            # Determine split based on signer
            if signer in train_signers:
                split = 'train'
            elif signer in val_signers:
                split = 'val'
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
            
            # Build item (use absolute path)
            item = {
                "file_path": str(f.resolve()),
                "signer": signer,
                "label_italian": label_it,
                "labels_english": english_labels,
                "split": split
            }
            
            if category:
                item["category"] = category
            
            dataset.append(item)
    
    return dataset

def get_split_info():
    """
    Get information about the train/test split.
    
    Returns:
        Dictionary with split statistics
    """
    split_config = load_split_config()
    if not split_config:
        return None
    
    train_data = get_dataset(split_filter='train')
    val_data   = get_dataset(split_filter='val')
    test_data  = get_dataset(split_filter='test')
    
    from collections import Counter
    
    train_signers = Counter([d['signer'] for d in train_data])
    val_signers   = Counter([d['signer'] for d in val_data])
    test_signers  = Counter([d['signer'] for d in test_data])
    
    return {
        'strategy': split_config.get('split_strategy', 'unknown'),
        'train_signers': sorted(train_signers.keys()),
        'val_signers':   sorted(val_signers.keys()),
        'test_signers':  sorted(test_signers.keys()),
        'train_count': len(train_data),
        'val_count':   len(val_data),
        'test_count':  len(test_data),
        'total_count': len(train_data) + len(val_data) + len(test_data),
        'train_ratio': len(train_data) / (len(train_data) + len(val_data) + len(test_data))
            if (len(train_data) + len(val_data) + len(test_data)) > 0 else 0
    }

# Expose main functions
__all__ = ['get_dataset', 'get_split_info', 'load_dictionary', 'load_split_config']
