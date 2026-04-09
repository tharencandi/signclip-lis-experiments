# I USED THIS LOCALLY, NEEDS TWEAKS FOR COLAB (AT LEAST THE PROPER PATHS)
# Loads the A3lis pose dataset

import pandas as pd
import os
from glob import glob

dataset_path = "../dataset_poses"
csv_path = "../sign_dictionary.csv"

def load_dictionary():
    """Loads the CSV and creates a fast lookup dictionary."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}!")
        
    df = pd.read_csv(csv_path)
    lookup = {}
    
    # Create dictionary: 'casa' -> ['house', 'home']
    for index, row in df.iterrows():
        italian = row['label_italian']
        english_raw = str(row['label_english'])
        # Handle multiple labels split by semicolon
        english_list = [word.strip() for word in english_raw.split(';')]
        lookup[italian] = english_list
        
    return lookup

def get_dataset():
    """
    Scans the folders and links every file to its English labels.
    Returns a list of items: [{'path': ..., 'labels': [...]}, ...]
    """
    # 1. Load the dictionary
    it_to_en = load_dictionary()
    
    # 2. Find all files
    # sorted not really useful, just to test if the first one gets printed, removable
    files = sorted(glob(os.path.join(dataset_path, "**", "*.pose"), recursive=True))
    
    dataset = []
    
    for f in files:
        # Extract Italian Gloss from filename
        filename = os.path.basename(f)
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        
        # Logic: signer_word -> word
        if len(parts) >= 2:
            label_it = "_".join(parts[1:])
            
            # 3. THE LINKING HAPPENS HERE
            english_labels = it_to_en.get(label_it, ["UNKNOWN"])
            
            dataset.append({
                "file_path": f,
                "signer": parts[0],
                "label_italian": label_it,
                "labels_english": english_labels
            })
            
    return dataset

""" HOW TO LOAD THE DATA USING THIS

from data_loader import get_dataset

# Load everything in one command
my_data = get_dataset()

# Print the first item to see if it worked
print(f"Loaded {len(my_data)} videos.")
print(my_data[0]) 
# Output: {'file_path': 'dataset_poses/marco_casa.pose', 'labels_english': ['house'], ...}

"""