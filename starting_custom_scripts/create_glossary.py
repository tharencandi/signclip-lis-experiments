# I USED THIS LOCALLY, NEEDS TWEAKS FOR COLAB (AT LEAST THE PROPER PATHS)
# Creates the A3lis csv

import os
import pandas as pd
from glob import glob

# --- CONFIGURATION ---
POSE_FOLDER = "../dataset_poses"
OUTPUT_CSV = "../sign_dictionary.csv"
# ---------------------

def main():
    # 1. Find all pose files RECURSIVELY
    # The "**" tells it to look inside any subfolder depth
    # recursive=True is the key flag here
    pose_files = glob(os.path.join(POSE_FOLDER, "**", "*.pose"), recursive=True)
    
    if not pose_files:
        print(f"No .pose files found in '{POSE_FOLDER}' or its subfolders!")
        return

    print(f"Scanning {len(pose_files)} files recursively...")

    unique_signs = set()

    for file_path in pose_files:
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]

        # Splitting logic: "signer_signWord.pose"
        parts = name_no_ext.split('_')
        
        if len(parts) >= 2:
            # Everything after the first underscore is the sign
            sign_italian = "_".join(parts[1:])
            unique_signs.add(sign_italian)
        else:
            # Optional: handle odd filenames if necessary
            pass

    # 2. Create and Save DataFrame
    df = pd.DataFrame(sorted(list(unique_signs)), columns=["label_italian"])
    df["label_english"] = "" 

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Created '{OUTPUT_CSV}' with {len(df)} unique signs.")
    print("Action: Open this file and fill in the 'label_english' column.")

if __name__ == "__main__":
    main()