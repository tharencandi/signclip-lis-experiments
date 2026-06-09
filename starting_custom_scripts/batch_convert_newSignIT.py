# Converts videos to poses and renames them based on CSV metadata

import os
import subprocess
import csv
from glob import glob
from tqdm import tqdm  # progress bar

INPUT_FOLDER = "../dataset_videos"
OUTPUT_FOLDER = "../dataset_poses"
CSV_PATH = "../dataset_videos/Videos.csv"
EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]


def main():
    # 1. Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    # 2. Load the CSV Data into a Dictionary
    video_metadata = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # We use 'name_video' as the key to look up data later
                video_metadata[row['name_video']] = row
        print(f"Loaded metadata for {len(video_metadata)} videos from CSV.")
    else:
        print(f"⚠️ Warning: CSV file not found at '{CSV_PATH}'. Check the path!")

    # 3. Find all video files
    video_files = []
    for ext in EXTENSIONS:
        # Recursive search (looks inside subfolders too)
        found = glob(os.path.join(INPUT_FOLDER, "**", ext), recursive=True)
        video_files.extend(found)

    if len(video_files) == 0:
        print(f"No videos found in '{INPUT_FOLDER}'.")
        return

    print(f"Found {len(video_files)} videos. Starting conversion...\n")

    # 4. Iterate through videos with progress bar
    for video_path in tqdm(video_files, desc="Processing"):
        
        # Get filename (e.g., 'asino_donkey_seq1_00565-00707.mp4')
        filename = os.path.basename(video_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Check if this video exists in our CSV dictionary
        if filename in video_metadata:
            row = video_metadata[filename]
            
            # Extract the specific columns
            it_macro = row['it_macro']
            eng_macro = row['eng_macro']
            it_label = row['it_label']
            eng_label = row['eng_label']
            frame_start_csv = row['frame_start_csv']
            frame_end_csv = row['frame_end_csv']
            split = row['split']
            
            # Build the new filename! You can rearrange these variables however you want.
            new_pose_name = f"{name_no_ext}_{it_macro}_{eng_macro}_{it_label}_{eng_label}_{frame_start_csv}_{frame_end_csv}_{split}.pose"
        else:
            # Fallback just in case a video is NOT in the CSV
            new_pose_name = f"{name_no_ext}.pose"

        # Create explicit output path
        pose_output_path = os.path.join(OUTPUT_FOLDER, new_pose_name)

        # SKIP if file already exists (allows you to stop and resume later)
        if os.path.exists(pose_output_path):
            continue

        # Prepare the command
        cmd = [
            "video_to_pose",
            "--format", "mediapipe",
            "-i", video_path,
            "-o", pose_output_path
        ]

        # Set Environment Variable to prevent WSL crashing
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"

        try:
            # Run the command and hide the spammy output
            subprocess.run(cmd, env=env, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"\nError processing {filename}")
            # print(e.stderr.decode()) # Uncomment to see error details

    print(f"\nDone! Poses saved in '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()