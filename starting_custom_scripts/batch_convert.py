# I USED THIS LOCALLY, NEEDS TWEAKS FOR COLAB (AT LEAST THE PROPER PATHS)
# Converts videos to poses

import os
import subprocess
import sys
from glob import glob
from tqdm import tqdm  # progress bar


INPUT_FOLDER = "../dataset_videos"
OUTPUT_FOLDER = "../dataset_poses"
EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]


def main():
    # 1. Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    # 2. Find all video files
    video_files = []
    for ext in EXTENSIONS:
        # Recursive search (looks inside subfolders too)
        # remove recursive=True if you only want the top folder
        found = glob(os.path.join(INPUT_FOLDER, "**", ext), recursive=True)
        video_files.extend(found)

    if len(video_files) == 0:
        print(f"No videos found in '{INPUT_FOLDER}'.")
        print("   Please create the folder and add some video files.")
        return

    print(f"Found {len(video_files)} videos. Starting conversion...")

    # 3. Iterate through videos with progress bar
    for video_path in tqdm(video_files, desc="Processing"):
        
        # Get filename (e.g., 'video01.mp4')
        filename = os.path.basename(video_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Create output path (e.g., 'dataset_poses/video01.pose')
        pose_output_path = os.path.join(OUTPUT_FOLDER, name_no_ext + ".pose")

        # SKIP if file already exists (allows you to stop and resume later)
        if os.path.exists(pose_output_path):
            continue

        # Prepare the command
        # We use the same 'video_to_pose' command you verified earlier
        cmd = [
            "video_to_pose",
            "--format", "mediapipe",
            "-i", video_path,
            "-o", pose_output_path
        ]

        # Set Environment Variable to prevent WSL crashing (Headless mode)
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"

        try:
            # Run the command and hide the spammy output (capture_output=True)
            # Change to False if you want to see errors
            subprocess.run(cmd, env=env, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"\nError processing {filename}")
            # print(e.stderr.decode()) # Uncomment to see error details

    print(f"\nDone! Poses saved in '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()