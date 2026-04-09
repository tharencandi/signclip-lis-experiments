# I USED THIS LOCALLY, NEEDS TWEAKS FOR COLAB (AT LEAST THE PROPER PATHS)
# Converts videos to poses using more cores at the same time (change NUM_WORKERS)

import os
import subprocess
import shutil
from glob import glob

# The folder containing videos (searches all subfolders automatically)
INPUT_FOLDER = "../dataset_videos" 
# Folder that will contain poses
OUTPUT_FOLDER = "../signIT_poses"
# Number of videos to process at the exact same time. 
# If PC freezes, lower.
NUM_WORKERS = 4 

def main():
    # 1. Apply the WSL crash fix
    os.environ["MPLBACKEND"] = "Agg"
    
    # 2. Verify the input folder exists and creates output one
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: The directory '{INPUT_FOLDER}' does not exist.")
        return
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Starting fast batch conversion on: '{INPUT_FOLDER}'")
    print(f"Using {NUM_WORKERS} parallel workers.")
    print("Please wait, this might take a while...\n")
    
    # 3. Build the terminal command
    command = [
        "videos_to_poses",
        "--format", "mediapipe",
        "--num-workers", str(NUM_WORKERS),
        "--recursive",
        "--directory", INPUT_FOLDER
    ]
    
    # 4. Execute the command
    try:
        # subprocess.run acts exactly like typing it into the terminal
        subprocess.run(command, check=True)
        print("\nAll videos processed successfully!")

        # 4. Clean up: Move all .pose files to the output directory
        # We search the input folder recursively for the generated poses
        generated_poses = glob(os.path.join(INPUT_FOLDER, "**", "*.pose"), recursive=True)
        
        moved_count = 0
        for pose_path in generated_poses:
            filename = os.path.basename(pose_path)
            destination = os.path.join(OUTPUT_FOLDER, filename)
            
            # Move the file (overwrites if it already exists in the output folder)
            shutil.move(pose_path, destination)
            moved_count += 1
            
        print(f"Successfully moved {moved_count} .pose files to '{OUTPUT_FOLDER}'.")
        
    except subprocess.CalledProcessError as e:
        print(f"\nThe conversion tool crashed. Error details: {e}")
    except FileNotFoundError:
        print("\nError: 'videos_to_poses' command not found.")
        print("   Make sure you have activated your environment: conda activate signclip")

if __name__ == "__main__":
    main()