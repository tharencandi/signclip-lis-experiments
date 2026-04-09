# I USED THIS LOCALLY, NEEDS TWEAKS FOR COLAB (AT LEAST THE PROPER PATHS)
# Loads and performs retrieval experiment for signIT

import os
import random
from glob import glob
from tqdm import tqdm
from pose_format import Pose

import sys
# This tells Python to look in the parent folder (MMPT) for demo_sign.py
sys.path.append(os.path.abspath(".."))

from demo_sign import score_pose_and_text

INPUT_FOLDER = "../signIT_poses" 

def get_signIT_dataset():
    """
    Scans the folder and extracts labels directly from filenames.
    Filename format: 01_animals_bear_test_... -> Label: 'bear'
    """
    files = sorted(glob(os.path.join(INPUT_FOLDER, "**", "*.pose"), recursive=True))
    
    dataset = []
    
    for f in files:
        filename = os.path.basename(f)
        # Split "01_animals_bear_test..." by "_"
        parts = filename.split('_')
        
        if len(parts) > 2:
            # The 3rd word is the word itself (index 2)
            label = parts[2]
            
            dataset.append({
                "file_path": f,
                "label": label
            })
            
    return dataset

def run_experiment(target_word, num_distractors=9):
    print(f"\nMq EXPERIMENT: Looking for '{target_word}'...")

    # 1. Load Dataset
    full_dataset = get_signIT_dataset()
    
    if not full_dataset:
        print(f"❌ No files found in '{INPUT_FOLDER}'!")
        return

    # 2. Find Correct Video (Positive)
    correct_videos = [d for d in full_dataset if d['label'] == target_word]
    
    if not correct_videos:
        print(f"❌ No videos found for word '{target_word}'. Check your spelling!")
        # Print a few examples of what it DID find to help debug
        print(f"   (Found labels like: {[d['label'] for d in full_dataset[:5]]}...)")
        return

    target_video = random.choice(correct_videos)

    # 3. Find Wrong Videos (Negative)
    other_videos = [d for d in full_dataset if d['label'] != target_word]
    distractors = random.sample(other_videos, min(num_distractors, len(other_videos)))

    # 4. Mix and Shuffle
    test_batch = [target_video] + distractors
    random.shuffle(test_batch)

    print(f"   Comparing 1 '{target_word}' vs {len(distractors)} other words...")

    # 5. Run AI Score
    results = []
    # Note: We use <en> <ase> because the model is pretrained on ASL
    text_query = f"<en> <ase> {target_word}"

    for item in tqdm(test_batch, desc="Scoring"):
        try:
            with open(item['file_path'], "rb") as f:
                pose = Pose.read(f.read())
            
            # We force max_frames=256 to prevent crashes on long videos
            _, score = score_pose_and_text(pose, text_query, max_frames=256)
            
            results.append({
                "path": item['file_path'],
                "true_label": item['label'],
                "score": score
            })
        except Exception as e:
            print(f"Skipping bad file: {item['file_path']}")

    # 6. Sort and Print
    results.sort(key=lambda x: x['score'], reverse=True)

    print("\n🏆 RESULTS:")
    print("-" * 60)
    print(f"{'RANK':<5} | {'SCORE':<10} | {'ACTUAL WORD':<20} | {'FILE'}")
    print("-" * 60)
    
    for i, res in enumerate(results):
        rank = i + 1
        is_correct = (res['true_label'] == target_word)
        marker = "✅" if is_correct else "❌"
        
        print(f"#{rank:<4} | {res['score']:.4f}     | {marker} {res['true_label']:<15} | {os.path.basename(res['path'])}")

    if results[0]['true_label'] == target_word:
        print(f"\nSUCCESS! 🚀 Found the {target_word}.")
    else:
        print(f"\nFAIL. 📉 Confused.")

if __name__ == "__main__":
    # Change this to whatever word you want to test
    run_experiment("bear")
    run_experiment("bird")
    run_experiment("cat")
    run_experiment("blue")
    run_experiment("brother")
    run_experiment("brother-in-law")
    run_experiment("dad")
    run_experiment("wife")
    run_experiment("pink")
    run_experiment("sadness")
    run_experiment("pineapple")
    run_experiment("pizza")
    