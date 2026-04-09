# I USED THIS LOCALLY, NEEDS TWEAKS FOR COLAB (AT LEAST THE PROPER PATHS)
# Does retrieval experiment for a3lis for 

import random
import torch
import numpy as np
from pose_format import Pose
from tqdm import tqdm

import sys
import os
# This tells Python to also look in the parent folder (MMPT) for files
sys.path.append(os.path.abspath(".."))

# Import your tools
from data_loader import get_dataset
# We import the scoring function from your existing demo script
from demo_sign import score_pose_and_text

def run_experiment(target_english_word, num_distractors=9):
    print(f"\nEXPERIMENT: Looking for '{target_english_word}'...")

    # 1. Load the full dataset
    full_dataset = get_dataset()
    
    # 2. Find the "Correct" video (Positive Sample)
    # We look for ANY video that has the target word in its English labels
    correct_videos = [d for d in full_dataset if target_english_word in d['labels_english']]
    
    if not correct_videos:
        print(f"❌ Error: No videos found for label '{target_english_word}' in the dataset!")
        print("   Check your sign_dictionary.csv to make sure the spelling matches.")
        return

    # Pick one random correct video
    target_video = random.choice(correct_videos)
    
    # 3. Find "Wrong" videos (Negative Samples)
    # Videos that DO NOT contain the target word
    other_videos = [d for d in full_dataset if target_english_word not in d['labels_english']]
    
    # Pick random distractors
    distractors = random.sample(other_videos, min(num_distractors, len(other_videos)))
    
    # Combine them into a test batch
    test_batch = [target_video] + distractors
    # Shuffle so we don't know where the answer is (to be fair)
    random.shuffle(test_batch)

    print(f"   Comparing 1 Correct Video vs {len(distractors)} Wrong Videos...")
    
    # 4. Run the AI Score
    results = []
    
    # Wrap text in the specific tags SignCLIP expects
    text_query = f"<en> <ase> {target_english_word}"
    
    for item in tqdm(test_batch, desc="Scoring"):
        # Load the pose file
        with open(item['file_path'], "rb") as f:
            pose = Pose.read(f.read())
        
        # Calculate Score
        # We reuse the function you already have in demo_sign.py
        # We need to set max frames to 256 because that's the limit of signclip
        # The extra frames will be truncated
        _, score = score_pose_and_text(pose, text_query, max_frames=256)
        
        results.append({
            "path": item['file_path'],
            "true_label": item['labels_english'],
            "score": score
        })

    # 5. Sort by Score (Highest confidence first)
    results.sort(key=lambda x: x['score'], reverse=True)

    # 6. Print Results
    print("\n🏆 RESULTS (Ranked by AI confidence):")
    print("-" * 60)
    print(f"{'RANK':<5} | {'SCORE':<10} | {'ACTUAL LABEL':<30} | {'FILE'}")
    print("-" * 60)
    
    for i, res in enumerate(results):
        rank = i + 1
        is_correct = target_english_word in res['true_label']
        marker = "✅" if is_correct else "❌"
        
        # Make the correct answer bold/visible
        print(f"#{rank:<4} | {res['score']:.4f}     | {marker} {str(res['true_label']):<25} | {res['path']}")

    # Final Verdict
    top_result_is_correct = target_english_word in results[0]['true_label']
    if top_result_is_correct:
        print(f"\nSUCCESS! 🚀 The AI found '{target_english_word}' correctly!")
    else:
        print(f"\nFAIL. 📉 The AI was confused.")

if __name__ == "__main__":
    # You can change this word to anything in your dictionary
    # e.g., "house", "tree", "car"
    run_experiment("house")
    run_experiment("administration")
    run_experiment("monday")
    run_experiment("hot")
    run_experiment("cold")
    run_experiment("exam")
    run_experiment("phone")