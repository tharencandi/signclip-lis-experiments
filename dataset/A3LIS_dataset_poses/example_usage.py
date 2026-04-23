"""
Example usage of the A3LIS dataset package

Run from project root:
    python dataset/A3LIS_dataset_poses/example_usage.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from A3LIS_dataset_poses.a3lis import get_dataset, get_split_info
from collections import Counter

def main():
    print("="*60)
    print("A3LIS Dataset Example")
    print("="*60)
    
    # 1. Get split information
    print("\n1. Split Information:")
    split_info = get_split_info()
    if split_info:
        print(f"   Strategy: {split_info['strategy']}")
        print(f"   Train: {split_info['train_count']} samples from signers {split_info['train_signers']}")
        print(f"   Test: {split_info['test_count']} samples from signers {split_info['test_signers']}")
        print(f"   Ratio: {split_info['train_ratio']:.1%} train / {1-split_info['train_ratio']:.1%} test")
    
    # 2. Load all data
    print("\n2. Load All Data:")
    all_data = get_dataset()
    print(f"   Total samples: {len(all_data)}")
    
    # 3. Load only training data
    print("\n3. Load Training Data:")
    train_data = get_dataset(split_filter='train')
    print(f"   Training samples: {len(train_data)}")
    
    # 4. Load only test data
    print("\n4. Load Test Data:")
    test_data = get_dataset(split_filter='test')
    print(f"   Test samples: {len(test_data)}")
    
    # 5. Show first sample
    print("\n5. First Training Sample:")
    if train_data:
        sample = train_data[0]
        for key, value in sample.items():
            print(f"   {key}: {value}")
    
    # 6. Category breakdown
    print("\n6. Category Breakdown (train + test):")
    categories = Counter([d.get('category', 'no_category') for d in all_data])
    for cat, count in sorted(categories.items()):
        if cat != 'no_category':
            print(f"   {cat:<20} {count:>4} samples")
    
    # 7. Signs per category
    print("\n7. Unique Signs per Category:")
    from collections import defaultdict
    signs_by_cat = defaultdict(set)
    for item in all_data:
        cat = item.get('category')
        sign = item['label_italian']
        if cat:
            signs_by_cat[cat].add(sign)
    
    for cat, signs in sorted(signs_by_cat.items()):
        print(f"   {cat:<20} {len(signs):>3} unique signs")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
