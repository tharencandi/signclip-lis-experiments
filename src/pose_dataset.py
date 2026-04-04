import os
import re
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

class PoseDataset:
    """Flexible pose dataset for zero-shot, few-shot, and fine-tuning."""
    
    def __init__(self, data_dir: str, label_type: str = 'micro'):
        """
        Args:
            data_dir: Path to folder containing .pose files
            label_type: 'micro' or 'macro' for label granularity
        """
        self.data_dir = Path(data_dir)
        self.label_type = label_type
        self.samples = self._parse_all_files()
        self.class_to_samples = self._group_by_class()
    
    def _parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse: [video_id]_[macro]_[micro]_[split]_[start]_[end].pose"""
        pattern = r'(.+?)_(.+?)_(.+?)_(train|test)_(\d+)_(\d+)\.pose$'
        match = re.match(pattern, filename)
        
        if not match:
            return None
        
        video_id, macro, micro, split, start, end = match.groups()
        return {
            'filename': filename,
            'path': self.data_dir / filename,
            'video_id': video_id,
            'macro_label': macro,
            'micro_label': micro,
            'label': micro if self.label_type == 'micro' else macro,
            'split': split,
            'start_frame': int(start),
            'end_frame': int(end),
        }
    
    def _parse_all_files(self) -> List[Dict]:
        """Parse all .pose files in directory."""
        samples = []
        for f in sorted(self.data_dir.glob('*.pose')):
            meta = self._parse_filename(f.name)
            if meta:
                samples.append(meta)
        return samples
    
    def _group_by_class(self) -> Dict[str, List[int]]:
        """Map class labels to sample indices."""
        grouped = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            grouped[sample['label']].append(idx)
        return dict(grouped)
    
    # === FILTERING ===
    
    def filter_split(self, split: str) -> 'PoseDataset':
        """Create new dataset with only train or test samples."""
        filtered = PoseDataset.__new__(PoseDataset)
        filtered.data_dir = self.data_dir
        filtered.label_type = self.label_type
        filtered.samples = [s for s in self.samples if s['split'] == split]
        filtered.class_to_samples = filtered._group_by_class()
        return filtered
    
    def filter_classes(self, classes: List[str]) -> 'PoseDataset':
        """Keep only specified classes."""
        filtered = PoseDataset.__new__(PoseDataset)
        filtered.data_dir = self.data_dir
        filtered.label_type = self.label_type
        filtered.samples = [s for s in self.samples if s['label'] in classes]
        filtered.class_to_samples = filtered._group_by_class()
        return filtered
    
    # === FEW-SHOT SAMPLING ===
    
    def sample_k_shot(self, k: int, seed: Optional[int] = None) -> 'PoseDataset':
        """Sample up to k examples per class."""
        if seed is not None:
            random.seed(seed)
        
        sampled_indices = []
        for label, indices in self.class_to_samples.items():
            if len(indices) <= k:
                sampled_indices.extend(indices)
            else:
                sampled_indices.extend(random.sample(indices, k))
        
        filtered = PoseDataset.__new__(PoseDataset)
        filtered.data_dir = self.data_dir
        filtered.label_type = self.label_type
        filtered.samples = [self.samples[i] for i in sorted(sampled_indices)]
        filtered.class_to_samples = filtered._group_by_class()
        return filtered
    
    # === DATA ACCESS ===
    
    def load_pose(self, idx: int):
        """Load pose data for given sample index."""
        from pose_format import Pose
        
        path = self.samples[idx]['path']
        with open(path, 'rb') as f:
            pose = Pose.read(f.read())
        return pose
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[object, str, Dict]:
        """Returns (pose_data, label, metadata)."""
        sample = self.samples[idx]
        pose_data = self.load_pose(idx)
        return pose_data, sample['label'], sample
    
    @property
    def classes(self) -> List[str]:
        """Get sorted list of unique classes."""
        return sorted(self.class_to_samples.keys())
    
    @property
    def num_classes(self) -> int:
        return len(self.class_to_samples)
    
    def load_embedding(self, idx: int, embedding_dir: Optional[Path] = None) -> np.ndarray:
        """Load precomputed embedding if available."""
        if embedding_dir:
            # Match filename pattern: same basename but .npy instead of .pose
            emb_path = embedding_dir / self.samples[idx]['filename'].replace('.pose', '.npy')
            if emb_path.exists():
                return np.load(emb_path)
        raise FileNotFoundError(f"No precomputed embedding found for sample {idx}")
    
    def get_unique_labels(self) -> List[str]:
        """Get list of unique labels in dataset."""
        return self.classes