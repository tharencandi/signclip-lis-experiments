# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle
import torch
from transformers import AutoTokenizer
import pandas as pd
import json
from glob import glob
from signclip.datasets.mmdataset import MMDataset
from pose_format import Pose
# Utility: load split config (from data_loader.py)
def load_split_config(split_config_path):
    if not os.path.exists(split_config_path):
        print(f"Warning: {split_config_path} not found. All samples will be marked as 'unknown' split.")
        return None
    with open(split_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config
"""
    for a3lis-147 fine tuning.


    10 signers, 147 classes, 10*147=1470 videos.

    70-10-20 split for train-val-test.
    make sure one male and one female for test.
     
"""
# --- Custom Processors ---
class A3LISMetaProcessor:
    def __init__(self, poses_root, csv_path, split_config_path=None, split_filter=None):
        self.samples = []
        self.split = None  # can be set externally if needed
        # Load label mapping from CSV (also English and category if available)
        df = pd.read_csv(csv_path)
        self.label_map = {row['label_italian']: idx for idx, row in df.iterrows()}
        # Parse English labels as list, strip whitespace
        self.english_map = {row['label_italian']: [lbl.strip() for lbl in str(row['label_english']).split(';')] if pd.notna(row['label_english']) else ["UNKNOWN"] for idx, row in df.iterrows()}
        self.category_map = {row['label_italian']: row['category'] for idx, row in df.iterrows() if 'category' in row and pd.notna(row['category'])}

        # Load split config if provided
        split_config = load_split_config(split_config_path) if split_config_path else None
        train_signers = set(split_config['train_signers']) if split_config and 'train_signers' in split_config else set()
        val_signers = set(split_config['val_signers']) if split_config and 'val_signers' in split_config else set()
        test_signers = set(split_config['test_signers']) if split_config and 'test_signers' in split_config else set()

        # Recursively find all .pose files
        pose_files = glob(os.path.join(poses_root, '**', '*.pose'), recursive=True)
        for pose_path in pose_files:
            fname = os.path.basename(pose_path)
            signer, label = fname[:-5].split('_', 1)
            # Determine split
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
            # Get English and category if available
            english_labels = self.english_map.get(label, ["UNKNOWN"])
            # Use only the first English label
            english_label = english_labels[0]
            category = self.category_map.get(label, None)
            self.samples.append({
                'signer': signer,
                'label': label,
                'label_italian': label,
                'label_english': english_label,
                'pose_path': pose_path,
                'split': split,
                'labels_english': english_labels,
                'category': category
            })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample

class A3LISVideoProcessor:
    def __init__(self):
        
        self.Pose = Pose
    def __call__(self, pose_path):
        with open(pose_path, "rb") as f:
            return self.Pose.read(f.read())

class A3LISTextProcessor:
    def __init__(self, label_map):
        self.label_map = label_map
    def __call__(self, label):
        return self.label_map[label]

class A3LISAlignProcessor:
    def __init__(self, tokenizer, max_text_len=64, pretokenized_labels=None):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.pretokenized_labels = pretokenized_labels
        # Cache special token IDs for _build_text_seq-compatible formatting
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

    def _build_caps(self, raw_ids):
        """Mirror Aligner._build_text_seq: [CLS, SEP, raw_ids, SEP, PAD...].

        MMFusionSeparate.forward_video expects caps[:, :2] == [CLS, SEP].
        MMFusionSeparate.forward_text skips position 1 and reads caps[:, 0] + caps[:, 2:],
        which gives [CLS, raw_ids..., SEP, PAD...] — standard BERT input.
        """
        max_raw = self.max_text_len - 3
        raw_ids = raw_ids[:max_raw]
        full = [self.cls_id, self.sep_id] + raw_ids + [self.sep_id]
        pad_len = self.max_text_len - len(full)
        full = full + [self.pad_id] * pad_len
        caps = torch.tensor(full, dtype=torch.long)
        cmasks = torch.zeros(self.max_text_len, dtype=torch.long)
        cmasks[:len(raw_ids) + 3] = 1  # CLS + SEP + raw + SEP
        return caps, cmasks

    def __call__(self, sample, video_feature, text_feature):
        # Use English label for tokenization
        pose_path = sample['pose_path']
        label = sample['label']
        label_italian = sample.get('label_italian', label)
        label_english = sample.get('label_english', label)  # fallback to label if missing
        # Use pretokenized if available, else tokenize on the fly
        if self.pretokenized_labels and label_english in self.pretokenized_labels:
            raw_ids = self.pretokenized_labels[label_english]['raw_ids']
            caps, cmasks = self._build_caps(raw_ids)
        else:
            transformed_label = f"<en> <lis> {label_english}"
            raw_ids = self.tokenizer(transformed_label, add_special_tokens=False)["input_ids"]
            caps, cmasks = self._build_caps(raw_ids)
        out = {
            'pose': video_feature,
            'label': text_feature,
            'pose_path': pose_path,
            'label_italian': label_italian,
            'label_english': label_english,
            'caps': caps,
            'cmasks': cmasks
        }
        return out

# --- Subclass of MMDataset ---
class A3LISDataset(MMDataset):
    def __init__(self, poses_root, csv_path, split_config_path=None, split_filter=None, tokenizer_name='bert-base-cased', max_text_len=64, pretokenized_labels_path=None):
        meta_processor = A3LISMetaProcessor(poses_root, csv_path, split_config_path, split_filter)
        video_processor = A3LISVideoProcessor()
        text_processor = A3LISTextProcessor(meta_processor.label_map)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        pretokenized_labels = None
        if pretokenized_labels_path and os.path.exists(pretokenized_labels_path):
            with open(pretokenized_labels_path, 'rb') as f:
                pretokenized_labels = pickle.load(f)
        align_processor = A3LISAlignProcessor(tokenizer, max_text_len=max_text_len, pretokenized_labels=pretokenized_labels)
        super().__init__(meta_processor, video_processor, text_processor, align_processor)
