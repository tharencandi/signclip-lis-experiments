import os
import re
from glob import glob

import pandas as pd
import torch
from pose_format import Pose
from transformers import AutoTokenizer

from signclip.datasets.mmdataset import MMDataset


_FILENAME_RE = re.compile(
    r'^(.+)'
    r'_(animali|cibo|colori|emozioni|famiglia)'
    r'_(animals|food|colors|emotions|family)'
    r'_([a-z]+)'
    r'_(.+)'
    r'_(\d+)_(\d+)'
    r'_(train|test|val)\.pose$'
)


def parse_signit_filename(filename):
    match = _FILENAME_RE.match(filename)
    if not match:
        return None
    return {
        'name_video_stem': match.group(1),
        'it_macro': match.group(2),
        'eng_macro': match.group(3),
        'it_label': match.group(4),
        'eng_label': match.group(5).lower(),
        'frame_start': int(match.group(6)),
        'frame_end': int(match.group(7)),
        'split': match.group(8),
    }


class SignITMetaProcessor:
    def __init__(self, poses_root, csv_path, split_filter=None):
        self.samples = []
        self.split = split_filter

        df = pd.read_csv(csv_path)
        self.label_map = {row['label_italian']: idx for idx, row in df.iterrows()}
        self.english_map = {
            row['label_italian']: [
                label.strip().lower()
                for label in str(row['label_english']).replace(';', ',').split(',')
                if label.strip()
            ]
            for _, row in df.iterrows()
        }
        self.category_map = {
            row['label_italian']: str(row['category']).strip().lower()
            for _, row in df.iterrows()
            if 'category' in row and pd.notna(row['category'])
        }

        pose_files = sorted(glob(os.path.join(poses_root, '*.pose')))
        for pose_path in pose_files:
            parsed = parse_signit_filename(os.path.basename(pose_path))
            if parsed is None:
                continue

            if split_filter and parsed['split'] != split_filter:
                continue

            it_label = parsed['it_label']
            english_labels = self.english_map.get(it_label, [parsed['eng_label']])
            english_label = english_labels[0]

            self.samples.append({
                'signer': parsed['name_video_stem'],
                'label': it_label,
                'label_italian': it_label,
                'label_english': english_label,
                'pose_path': pose_path,
                'split': parsed['split'],
                'labels_english': english_labels,
                'category': self.category_map.get(it_label, parsed['eng_macro']),
                'it_macro': parsed['it_macro'],
                'eng_macro': parsed['eng_macro'],
                'frame_start': parsed['frame_start'],
                'frame_end': parsed['frame_end'],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SignITVideoProcessor:
    def __init__(self):
        self.Pose = Pose

    def __call__(self, pose_path):
        with open(pose_path, 'rb') as handle:
            return self.Pose.read(handle.read())


class SignITTextProcessor:
    def __init__(self, label_map):
        self.label_map = label_map

    def __call__(self, label):
        clean_label = label.replace("’", "'")
        return self.label_map[clean_label]


class SignITAlignProcessor:
    def __init__(self, tokenizer, max_text_len=64):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

    def _build_caps(self, raw_ids):
        max_raw = self.max_text_len - 3
        raw_ids = raw_ids[:max_raw]
        full = [self.cls_id, self.sep_id] + raw_ids + [self.sep_id]
        pad_len = self.max_text_len - len(full)
        full = full + [self.pad_id] * pad_len
        caps = torch.tensor(full, dtype=torch.long)
        cmasks = torch.zeros(self.max_text_len, dtype=torch.long)
        cmasks[:len(raw_ids) + 3] = 1
        return caps, cmasks

    def __call__(self, sample, video_feature, text_feature):
        label = sample['label']
        label_italian = sample.get('label_italian', label)
        label_english = sample.get('label_english', label).lower()
        transformed_label = f"<en> <lis> {label_english}"
        raw_ids = self.tokenizer(transformed_label, add_special_tokens=False)["input_ids"]
        caps, cmasks = self._build_caps(raw_ids)

        return {
            'pose': video_feature,
            'label': text_feature,
            'pose_path': sample['pose_path'],
            'label_italian': label_italian,
            'label_english': label_english,
            'caps': caps,
            'cmasks': cmasks,
        }


class SignITDataset(MMDataset):
    def __init__(
        self,
        poses_root,
        csv_path,
        split_filter=None,
        tokenizer_name='bert-base-cased',
        max_text_len=64,
    ):
        meta_processor = SignITMetaProcessor(poses_root, csv_path, split_filter)
        video_processor = SignITVideoProcessor()
        text_processor = SignITTextProcessor(meta_processor.label_map)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        align_processor = SignITAlignProcessor(tokenizer, max_text_len=max_text_len)
        super().__init__(meta_processor, video_processor, text_processor, align_processor)
