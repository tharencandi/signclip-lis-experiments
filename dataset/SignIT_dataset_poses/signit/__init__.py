"""
SignIT Dataset Loader Package
Italian Sign Language pose dataset with train/test/val splits and categories
embedded in filenames.

New filename format (poses/ directory):
  {name_video_stem}_{it_macro}_{eng_macro}_{it_label}_{eng_label}_{fstart}_{fend}_{split}.pose

Example:
  acqua_water_seq1_31192-31263_cibo_food_acqua_Water_31192_31263_test.pose
  azzurro_light_blue_seq1_44945-45069_colori_colors_azzurro_Light Blue_44945_45069_train.pose
"""

import pandas as pd
import re
from pathlib import Path

# Get package directory (absolute path)
_PACKAGE_DIR = Path(__file__).parent.parent.resolve()

# --- CONFIG ---
POSES_DIR = (_PACKAGE_DIR / "poses").resolve()
CSV_PATH = (_PACKAGE_DIR / "sign_dictionary_with_categories.csv").resolve()
# --------------

# Known macro values used as regex anchors
_IT_MACROS = "animali|cibo|colori|emozioni|famiglia"
_EN_MACROS = "animals|food|colors|emotions|family"

# Anchored from the right so eng_label (which may contain spaces) is captured correctly.
_FILENAME_RE = re.compile(
    r'^(.+)'
    r'_(animali|cibo|colori|emozioni|famiglia)'
    r'_(animals|food|colors|emotions|family)'
    r'_([a-z]+)'
    r'_(.+)'
    r'_(\d+)_(\d+)'
    r'_(train|test|val)$'
)


def parse_signit_filename(filename: str):
    """
    Parse a SignIT pose filename into its metadata components.

    New format:
        {name_video_stem}_{it_macro}_{eng_macro}_{it_label}_{eng_label}_{fstart}_{fend}_{split}.pose

    Returns a dict or None if the filename does not match.
    """
    name = filename.removesuffix('.pose')
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return {
        'filename':        filename,
        'name_video_stem': m.group(1),
        'it_macro':        m.group(2),
        'eng_macro':       m.group(3),
        'it_label':        m.group(4),
        'eng_label':       m.group(5),
        'frame_start':     int(m.group(6)),
        'frame_end':       int(m.group(7)),
        'split':           m.group(8),
    }


def load_dictionary(use_categories=True):
    """
    Loads the CSV and creates lookup dictionaries.

    Returns:
        Tuple of (italian_to_english, italian_to_category)
    """
    if not CSV_PATH.exists():
        print(f"Warning: {CSV_PATH} not found. Labels will be taken from filenames only.")
        return {}, {}

    df = pd.read_csv(CSV_PATH)

    it_to_en = {}
    for _, row in df.iterrows():
        italian = row['label_italian']
        english_raw = str(row['label_english'])
        english_list = [w.strip() for w in english_raw.replace(';', ',').split(',')]
        it_to_en[italian] = english_list

    it_to_category = {}
    if 'category' in df.columns:
        for _, row in df.iterrows():
            italian = row['label_italian']
            category = str(row['category']).strip()
            if category and category not in ['nan', '']:
                it_to_category[italian] = category

    return it_to_en, it_to_category


def get_dataset(use_categories=True, split_filter=None):
    """
    Scans poses/ for .pose files and returns a list of metadata dicts.

    Args:
        use_categories: If True, load category information from CSV.
        split_filter: Optional 'train', 'test', 'val', or None for all.

    Returns:
        List of dicts with keys:
            file_path, name_video_stem, it_label, eng_label, labels_english,
            it_macro, eng_macro, category, split, frame_start, frame_end
    """
    it_to_en, it_to_category = load_dictionary(use_categories=use_categories)

    files = sorted(POSES_DIR.glob("*.pose"))
    if not files:
        print(f"Warning: No .pose files found in {POSES_DIR}")
        return []

    dataset = []
    for f in files:
        info = parse_signit_filename(f.name)
        if not info:
            print(f"Warning: Could not parse filename {f.name}")
            continue

        if split_filter and info['split'] != split_filter:
            continue

        it_label = info['it_label']
        eng_label_filename = info['eng_label'].lower()

        # Prefer CSV for English labels (may list multiple synonyms); fall back to filename
        english_labels = [l.lower() for l in it_to_en.get(it_label, [eng_label_filename])]

        # Prefer CSV for category; fall back to eng_macro from filename
        category = it_to_category.get(it_label, info['eng_macro'])

        dataset.append({
            'file_path':       str(f.resolve()),
            'name_video_stem': info['name_video_stem'],
            'it_label':        it_label,
            'eng_label':       eng_label_filename,
            'labels_english':  english_labels,
            'it_macro':        info['it_macro'],
            'eng_macro':       info['eng_macro'],
            'category':        category,
            'split':           info['split'],
            'frame_start':     info['frame_start'],
            'frame_end':       info['frame_end'],
        })

    return dataset


def get_split_info():
    """
    Returns split statistics for the dataset.
    """
    from collections import Counter

    all_data = get_dataset()
    by_split = {}
    for item in all_data:
        by_split.setdefault(item['split'], []).append(item)

    total = len(all_data)
    result = {'total_count': total}
    for split in ('train', 'test', 'val'):
        items = by_split.get(split, [])
        result[f'{split}_count'] = len(items)
        result[f'{split}_ratio'] = len(items) / total if total > 0 else 0
        result[f'{split}_labels'] = sorted({i['it_label'] for i in items})

    return result


__all__ = ['get_dataset', 'get_split_info', 'load_dictionary', 'parse_signit_filename',
           'POSES_DIR', 'CSV_PATH']
