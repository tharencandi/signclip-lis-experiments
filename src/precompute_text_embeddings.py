"""
Precompute text embeddings for SignCLIP and save reusable artifacts.

This mirrors the pose precompute workflow but operates on class labels from
A3LIS embeddings metadata, then writes:
- text_embeddings_*.npy
- text_labels_*.txt
- text_metadata_*.json

Usage:
    # Precompute text anchors from all labels in embeddings metadata
    python src/precompute_text_embeddings.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2 \
        --output_dir dataset/embeddings/text_cache \
        --model_name default \
        --prompt_type en_lis

    # Use macro categories as labels
    python src/precompute_text_embeddings.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2 \
        --output_dir dataset/embeddings/text_cache \
        --use_categories \
        --prompt_type en_lis

    # Custom prompt template (overrides --prompt_type)
    python src/precompute_text_embeddings.py \
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2 \
        --output_dir dataset/embeddings/text_cache \
        --text_template "<en> <lis> {}"
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.demo_sign import embed_text
from src.embedding_utils import PROMPT_TEMPLATES


def sanitize_for_filename(value: str) -> str:
    """Convert an arbitrary string into a filesystem-friendly token."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned.lower() or "custom"


def load_unique_labels_from_metadata(
    embedding_dir: Path,
    split: str = "all",
    label_language: str = "english",
    use_categories: bool = False,
) -> List[str]:
    """Load unique class labels from A3LIS embeddings_metadata.json."""
    metadata_path = embedding_dir / "embeddings_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    unique_labels = set()
    valid_splits = {"train", "val", "test", "unknown"}

    for item in metadata.get("embeddings", []):
        item_split = item.get("split", "unknown")
        if split != "all" and item_split != split:
            continue
        if split == "all" and item_split not in valid_splits:
            continue

        if use_categories:
            label = item.get("category", item.get("label_italian", ""))
        elif label_language == "italian":
            label = item.get("label_italian", "")
        else:
            english_labels = item.get("labels_english", [])
            label = english_labels[0] if english_labels else item.get("label_italian", "")

        label = str(label).strip()
        if label:
            unique_labels.add(label)

    return sorted(unique_labels)


def resolve_prompt_template(
    text_template: Optional[str],
    prompt_type: Optional[str],
) -> Tuple[Optional[str], str]:
    """Resolve template and template tag using zero-shot precedence."""
    if text_template:
        return text_template, "custom"

    if prompt_type and prompt_type in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[prompt_type], prompt_type

    return None, "raw"


def precompute_text_embeddings(
    pose_embeddings_dir: str,
    output_dir: str,
    model_name: str = "default",
    checkpoint_path: Optional[str] = None,
    split: str = "all",
    label_language: str = "english",
    use_categories: bool = False,
    text_template: Optional[str] = None,
    prompt_type: Optional[str] = None,
) -> Path:
    """Precompute and save text embeddings for class labels."""
    embedding_dir = Path(pose_embeddings_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels = load_unique_labels_from_metadata(
        embedding_dir=embedding_dir,
        split=split,
        label_language=label_language,
        use_categories=use_categories,
    )

    if not labels:
        raise ValueError(
            f"No labels found in {pose_embeddings_dir} for split='{split}', "
            f"label_language='{label_language}', use_categories={use_categories}."
        )

    template_to_use, template_name = resolve_prompt_template(text_template, prompt_type)
    if template_to_use:
        text_inputs = [template_to_use.format(label) for label in labels]
    else:
        text_inputs = labels

    print(f"\n{'=' * 60}")
    print("Precompute Text Embeddings")
    print(f"{'=' * 60}")
    print(f"Pose embeddings dir: {pose_embeddings_dir}")
    print(f"Output dir:          {output_dir}")
    print(f"Model:               {model_name}")
    print(f"Checkpoint:          {checkpoint_path if checkpoint_path else 'none'}")
    print(f"Split:               {split}")
    print(f"Label language:      {label_language}")
    print(f"Use categories:      {use_categories}")
    print(f"Prompt type:         {template_name}")
    if template_to_use:
        print(f"Template:            '{template_to_use}'")
    print(f"Unique labels:       {len(labels)}")
    print(f"Example input:       '{text_inputs[0]}'")

    # Compute text embeddings one label at a time to keep memory bounded.
    vectors = []
    for text_input in tqdm(text_inputs, desc="Embedding text labels"):
        vec = embed_text(
            [text_input],
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )[0]
        vectors.append(vec)

    text_embeddings = np.vstack(vectors)

    label_tag = "categories" if use_categories else label_language
    template_tag = sanitize_for_filename(template_name)
    model_tag = sanitize_for_filename(model_name)
    split_tag = sanitize_for_filename(split)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    stem = f"{model_tag}_{template_tag}_{label_tag}_{split_tag}_{timestamp}"
    embeddings_file = output_path / f"text_embeddings_{stem}.npy"
    labels_file = output_path / f"text_labels_{stem}.txt"
    metadata_file = output_path / f"text_metadata_{stem}.json"

    np.save(embeddings_file, text_embeddings)

    with open(labels_file, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")

    metadata = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "pose_embeddings_dir": str(pose_embeddings_dir),
        "split": split,
        "label_language": label_language,
        "use_categories": use_categories,
        "prompt_type": template_name,
        "text_template": template_to_use,
        "num_labels": len(labels),
        "embedding_dim": int(text_embeddings.shape[1]),
        "labels": labels,
        "text_inputs": text_inputs,
        "files": {
            "embeddings": str(embeddings_file.name),
            "labels": str(labels_file.name),
        },
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved text embeddings: {embeddings_file}")
    print(f"Saved text labels:     {labels_file}")
    print(f"Saved text metadata:   {metadata_file}")
    print(f"Embeddings shape:      {text_embeddings.shape}")
    print(f"{'=' * 60}\n")

    return embeddings_file


def main():
    parser = argparse.ArgumentParser(
        description="Precompute SignCLIP text embeddings from A3LIS metadata labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pose_embeddings_dir",
        type=str,
        required=True,
        help="Directory containing embeddings_metadata.json (same style as pose embeddings dir)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where text_embeddings/text_labels/text_metadata files will be saved",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="default",
        choices=["default", "asl_citizen", "asl_finetune", "suisse", "a3lis_finetune"],
        help="SignCLIP model used to embed text anchors",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional checkpoint path overlaid on top of --model_name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Which split labels to include when building the unique label set",
    )
    parser.add_argument(
        "--label_language",
        type=str,
        default="english",
        choices=["italian", "english"],
        help="Label language used to extract class names from metadata",
    )
    parser.add_argument(
        "--use_categories",
        action="store_true",
        help="Use macro categories instead of micro labels",
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt_type",
        type=str,
        default=None,
        choices=list(PROMPT_TEMPLATES.keys()),
        help="Predefined prompt template key",
    )
    prompt_group.add_argument(
        "--text_template",
        type=str,
        default=None,
        help="Custom prompt template containing '{}' placeholder",
    )

    args = parser.parse_args()

    precompute_text_embeddings(
        pose_embeddings_dir=args.pose_embeddings_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        split=args.split,
        label_language=args.label_language,
        use_categories=args.use_categories,
        text_template=args.text_template,
        prompt_type=args.prompt_type,
    )


if __name__ == "__main__":
    main()
