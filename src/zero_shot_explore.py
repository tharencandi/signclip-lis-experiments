"""
Zero-shot exploration: find the top-N closest words from the SpreadTheSign vocabulary
for each pose embedding — no task-specific label set required.

The SpreadTheSign vocabulary (~18k words and phrases) is embedded once using the same
prompt template as normal zero-shot ("<en> <lis> {word}"). The resulting matrix is
cached to disk so subsequent runs are instant.

For each pose embedding, the top-N closest words (by Euclidean distance, same
metric used throughout this codebase) are printed alongside the gold label.

Usage:
    # Explore test split, top-5 words per video
    python src/zero_shot_explore.py \\
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2_no_norm \\
        --split test

    # Use a fine-tuned checkpoint, show top-10, cache vocab embeddings in a custom dir
    python src/zero_shot_explore.py \\
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2_no_norm \\
        --model_name a3lis_finetune \\
        --split test \\
        --top_k 10 \\
        --vocab_cache_dir dataset/embeddings/vocab_cache

    # Limit to first 20 videos (quick sanity check)
    python src/zero_shot_explore.py \\
        --pose_embeddings_dir dataset/embeddings/a3lis_default_v2_no_norm \\
        --split test \\
        --max_samples 20
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.demo_sign import embed_text
from src.embedding_utils import load_a3lis_embeddings, EN_LIS_PROMPT_TEMPLATE

# Default vocabulary CSV (SpreadTheSign, single 'word' column, ~18k entries)
VOCAB_CSV_DEFAULT = str(project_root / "dataset" / "spreadthesign_vocabulary.csv")
# Embed one word at a time to avoid OOM on CPU/laptop — model stays loaded, overhead is small
# Override with --vocab_batch_size on GPU (e.g. 256 on Colab)
VOCAB_EMBED_BATCH_SIZE = 1


def load_vocab_from_csv(vocab_csv: str) -> list[str]:
    """Load word list from a single-column CSV (header: 'word')."""
    path = Path(vocab_csv)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary CSV not found: {vocab_csv}")
    words = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip()  # skip header
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    return words


def embed_vocab(
    words: list[str],
    model_name: str,
    checkpoint_path: Optional[str],
    prompt_template: str,
    batch_size: int = VOCAB_EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Embed every word in `words`. batch_size=1 for CPU/laptop, 256+ for GPU."""
    prompted = [prompt_template.format(w) for w in words]
    chunks = []
    for i in tqdm(range(0, len(prompted), batch_size), desc="Embedding vocab"):
        batch = prompted[i: i + batch_size]
        chunks.append(embed_text(batch, model_name=model_name, checkpoint_path=checkpoint_path))
    return np.vstack(chunks)


def load_or_build_vocab_embeddings(
    vocab_cache_dir: str,
    model_name: str,
    checkpoint_path: Optional[str],
    prompt_template: str,
    batch_size: int = VOCAB_EMBED_BATCH_SIZE,
    vocab_csv: str = VOCAB_CSV_DEFAULT,
) -> tuple[list[str], np.ndarray]:
    """Load vocab embeddings from cache, or build and save them if missing."""
    cache_dir = Path(vocab_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache key encodes model + checkpoint + prompt + vocab source so configs don't collide.
    ckpt_tag = Path(checkpoint_path).stem if checkpoint_path else "none"
    prompt_tag = prompt_template.replace(" ", "_").replace("<", "").replace(">", "").replace("{}", "WORD")
    vocab_tag = Path(vocab_csv).stem
    stem = f"vocab_{vocab_tag}_{model_name}_{ckpt_tag}_{prompt_tag}"
    emb_path = cache_dir / f"{stem}.npy"
    words_path = cache_dir / f"{stem}_words.json"

    if emb_path.exists() and words_path.exists():
        print(f"Loading cached vocab embeddings from {emb_path}")
        words = json.loads(words_path.read_text(encoding="utf-8"))
        embeddings = np.load(emb_path)
        print(f"  {len(words)} words, embedding dim={embeddings.shape[1]}")
        return words, embeddings

    print("Loading vocabulary from CSV...")
    words = load_vocab_from_csv(vocab_csv)
    print(f"  {len(words)} entries from {Path(vocab_csv).name}")

    print("Embedding vocabulary (this runs once and is cached)...")
    embeddings = embed_vocab(words, model_name, checkpoint_path, prompt_template, batch_size)
    np.save(emb_path, embeddings)
    words_path.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    print(f"Cached vocab embeddings to {emb_path}")

    return words, embeddings


def explore(
    pose_embeddings_dir: str,
    split: str = "test",
    label_language: str = "english",
    use_categories: bool = False,
    model_name: str = "default",
    checkpoint_path: Optional[str] = None,
    prompt_template: str = EN_LIS_PROMPT_TEMPLATE,
    top_k: int = 5,
    vocab_cache_dir: Optional[str] = None,
    vocab_batch_size: int = VOCAB_EMBED_BATCH_SIZE,
    vocab_csv: str = VOCAB_CSV_DEFAULT,
    max_samples: Optional[int] = None,
):
    embedding_dir = Path(pose_embeddings_dir)

    # Resolve vocab cache location
    cache_dir = vocab_cache_dir or str(embedding_dir / "vocab_cache")

    # --- 1. Load pose embeddings ---
    print(f"\nLoading {split} pose embeddings from {embedding_dir}...")
    pose_embs, labels, filenames, _ = load_a3lis_embeddings(
        embedding_dir, split=split, label_language=label_language, use_categories=use_categories
    )

    if len(pose_embs) == 0:
        print(f"No embeddings found for split='{split}'.")
        return

    if max_samples is not None:
        pose_embs = pose_embs[:max_samples]
        labels = labels[:max_samples]
        filenames = filenames[:max_samples]

    print(f"  {len(pose_embs)} samples, dim={pose_embs.shape[1]}")

    # --- 2. Load or build vocab embeddings ---
    words, vocab_embs = load_or_build_vocab_embeddings(
        cache_dir, model_name, checkpoint_path, prompt_template, vocab_batch_size, vocab_csv
    )
    vocab_embs = vocab_embs.astype(np.float32)
    pose_embs = pose_embs.astype(np.float32)

    # --- 3. Nearest-neighbour search (Euclidean, memory-efficient) ---
    # Use ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b to avoid the (B, V, D) broadcast
    # which would be ~1.6 GB at B=64, V=12k, D=512.
    print(f"\nSearching top-{top_k} vocab words per pose embedding (Euclidean)...")

    vocab_norms_sq = np.sum(vocab_embs ** 2, axis=1)  # (V,) — precompute once
    all_top_words = []
    all_top_dists = []

    for pose, in zip(pose_embs):
        pose_norm_sq = float(np.dot(pose, pose))
        dot = vocab_embs @ pose                          # (V,)
        dists_sq = pose_norm_sq + vocab_norms_sq - 2 * dot  # (V,)
        np.clip(dists_sq, 0, None, out=dists_sq)         # numerical safety
        top_idx = np.argpartition(dists_sq, top_k)[:top_k]
        top_idx = top_idx[np.argsort(dists_sq[top_idx])]
        all_top_words.append([words[j] for j in top_idx])
        all_top_dists.append([float(np.sqrt(dists_sq[j])) for j in top_idx])

    # --- 4. Print results ---
    hit_1 = 0
    hit_k = 0
    print(f"\n{'='*70}")
    print(f"Vocab Nearest-Neighbour Exploration  |  split={split}  |  top_k={top_k}")
    print(f"Prompt template: \"{prompt_template}\"")
    print(f"{'='*70}\n")

    for i, (gold, fname, top_words, top_dists) in enumerate(
        zip(labels, filenames, all_top_words, all_top_dists)
    ):
        in_top1 = gold.lower() == top_words[0].lower()
        in_topk = any(gold.lower() == w.lower() for w in top_words)
        if in_top1:
            hit_1 += 1
        if in_topk:
            hit_k += 1

        marker = "✓" if in_top1 else ("~" if in_topk else "✗")
        print(f"{marker} [{i+1:>4}] gold={gold!r:<25}  file={Path(fname).name}")
        for rank, (w, d) in enumerate(zip(top_words, top_dists), 1):
            hit_marker = "***" if w.lower() == gold.lower() else "   "
            print(f"       {rank}. {w:<20} dist={d:.4f} {hit_marker}")
        print()

    n = len(labels)
    print(f"{'='*70}")
    print(f"Summary  ({n} samples)")
    print(f"  Gold in top-1:   {hit_1:>5} / {n}  ({hit_1/n:.1%})")
    print(f"  Gold in top-{top_k}:   {hit_k:>5} / {n}  ({hit_k/n:.1%})")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Explore what vocabulary words pose embeddings are closest to.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pose_embeddings_dir", required=True,
                        help="Directory containing precomputed pose embeddings + metadata")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to explore (default: test)")
    parser.add_argument("--label_language", default="english", choices=["english", "italian"],
                        help="Label language for gold labels (default: english)")
    parser.add_argument("--use_categories", action="store_true",
                        help="Use macro categories as gold labels instead of micro signs")
    parser.add_argument("--model_name", default="default",
                        choices=["default", "asl_citizen", "asl_finetune", "suisse", "a3lis_finetune"],
                        help="SignCLIP model to use (default: default)")
    parser.add_argument("--model_checkpoint_path", default=None,
                        help="Optional checkpoint path overlaid on --model_name")
    parser.add_argument("--prompt_template", default=EN_LIS_PROMPT_TEMPLATE,
                        help=f"Text prompt template, use {{}} as word placeholder "
                             f"(default: \"{EN_LIS_PROMPT_TEMPLATE}\")")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of closest vocab words to show per video (default: 5)")
    parser.add_argument("--vocab_cache_dir", default=None,
                        help="Directory to cache vocab embeddings. Defaults to "
                             "<pose_embeddings_dir>/vocab_cache/")
    parser.add_argument("--vocab_csv", default=VOCAB_CSV_DEFAULT,
                        help="Path to vocabulary CSV (single 'word' column). "
                             f"Default: dataset/spreadthesign_vocabulary.csv")
    parser.add_argument("--vocab_batch_size", type=int, default=VOCAB_EMBED_BATCH_SIZE,
                        help="Batch size for embedding the vocab (default: 1 for CPU/laptop, "
                             "use 256+ on Colab/GPU)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to first N samples (useful for quick sanity checks)")

    args = parser.parse_args()

    explore(
        pose_embeddings_dir=args.pose_embeddings_dir,
        split=args.split,
        label_language=args.label_language,
        use_categories=args.use_categories,
        model_name=args.model_name,
        checkpoint_path=args.model_checkpoint_path,
        prompt_template=args.prompt_template,
        top_k=args.top_k,
        vocab_cache_dir=args.vocab_cache_dir,
        vocab_batch_size=args.vocab_batch_size,
        vocab_csv=args.vocab_csv,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
