"""
Evaluate the fine-tuned A3LIS model on the held-out test set.

Reproduces exactly the R@1/R@5/R@10/MedianK metrics from training:
  - Model loaded from checkpoint (same weights used at training time)
  - Test set: signer-independent hold-out (mrlb, msf — ~294 videos)
  - 147-class retrieval: each video ranked against all 147 frozen class text embeddings
  - Loss: batch-local bidirectional NCE (MMContraLoss)

Usage:
    python src/eval_finetune.py
    python src/eval_finetune.py --checkpoint runs/signclip_a3lis_finetune/checkpoint_best.pt
    python src/eval_finetune.py --config configs/signclip_v1_1/a3lis_finetune.yaml \\
                                --checkpoint runs/signclip_a3lis_finetune/best_checkpoint_26_04.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from signclip.tasks.a3lis_finetune import fineTuneA3LIS
from signclip.datasets.a3lis_dataset import A3LISDataset
from signclip.utils.a3lis_paths import (
    POSES_ROOT,
    CSV_PATH,
    SPLIT_CONFIG_PATH,
    PRETOKENIZED_LABELS_PATH,
    DEFAULT_CHECKPOINT_PATH,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned A3LIS model on the test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use checkpoint and config from default paths
  python src/eval_finetune.py

  # Specific checkpoint
  python src/eval_finetune.py \\
      --checkpoint runs/signclip_a3lis_finetune/checkpoint_best.pt

  # Compare two checkpoints
  python src/eval_finetune.py \\
      --checkpoint runs/signclip_a3lis_finetune/best_checkpoint_26_04.pt
"""
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/signclip_v1_1/a3lis_finetune.yaml',
        help='Path to training config YAML (default: configs/signclip_v1_1/a3lis_finetune.yaml)'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to fine-tuned checkpoint .pt file. Overrides config restore_file.'
             ' Default: runs/signclip_a3lis_finetune/checkpoint_best.pt'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Batch size for evaluation (default: from config)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save results JSON. Default: no file saved.'
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    checkpoint_path = (
        args.checkpoint
        or config.fairseq.checkpoint.get('restore_file', None)
        or DEFAULT_CHECKPOINT_PATH
    )

    print(f"\n{'='*60}")
    print(f"A3LIS Fine-tune — Test Set Evaluation")
    print(f"{'='*60}")
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        print("Run scripts/run_finetune.py first to produce a checkpoint.")
        sys.exit(1)

    # Instantiate the finetuner. This:
    #   1. Loads pretrained SignCLIP weights
    #   2. Overlays the fine-tuned checkpoint
    #   3. Builds train/val dataloaders (needed for precomputing text embeddings)
    #   4. Pre-computes all 147 class text embeddings (frozen encoder)
    print("\nBuilding model and precomputing class text embeddings...")
    finetuner = fineTuneA3LIS(config, checkpoint_path=checkpoint_path)

    # Build test DataLoader with the same settings as training
    max_text_len = getattr(config.dataset, 'max_len', 64)
    batch_size = args.batch_size or getattr(config.fairseq.dataset, 'batch_size', 16)
    num_workers = getattr(config.fairseq.dataset, 'num_workers', 0)

    test_dataset = A3LISDataset(
        POSES_ROOT,
        CSV_PATH,
        split_config_path=SPLIT_CONFIG_PATH,
        split_filter='test',
        max_text_len=max_text_len,
        pretokenized_labels_path=PRETOKENIZED_LABELS_PATH
    )

    if len(test_dataset) == 0:
        print("\nERROR: No test samples found. Check split_config_path.")
        sys.exit(1)

    print(f"\nTest set:   {len(test_dataset)} samples "
          f"({len(test_dataset.meta_processor.label_map)} classes)")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=finetuner.pose_collate_fn,
        num_workers=num_workers
    )

    # Swap val_data → test_loader so eval_with_metrics runs on the test set.
    # This reuses the exact same evaluation loop used during training.
    original_val_data = finetuner.val_data
    finetuner.val_data = test_loader

    print("\nRunning evaluation on test set...")
    test_loss, r1, r5, r10, medK = finetuner.eval_with_metrics()

    finetuner.val_data = original_val_data  # restore

    n = len(test_dataset)
    num_classes = len(test_dataset.meta_processor.label_map)

    print(f"\n{'='*60}")
    print(f"Test Set Results")
    print(f"{'='*60}")
    print(f"  Checkpoint: {Path(checkpoint_path).name}")
    print(f"  Samples:    {n}")
    print(f"  Classes:    {num_classes}")
    print()
    print(f"  Loss:       {test_loss:.4f}")
    print(f"  R@1↑:       {r1:>6.2f}%  ({round(r1 / 100 * n):>3}/{n})")
    print(f"  R@5↑:       {r5:>6.2f}%  ({round(r5 / 100 * n):>3}/{n})")
    print(f"  R@10↑:      {r10:>6.2f}%  ({round(r10 / 100 * n):>3}/{n})")
    print(f"  MedianK↓:   {medK:>6.1f}  (random baseline: {num_classes // 2})")
    print(f"{'='*60}\n")

    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_path / f"eval_finetune_{timestamp}.json"
        results = {
            'checkpoint': str(checkpoint_path),
            'num_samples': n,
            'num_classes': num_classes,
            'loss': float(test_loss),
            'recall@1': r1 / 100,
            'recall@5': r5 / 100,
            'recall@10': r10 / 100,
            'median_rank': float(medK),
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
