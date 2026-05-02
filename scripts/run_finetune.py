import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from signclip.tasks.a3lis_finetune import fineTuneA3LIS


def main():
    parser = argparse.ArgumentParser(description="A3LIS Fine-tuning Launcher")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/signclip_v1_1/a3lis_finetune.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--jobtype",
        type=str,
        default="local_single",
        choices=["local_single", "local_predict"],
        help="local_single: train + eval loop | local_predict: eval only",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    checkpoint_path = config.fairseq.checkpoint.get("restore_file", None)
    save_dir = config.fairseq.checkpoint.get("save_dir", "runs/a3lis_finetune")
    os.makedirs(save_dir, exist_ok=True)

    finetuner = fineTuneA3LIS(config, checkpoint_path=checkpoint_path)

    if args.jobtype == "local_single":
        num_epochs = config.fairseq.optimization.get("max_epoch", 10)
        best_ckpt_path = os.path.join(save_dir, "best_checkpoint.pt")
        writer = SummaryWriter(log_dir=save_dir)
        best_score = -1.0
        num_classes = config.get('num_classes', 147) or 147

        def retrieval_score(r1, r5, r10, medK):
            """Composite retrieval score in [0, 1].
            R@1 dominates (0.5) but R@5, R@10, and medK prevent saving on
            single-video R@1 spikes. medK term: lower rank = higher score.
            More stable than R@1 alone with small val sets (~150 videos,
            min jump = 0.67%).
            """
            return (0.500 * r1 / 100.
                  + 0.250 * r5 / 100.
                  + 0.125 * r10 / 100.
                  + 0.125 * (1. - medK / num_classes))

        for epoch in range(num_epochs):
            train_loss, train_r1, train_r5, train_r10, train_medK = finetuner.train_step_with_metrics()
            val_loss,   val_r1,   val_r5,   val_r10,   val_medK   = finetuner.eval_with_metrics()

            # TensorBoard
            writer.add_scalar("Loss/train",      train_loss, epoch)
            writer.add_scalar("Loss/val",         val_loss,   epoch)
            writer.add_scalar("Recall@1/train",   train_r1,   epoch)
            writer.add_scalar("Recall@1/val",     val_r1,     epoch)
            writer.add_scalar("Recall@5/train",   train_r5,   epoch)
            writer.add_scalar("Recall@5/val",     val_r5,     epoch)
            writer.add_scalar("Recall@10/train",  train_r10,  epoch)
            writer.add_scalar("Recall@10/val",    val_r10,    epoch)
            writer.add_scalar("MedianK/train",    train_medK, epoch)
            writer.add_scalar("MedianK/val",      val_medK,   epoch)
            val_score = retrieval_score(val_r1, val_r5, val_r10, val_medK)
            writer.add_scalar("Score/val",         val_score,  epoch)
            writer.flush()

            # Terminal
            print(
                f"[Epoch {epoch+1:>3}/{num_epochs}] "
                f"Train — Loss: {train_loss:.4f} R@1: {train_r1:.2f}% R@5: {train_r5:.2f}% "
                f"R@10: {train_r10:.2f}% medK: {train_medK:.1f}"
            )
            print(
                f"[Epoch {epoch+1:>3}/{num_epochs}] "
                f"Val   — Loss: {val_loss:.4f}  R@1: {val_r1:.2f}% R@5: {val_r5:.2f}% "
                f"R@10: {val_r10:.2f}% medK: {val_medK:.1f}"
            )

            # Save best checkpoint by composite retrieval score
            if val_score > best_score:
                best_score = val_score
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": finetuner.model.state_dict(),
                        "optimizer_state_dict": finetuner.optimizer.state_dict(),
                        "val_r1": val_r1,
                        "val_r5": val_r5,
                        "val_r10": val_r10,
                        "val_medK": val_medK,
                        "val_loss": val_loss,
                        "score": val_score,
                    },
                    best_ckpt_path,
                )
                print(f"  ✓ Saved best checkpoint (score: {val_score:.4f} — R@1: {val_r1:.2f}% R@5: {val_r5:.2f}% medK: {val_medK:.1f} at epoch {epoch+1})")

        writer.close()
        print(f"\nTraining complete. Best score: {best_score:.4f} — checkpoint at {best_ckpt_path}")

    elif args.jobtype == "local_predict":
        val_loss, val_r1, val_r5, val_r10, val_medK = finetuner.eval_with_metrics()
        print(
            f"Eval — Loss: {val_loss:.4f} | R@1: {val_r1:.2f}% | R@5: {val_r5:.2f}% | "
            f"R@10: {val_r10:.2f}% | medK: {val_medK:.1f}"
        )


if __name__ == "__main__":
    main()