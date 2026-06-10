import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from signclip.tasks.signit_finetune import fineTuneSignIT


def main():
    parser = argparse.ArgumentParser(description="SignIT Fine-tuning Launcher")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/signclip_v1_1/signit_finetune.yaml",
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
    save_dir = config.fairseq.checkpoint.get("save_dir", "runs/signclip_signit_finetune")
    os.makedirs(save_dir, exist_ok=True)

    finetuner = fineTuneSignIT(config, checkpoint_path=checkpoint_path)

    if args.jobtype == "local_single":
        num_epochs = config.fairseq.optimization.get("max_epoch", 10)
        writer = SummaryWriter(log_dir=save_dir)
        num_classes = len(finetuner.train_dataset.meta_processor.label_map)

        def retrieval_score(r1, r5, r10, medK):
            return (
                0.500 * r1 / 100.
                + 0.250 * r5 / 100.
                + 0.125 * r10 / 100.
                + 0.125 * (1. - medK / num_classes)
            )

        def equal_r1_r5_r10_score(r1, r5, r10):
            return (r1 + r5 + r10) / 300.0

        criteria = {
            "loss": {
                "mode": "min",
                "best": float("inf"),
                "path": os.path.join(save_dir, "best_by_loss_checkpoint.pt"),
            },
            "r1": {
                "mode": "max",
                "best": float("-inf"),
                "path": os.path.join(save_dir, "best_by_r1_checkpoint.pt"),
            },
            "r1_r5_r10_equal": {
                "mode": "max",
                "best": float("-inf"),
                "path": os.path.join(save_dir, "best_by_r1_r5_r10_equal_checkpoint.pt"),
            },
            "regime_score": {
                "mode": "max",
                "best": float("-inf"),
                "path": os.path.join(save_dir, "best_checkpoint.pt"),
            },
        }

        for epoch in range(num_epochs):
            train_loss, train_r1, train_r5, train_r10, train_medK = finetuner.train_step_with_metrics()
            val_loss, val_r1, val_r5, val_r10, val_medK = finetuner.eval_with_metrics()

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Recall@1/train", train_r1, epoch)
            writer.add_scalar("Recall@1/val", val_r1, epoch)
            writer.add_scalar("Recall@5/train", train_r5, epoch)
            writer.add_scalar("Recall@5/val", val_r5, epoch)
            writer.add_scalar("Recall@10/train", train_r10, epoch)
            writer.add_scalar("Recall@10/val", val_r10, epoch)
            writer.add_scalar("MedianK/train", train_medK, epoch)
            writer.add_scalar("MedianK/val", val_medK, epoch)
            val_score = retrieval_score(val_r1, val_r5, val_r10, val_medK)
            val_equal_score = equal_r1_r5_r10_score(val_r1, val_r5, val_r10)
            writer.add_scalar("Score/val", val_score, epoch)
            writer.add_scalar("Score/val_r1_r5_r10_equal", val_equal_score, epoch)
            writer.flush()

            print(
                f"[Epoch {epoch + 1:>3}/{num_epochs}] "
                f"Train - Loss: {train_loss:.4f} R@1: {train_r1:.2f}% R@5: {train_r5:.2f}% "
                f"R@10: {train_r10:.2f}% medK: {train_medK:.1f}"
            )
            print(
                f"[Epoch {epoch + 1:>3}/{num_epochs}] "
                f"Val   - Loss: {val_loss:.4f}  R@1: {val_r1:.2f}% R@5: {val_r5:.2f}% "
                f"R@10: {val_r10:.2f}% medK: {val_medK:.1f}"
            )

            values = {
                "loss": val_loss,
                "r1": val_r1,
                "r1_r5_r10_equal": val_equal_score,
                "regime_score": val_score,
            }

            for name, spec in criteria.items():
                current_value = values[name]
                improved = (
                    current_value < spec["best"]
                    if spec["mode"] == "min"
                    else current_value > spec["best"]
                )
                if not improved:
                    continue

                spec["best"] = current_value
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
                        "equal_score": val_equal_score,
                        "selection_metric": name,
                        "selection_value": current_value,
                    },
                    spec["path"],
                )
                print(f"  Saved best-{name} checkpoint ({current_value:.4f})")

        writer.close()
        print("\nTraining complete. Best checkpoints:")
        for name, spec in criteria.items():
            print(f"  - {name}: {spec['best']:.4f} @ {spec['path']}")

    elif args.jobtype == "local_predict":
        val_loss, val_r1, val_r5, val_r10, val_medK = finetuner.eval_with_metrics()
        print(
            f"Eval - Loss: {val_loss:.4f} | R@1: {val_r1:.2f}% | R@5: {val_r5:.2f}% | "
            f"R@10: {val_r10:.2f}% | medK: {val_medK:.1f}"
        )


if __name__ == "__main__":
    main()
