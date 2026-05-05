import torch
from signclip.tasks.a3lis_finetune import fineTuneA3LIS
from omegaconf import OmegaConf

# This script loads the best checkpoint and runs validation on the A3LIS dataset.

def main():
    config = OmegaConf.load("configs/signclip_v1_1/a3lis_finetune.yaml")
    checkpoint_path = config.fairseq.checkpoint.save_dir + "/checkpoint_epoch_" + str(config.fairseq.optimization.max_epoch) + ".pt"
    finetuner = fineTuneA3LIS(config, checkpoint_path=checkpoint_path)
    val_loss, val_acc = finetuner.eval_with_metrics()
    print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()
