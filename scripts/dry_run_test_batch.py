# Dry-run/test-batch script for A3LIS fine-tuning pipeline
# This script will run a single batch through the data loader, collate, model, and loss to catch errors before full training.

from signclip.tasks.a3lis_finetune import fineTuneA3LIS
from omegaconf import OmegaConf
import torch

config = OmegaConf.load("configs/signclip_v1_1/a3lis_finetune.yaml")
checkpoint_path = "pretrained_models/baseline_temporal_checkpoint_best.pt"

finetuner = fineTuneA3LIS(config, checkpoint_path=checkpoint_path)

# Get a single batch from the train loader
batch_iter = iter(finetuner.train_data)
batch = next(batch_iter)

print("Batch keys:", batch.keys())
for k, v in batch.items():
    if torch.is_tensor(v):
        print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
    else:
        print(f"{k}: type={type(v)}")

# Move to device
batch = finetuner.move_to_device(batch)

# Model forward pass
try:
    logits = finetuner.model(batch['caps'], batch['cmasks'], batch['pose'], batch['vmasks'])
    print("Model output shape:", logits.shape)
except Exception as e:
    print("Model forward error:", e)

# Loss computation
try:
    loss = finetuner.criterion(logits, batch['label'])
    print("Loss value:", loss.item())
except Exception as e:
    print("Loss computation error:", e)

print("Dry-run/test-batch completed.")
