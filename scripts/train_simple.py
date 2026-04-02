"""Simple PyTorch training loop for SignCLIP fine-tuning.

No fairseq dependencies - just pure PyTorch.
"""
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def simple_train(
    model,
    train_loader,
    val_loader=None,
    epochs=10,
    lr=5e-5,
    save_dir="checkpoints",
    device="cuda"
):
    """Simple training loop for fine-tuning SignCLIP.
    
    Args:
        model: SignCLIP model
        train_loader: Training data loader
        val_loader: Optional validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints
        device: Device to train on
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress:
            # Move batch to device
            video = batch['video'].to(device)
            text = batch['text'].to(device)
            text_mask = batch['text_mask'].to(device)
            video_mask = batch['video_mask'].to(device)
            
            # Forward pass
            output = model(text, text_mask, video, video_mask)
            loss = output['loss'] if isinstance(output, dict) else output
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress
            total_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "checkpoint_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    video = batch['video'].to(device)
                    text = batch['text'].to(device)
                    text_mask = batch['text_mask'].to(device)
                    video_mask = batch['video_mask'].to(device)
                    
                    output = model(text, text_mask, video, video_mask)
                    loss = output['loss'] if isinstance(output, dict) else output
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            model.train()
    
    print(f"Training complete! Best loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple SignCLIP Training")
    parser.add_argument("--config", type=str, required=True, help="Model config path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    
    args = parser.parse_args()
    
    print("Load your model and data here, then call simple_train()")
    print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
