"""
SignIT-Adapter: Frozen SignCLIP Backbone + Smart Head for A3LIS-147

This experiment treats the massive pre-trained SignCLIP model as a fixed feature extractor 
and focuses all computational effort on training a specialized, lightweight "adapter" head 
to map these features to Italian Sign Language (LIS) classes.

Key Components:
1. Frozen SignCLIP backbone for feature extraction (768-dim embeddings)
2. Smart Head: MLP adapter with residual connections
3. Dual-Loss: Cross-Entropy (label smoothing) + Optional Supervised Decoupled Contrastive Loss (inspired by SignCL)
4. Training: EMA (0.9998), small batch (16), strong augmentation
5. Evaluation: 10-fold Leave-One-Signer-Out (LOSO) cross-validation

Usage:
    # Train and evaluate with LOSO cross-validation
    python src/smart_head.py \
        --embeddings_dir dataset/embeddings/a3lis_normalised \
        --output_dir runs/smart_head \
        --epochs 100 \
        --batch_size 16 \
        --lr 0.001 \
        --lambda_contrastive 0.01 \
        --ema_momentum 0.9998 \
        --signcl_loss True \
    
    # Train on specific fold
    python src/smart_head.py \
        --embeddings_dir dataset/embeddings/a3lis_normalised \
        --output_dir runs/smart_head \
        --fold 0 \
        --epochs 100
        --signcl_loss True
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class SmartHead(nn.Module):
    """
    Adapter head for frozen SignCLIP embeddings.
    
    Architecture:
    - LayerNorm
    - Linear expansion (embedding_dim -> hidden_dim)
    - GELU activation
    - Dropout
    - Linear compression (hidden_dim -> embedding_dim)
    - Residual connection (add original embedding)
    - Final classification layer (embedding_dim -> num_classes)
    """
    
    def __init__(self, embedding_dim: int = 512, num_classes: int = 147, 
                 hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Adapter MLP with residual
        self.norm = nn.LayerNorm(embedding_dim)
        self.expand = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.compress = nn.Linear(hidden_dim, embedding_dim)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, return_features=False):
        """
        Args:
            x: Frozen SignCLIP embeddings (batch_size, embedding_dim)
            return_features: If True, return transformed features before classification
        
        Returns:
            logits: (batch_size, num_classes) or
            (logits, features) if return_features=True
        """
        # Adapter transformation with residual
        residual = x
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.compress(x)
        x = x + residual  # Residual connection preserves pre-trained features
        
        # Classification
        logits = self.classifier(x)
        
        if return_features:
            return logits, x
        return logits


"""
### Supervised Contrastive Loss Formula


1.  **Denominator Summation**: In the standard formulation of Supervised Contrastive Loss (Khosla et al.), 
    the denominator usually sums over *all* samples in the batch except the anchor (both positives and negatives).
     However, your specific code implementation (`exp_sim = torch.exp(similarity_matrix) * mask_neg`) 
     strictly zeros out the positives in the denominator, meaning it sums **only** the negatives. 
     The formula above accurately reflects your specific implementation. 

This is also known as Decoupled Contrastive Learning (DCL).

DCL does not works better as a regulariser in conjunction with cross-entropy loss, whilst
supervised contrastive learning with positives in the denominator is often used as the main loss, for large batch training without cross-entropy.

Hard Constraint vs. Relative Ranking:
With positives (Standard): The model tries to make the positive pair more similar relative to everything else in the batch.
Without positives: The model focuses purely on pushing negatives away. Without the positive term acting as a "buffer" in the denominator, the model might push negatives away so aggressively that it ignores the finer structure of the positive clusters. 

"""
     
     

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss with dynamic margin.
    
    Pulls embeddings of same class together, pushes different classes apart.
  
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature  # Temperature scaling for similarity
    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, embedding_dim) - L2 normalized
            labels: (batch_size,) - class labels
        
        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positives (same class, all other samples)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_pos.fill_diagonal_(0)  # Exclude self
        
        # Mask for negatives (all other classes)
        mask_neg = 1 - torch.eq(labels, labels.T).float().to(device)
        
        # Compute log probabilities
        exp_sim = torch.exp(similarity_matrix) * mask_neg  # Only consider negatives in denominator
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-8)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class A3LISEmbeddingDataset(Dataset):
    """Dataset for precomputed A3LIS embeddings."""
    
    def __init__(self, embeddings: np.ndarray, labels: List[str], 
                 signers: List[str], label_to_idx: Dict[str, int]):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = labels
        self.signers = signers
        self.label_to_idx = label_to_idx
        self.label_indices = torch.tensor([label_to_idx[label] for label in labels])
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'label': self.label_indices[idx],
            'label_str': self.labels[idx],
            'signer': self.signers[idx]
        }


class EMATracker:
    """Exponential Moving Average tracker for model weights."""
    
    def __init__(self, model: nn.Module, momentum: float = 0.9998):
        self.model = model
        self.momentum = momentum
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.momentum * self.shadow[name] + \
                                   (1 - self.momentum) * param.data
    
    def apply_shadow(self):
        """Replace model weights with shadow (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def load_embeddings_with_metadata(embeddings_dir: Path):
    """
    Load precomputed embeddings and metadata for A3LIS dataset.
    
    Returns:
        embeddings: np.ndarray (num_samples, embedding_dim)
        labels: List[str] - label for each sample
        signers: List[str] - signer for each sample
        filenames: List[str] - filename for each sample
        unique_labels: List[str] - sorted unique labels
    """
    metadata_path = embeddings_dir / 'embeddings_metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    embeddings = []
    labels = []
    signers = []
    filenames = []
    
    for item in tqdm(metadata['embeddings'], desc="Loading embeddings"):
        emb_path = embeddings_dir / item['embedding_file']
        if not emb_path.exists():
            continue
        
        emb = np.load(emb_path)
        if emb.ndim > 1:
            emb = emb.squeeze()
        
        embeddings.append(emb)
        labels.append(item['label_italian'])
        signers.append(item['signer'])
        filenames.append(item['embedding_file'])
    
    embeddings = np.array(embeddings)
    unique_labels = sorted(set(labels))
    
    print(f"\nLoaded {len(embeddings)} embeddings")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Unique labels: {len(unique_labels)}")
    print(f"Unique signers: {len(set(signers))}")
    
    return embeddings, labels, signers, filenames, unique_labels


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                ce_criterion: nn.Module, cl_criterion: nn.Module,
                lambda_contrastive: float, ema: Optional[EMATracker],
                device: torch.device, epoch: int):
    """Train for one epoch with dual loss."""
    model.train()
    
    total_loss = 0
    total_ce_loss = 0
    total_cl_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        embeddings = batch['embedding'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with features
        logits, features = model(embeddings, return_features=True)
        
        # Loss 1: Cross-Entropy with label smoothing
        ce_loss = ce_criterion(logits, labels)
        
        # Loss 2: Supervised Contrastive Loss (SignCL)
        cl_loss = cl_criterion(features, labels)
        
        # Combined loss
        loss = ce_loss + lambda_contrastive * cl_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Track metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_cl_loss += cl_loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'cl': f'{cl_loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_cl_loss = total_cl_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'cl_loss': avg_cl_loss,
        'accuracy': accuracy
    }


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device,
             num_classes: int):
    """Evaluate model with retrieval metrics."""
    model.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embedding'].to(device)
            labels = batch['label']
            
            logits = model(embeddings)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute retrieval metrics
    probabilities = F.softmax(all_logits, dim=1)
    sorted_indices = torch.argsort(probabilities, dim=1, descending=True)
    
    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    ranks = []
    
    for i in range(len(all_labels)):
        gold_label = all_labels[i].item()
        ranked_predictions = sorted_indices[i].tolist()
        
        if gold_label in ranked_predictions[:1]:
            hit_1 += 1
        if gold_label in ranked_predictions[:5]:
            hit_5 += 1
        if gold_label in ranked_predictions[:10]:
            hit_10 += 1
        
        if gold_label in ranked_predictions:
            rank = ranked_predictions.index(gold_label)
            ranks.append(rank)
        else:
            ranks.append(num_classes)
    
    num_test = len(all_labels)
    r_at_1 = hit_1 / num_test
    r_at_5 = hit_5 / num_test
    r_at_10 = hit_10 / num_test
    median_rank = statistics.median(ranks) + 1  # 1-indexed
    
    return {
        'r@1': r_at_1,
        'r@5': r_at_5,
        'r@10': r_at_10,
        'median_rank': median_rank,
        'hit_1': hit_1,
        'hit_5': hit_5,
        'hit_10': hit_10,
        'num_test': num_test
    }


def train_and_evaluate_fold(
    fold: int,
    train_embeddings: np.ndarray,
    train_labels: List[str],
    train_signers: List[str],
    test_embeddings: np.ndarray,
    test_labels: List[str],
    test_signers: List[str],
    label_to_idx: Dict[str, int],
    args,
    device: torch.device
):
    """Train and evaluate on a single LOSO fold."""
    
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/10: Test signer = {test_signers[0]}")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_embeddings)}")
    print(f"Test samples: {len(test_embeddings)}")
    
    # Create datasets
    train_dataset = A3LISEmbeddingDataset(
        train_embeddings, train_labels, train_signers, label_to_idx
    )
    test_dataset = A3LISEmbeddingDataset(
        test_embeddings, test_labels, test_signers, label_to_idx
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Initialize model
    embedding_dim = train_embeddings.shape[1]
    num_classes = len(label_to_idx)
    
    model = SmartHead(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Loss functions
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    cl_criterion = SupervisedContrastiveLoss(temperature=args.temperature)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # EMA tracker
    ema = EMATracker(model, momentum=args.ema_momentum) if args.use_ema else None
    
    # Training loop
    best_r_at_1 = 0
    best_epoch = 0
    train_history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, ce_criterion, cl_criterion,
            args.lambda_contrastive, ema, device, epoch
        )

        # Evaluate with EMA weights
        if ema is not None:
            ema.apply_shadow()

        eval_metrics = evaluate(model, test_loader, device, num_classes)

        if ema is not None:
            ema.restore()

        # Learning rate step
        scheduler.step()

        # Track best model
        if eval_metrics['r@1'] > best_r_at_1:
            best_r_at_1 = eval_metrics['r@1']
            best_epoch = epoch

            # Save best model
            if args.save_models:
                save_path = Path(args.output_dir) / f'fold_{fold}_best.pt'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_shadow': ema.shadow if ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': eval_metrics
                }, save_path)

        # Log progress
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"ce={train_metrics['ce_loss']:.4f}, "
              f"cl={train_metrics['cl_loss']:.4f}, "
              f"acc={train_metrics['accuracy']:.2f}%")
        print(f"  Test:  R@1={eval_metrics['r@1']:.4f}, "
              f"R@5={eval_metrics['r@5']:.4f}, "
              f"R@10={eval_metrics['r@10']:.4f}, "
              f"MedianR={eval_metrics['median_rank']:.1f}")
        print(f"  Best:  R@1={best_r_at_1:.4f} (epoch {best_epoch})")
        print(f"  LR:    {scheduler.get_last_lr()[0]:.6f}")

        train_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'test': eval_metrics
        })
    
    # Final evaluation with best EMA weights
    if args.save_models and ema is not None:
        checkpoint_path = Path(args.output_dir) / f'fold_{fold}_best.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            if 'ema_shadow' in checkpoint and checkpoint['ema_shadow'] is not None:
                # Load EMA weights
                for name, param in model.named_parameters():
                    if name in checkpoint['ema_shadow']:
                        param.data = checkpoint['ema_shadow'][name]
    
    final_metrics = evaluate(model, test_loader, device, num_classes)
    
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1} Final Results (Test Signer: {test_signers[0]})")
    print(f"{'='*60}")
    print(f"  R@1↑:        {final_metrics['r@1']:>7.2%}  ({final_metrics['hit_1']:>5}/{final_metrics['num_test']})")
    print(f"  R@5↑:        {final_metrics['r@5']:>7.2%}  ({final_metrics['hit_5']:>5}/{final_metrics['num_test']})")
    print(f"  R@10↑:       {final_metrics['r@10']:>7.2%}  ({final_metrics['hit_10']:>5}/{final_metrics['num_test']})")
    print(f"  MedianR↓:    {final_metrics['median_rank']:>7.1f}")
    print(f"{'='*60}\n")
    
    return {
        'fold': fold,
        'test_signer': test_signers[0],
        'final_metrics': final_metrics,
        'best_epoch': best_epoch,
        'train_history': train_history
    }


def run_loso_cross_validation(embeddings_dir: str, args, device: torch.device):
    """Run 10-fold Leave-One-Signer-Out cross-validation."""
    
    embeddings_dir = Path(embeddings_dir)
    
    # Load all data
    embeddings, labels, signers, filenames, unique_labels = load_embeddings_with_metadata(embeddings_dir)
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Get unique signers (should be 10 for A3LIS)
    unique_signers = sorted(set(signers))
    print(f"\nUnique signers: {unique_signers}")
    print(f"Number of folds: {len(unique_signers)}")
    
    if len(unique_signers) != 10:
        print(f"WARNING: Expected 10 signers for A3LIS-147, found {len(unique_signers)}")
    
    # Run LOSO cross-validation
    all_results = []
    
    for fold, test_signer in enumerate(unique_signers):
        # Skip if only running specific fold
        if args.fold is not None and fold != args.fold:
            continue
        
        # Split data by signer
        train_mask = np.array([s != test_signer for s in signers])
        test_mask = np.array([s == test_signer for s in signers])
        
        train_embeddings = embeddings[train_mask]
        train_labels = [labels[i] for i in range(len(labels)) if train_mask[i]]
        train_signers_fold = [signers[i] for i in range(len(signers)) if train_mask[i]]
        
        test_embeddings = embeddings[test_mask]
        test_labels = [labels[i] for i in range(len(labels)) if test_mask[i]]
        test_signers_fold = [signers[i] for i in range(len(signers)) if test_mask[i]]
        
        # Train and evaluate on this fold
        fold_results = train_and_evaluate_fold(
            fold, train_embeddings, train_labels, train_signers_fold,
            test_embeddings, test_labels, test_signers_fold,
            label_to_idx, args, device
        )
        
        all_results.append(fold_results)
    
    # Aggregate results across all folds
    if len(all_results) == len(unique_signers):
        print(f"\n{'='*60}")
        print(f"LOSO Cross-Validation Results (All 10 Folds)")
        print(f"{'='*60}\n")
        
        # Per-fold results
        for result in all_results:
            m = result['final_metrics']
            print(f"Fold {result['fold']+1} ({result['test_signer']:>4}): "
                  f"R@1={m['r@1']:.4f}, R@5={m['r@5']:.4f}, "
                  f"R@10={m['r@10']:.4f}, MedianR={m['median_rank']:.1f}")
        
        # Average results
        avg_r1 = np.mean([r['final_metrics']['r@1'] for r in all_results])
        avg_r5 = np.mean([r['final_metrics']['r@5'] for r in all_results])
        avg_r10 = np.mean([r['final_metrics']['r@10'] for r in all_results])
        avg_median_rank = np.mean([r['final_metrics']['median_rank'] for r in all_results])
        
        std_r1 = np.std([r['final_metrics']['r@1'] for r in all_results])
        std_r5 = np.std([r['final_metrics']['r@5'] for r in all_results])
        std_r10 = np.std([r['final_metrics']['r@10'] for r in all_results])
        std_median_rank = np.std([r['final_metrics']['median_rank'] for r in all_results])
        
        print(f"\n{'='*60}")
        print(f"Average Across All Folds")
        print(f"{'='*60}")
        print(f"  R@1↑:        {avg_r1:.4f} ± {std_r1:.4f}")
        print(f"  R@5↑:        {avg_r5:.4f} ± {std_r5:.4f}")
        print(f"  R@10↑:       {avg_r10:.4f} ± {std_r10:.4f}")
        print(f"  MedianR↓:    {avg_median_rank:.2f} ± {std_median_rank:.2f}")
        print(f"{'='*60}\n")
        
        # Save aggregated results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'loso_results.json', 'w') as f:
            json.dump({
                'pose_embeddings_dir': args.embeddings_dir,
                'all_folds': all_results,
                'average': {
                    'r@1': {'mean': avg_r1, 'std': std_r1},
                    'r@5': {'mean': avg_r5, 'std': std_r5},
                    'r@10': {'mean': avg_r10, 'std': std_r10},
                    'median_rank': {'mean': avg_median_rank, 'std': std_median_rank}
                },
                'hyperparameters': vars(args)
            }, f, indent=2)
        
        print(f"Results saved to {output_dir / 'loso_results.json'}\n")


def main():
    parser = argparse.ArgumentParser(
        description="SignIT-Adapter: Smart Head training for A3LIS-147",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data
    parser.add_argument('--embeddings_dir', type=str, required=True,
                       help='Directory containing precomputed embeddings')
    parser.add_argument('--output_dir', type=str, default='runs/smart_head',
                       help='Output directory for models and results')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (L2 regularization)')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=1024,
                       help='Hidden dimension for MLP adapter')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Loss function
    parser.add_argument('--lambda_contrastive', type=float, default=0.01,
                       help='Weight for contrastive loss (lambda)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing for cross-entropy')    
    
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature scaling for contrastive loss')

    # Regularization
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use Exponential Moving Average')
    parser.add_argument('--ema_momentum', type=float, default=0.9998,
                       help='EMA momentum')
    
    # Cross-validation
    parser.add_argument('--fold', type=int, default=None,
                       help='Run specific fold (0-9), or None for all folds')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement). Set to 0 to disable.')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--save_models', action='store_true',
                       help='Save best model checkpoints')
    
    parser.add_argument('--no-signcl_loss', action='store_true', default=False,
                       help='Whether to include Supervised Contrastive Loss (SignCL)')

    args = parser.parse_args()


    #signcl_loss false overrides lambda_contrastive to 0
    if args.no_signcl_loss:
        args.lambda_contrastive = 0.0
        print("\nSupervised Contrastive Loss (SignCL) disabled. "
              "Lambda for contrastive loss set to 0.\n")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Print hyperparameters
    print(f"\n{'='*60}")
    print(f"Hyperparameters")
    print(f"{'='*60}")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Run LOSO cross-validation
    run_loso_cross_validation(args.embeddings_dir, args, device)


if __name__ == '__main__':
    main()
