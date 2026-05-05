import torch
from torch import nn
from .loss import Loss

"""
Decoupled Supervised Contrastive Loss for fine-tuning SignCLIP on A3LIS.

Key design decisions:
- sim_matrix is passed in pre-scaled (externally multiplied by logit_scale).
  Do NOT re-normalize or re-scale inside this function.
- The "decoupled" property means: the denominator contains ONLY negatives
  (different-class samples), excluding both self and same-class positives.
  This eliminates intra-class repulsion, which is the core improvement over
  standard SupCon.
- sim_matrix may be rectangular [N_videos x N_unique_classes] when called
  from fineTuneA3LIS, where text embeddings are deduplicated per class.
  In that case, targets must be indices into the columns (i.e. inverse_indices).
"""


class DecoupledSupCon(Loss):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def __call__(self, sim_matrix, targets, **kwargs):
        """
        Args:
            sim_matrix: [N x C] pre-scaled similarity matrix.
                        N = number of video samples in batch.
                        C = number of unique text/class embeddings (columns).
                        Already multiplied by logit_scale externally.
            targets: [N] integer tensor — single positive column per row.
                     OR a [N x C] boolean/float pos_mask for multi-positive use.
        """
        device = sim_matrix.device
        n, c = sim_matrix.shape

        # Accept either integer targets (one positive per row) or a pre-built pos_mask
        if targets.dim() == 2:
            pos_mask = targets.float().to(device)
        else:
            pos_mask = torch.zeros(n, c, device=device)
            pos_mask[torch.arange(n, device=device), targets] = 1.0

        # Negative mask [N x C]: 1 where column j is a DIFFERENT class from row i
        neg_mask = 1.0 - pos_mask

        num_pos = pos_mask.sum().item()
        if num_pos == 0:
            # Fallback: standard InfoNCE on diagonal (should not happen with dedup logic)
            diag_targets = torch.arange(min(n, c), device=device)
            return nn.CrossEntropyLoss()(sim_matrix[:len(diag_targets), :len(diag_targets)], diag_targets)

        # Decoupled denominator: sum over negatives only
        exp_sim = torch.exp(sim_matrix) * neg_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-10)

        # Mean log-likelihood over positive pairs per sample
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-10)
        loss = -mean_log_prob_pos.mean()
        return loss