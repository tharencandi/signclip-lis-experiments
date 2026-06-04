# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Decoupled Hard-Negative NCE (DHN-NCE) for cross-modal contrastive learning."""

import torch

from .loss import Loss


class DHNNCELoss(Loss):
    """MedCLIP-style decoupled hard-negative NCE for video/text pairs.

    Given aligned batches (video_i, text_i), this removes positives from the
    denominator and reweights negatives using beta-controlled hardness weights.
    """

    def __init__(self, temperature=0.07, beta1=0.5, beta2=0.5, eps=1e-8):
        self.temperature = temperature
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def __call__(self, pooled_video, pooled_text, **kwargs):
        bsz = pooled_video.size(0)
        if bsz <= 1:
            # No negatives available in-batch.
            return pooled_video.new_zeros(())

        sim = (pooled_video @ pooled_text.t()) / self.temperature
        diag_mask = torch.eye(bsz, dtype=torch.bool, device=sim.device)

        # Positive logits (i,i)
        pos_v2t = sim.diag()
        pos_t2v = sim.t().diag()

        # Reweight negatives for v->t.
        neg_weight_v2t = torch.exp(self.beta1 * sim).masked_fill(diag_mask, 0.0)
        neg_weight_v2t = (bsz - 1) * neg_weight_v2t / (
            neg_weight_v2t.sum(dim=1, keepdim=True) + self.eps
        )

        # Reweight negatives for t->v.
        sim_t = sim.t()
        neg_weight_t2v = torch.exp(self.beta2 * sim_t).masked_fill(diag_mask, 0.0)
        neg_weight_t2v = (bsz - 1) * neg_weight_t2v / (
            neg_weight_t2v.sum(dim=1, keepdim=True) + self.eps
        )

        # Decoupled denominators: exclude diagonal by masking to -inf before exp.
        neg_sim_v2t = sim.masked_fill(diag_mask, float("-inf"))
        neg_sim_t2v = sim_t.masked_fill(diag_mask, float("-inf"))

        loss_v2t = -pos_v2t + torch.log(
            (torch.exp(neg_sim_v2t) * neg_weight_v2t).sum(dim=1) + self.eps
        )
        loss_t2v = -pos_t2v + torch.log(
            (torch.exp(neg_sim_t2v) * neg_weight_t2v).sum(dim=1) + self.eps
        )

        return (loss_v2t + loss_t2v).mean()
