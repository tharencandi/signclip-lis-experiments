"""
Retrieval evaluation metrics for sign language recognition.

Used by:
  - signclip/tasks/a3lis_finetune.py  (training / val loop)
  - src/eval_finetune.py              (test-set evaluation)
"""

import torch


def compute_retrieval_metrics(sim_matrix: torch.Tensor, label_tensor: torch.Tensor):
    """Compute R@1, R@5, R@10 and median rank from a similarity matrix.

    Args:
        sim_matrix:   Float tensor of shape (N, num_classes).
                      Higher values = more similar.
        label_tensor: Long tensor of shape (N,) with ground-truth class indices.

    Returns:
        r1       (float): Recall@1  in percent  [0, 100]
        r5       (float): Recall@5  in percent  [0, 100]
        r10      (float): Recall@10 in percent  [0, 100]
        medianK  (float): Median rank of the correct class (1-indexed)
        ranks    (list[int]): Per-sample ranks (1-indexed)
    """
    recall_at_1 = recall_at_5 = recall_at_10 = 0
    ranks = []

    for i in range(sim_matrix.size(0)):
        sorted_cols = torch.argsort(sim_matrix[i], descending=True)
        correct_col = label_tensor[i].item()
        rank = (sorted_cols == correct_col).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
        if rank == 1:
            recall_at_1 += 1
        if rank <= 5:
            recall_at_5 += 1
        if rank <= 10:
            recall_at_10 += 1

    n = sim_matrix.size(0)
    medianK = float(torch.median(torch.tensor(ranks, dtype=torch.float)).item())
    r1  = 100. * recall_at_1  / n
    r5  = 100. * recall_at_5  / n
    r10 = 100. * recall_at_10 / n

    return r1, r5, r10, medianK, ranks
