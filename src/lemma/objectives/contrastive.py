"""Symmetric InfoNCE (CLIP-style) for gloss<->statement alignment.

Given L2-normalized gloss embeddings G (B,d) and statement embeddings S (B,d)
where row i of G pairs with row i of S, every other row in the batch is a
negative. Loss is the mean of the gloss->statement and statement->gloss
cross-entropies over the scaled similarity matrix.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(g: torch.Tensor, s: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale.exp() * (g @ s.T)
    targets = torch.arange(g.size(0), device=g.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


@torch.no_grad()
def in_batch_accuracy(g: torch.Tensor, s: torch.Tensor) -> float:
    """Fraction of glosses whose top-1 statement in the batch is its own pair."""
    return (torch.argmax(g @ s.T, dim=1) == torch.arange(g.size(0), device=g.device)).float().mean().item()
