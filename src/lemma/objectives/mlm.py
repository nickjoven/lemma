"""S2 objective: span-corruption masked LM.

15% of tokens corrupted in spans of mean length 3; 80/10/10 mask/random/keep.
The falsifiable floor is masked-token accuracy against a unigram-frequency
baseline computed on the same split (evaluate.py owns that comparison).
"""

from __future__ import annotations

import torch


def span_mask(ids: torch.Tensor, pad_id: int, mask_id: int, vocab_size: int,
              rate: float = 0.15, mean_span: float = 3.0,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (corrupted_ids, labels) with labels = -100 off the masked positions."""
    B, T = ids.shape
    valid = ids.ne(pad_id)
    # Geometric span sampling via a Bernoulli "span start" process.
    start_p = rate / mean_span
    starts = torch.rand(B, T, generator=generator, device=ids.device) < start_p
    lengths = torch.clamp(
        torch.poisson(torch.full((B, T), mean_span - 1, device=ids.device),
                      generator=generator).long() + 1, max=8)
    mask = torch.zeros_like(valid)
    for offset in range(int(lengths.max().item())):
        step = starts & (lengths > offset)
        mask |= step.roll(shifts=offset, dims=1)
    mask &= valid

    labels = torch.where(mask, ids, torch.full_like(ids, -100))
    corrupted = ids.clone()
    r = torch.rand(B, T, generator=generator, device=ids.device)
    corrupted[mask & (r < 0.8)] = mask_id
    random_slot = mask & (r >= 0.8) & (r < 0.9)
    corrupted[random_slot] = torch.randint(
        4, vocab_size, (int(random_slot.sum()),), generator=generator, device=ids.device)
    return corrupted, labels


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    keep = labels.ne(-100)
    if not keep.any():
        return 0.0
    return (logits.argmax(-1)[keep] == labels[keep]).float().mean().item()
