"""The shared statement encoder: ~33M params, SDPA attention, RoPE, pre-norm.

Every stage (MLM, gloss-fidelity, verdict head, contrastive) fine-tunes this
one architecture; heads live in heads.py. SDPA dispatches to a fused
FlashAttention-class kernel on sm89 for bf16 — no flash-attn build needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    vocab_size: int = 24_000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_seq: int = 512
    dropout: float = 0.1
    pad_id: int = 0


def rope_cache(max_seq: int, head_dim: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (10_000 ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    t = torch.arange(max_seq, device=device)
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D). Rotate pairs.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.size(-2)], sin[: x.size(-2)]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, pad_mask):
        B, T, C = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = apply_rope(q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask)
        x = x + self.dropout(self.proj(att.transpose(1, 2).reshape(B, T, C)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        cos, sin = rope_cache(cfg.max_seq, cfg.d_model // cfg.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) -> hidden states (B, T, C)."""
        pad = ids.eq(self.cfg.pad_id)
        # SDPA additive mask: (B, 1, 1, T), -inf at pad positions.
        mask = torch.zeros_like(pad, dtype=self.embed.weight.dtype)
        mask = mask.masked_fill(pad, float("-inf"))[:, None, None, :]
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin, mask)
        return self.norm(x)

    def pool(self, ids: torch.Tensor) -> torch.Tensor:
        """Mean over non-pad positions -> (B, C) statement embedding."""
        h = self.forward(ids)
        keep = ids.ne(self.cfg.pad_id).unsqueeze(-1)
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1)
