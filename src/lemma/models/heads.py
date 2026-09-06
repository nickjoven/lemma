"""Task heads over the shared encoder."""

from __future__ import annotations

import torch.nn as nn

from .encoder import Encoder, EncoderConfig


class MLMHead(nn.Module):
    """Tied-embedding masked-LM head for S2 pretraining."""

    def __init__(self, encoder: Encoder):
        super().__init__()
        cfg = encoder.cfg
        self.transform = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model)
        )
        self.decoder = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        self.decoder.weight = encoder.embed.weight

    def forward(self, hidden):
        return self.decoder(self.transform(hidden))


class ProjectionHead(nn.Module):
    """For contrastive objectives (S3 gloss-fidelity, S5 equivalence)."""

    def __init__(self, cfg: EncoderConfig, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, out_dim)
        )

    def forward(self, pooled):
        return self.net(pooled)


class ClassifierHead(nn.Module):
    """S4 verdict predictor: pooled embedding -> verdict class logits."""

    def __init__(self, cfg: EncoderConfig, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(cfg.d_model, n_classes),
        )

    def forward(self, pooled):
        return self.net(pooled)
