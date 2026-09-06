"""Shape, masking, and size sanity for the shared encoder."""

import torch

from lemma.models.encoder import Encoder, EncoderConfig
from lemma.models.heads import ClassifierHead, MLMHead


def small():
    return EncoderConfig(vocab_size=1000, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq=32)


def test_forward_shapes():
    cfg = small()
    enc = Encoder(cfg)
    ids = torch.randint(1, cfg.vocab_size, (3, 16))
    assert enc(ids).shape == (3, 16, cfg.d_model)
    assert enc.pool(ids).shape == (3, cfg.d_model)


def test_pad_does_not_leak_into_pool():
    cfg = small()
    enc = Encoder(cfg).eval()
    ids = torch.randint(1, cfg.vocab_size, (1, 8))
    padded = torch.cat([ids, torch.zeros(1, 8, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a, b = enc.pool(ids), enc.pool(padded)
    assert torch.allclose(a, b, atol=1e-5)


def test_heads_shapes():
    cfg = small()
    enc = Encoder(cfg)
    ids = torch.randint(1, cfg.vocab_size, (2, 16))
    hidden = enc(ids)
    assert MLMHead(enc)(hidden).shape == (2, 16, cfg.vocab_size)
    assert ClassifierHead(cfg, 5)(enc.pool(ids)).shape == (2, 5)


def test_full_config_param_count_in_budget():
    n = sum(p.numel() for p in Encoder(EncoderConfig()).parameters())
    assert 25e6 < n < 45e6, f"{n/1e6:.1f}M params out of the 33M-class budget"
