"""Cache round-trip and MLM corruption invariants."""

import numpy as np
import torch

from lemma.data.cache import CachedStatements, build
from lemma.objectives.mlm import mlm_loss, span_mask


class TinyTok:
    def encode(self, s):
        class E:
            pass
        e = E()
        e.ids = [min(ord(c), 60_000) for c in s]
        return e


def test_cache_round_trip(tmp_path):
    stmts = ["abc", "de", "x" * 700]
    stats = build(stmts, TinyTok(), tmp_path, "t", max_seq=512)
    assert stats["sequences"] == 3
    assert stats["truncated"] == 1
    cache = CachedStatements(tmp_path, "t")
    assert len(cache) == 3
    assert cache[0].tolist() == [ord(c) for c in "abc"]
    assert len(cache[2]) == 512


def test_span_mask_never_touches_pad():
    ids = torch.randint(4, 1000, (4, 32))
    ids[:, 24:] = 0  # pad
    g = torch.Generator().manual_seed(0)
    corrupted, labels = span_mask(ids, pad_id=0, mask_id=1, vocab_size=1000, generator=g)
    assert (labels[:, 24:] == -100).all()
    assert (corrupted[:, 24:] == 0).all()


def test_mlm_loss_finite_and_labels_sparse():
    ids = torch.randint(4, 1000, (4, 32))
    g = torch.Generator().manual_seed(0)
    corrupted, labels = span_mask(ids, pad_id=0, mask_id=1, vocab_size=1000, generator=g)
    frac = labels.ne(-100).float().mean().item()
    assert 0.02 < frac < 0.45, f"corruption rate {frac} out of band"
    logits = torch.randn(4, 32, 1000)
    assert torch.isfinite(mlm_loss(logits, labels))
