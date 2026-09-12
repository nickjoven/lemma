"""W0 model plumbing on a tiny encoder (CPU): shapes, unknown tactic, censored
cost loss finite and ordering-correct."""

import math

import torch

from lemma.models.encoder import Encoder, EncoderConfig
from lemma.train_w0 import OUTCOMES, WorldModel, tobit_nll


def tiny():
    return Encoder(EncoderConfig(vocab_size=64, max_seq=16, d_model=32, n_layers=1, n_heads=2))


def test_forward_shapes_and_unknown_tactic():
    m = WorldModel(tiny(), ["simp", "aesop"], d_tac=8)
    ids = torch.randint(1, 64, (3, 10))
    z, o, c = m(ids, ["simp", "aesop", "never-seen"])
    assert z.shape == (3, 32) and o.shape == (3, len(OUTCOMES)) and c.shape == (3, 2)
    assert m.tac_idx(["never-seen"], ids.device).item() == 2


def test_tobit_censored_rows_prefer_higher_predictions():
    y = torch.tensor([math.log(1000.0)])                       # cap
    cens = torch.tensor([True])
    low = tobit_nll(torch.tensor([math.log(10.0)]), torch.zeros(1), y, cens)
    high = tobit_nll(torch.tensor([math.log(5000.0)]), torch.zeros(1), y, cens)
    assert torch.isfinite(low) and torch.isfinite(high) and high < low
    obs = tobit_nll(torch.tensor([math.log(1000.0)]), torch.zeros(1), y, torch.tensor([False]))
    assert torch.isfinite(obs)
