"""W0 model plumbing on a tiny encoder (CPU): shapes, unknown tactic, censored
cost loss finite and ordering-correct."""

import math

import torch

from lemma.models.encoder import Encoder, EncoderConfig
import copy

from safetensors.torch import load_file

from lemma.train_w0 import OUTCOMES, BestPair, WorldModel, tobit_nll


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


def test_best_pair_saves_online_and_target_from_the_same_step(tmp_path):
    """lemma #9: with the best step before the final step, the saved pair must be
    the online model AND the EMA target as they were at that step, not the
    final target."""
    torch.manual_seed(0)
    model = WorldModel(tiny(), ["simp"], d_tac=8)
    target = copy.deepcopy(model.enc)
    pair = BestPair(tmp_path / "online.safetensors", tmp_path / "target.safetensors")

    def perturb():
        with torch.no_grad():
            for p in list(model.parameters()) + list(target.parameters()):
                p.add_(torch.randn_like(p))

    assert pair.consider(0.5, 1, model, target)
    perturb()
    assert pair.consider(0.7, 2, model, target)          # the best step
    best_online = {k: v.clone() for k, v in model.state_dict().items()}
    best_target = {k: v.clone() for k, v in target.state_dict().items()}
    perturb()
    assert not pair.consider(0.6, 3, model, target)      # worse: nothing overwritten
    assert pair.best_step == 2 and pair.best == 0.7

    saved_online = load_file(str(pair.online_path))
    saved_target = load_file(str(pair.target_path))
    for k, v in best_online.items():
        assert torch.equal(saved_online[k], v), k
    for k, v in best_target.items():
        assert torch.equal(saved_target[k], v), k
    # and NOT the final (perturbed) target, which is what the old code released
    assert any(not torch.equal(saved_target[k], v) for k, v in target.state_dict().items())
