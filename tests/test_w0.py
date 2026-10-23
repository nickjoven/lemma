"""W0 model plumbing on a tiny encoder (CPU): shapes, unknown tactic, censored
cost loss finite and ordering-correct."""

import copy
import math

import torch
from safetensors.torch import load_file

from lemma.models.encoder import Encoder, EncoderConfig
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


def test_train_releases_the_best_pair_with_role_labelled_cids(tmp_path, monkeypatch, ket_store):
    """Review of #10 test gap: drive train_w0.train() end to end on CPU and
    check the end-of-training moves (best files become the released files),
    the CID roles ([0] online, [1] EMA target), the sealed manifest, the
    .gitignore lines, and that the run records the split it inherited."""
    import argparse
    import json

    import blake3
    import yaml
    from tokenizers import Tokenizer, models, pre_tokenizers

    from lemma import ledger, train_w0
    from lemma.data.split_binding import Binding
    from lemma.schemas import RunInputs, RunRecord

    words = [f"w{i}" for i in range(40)]
    vocab = {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    monkeypatch.setattr(train_w0.Tokenizer, "from_file", staticmethod(lambda _p: tok))
    monkeypatch.setattr(train_w0, "load_s2_encoder",
                        lambda ckpt, vocab_size, max_seq: Encoder(EncoderConfig(vocab_size=vocab_size, max_seq=max_seq,
                                                                                d_model=16, n_layers=1, n_heads=2, d_ff=32)))

    def row(i):
        return {"before": f"w{i % 40} w{(i * 7) % 40}", "after": f"w{(i + 3) % 40} w{(i * 5) % 40}", "single": True,
                "tactic": ["simp", "aesop"][i % 2], "outcome": i % 4, "logcost": 2.0, "censored": i % 9 == 0,
                "before_lock": f"b{i}", "after_lock": f"a{i}"}
    data = {"train": [row(i) for i in range(120)], "val": [row(i) for i in range(200, 225)], "test": []}
    binding = Binding({}, "v1-module-lock-follows-first", ledger.ket_put(b"fixture certificate"), "pinned_rule_reconstructed")
    monkeypatch.setattr(train_w0, "load_data", lambda cfg, run: (data, ["aesop", "simp"], binding))

    s2 = RunRecord(run_id="s2-fixture", kind="train", stage="S2", started=ledger.now_iso(),
                   inputs=RunInputs(corpus_cids=["corpus"], config_cid="cfg", code_git=ledger.git_head(), mathlib_pin="pin"))
    s2.checkpoint_cids.append(ledger.ket_put(b"s2 weights"))
    ledger.finish_run(s2, {"masked_acc": 0.9})

    root = tmp_path / "w0root"
    root.mkdir()
    (root / ".gitignore").write_text("")
    monkeypatch.setattr(train_w0, "REPO_ROOT", root)
    cfg = {"s2_run": "s2-fixture", "tokenizer_path": "unused", "max_seq": 8, "seed": 0, "batch_size": 8,
           "steps": 6, "val_every": 2, "lr_encoder": 1e-3, "lr_heads": 1e-2, "ema": 0.5}
    (tmp_path / "w0.yaml").write_text(yaml.safe_dump(cfg))
    assert train_w0.train(argparse.Namespace(config=tmp_path / "w0.yaml", run_id="w0-fixture")) == 0

    ckpts = root / "runs" / "ckpts"
    online, target = ckpts / "w0-fixture.safetensors", ckpts / "w0-fixture-target.safetensors"
    assert online.exists() and target.exists()
    assert not (ckpts / "w0-fixture-best.safetensors").exists()
    assert not (ckpts / "w0-fixture-best-target.safetensors").exists()
    rec = ledger.citable("w0-fixture")
    assert rec.checkpoint_cids == [blake3.blake3(online.read_bytes()).hexdigest(),
                                   blake3.blake3(target.read_bytes()).hexdigest()]
    assert rec.inputs.split_manifest_cid == binding.certificate_cid
    metrics = json.loads(ledger.ket_get(rec.metrics_cid))["metrics"]
    assert metrics["checkpoint_roles"] == {"online": rec.checkpoint_cids[0], "ema_target": rec.checkpoint_cids[1]}
    manifest = json.loads(ledger.ket_get_verified(metrics["checkpoint_manifest_cid"]))
    assert [c["role"] for c in manifest["checkpoints"]] == ["online", "ema_target"]
    assert manifest["best_step"] == metrics["best_step"] and manifest["steps"] == 6
    assert json.loads(ledger.ket_get_verified(manifest["tactics_cid"])) == ["aesop", "simp"]
    ignored = (root / ".gitignore").read_text().splitlines()
    assert all(f".ket/cas/{c}" in ignored for c in rec.checkpoint_cids)
