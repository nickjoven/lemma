"""S3: gloss-fidelity — align docstrings with their statements.

Shared encoder (warm-started from the S2 checkpoint) with two projection
heads, trained with symmetric InfoNCE over (docstring, readable statement)
pairs drawn from the declarations corpus. Splits are the module-level
leakage-safe assignment, so eval glosses come from held-out modules.

    uv run python -u -m lemma.train_s3 --config configs/s3_gloss.yaml
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from . import ledger
from .data import splits
from .data.corpus import encoder_text, iter_rows
from .models.encoder import Encoder, EncoderConfig
from .models.heads import ProjectionHead
from .objectives.contrastive import in_batch_accuracy, info_nce

REPO_ROOT = Path(__file__).resolve().parents[2]


class GlossModel(nn.Module):
    def __init__(self, enc: Encoder, out_dim: int):
        super().__init__()
        self.enc = enc
        self.proj_g = ProjectionHead(enc.cfg, out_dim)
        self.proj_s = ProjectionHead(enc.cfg, out_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

    def encode(self, ids: torch.Tensor, which: str) -> torch.Tensor:
        z = self.enc.pool(ids)
        z = self.proj_g(z) if which == "g" else self.proj_s(z)
        return nn.functional.normalize(z, dim=-1)


def load_s2_encoder(ckpt: Path, vocab_size: int, max_seq: int) -> Encoder:
    cfg = EncoderConfig(vocab_size=vocab_size, max_seq=max_seq)
    enc = Encoder(cfg)
    sd = load_file(str(ckpt))
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    return enc


def gloss_pairs_by_split() -> dict[str, list[tuple[str, str, str]]]:
    """split -> [(docstring, statement, module)] for declarations with a docstring.
    One corpus pass for all splits (the pass is the expensive part)."""
    rows = list(iter_rows("declarations", verify=False))
    assign = splits.assign(rows)
    out: dict[str, list] = {s: [] for s in splits.SPLITS}
    for r in rows:
        if r.docstring:
            out[assign[r.name]].append((r.docstring, encoder_text(r), r.module))
    return out


def gloss_pairs(split: str) -> list[tuple[str, str, str]]:
    return gloss_pairs_by_split()[split]


def module_index(pairs: list[tuple[str, str, str]]) -> dict[str, list[int]]:
    by_mod: dict[str, list[int]] = {}
    for i, (_, _, m) in enumerate(pairs):
        by_mod.setdefault(m, []).append(i)
    return by_mod


def sample_hard_batch(rng: np.random.Generator, by_mod: dict[str, list[int]], mods: list[str],
                      bs: int, per_module: int) -> list[int]:
    """Module-grouped batch: several pairs from each sampled module, so
    same-module statements are in-batch HARD negatives for each other. v1
    used uniform random pairs and never exercised the same-module case — the
    exact case its hard-drift AUC (0.767) failed on."""
    idx: list[int] = []
    while len(idx) < bs:
        m = mods[rng.integers(0, len(mods))]
        members = by_mod[m]
        take = min(per_module, len(members), bs - len(idx))
        idx.extend(rng.choice(members, size=take, replace=False).tolist())
    return idx


@torch.no_grad()
def val_hard_auc(model: "GlossModel", tok: Tokenizer, val: list[tuple[str, str, str]],
                 max_seq: int, device: str, rng: random.Random) -> float:
    """Same-module hard-negative drift AUC on a fixed val subset — the S3 gate
    metric, probed during training for early stopping."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    pad = model.enc.cfg.pad_id
    G, S = [], []
    for i in range(0, len(val), 256):
        chunk = val[i:i + 256]
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            G.append(model.encode(tokenize(tok, [c[0] for c in chunk], max_seq, pad).to(device), "g").float().cpu())
            S.append(model.encode(tokenize(tok, [c[1] for c in chunk], max_seq, pad).to(device), "s").float().cpu())
    G, S = torch.cat(G), torch.cat(S)
    sims = G @ S.T
    by_mod = module_index(val)
    pos, neg = [], []
    for i, (_, _, m) in enumerate(val):
        others = [k for k in by_mod[m] if k != i]
        if others:
            pos.append(sims[i, i].item())
            neg.append(sims[i, rng.choice(others)].item())
    model.train()
    if len(pos) < 20:
        return float("nan")
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))


def tokenize(tok: Tokenizer, texts: list[str], max_seq: int, pad_id: int) -> torch.Tensor:
    batch = [tok.encode(t).ids[:max_seq] for t in texts]
    width = max(len(b) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b)] = torch.tensor(b)
    return ids


def save_ckpt(model: GlossModel, path: Path) -> None:
    d = model.state_dict()
    save_file({k: v.contiguous() for k, v in d.items()}, str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    config_cid = ledger.ket_put(args.config.read_bytes())
    run_id = args.run_id or f"s3-gloss-{time.strftime('%Y%m%d-%H%M%S')}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 1337))

    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    s2_rec = ledger.citable(cfg["s2_run"])
    s2_ckpt = REPO_ROOT / "runs" / "ckpts" / f"{cfg['s2_run']}.safetensors"
    enc = load_s2_encoder(s2_ckpt, tok.get_vocab_size(), cfg["max_seq"])
    model = GlossModel(enc, cfg.get("proj_dim", 256)).to(device)
    pad = enc.cfg.pad_id

    by_split = gloss_pairs_by_split()
    train = by_split["train"]
    prng = random.Random(cfg.get("seed", 1337))
    val = prng.sample(by_split["val"], min(cfg.get("val_pairs", 1500), len(by_split["val"])))
    by_mod = module_index(train)
    mods = list(by_mod)
    print(f"train gloss pairs: {len(train)} over {len(mods)} modules; val probe: {len(val)}", flush=True)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": float(cfg.get("lr_encoder", 5e-5))},
        {"params": [*model.proj_g.parameters(), *model.proj_s.parameters(), model.logit_scale],
         "lr": float(cfg.get("lr_heads", 5e-4))},
    ], weight_decay=0.01, betas=(0.9, 0.98))

    rec = ledger.start_run(ledger.RunRecord(
        run_id=run_id, kind="train", stage="S3", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=s2_rec.inputs.corpus_cids,
                                tokenizer_cid=s2_rec.inputs.tokenizer_cid,
                                config_cid=config_cid,
                                init_checkpoint_cid=s2_rec.checkpoint_cids[-1],
                                code_git=ledger.git_head(), mathlib_pin=s2_rec.inputs.mathlib_pin),
        seed=cfg.get("seed", 1337),
        hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                  "torch": torch.__version__}))

    t0 = time.monotonic()
    rng = np.random.default_rng(cfg.get("seed", 1337))
    bs, steps, ckpt_every = cfg.get("batch_size", 128), cfg.get("steps", 8000), cfg.get("ckpt_every", 1000)
    ckpt_dir = REPO_ROOT / "runs" / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rolling = ckpt_dir / f"{run_id}-latest.safetensors"
    best_path = ckpt_dir / f"{run_id}-best.safetensors"
    val_every, per_module = cfg.get("val_every", 1000), cfg.get("per_module", 4)
    best_auc, best_step = -1.0, -1
    losses, accs = [], []
    model.train()
    for step in range(steps):
        if step > 0 and step % ckpt_every == 0:
            save_ckpt(model, rolling)
            print(f"step {step}: rolling checkpoint", flush=True)
        if step > 0 and step % val_every == 0:
            auc = val_hard_auc(model, tok, val, cfg["max_seq"], device, prng)
            improved = auc > best_auc
            if improved:
                best_auc, best_step = auc, step
                save_ckpt(model, best_path)
            print(f"step {step}: val hard-AUC {auc:.4f}{' (best, saved)' if improved else ''}", flush=True)
        idx = sample_hard_batch(rng, by_mod, mods, bs, per_module)
        g_ids = tokenize(tok, [train[i][0] for i in idx], cfg["max_seq"], pad).to(device)
        s_ids = tokenize(tok, [train[i][1] for i in idx], cfg["max_seq"], pad).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            g = model.encode(g_ids, "g")
            s = model.encode(s_ids, "s")
            loss = info_nce(g.float(), s.float(), model.logit_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, float(np.log(100)))
        losses.append(loss.item())
        if step % 100 == 0:
            accs.append(in_batch_accuracy(g.float(), s.float()))
            print(f"step {step}: loss {np.mean(losses[-100:]):.4f} in-batch@1 {accs[-1]:.4f} "
                  f"scale {model.logit_scale.exp().item():.1f}", flush=True)

    # Early stopping: the sealed checkpoint is the one with the best val
    # hard-AUC, not the last step (v1 overfit: train loss 0.063, held-out R@1 28%).
    auc = val_hard_auc(model, tok, val, cfg["max_seq"], device, prng)
    if auc > best_auc:
        best_auc, best_step = auc, steps
        save_ckpt(model, best_path)
    ckpt = ckpt_dir / f"{run_id}.safetensors"
    best_path.replace(ckpt)
    cid = ledger.ket_put(ckpt.read_bytes())
    rec.checkpoint_cids.append(cid)
    rolling.unlink(missing_ok=True)
    gi = REPO_ROOT / ".gitignore"
    line = f".ket/cas/{cid}"
    if line not in gi.read_text().splitlines():
        with open(gi, "a") as f:
            f.write(f"{line}\n")

    ledger.finish_run(rec, metrics={
        "final_loss": float(np.mean(losses[-100:])),
        "in_batch_acc": float(accs[-1]) if accs else 0.0,
        "steps": steps, "train_pairs": len(train),
        "logit_scale": model.logit_scale.exp().item(),
        "val_hard_auc_best": round(best_auc, 4), "best_step": best_step,
        "hard_negatives": f"module-grouped batches, {per_module} per module",
    }, started_monotonic=t0)
    print(f"run {run_id} sealed; metrics_cid {rec.metrics_cid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
