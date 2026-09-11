"""S5: equivalence and reduction discovery.

One statement tower (warm-started from the sealed S2 encoder) with a single
projection head, trained with symmetric InfoNCE over statement<->statement
positives: the two sides of every Iff theorem, and same-lock dedup pairs
(the same theorem stated under two names). Module-grouped batches supply
same-module hard negatives. Early stopping on held-out known-Iff retrieval
MRR, the S5 gate metric.

    uv run python -u -m lemma.train_s5 --config configs/s5_equiv.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file
from tokenizers import Tokenizer

from . import ledger
from .data import splits
from .data.corpus import CORPORA, encoder_text, iter_rows
from .models.encoder import Encoder
from .models.heads import ProjectionHead
from .objectives.contrastive import in_batch_accuracy, info_nce
from .train_s3 import load_s2_encoder, module_index, sample_hard_batch, tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]


class EquivModel(nn.Module):
    def __init__(self, enc: Encoder, out_dim: int):
        super().__init__()
        self.enc = enc
        self.proj = ProjectionHead(enc.cfg, out_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.proj(self.enc.pool(ids)), dim=-1)


Pair = tuple[str, str, str, str]  # (a, b, module, kind: "iff" | "dedup")


def equivalence_pairs_by_split() -> dict[str, list[Pair]]:
    """Iff sides + same-lock dedup pairs, assigned to the module-level split.
    A dedup pair whose two members fall in different splits is dropped (leak)."""
    rows = list(iter_rows("declarations", verify=False))
    assign = splits.assign(rows)
    out: dict[str, list[Pair]] = {s: [] for s in splits.SPLITS}
    with open(CORPORA / "iff_pairs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out[splits.split_of_area(r["module"])].append((r["lhs"], r["rhs"], r["module"], "iff"))
    by_lock: dict[str, list] = defaultdict(list)
    for r in rows:
        by_lock[r.lock].append(r)
    for group in by_lock.values():
        if len(group) < 2:
            continue
        a = group[0]
        for b in group[1:]:
            if assign[a.name] == assign[b.name]:
                out[assign[a.name]].append((encoder_text(a), encoder_text(b), a.module, "dedup"))
    return out


@torch.no_grad()
def val_iff_mrr(model: EquivModel, tok: Tokenizer, val: list[Pair], max_seq: int, device: str) -> float:
    """lhs -> rhs retrieval MRR among all val rhs (known-Iff pairs only)."""
    iff = [p for p in val if p[3] == "iff"]
    if len(iff) < 20:
        return float("nan")
    model.eval()
    pad = model.enc.cfg.pad_id
    A, B = [], []
    for i in range(0, len(iff), 256):
        chunk = iff[i:i + 256]
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            A.append(model.encode(tokenize(tok, [c[0] for c in chunk], max_seq, pad).to(device)).float().cpu())
            B.append(model.encode(tokenize(tok, [c[1] for c in chunk], max_seq, pad).to(device)).float().cpu())
    A, B = torch.cat(A), torch.cat(B)
    sims = A @ B.T
    ranks = (sims > sims.diagonal()[:, None]).sum(1) + 1
    model.train()
    return float((1.0 / ranks.float()).mean())


def save_ckpt(model: EquivModel, path: Path) -> None:
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    config_cid = ledger.ket_put(args.config.read_bytes())
    run_id = args.run_id or f"s5-equiv-{time.strftime('%Y%m%d-%H%M%S')}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 1337))

    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    s2_rec = ledger.citable(cfg["s2_run"])
    enc = load_s2_encoder(REPO_ROOT / "runs" / "ckpts" / f"{cfg['s2_run']}.safetensors",
                          tok.get_vocab_size(), cfg["max_seq"])
    model = EquivModel(enc, cfg.get("proj_dim", 256)).to(device)
    pad = enc.cfg.pad_id

    by_split = equivalence_pairs_by_split()
    train = by_split["train"]
    prng = random.Random(cfg.get("seed", 1337))
    val = prng.sample(by_split["val"], min(cfg.get("val_pairs", 1500), len(by_split["val"])))
    by_mod = module_index([(a, b, m) for a, b, m, _ in train])
    mods = list(by_mod)
    n_iff = sum(1 for p in train if p[3] == "iff")
    print(f"train pairs: {len(train)} ({n_iff} iff, {len(train) - n_iff} dedup) over {len(mods)} modules; "
          f"val probe: {len(val)}", flush=True)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": float(cfg.get("lr_encoder", 5e-5))},
        {"params": [*model.proj.parameters(), model.logit_scale], "lr": float(cfg.get("lr_heads", 5e-4))},
    ], weight_decay=0.01, betas=(0.9, 0.98))

    rec = ledger.start_run(ledger.RunRecord(
        run_id=run_id, kind="train", stage="S5", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=s2_rec.inputs.corpus_cids, tokenizer_cid=s2_rec.inputs.tokenizer_cid,
                                config_cid=config_cid, init_checkpoint_cid=s2_rec.checkpoint_cids[-1],
                                code_git=ledger.git_head(), mathlib_pin=s2_rec.inputs.mathlib_pin),
        seed=cfg.get("seed", 1337),
        hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__}))

    t0 = time.monotonic()
    rng = np.random.default_rng(cfg.get("seed", 1337))
    bs, steps = cfg.get("batch_size", 128), cfg.get("steps", 6000)
    ckpt_every, val_every, per_module = cfg.get("ckpt_every", 1000), cfg.get("val_every", 1000), cfg.get("per_module", 4)
    ckpt_dir = REPO_ROOT / "runs" / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rolling, best_path = ckpt_dir / f"{run_id}-latest.safetensors", ckpt_dir / f"{run_id}-best.safetensors"
    best_mrr, best_step = -1.0, -1
    losses, accs = [], []
    model.train()
    for step in range(steps):
        if step > 0 and step % ckpt_every == 0:
            save_ckpt(model, rolling)
            print(f"step {step}: rolling checkpoint", flush=True)
        if step > 0 and step % val_every == 0:
            mrr = val_iff_mrr(model, tok, val, cfg["max_seq"], device)
            improved = mrr > best_mrr
            if improved:
                best_mrr, best_step = mrr, step
                save_ckpt(model, best_path)
            print(f"step {step}: val iff-MRR {mrr:.4f}{' (best, saved)' if improved else ''}", flush=True)
        idx = sample_hard_batch(rng, by_mod, mods, bs, per_module)
        a_ids = tokenize(tok, [train[i][0] for i in idx], cfg["max_seq"], pad).to(device)
        b_ids = tokenize(tok, [train[i][1] for i in idx], cfg["max_seq"], pad).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            a, b = model.encode(a_ids), model.encode(b_ids)
            loss = info_nce(a.float(), b.float(), model.logit_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, float(np.log(100)))
        losses.append(loss.item())
        if step % 100 == 0:
            accs.append(in_batch_accuracy(a.float(), b.float()))
            print(f"step {step}: loss {np.mean(losses[-100:]):.4f} in-batch@1 {accs[-1]:.4f} "
                  f"scale {model.logit_scale.exp().item():.1f}", flush=True)

    mrr = val_iff_mrr(model, tok, val, cfg["max_seq"], device)
    if mrr > best_mrr:
        best_mrr, best_step = mrr, steps
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
        "final_loss": float(np.mean(losses[-100:])), "in_batch_acc": float(accs[-1]) if accs else 0.0,
        "steps": steps, "train_pairs": len(train), "train_iff": n_iff,
        "val_iff_mrr_best": round(best_mrr, 4), "best_step": best_step,
        "hard_negatives": f"module-grouped batches, {per_module} per module",
    }, started_monotonic=t0)
    print(f"run {run_id} sealed; metrics_cid {rec.metrics_cid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
