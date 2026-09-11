"""S4 v1: mutant verdict predictor on Phase-3a labels.

The full gate verdict for statement mutants needs Phase-3b proof attempts;
what Phase-3a gives per mutant is `elaborates` (is the mutated statement
well-typed — a real, gate-adjacent judgement that requires reading types) and
the operator class. S4 v1 is a two-head classifier over the S2 encoder:
  elaborates : {False, True}                    (class-balanced CE)
  op_class   : {hyp_del, swap_*, proof_side}    (CE)
Mutants inherit their parent theorem's module-level split (the single most
important leakage rule for S4). Early stopping on val macro-F1(elaborates).
An operator class can be held out entirely (ood_operator) so the eval can
report generalization to an unseen mutation type.

    uv run python -u -m lemma.train_s4 --config configs/s4_verdict.yaml
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
from safetensors.torch import save_file
from sklearn.metrics import f1_score
from tokenizers import Tokenizer

from . import ledger
from .data import splits
from .data.corpus import iter_mutants, iter_rows
from .models.encoder import Encoder
from .models.heads import ClassifierHead
from .train_s3 import load_s2_encoder, tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]
# Statement-mutation operators only. The proof-side records (sorry/axiom
# injection) leave the statement UNCHANGED, so their elaborates label is
# trivially True and carries no mutation signal: including them made 95% of
# the elaborates task "predict True" (smoke val F1 0.4994 = the degenerate
# one-class answer). S4 v1 trains on the 8,187 statement mutants (75/25
# typed/ill-typed) and identifies among the six statement operators.
OP_CLASSES = ["hyp_del", "swap_and_or", "swap_or_and", "swap_eq_ne", "swap_ne_eq", "swap_lit_01"]


class VerdictModel(nn.Module):
    def __init__(self, enc: Encoder):
        super().__init__()
        self.enc = enc
        self.head_elab = ClassifierHead(enc.cfg, 2)
        self.head_op = ClassifierHead(enc.cfg, len(OP_CLASSES))

    def forward(self, ids):
        z = self.enc.pool(ids)
        return self.head_elab(z), self.head_op(z)


def mutants_by_split(ood_operator: str | None):
    """split -> [(text, elab, op_idx, op_class)]; mutants inherit the parent's split.
    The ood_operator class is removed from train/val and kept only in test."""
    rows = list(iter_rows("declarations", verify=False))
    assign = splits.assign(rows)
    out = {s: [] for s in splits.SPLITS}
    for m in iter_mutants():
        s = assign.get(m.parent_name)
        if s is None or m.elaborates is None:
            continue
        oc = m.op_class
        if oc == "proof_side":
            continue
        if ood_operator and oc == ood_operator and s != "test":
            continue
        out[s].append((m.mutated_type, int(m.elaborates), OP_CLASSES.index(oc), oc))
    return out


@torch.no_grad()
def predict(model, tok, items, max_seq, device, bs=256):
    model.eval()
    pe, po = [], []
    for i in range(0, len(items), bs):
        ids = tokenize(tok, [t[0] for t in items[i:i + bs]], max_seq, model.enc.cfg.pad_id).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            le, lo = model(ids)
        pe.append(le.argmax(1).cpu()); po.append(lo.argmax(1).cpu())
    model.train()
    return torch.cat(pe).numpy(), torch.cat(po).numpy()


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    config_cid = ledger.ket_put(args.config.read_bytes())
    run_id = args.run_id or f"s4-verdict-{time.strftime('%Y%m%d-%H%M%S')}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 1337))

    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    s2_rec = ledger.citable(cfg["s2_run"])
    enc = load_s2_encoder(REPO_ROOT / "runs" / "ckpts" / f"{cfg['s2_run']}.safetensors",
                          tok.get_vocab_size(), cfg["max_seq"])
    model = VerdictModel(enc).to(device)
    pad = enc.cfg.pad_id

    data = mutants_by_split(cfg.get("ood_operator"))
    train, val = data["train"], data["val"]
    n_pos = sum(t[1] for t in train)
    w = torch.tensor([len(train) / max(len(train) - n_pos, 1), len(train) / max(n_pos, 1)], device=device)
    w = w / w.sum() * 2
    print(f"train mutants: {len(train)} (elaborates={n_pos}) val: {len(val)} test: {len(data['test'])} "
          f"| class weights {w.tolist()} | ood_operator={cfg.get('ood_operator')}", flush=True)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": float(cfg.get("lr_encoder", 3e-5))},
        {"params": [*model.head_elab.parameters(), *model.head_op.parameters()], "lr": float(cfg.get("lr_heads", 5e-4))},
    ], weight_decay=0.01)

    rec = ledger.start_run(ledger.RunRecord(
        run_id=run_id, kind="train", stage="S4", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=[*s2_rec.inputs.corpus_cids, *cfg.get("mutant_corpus_cids", [])],
                                tokenizer_cid=s2_rec.inputs.tokenizer_cid, config_cid=config_cid,
                                init_checkpoint_cid=s2_rec.checkpoint_cids[-1],
                                code_git=ledger.git_head(), mathlib_pin=s2_rec.inputs.mathlib_pin),
        seed=cfg.get("seed", 1337),
        hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__}))

    t0 = time.monotonic()
    rng = np.random.default_rng(cfg.get("seed", 1337))
    bs, steps, val_every = cfg.get("batch_size", 64), cfg.get("steps", 3000), cfg.get("val_every", 500)
    ckpt_dir = REPO_ROOT / "runs" / "ckpts"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_id}-best.safetensors"
    best_f1, best_step, losses = -1.0, -1, []
    ce_e, ce_o = nn.CrossEntropyLoss(weight=w), nn.CrossEntropyLoss()
    model.train()
    for step in range(steps):
        if step > 0 and step % val_every == 0:
            pe, _ = predict(model, tok, val, cfg["max_seq"], device)
            f1 = macro_f1([t[1] for t in val], pe)
            if f1 > best_f1:
                best_f1, best_step = f1, step
                save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(best_path))
            print(f"step {step}: val macro-F1(elaborates) {f1:.4f}{' (best, saved)' if f1 == best_f1 and best_step == step else ''}", flush=True)
        idx = rng.integers(0, len(train), size=bs)
        batch = [train[i] for i in idx]
        ids = tokenize(tok, [b[0] for b in batch], cfg["max_seq"], pad).to(device)
        ye = torch.tensor([b[1] for b in batch], device=device)
        yo = torch.tensor([b[2] for b in batch], device=device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            le, lo = model(ids)
            loss = ce_e(le.float(), ye) + 0.5 * ce_o(lo.float(), yo)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"step {step}: loss {np.mean(losses[-100:]):.4f}", flush=True)

    pe, _ = predict(model, tok, val, cfg["max_seq"], device)
    f1 = macro_f1([t[1] for t in val], pe)
    if f1 > best_f1:
        best_f1, best_step = f1, steps
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(best_path))
    ckpt = ckpt_dir / f"{run_id}.safetensors"; best_path.replace(ckpt)
    cid = ledger.ket_put(ckpt.read_bytes()); rec.checkpoint_cids.append(cid)
    gi = REPO_ROOT / ".gitignore"; line = f".ket/cas/{cid}"
    if line not in gi.read_text().splitlines():
        with open(gi, "a") as f: f.write(f"{line}\n")
    ledger.finish_run(rec, metrics={"final_loss": float(np.mean(losses[-100:])), "steps": steps,
                                    "train_mutants": len(train), "val_macro_f1_elab_best": round(best_f1, 4),
                                    "best_step": best_step, "ood_operator": cfg.get("ood_operator")},
                      started_monotonic=t0)
    print(f"run {run_id} sealed; metrics_cid {rec.metrics_cid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
