"""S4 v2: mutant verdict predictor on Phase-3b PROVER-RELATIVE labels.

The label for a statement mutant is the three-way `gate_verdict` of a quod
attempt (docs/prover-loop.md §2): proven | no_proof_found | rejected — for
the named prover, within its recorded budget; never "true"/"false". Rows come
from `labels.join_mutant_attempts`, which only admits an attempt whose rebuilt
mutant has the corpus mutant's lock. `labels.collapse_rule` decides whether
`rejected` has enough support to be its own class or folds into proven-vs-not.

Mutants inherit their parent theorem's module-level split (the S4 leakage
rule); an operator class can be held out entirely (ood_operator) so the eval
reports generalization to an unseen mutation type; attempts can be restricted
to one prover so the label has a single, stated meaning.

    uv run python -u -m lemma.train_s4v2 --config configs/s4v2_verdict.yaml
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file
from tokenizers import Tokenizer

from . import ledger
from .data import splits
from .data.corpus import iter_attempts, iter_mutants, iter_rows
from .data.labels import collapse_rule, join_mutant_attempts
from .models.encoder import Encoder
from .models.heads import ClassifierHead
from .train_s3 import load_s2_encoder, tokenize
from .train_s4 import macro_f1

REPO_ROOT = Path(__file__).resolve().parents[2]


class VerdictModelV2(nn.Module):
    def __init__(self, enc: Encoder, n_classes: int):
        super().__init__()
        self.enc = enc
        self.head = ClassifierHead(enc.cfg, n_classes)

    def forward(self, ids):
        return self.head(self.enc.pool(ids))


def labelled_by_split(rows: list[dict], assign: dict[str, str], classes: list[str],
                      ood_operator: str | None) -> dict[str, list[tuple]]:
    """split -> [(text, class_idx, op_class, label, prover)]. Mutants inherit the
    parent's split; the ood_operator class is removed from train/val, kept in test."""
    out = {s: [] for s in splits.SPLITS}
    for r in rows:
        s = assign.get(r["parent"])
        if s is None:
            continue
        if ood_operator and r["op_class"] == ood_operator and s != "test":
            continue
        out[s].append((r["text"], classes.index(r["label"]), r["op_class"], r["label"], r["prover"]))
    return out


def load_labelled(cfg: dict) -> tuple[dict[str, list[tuple]], list[str], dict]:
    """-> (by_split, classes, audit). Every filter applied here is recorded in the audit."""
    prover = cfg.get("prover")
    attempts = [a for a in iter_attempts(cfg.get("attempts_corpus", "attempts"))
                if not prover or a.prover == prover]
    rows, audit = join_mutant_attempts(iter_mutants(), attempts)
    classes, support = collapse_rule(rows, min_support=int(cfg.get("min_support", 30)))
    decl_rows = list(iter_rows("declarations", verify=False))
    data = labelled_by_split(rows, splits.assign(decl_rows), classes, cfg.get("ood_operator"))
    audit.update({"attempts_considered": len(attempts), "prover_filter": prover, "classes": classes,
                  "support3": support, "per_split": {s: len(v) for s, v in data.items()}})
    return data, classes, audit


@torch.no_grad()
def predict(model, tok, items, max_seq, device, bs=256):
    model.eval()
    preds = []
    for i in range(0, len(items), bs):
        ids = tokenize(tok, [t[0] for t in items[i:i + bs]], max_seq, model.enc.cfg.pad_id).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            preds.append(model(ids).argmax(1).cpu())
    model.train()
    return torch.cat(preds).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    config_cid = ledger.ket_put(args.config.read_bytes())
    run_id = args.run_id or f"s4v2-verdict-{time.strftime('%Y%m%d-%H%M%S')}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 1337))

    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    s2_rec = ledger.citable(cfg["s2_run"])
    enc = load_s2_encoder(REPO_ROOT / "runs" / "ckpts" / f"{cfg['s2_run']}.safetensors",
                          tok.get_vocab_size(), cfg["max_seq"])

    data, classes, audit = load_labelled(cfg)
    train, val = data["train"], data["val"]
    if not train or not val:
        raise SystemExit(f"S4 v2: empty split ({audit['per_split']}); nothing to train on")
    counts = Counter(t[1] for t in train)
    w = torch.tensor([len(train) / max(counts.get(c, 0), 1) for c in range(len(classes))], device=device)
    w = w / w.sum() * len(classes)
    print(f"classes {classes} | train {len(train)} {dict(counts)} | val {len(val)} | test {len(data['test'])} "
          f"| weights {[round(x, 3) for x in w.tolist()]} | audit {audit}", flush=True)
    model = VerdictModelV2(enc, len(classes)).to(device)
    pad = enc.cfg.pad_id
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": float(cfg.get("lr_encoder", 3e-5))},
        {"params": model.head.parameters(), "lr": float(cfg.get("lr_heads", 5e-4))},
    ], weight_decay=0.01)

    rec = ledger.start_run(ledger.RunRecord(
        run_id=run_id, kind="train", stage="S4", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=[*s2_rec.inputs.corpus_cids, *cfg.get("mutant_corpus_cids", []),
                                             *cfg.get("attempts_corpus_cids", [])],
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
    ce = nn.CrossEntropyLoss(weight=w)
    yv = [t[1] for t in val]

    def validate(step):
        nonlocal best_f1, best_step
        f1 = macro_f1(yv, predict(model, tok, val, cfg["max_seq"], device))
        if f1 > best_f1:
            best_f1, best_step = f1, step
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(best_path))
        print(f"step {step}: val macro-F1 {f1:.4f}{' (best, saved)' if best_step == step else ''}", flush=True)

    model.train()
    for step in range(steps):
        if step > 0 and step % val_every == 0:
            validate(step)
        idx = rng.integers(0, len(train), size=bs)
        batch = [train[i] for i in idx]
        ids = tokenize(tok, [b[0] for b in batch], cfg["max_seq"], pad).to(device)
        y = torch.tensor([b[1] for b in batch], device=device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = ce(model(ids).float(), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"step {step}: loss {np.mean(losses[-100:]):.4f}", flush=True)
    validate(steps)

    ckpt = ckpt_dir / f"{run_id}.safetensors"; best_path.replace(ckpt)
    cid = ledger.ket_put(ckpt.read_bytes()); rec.checkpoint_cids.append(cid)
    gi = REPO_ROOT / ".gitignore"; line = f".ket/cas/{cid}"
    if line not in gi.read_text().splitlines():
        with open(gi, "a") as f: f.write(f"{line}\n")
    ledger.finish_run(rec, metrics={"final_loss": float(np.mean(losses[-100:])), "steps": steps,
                                    "classes": classes, "train_rows": len(train), "train_class_counts": dict(counts),
                                    "val_macro_f1_best": round(best_f1, 4), "best_step": best_step,
                                    "ood_operator": cfg.get("ood_operator"), "label_audit": audit},
                      started_monotonic=t0)
    print(f"run {run_id} sealed; metrics_cid {rec.metrics_cid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
