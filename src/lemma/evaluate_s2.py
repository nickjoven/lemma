"""S2 success-criterion eval: same-lock retrieval MRR, with controls.

The plan's falsifiable gate for the S2 encoder: mean-pooled embeddings must
retrieve exact-duplicate statements (declarations sharing a LOCK — the same
theorem stated under two names) at MRR >= 0.95. Controls, per the doctrine
that a metric without its controls is not citable:
  positive  same-lock pairs -> should retrieve each other at rank ~1
  negative  random cross-module pairs -> should NOT be nearest neighbours
  baseline  masked-token accuracy vs. a unigram-majority predictor

Every number is written to the ledger (metrics_cid) or it does not exist.

    uv run python -m lemma.evaluate_s2 --run s2-mlm-20260909-141736
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
from safetensors.torch import load_file
from tokenizers import Tokenizer

from . import ledger
from .data.corpus import encoder_text, iter_rows
from .models.encoder import Encoder, EncoderConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
MRR_GATE = 0.95


def load_encoder(ckpt: Path, vocab_size: int, device: str) -> Encoder:
    cfg = EncoderConfig(vocab_size=vocab_size)
    enc = Encoder(cfg)
    sd = load_file(str(ckpt))
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    return enc.to(device).eval()


@torch.no_grad()
def embed(enc: Encoder, tok: Tokenizer, texts: list[str], device: str, bs: int = 256) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), bs):
        batch = [tok.encode(t).ids[:enc.cfg.max_seq] for t in texts[i:i + bs]]
        width = max(len(b) for b in batch)
        ids = torch.full((len(batch), width), enc.cfg.pad_id, dtype=torch.long)
        for j, b in enumerate(batch):
            ids[j, :len(b)] = torch.tensor(b)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            z = enc.pool(ids.to(device)).float()
        out.append(torch.nn.functional.normalize(z, dim=-1).cpu())
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="sealed S2 train run_id")
    ap.add_argument("--groups", type=int, default=2000, help="same-lock groups to sample")
    ap.add_argument("--distractors", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    train_rec = ledger.citable(args.run)
    ckpt = REPO_ROOT / "runs" / "ckpts" / f"{args.run}.safetensors"
    tok = Tokenizer.from_file(str(REPO_ROOT / "runs" / "tokenizer" / "lean-unigram-32k.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_encoder(ckpt, tok.get_vocab_size(), device)

    rows = list(iter_rows("declarations", verify=False))
    by_lock: dict[str, list] = defaultdict(list)
    for r in rows:
        by_lock[r.lock].append(r)
    groups = [g for g in by_lock.values() if len(g) >= 2]
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    groups = groups[:args.groups]
    members = [r for g in groups for r in g]
    member_names = {r.name for r in members}
    distractors = rng.sample([r for r in rows if r.name not in member_names], args.distractors)
    pool = members + distractors
    texts = [encoder_text(r) for r in pool]
    locks = [r.lock for r in pool]
    modules = [r.module for r in pool]

    t0 = time.monotonic()
    Z = embed(enc, tok, texts, device)
    sims = Z @ Z.T
    sims.fill_diagonal_(-1.0)

    # positive control: rank of the nearest OTHER same-lock statement
    rr = []
    for i in range(len(members)):
        order = torch.argsort(sims[i], descending=True)
        for rank, j in enumerate(order.tolist(), start=1):
            if locks[j] == locks[i]:
                rr.append(1.0 / rank)
                break
    mrr = float(np.mean(rr))
    hit1 = float(np.mean([r == 1.0 for r in rr]))

    # negative control: a random cross-module pair should not be top-1 for each other
    neg_top1 = 0
    n_neg = 2000
    for _ in range(n_neg):
        i, j = rng.sample(range(len(pool)), 2)
        if modules[i] == modules[j] or locks[i] == locks[j]:
            continue
        if torch.argmax(sims[i]).item() == j:
            neg_top1 += 1
    neg_rate = neg_top1 / n_neg

    # baseline: unigram-majority masked-token accuracy on the train cache
    cache_ids = np.load(REPO_ROOT / "corpora" / "cache" / "declarations-train.ids.npy", mmap_mode="r")
    counts = np.bincount(np.asarray(cache_ids[:5_000_000]), minlength=tok.get_vocab_size())
    counts[:4] = 0  # specials
    unigram_acc = float(counts.max() / counts.sum())
    model_acc = json.loads(ledger.ket_get(train_rec.metrics_cid))["metrics"]["masked_acc"]

    metrics = {
        "same_lock_mrr": round(mrr, 4), "same_lock_hit@1": round(hit1, 4),
        "queries": len(members), "pool": len(pool), "groups": len(groups),
        "neg_cross_module_top1_rate": round(neg_rate, 4),
        "masked_acc_model": round(model_acc, 4), "masked_acc_unigram_baseline": round(unigram_acc, 4),
        "masked_acc_margin_pts": round(100 * (model_acc - unigram_acc), 1),
        "mrr_gate": MRR_GATE, "pass_mrr": mrr >= MRR_GATE,
        "pass_margin": (model_acc - unigram_acc) >= 0.25,
    }
    calib = {"pass": metrics["pass_mrr"] and neg_rate < 0.05 and metrics["pass_margin"],
             "controls": [
                 {"id": "same-lock-retrieval", "expect": f"MRR>={MRR_GATE}", "observed": mrr, "ok": metrics["pass_mrr"]},
                 {"id": "cross-module-negative", "expect": "top1<0.05", "observed": neg_rate, "ok": neg_rate < 0.05},
                 {"id": "beats-unigram-by-25pts", "expect": ">=25", "observed": metrics["masked_acc_margin_pts"], "ok": metrics["pass_margin"]},
             ]}

    rec = ledger.RunRecord(
        run_id=f"s2-eval-{args.run}", kind="eval", stage="S2", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=train_rec.inputs.corpus_cids,
                                tokenizer_cid=train_rec.inputs.tokenizer_cid,
                                config_cid=train_rec.inputs.config_cid,
                                init_checkpoint_cid=train_rec.checkpoint_cids[-1],
                                code_git=ledger.git_head(), mathlib_pin=train_rec.inputs.mathlib_pin),
        seed=args.seed, hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    ledger.start_run(rec)
    rec.calibration_record_cid = ledger.ket_put_json(calib)
    ledger.finish_run(rec, metrics, started_monotonic=t0,
                      status="completed" if calib["pass"] else "calibration_failed")
    print(json.dumps({**metrics, "calibration_pass": calib["pass"],
                      "metrics_cid": rec.metrics_cid}, indent=1))
    return 0 if calib["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
