"""S3 gloss-fidelity eval on held-out (test-split) modules, with controls.

  retrieval   gloss -> its statement among all test statements (MRR, R@1, R@10)
  drift-easy  ROC-AUC: matched pair vs docstring x random other statement
  drift-hard  ROC-AUC: matched pair vs docstring x random statement from the
              SAME module — the strongest mutant-free proxy for "the prose no
              longer describes this statement". This is the S3 GATE (>= 0.90)
              until the Phase-3 mutant corpus enables the plan's exact test
              (docstring x gate-verified mutant), which is recorded as deferred.
  top-decile  fraction of matched pairs scoring in the top 10% of all scores
  permutation sanity: shuffle labels -> AUC must collapse to ~0.5

Every number is ledgered (metrics_cid) or it does not exist.

    uv run python -m lemma.evaluate_s3 --run <s3 run_id>
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
from sklearn.metrics import roc_auc_score
from tokenizers import Tokenizer

from . import ledger
from .models.encoder import Encoder, EncoderConfig
from .train_s3 import GlossModel, gloss_pairs, tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]
HARD_AUC_GATE = 0.90


@torch.no_grad()
def encode_all(model: GlossModel, tok: Tokenizer, texts: list[str], which: str,
               max_seq: int, device: str, bs: int = 256) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), bs):
        ids = tokenize(tok, texts[i:i + bs], max_seq, model.enc.cfg.pad_id).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model.encode(ids, which).float().cpu())
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train_rec = ledger.citable(args.run)
    cfg_max_seq = 256
    tok = Tokenizer.from_file(str(REPO_ROOT / "runs" / "tokenizer" / "lean-unigram-32k.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(EncoderConfig(vocab_size=tok.get_vocab_size(), max_seq=cfg_max_seq))
    model = GlossModel(enc, 256)
    model.load_state_dict(load_file(str(REPO_ROOT / "runs" / "ckpts" / f"{args.run}.safetensors")))
    model = model.to(device).eval()

    test = gloss_pairs("test")
    glosses = [t[0] for t in test]
    stmts = [t[1] for t in test]
    mods = [t[2] for t in test]
    t0 = time.monotonic()
    G = encode_all(model, tok, glosses, "g", cfg_max_seq, device)
    S = encode_all(model, tok, stmts, "s", cfg_max_seq, device)
    sims = G @ S.T  # (n_gloss, n_stmt); diagonal = matched

    # retrieval
    n = len(test)
    ranks = []
    for i in range(n):
        order = torch.argsort(sims[i], descending=True)
        ranks.append(int((order == i).nonzero()[0].item()) + 1)
    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    r1, r10 = float(np.mean(ranks == 1)), float(np.mean(ranks <= 10))

    # drift: matched score vs mismatched score
    matched = sims.diagonal().numpy()
    by_mod: dict[str, list[int]] = defaultdict(list)
    for j, m in enumerate(mods):
        by_mod[m].append(j)
    easy_neg, hard_neg = [], []
    for i in range(n):
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)
        easy_neg.append(sims[i, j].item())
        same = [k for k in by_mod[mods[i]] if k != i]
        if same:
            hard_neg.append(sims[i, rng.choice(same)].item())
    y_easy = np.r_[np.ones(n), np.zeros(n)]
    auc_easy = float(roc_auc_score(y_easy, np.r_[matched, easy_neg]))
    hard_pos = matched[:len(hard_neg)] if hard_neg else matched
    y_hard = np.r_[np.ones(len(hard_neg)), np.zeros(len(hard_neg))]
    auc_hard = float(roc_auc_score(y_hard, np.r_[matched[[i for i in range(n)
                                                             if len(by_mod[mods[i]]) > 1]][:len(hard_neg)],
                                                  hard_neg])) if hard_neg else float("nan")

    # top-decile: matched pairs among the top 10% of all (matched ∪ easy-neg) scores
    allscores = np.r_[matched, easy_neg]
    thresh = np.quantile(allscores, 0.90)
    top_decile = float(np.mean(matched >= thresh))

    # permutation sanity: destroy labels -> AUC ~ 0.5
    perm = np.r_[matched, easy_neg].copy()
    yp = y_easy.copy()
    rng.shuffle(yp)
    auc_perm = float(roc_auc_score(yp, perm))

    metrics = {
        "test_pairs": n, "retrieval_mrr": round(mrr, 4), "recall@1": round(r1, 4), "recall@10": round(r10, 4),
        "drift_auc_easy": round(auc_easy, 4), "drift_auc_hard_same_module": round(auc_hard, 4),
        "hard_negatives": len(hard_neg), "matched_top_decile": round(top_decile, 4),
        "permutation_auc": round(auc_perm, 4),
        "hard_auc_gate": HARD_AUC_GATE, "pass_hard_auc": auc_hard >= HARD_AUC_GATE,
        "deferred": "docstring x gate-verified mutant AUC (needs Phase-3 mutant corpus)",
    }
    calib = {"pass": metrics["pass_hard_auc"] and 0.45 <= auc_perm <= 0.55,
             "controls": [
                 {"id": "drift-hard-same-module", "expect": f"AUC>={HARD_AUC_GATE}", "observed": auc_hard, "ok": metrics["pass_hard_auc"]},
                 {"id": "permutation-sanity", "expect": "0.45..0.55", "observed": auc_perm, "ok": 0.45 <= auc_perm <= 0.55},
                 {"id": "matched-top-decile", "expect": "reported", "observed": top_decile, "ok": True},
             ]}
    rec = ledger.RunRecord(
        run_id=f"s3-eval-{args.run}", kind="eval", stage="S3", started=ledger.now_iso(),
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
    print(json.dumps({**metrics, "calibration_pass": calib["pass"], "metrics_cid": rec.metrics_cid}, indent=1))
    return 0 if calib["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
