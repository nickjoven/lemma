"""S5 equivalence-discovery eval on held-out modules, with controls.

  known-iff   lhs -> rhs retrieval among all test rhs PLUS a distractor pool of
              test-split statements: MRR (the S5 GATE, >= 0.5), R@1, R@10
  dedup       same-lock retrieval (should stay ~S2's 0.99: a regression check)
  permutation sanity: shuffled labels -> AUC ~0.5
  proposals   the "real test" input: highest-similarity pairs of DISTINCT test
              statements that are NOT known same-lock and NOT a known Iff —
              candidate equivalences written to runs/s5_proposals-<run>.jsonl
              (content-addressed). Discharging them is quod's job: each is a
              demonstrandum `A <-> B` for the gates; the plan's bar is >= 10 of
              the top 100 discharged. That submission is a separate quod-side
              stage and is recorded here as pending, never assumed.

    uv run python -m lemma.evaluate_s5 --run <s5 run_id>
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from tokenizers import Tokenizer

from . import ledger
from .data import splits
from .data.corpus import encoder_text, iter_rows
from .models.encoder import Encoder, EncoderConfig
from .train_s3 import tokenize
from .train_s5 import EquivModel, equivalence_pairs_by_split

REPO_ROOT = Path(__file__).resolve().parents[2]
MRR_GATE = 0.5


@torch.no_grad()
def encode_all(model: EquivModel, tok: Tokenizer, texts: list[str], max_seq: int, device: str) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), 256):
        ids = tokenize(tok, texts[i:i + 256], max_seq, model.enc.cfg.pad_id).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model.encode(ids).float().cpu())
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--distractors", type=int, default=10000)
    ap.add_argument("--proposals", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    max_seq = 256

    train_rec = ledger.citable(args.run)
    tok = Tokenizer.from_file(str(REPO_ROOT / "runs" / "tokenizer" / "lean-unigram-32k.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EquivModel(Encoder(EncoderConfig(vocab_size=tok.get_vocab_size(), max_seq=max_seq)), 256)
    model.load_state_dict(load_file(str(REPO_ROOT / "runs" / "ckpts" / f"{args.run}.safetensors")))
    model = model.to(device).eval()

    test = equivalence_pairs_by_split()["test"]
    iff = [p for p in test if p[3] == "iff"]
    dedup = [p for p in test if p[3] == "dedup"]
    rows = list(iter_rows("declarations", verify=False))
    assign = splits.assign(rows)
    test_rows = [r for r in rows if assign[r.name] == "test"]
    distractor_rows = rng.sample(test_rows, min(args.distractors, len(test_rows)))
    t0 = time.monotonic()

    # known-Iff retrieval with distractors
    L = encode_all(model, tok, [p[0] for p in iff], max_seq, device)
    R = encode_all(model, tok, [p[1] for p in iff], max_seq, device)
    D = encode_all(model, tok, [encoder_text(r) for r in distractor_rows], max_seq, device)
    pool = torch.cat([R, D])
    sims = L @ pool.T
    ranks = (sims > sims[torch.arange(len(iff)), torch.arange(len(iff))][:, None]).sum(1) + 1
    ranks = ranks.numpy()
    mrr, r1, r10 = float(np.mean(1 / ranks)), float(np.mean(ranks == 1)), float(np.mean(ranks <= 10))

    # dedup regression check
    dd_mrr = float("nan")
    if len(dedup) >= 20:
        A = encode_all(model, tok, [p[0] for p in dedup], max_seq, device)
        B = encode_all(model, tok, [p[1] for p in dedup], max_seq, device)
        s2 = A @ torch.cat([B, D]).T
        rk = (s2 > s2[torch.arange(len(dedup)), torch.arange(len(dedup))][:, None]).sum(1) + 1
        dd_mrr = float(np.mean(1 / rk.numpy()))

    # permutation sanity on matched-vs-random scores
    pos = sims[torch.arange(len(iff)), torch.arange(len(iff))].numpy()
    neg = np.array([sims[i, rng.randrange(len(iff), pool.size(0))].item() for i in range(len(iff))])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    auc_real = float(roc_auc_score(y, np.r_[pos, neg]))
    yp = y.copy()
    rng.shuffle(yp)
    auc_perm = float(roc_auc_score(yp, np.r_[pos, neg]))

    # novel proposals: top cross-declaration similarities among distractor statements,
    # excluding known same-lock pairs (the only "known equivalence" available here)
    DD = D @ D.T
    DD.fill_diagonal_(-1.0)
    locks = [r.lock for r in distractor_rows]
    flat = torch.argsort(DD.flatten(), descending=True)
    proposals, seen = [], set()
    for f in flat.tolist():
        i, j = divmod(f, DD.size(1))
        if i >= j or locks[i] == locks[j]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        a, b = distractor_rows[i], distractor_rows[j]
        proposals.append({"score": round(DD[i, j].item(), 4), "a": a.name, "b": b.name,
                          "a_stmt": encoder_text(a), "b_stmt": encoder_text(b),
                          "a_module": a.module, "b_module": b.module})
        if len(proposals) >= args.proposals:
            break
    prop_path = REPO_ROOT / "runs" / f"s5_proposals-{args.run}.jsonl"
    prop_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in proposals))
    prop_cid = ledger.ket_put(prop_path.read_bytes())

    metrics = {
        "test_iff": len(iff), "distractors": len(distractor_rows),
        "iff_mrr": round(mrr, 4), "iff_recall@1": round(r1, 4), "iff_recall@10": round(r10, 4),
        "dedup_mrr": round(dd_mrr, 4) if dd_mrr == dd_mrr else None,
        "matched_vs_random_auc": round(auc_real, 4), "permutation_auc": round(auc_perm, 4),
        "mrr_gate": MRR_GATE, "pass_mrr": mrr >= MRR_GATE,
        "proposals": len(proposals), "proposals_cid": prop_cid,
        "gate_submission": "pending: submit proposals as A<->B demonstranda to quod's gates (>=10/100 discharged is the plan's bar)",
    }
    calib = {"pass": metrics["pass_mrr"] and 0.45 <= auc_perm <= 0.55,
             "controls": [
                 {"id": "known-iff-retrieval", "expect": f"MRR>={MRR_GATE}", "observed": mrr, "ok": metrics["pass_mrr"]},
                 {"id": "permutation-sanity", "expect": "0.45..0.55", "observed": auc_perm, "ok": 0.45 <= auc_perm <= 0.55},
                 {"id": "dedup-regression", "expect": "reported (S2 was 0.99)", "observed": dd_mrr, "ok": True},
             ]}
    rec = ledger.RunRecord(
        run_id=f"s5-eval-{args.run}", kind="eval", stage="S5", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=train_rec.inputs.corpus_cids, tokenizer_cid=train_rec.inputs.tokenizer_cid,
                                config_cid=train_rec.inputs.config_cid, init_checkpoint_cid=train_rec.checkpoint_cids[-1],
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
