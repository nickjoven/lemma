"""W0 gates that come BEFORE W0 trains (priority-0 list, items 5 and 6).

Translation test (item 5): for each tactic, over the transitions where it
made progress (one goal in, one goal out, different propositions), the
displacement embed(goal_after) - embed(goal_before) under the sealed S2
encoder, its mean and its spread. The per-tactic mean offset IS the
mandatory W0 baseline "state + learned per-tactic offset": next-state
retrieval MRR among the held-out candidates for
    identity      predict z_before
    offset        predict z_before + mu_tactic     (mu fitted on train split)
    lexical       TF-IDF cosine of the before-goal text against candidate texts
W0's gate is to beat offset AND lexical by >= 10 points on module-held-out
transitions. A tactic whose offset baseline is within `translation_margin`
of the best learned predictor is recorded as "translation" (no GPU spent).

Energy targets (item 6): per goal state, the empirical closure rate over the
ladder's rungs that were tried on it, energy = -log(rate) (+inf when no rung
closed it, recorded as null), the outcome class and the measured cost per
transition (budget rows CENSORED: their cost is a lower bound, never a
value). Both tables are sealed with a metrics_cid; prose agreement (quod Q-5)
is a human-read descriptor and is not a target here.

    uv run python -m lemma.w0_baselines --s2-run <s2 run_id> [--max-candidates 2000]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer

from . import ledger
from .data import splits
from .data.corpus import iter_goals, iter_rows, iter_transitions
from .train_s3 import load_s2_encoder, tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- pure parts

def single_step_rows(rows) -> list:
    """Progressed transitions with exactly one goal in and one goal out that
    differ — the rows a displacement is defined on."""
    return [t for t in rows if t.outcome == "open" and isinstance(t.goal_after, list)
            and len(t.goal_after) == 1 and t.goal_after[0] != t.goal_before]


def energy_targets(rows) -> dict[str, dict]:
    """goal_before lock -> {rungs, closed, closure_rate, energy, censored,
    cost_mean (uncensored heartbeats), outcomes}. Only rung steps count as
    trials (the intros step is not a proof attempt)."""
    by = defaultdict(lambda: {"rungs": 0, "closed": 0, "censored": 0, "costs": [], "outcomes": Counter()})
    for t in rows:
        if t.kind != "rung":
            continue
        s = by[t.goal_before]
        s["rungs"] += 1
        s["closed"] += t.outcome == "closed"
        s["censored"] += t.censored
        s["outcomes"][t.outcome] += 1
        if not t.censored:
            s["costs"].append(t.heartbeats)
    out = {}
    for lock, s in by.items():
        rate = s["closed"] / s["rungs"]
        out[lock] = {"rungs": s["rungs"], "closed": s["closed"], "closure_rate": round(rate, 4),
                     "energy": round(-math.log(rate), 4) if rate > 0 else None,
                     "censored": s["censored"], "cost_mean": round(float(np.mean(s["costs"])), 1) if s["costs"] else None,
                     "outcomes": dict(s["outcomes"])}
    return out


def mrr(pred: np.ndarray, cands: np.ndarray, target_idx: np.ndarray) -> float:
    """Mean reciprocal rank of the target among the candidates by cosine."""
    p = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
    c = cands / (np.linalg.norm(cands, axis=1, keepdims=True) + 1e-9)
    sims = p @ c.T
    tgt = sims[np.arange(len(pred)), target_idx]
    ranks = (sims > tgt[:, None]).sum(1) + 1
    return float(np.mean(1.0 / ranks))


def offsets_by_tactic(z_before: np.ndarray, z_after: np.ndarray, tactics: list[str]) -> dict[str, dict]:
    """Per tactic: n, mean displacement (vector), mean norm, spread = mean
    squared distance of displacements from their mean (the translation test)."""
    out = {}
    d = z_after - z_before
    for tac in sorted(set(tactics)):
        idx = [i for i, t in enumerate(tactics) if t == tac]
        di = d[idx]
        mu = di.mean(0)
        out[tac] = {"n": len(idx), "mu": mu, "mean_norm": float(np.linalg.norm(di, axis=1).mean()),
                    "spread": float(((di - mu) ** 2).sum(1).mean()),
                    "mu_norm": float(np.linalg.norm(mu))}
    return out


# ---------------------------------------------------------------- the run

@torch.no_grad()
def embed(enc, tok, texts, max_seq, device, bs=128) -> np.ndarray:
    out = []
    for i in range(0, len(texts), bs):
        ids = tokenize(tok, texts[i:i + bs], max_seq, enc.cfg.pad_id).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(enc.pool(ids).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, enc.cfg.d_model))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-run", required=True)
    ap.add_argument("--tokenizer", default="runs/tokenizer/lean-unigram-32k.json")
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--max-candidates", type=int, default=2000)
    ap.add_argument("--translation-margin", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    t0 = time.monotonic()

    s2_rec = ledger.citable(args.s2_run)
    tok = Tokenizer.from_file(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_s2_encoder(REPO_ROOT / "runs" / "ckpts" / f"{args.s2_run}.safetensors", tok.get_vocab_size(), args.max_seq).to(device).eval()

    goals = {g.lock: g.readable for g in iter_goals()}
    rows = list(iter_transitions())
    decl_rows = list(iter_rows("declarations", verify=False))
    assign = splits.assign(decl_rows)
    split_of = splits.assign_transitions(rows, assign)
    steps = [(i, t) for i, t in enumerate(single_step_rows(rows)) if t.goal_before in goals and t.goal_after[0] in goals]
    # re-index split by original row identity
    row_index = {id(t): i for i, t in enumerate(rows)}
    train = [t for _, t in steps if split_of.get(row_index[id(t)]) == "train"]
    test = [t for _, t in steps if split_of.get(row_index[id(t)]) == "test"]
    if len(test) > args.max_candidates:
        test = rng.sample(test, args.max_candidates)

    def emb_pair(ts):
        zb = embed(enc, tok, [goals[t.goal_before] for t in ts], args.max_seq, device)
        za = embed(enc, tok, [goals[t.goal_after[0]] for t in ts], args.max_seq, device)
        return zb, za

    zb_tr, za_tr = emb_pair(train)
    off = offsets_by_tactic(zb_tr, za_tr, [t.tactic for t in train]) if train else {}
    zb_te, za_te = emb_pair(test)
    tgt = np.arange(len(test))
    results = {"n_train_steps": len(train), "n_test_steps": len(test)}
    if test:
        pred_off = np.stack([zb_te[i] + off.get(t.tactic, {"mu": np.zeros(zb_te.shape[1])})["mu"] for i, t in enumerate(test)])
        results["mrr_identity"] = round(mrr(zb_te, za_te, tgt), 4)
        results["mrr_offset"] = round(mrr(pred_off, za_te, tgt), 4)
        vec = TfidfVectorizer(token_pattern=r"[^\s]+", min_df=1)
        X = vec.fit_transform([goals[t.goal_before] for t in test] + [goals[t.goal_after[0]] for t in test])
        Xb, Xa = X[:len(test)].toarray(), X[len(test):].toarray()
        results["mrr_lexical"] = round(mrr(Xb, Xa, tgt), 4)
        per_tac = {}
        for tac in sorted({t.tactic for t in test}):
            idx = [i for i, t in enumerate(test) if t.tactic == tac]
            if len(idx) < 5:
                continue
            per_tac[tac] = {"n_test": len(idx),
                            "mrr_identity": round(mrr(zb_te[idx], za_te, np.array(idx)), 4),
                            "mrr_offset": round(mrr(pred_off[idx], za_te, np.array(idx)), 4)}
        results["per_tactic_retrieval"] = per_tac
    results["translation_test"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "mu"}
                                   for k, v in off.items()}
    # permutation control: shuffle the target assignment
    if test:
        perm = np.array(tgt); rng.shuffle(perm.tolist())
        p = list(range(len(test))); rng.shuffle(p)
        results["mrr_offset_permuted"] = round(mrr(pred_off, za_te, np.array(p)), 4)

    energy = energy_targets(rows)
    en = [v["energy"] for v in energy.values() if v["energy"] is not None]
    results["energy"] = {"states": len(energy), "states_with_closure": len(en),
                         "energy_mean": round(float(np.mean(en)), 4) if en else None,
                         "censored_rows": sum(v["censored"] for v in energy.values()),
                         "outcome_totals": dict(sum((Counter(v["outcomes"]) for v in energy.values()), Counter()))}
    energy_path = REPO_ROOT / "runs" / f"w0-energy-{args.s2_run}.jsonl"
    energy_path.write_text("".join(json.dumps({"goal": k, **v}) + "\n" for k, v in energy.items()))
    results["energy_table_cid"] = ledger.ket_put(energy_path.read_bytes())
    results["gate_for_w0"] = {"beat_offset_by": 0.10, "beat_lexical_by": 0.10, "translation_margin": args.translation_margin,
                              "note": "W0 must exceed max(mrr_offset, mrr_lexical) + 0.10 on module-held-out single-step transitions; tactics whose offset MRR is within the margin of W0 are 'translation' and get no GPU"}

    cfg_cid = ledger.ket_put_json({"s2_run": args.s2_run, "max_seq": args.max_seq, "max_candidates": args.max_candidates, "seed": args.seed})
    rec = ledger.RunRecord(
        run_id=f"w0-baselines-{time.strftime('%Y%m%d-%H%M%S')}", kind="eval", stage="W0", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=s2_rec.inputs.corpus_cids, tokenizer_cid=s2_rec.inputs.tokenizer_cid,
                                config_cid=cfg_cid, init_checkpoint_cid=s2_rec.checkpoint_cids[-1],
                                code_git=ledger.git_head(), mathlib_pin=s2_rec.inputs.mathlib_pin),
        seed=args.seed, hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    ledger.start_run(rec)
    ledger.finish_run(rec, results, started_monotonic=t0)
    print(json.dumps({**results, "metrics_cid": rec.metrics_cid}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
