"""S2 success-criterion eval: same-lock retrieval MRR, with controls.

The plan's falsifiable gate for the S2 encoder: mean-pooled embeddings must
retrieve exact-duplicate statements (declarations sharing a LOCK — the same
theorem stated under two names) at MRR >= 0.95, on HELD-OUT data: queries and
candidates both come from the test split THE CHECKPOINT WAS TRAINED AGAINST
(split_binding.resolve: the certificate its train record names, or, for a run
sealed before #10, a rebuild under its pinned rule that must reproduce the
training cache's per-split counts; a fresh assignment under today's rule is
not held out relative to an older run, and is never used), and no lock of
that split occurs in train or val (lemma #5: the earlier evaluator
sampled the whole corpus, so its number is full-corpus duplicate retrieval).
Controls, per the doctrine that a metric without its controls is not citable:
  positive  same-lock pairs -> should retrieve each other at rank ~1
  negative  random cross-module pairs -> should NOT be nearest neighbours
  baseline  masked-token accuracy vs. a unigram-majority predictor
  baseline  exact-text retrieval: how much of the pool is retrievable by
            string equality alone, and an UNTRAINED encoder of the same shape
            on the same pool; learned generalization is the margin over both

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

import blake3
import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from . import ledger
from .data import split_binding, splits
from .data.corpus import CorpusError, encoder_text, iter_rows
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


def check_cache_report(report_path: Path, assignment: dict[str, str]) -> dict[str, int]:
    """The training cache's per-split sequence counts must be the resolved
    split's. A missing report is an error, not a skipped check (review of
    #10): for a pinned-rule rebuild these counts are the only evidence tying
    the rebuild to the cache the checkpoint read, and the unigram baseline
    below reads that cache too."""
    if not report_path.is_file():
        raise CorpusError(f"{report_path} is missing: the split cannot be checked against the training cache")
    want = {s: v["sequences"] for s, v in json.loads(report_path.read_text())["splits"].items()}
    got = {s: sum(1 for v in assignment.values() if v == s) for s in want}
    if got != want:
        raise CorpusError(f"the resolved split's counts {got} differ from the training cache {want}")
    return want


def check_manifest_file(manifest_path: Path, binding: split_binding.Binding) -> str:
    """Optional extra evidence: a split-manifest.json the owner believes the
    checkpoint trained against must name the resolved rule and hold exactly
    the resolved assignment. Returns its BLAKE3."""
    data = manifest_path.read_bytes()
    manifest = json.loads(data)
    rule = splits.rule_id(manifest.get("rule_id") or manifest.get("rule", "?"))
    if rule != binding.rule_id:
        raise CorpusError(f"{manifest_path} was built under {rule}, but the run trained under {binding.rule_id}")
    if manifest["assignment"] != binding.assignment:
        diff = sum(1 for k, v in binding.assignment.items() if manifest["assignment"].get(k) != v)
        raise CorpusError(f"{manifest_path} disagrees with the resolved split on {diff} rows")
    return blake3.blake3(data).hexdigest()


def heldout_pool(rows, assign: dict[str, str], n_groups: int, n_distractors: int,
                 rng: random.Random, split: str = "test"):
    """Queries and candidates drawn from `split` only. Returns (members,
    distractors, stats). Raises CorpusError if any lock of the split also
    occurs outside it: that would be leakage, and the split rule promises it
    cannot happen, so it is checked rather than trusted."""
    in_split = [r for r in rows if assign.get(r.name) == split]
    outside_locks = {r.lock for r in rows if assign.get(r.name) != split}
    # disclosed, not refused: a manifest built under the pre-#6 rule keeps locks
    # inside a split but can split a module, so its test split is held out by
    # declaration and lock, not by whole module; this count says how far
    outside_modules = {r.module for r in rows if assign.get(r.name) != split}
    shared_modules = {r.module for r in in_split} & outside_modules
    by_lock: dict[str, list] = defaultdict(list)
    for r in in_split:
        by_lock[r.lock].append(r)
    groups = [sorted(g, key=lambda r: r.name) for g in by_lock.values() if len(g) >= 2]
    leaked = sorted({r.lock for r in in_split} & outside_locks)   # every lock of the split, not only the queried ones
    if leaked:
        raise CorpusError(f"{len(leaked)} lock(s) of the {split} split also occur outside the {split} split: {leaked[:3]}")
    n_duplicate_groups = len(groups)            # before truncation to n_groups
    groups.sort(key=lambda g: g[0].lock)        # order independent of shard order
    rng.shuffle(groups)
    groups = groups[:n_groups]
    members = [r for g in groups for r in g]
    member_names = {r.name for r in members}
    candidates = sorted((r for r in in_split if r.name not in member_names), key=lambda r: r.name)
    distractors = rng.sample(candidates, min(n_distractors, len(candidates)))
    stats = {"split": split, "rows_in_split": len(in_split), "groups_available": len(by_lock),
             "duplicate_groups_available": n_duplicate_groups,
             "modules_in_split": len({r.module for r in in_split}),
             "modules_shared_with_other_splits": len(shared_modules),
             "groups": len(groups), "queries": len(members), "distractors": len(distractors),
             "query_names_cid": None, "pool_names_cid": None}
    return members, distractors, stats


def same_lock_mrr(sims: torch.Tensor, locks: list[str], n_queries: int) -> tuple[float, float]:
    """MRR and hit@1 of the nearest OTHER same-lock statement for each of the
    first n_queries rows; the diagonal must already be masked."""
    rr = []
    for i in range(n_queries):
        order = torch.argsort(sims[i], descending=True)
        for rank, j in enumerate(order.tolist(), start=1):
            if locks[j] == locks[i]:
                rr.append(1.0 / rank)
                break
    return float(np.mean(rr)), float(np.mean([r == 1.0 for r in rr]))


def expected_rr_first_of(m: int, k: int, offset: int = 0) -> float:
    """E[1 / (offset + R)] where R is the position of the first of k marked
    items in a uniformly random order of m items (1 <= k <= m). Exact:
    P(R = 1) = k/m and P(R = r+1) / P(R = r) = (m - r - k + 1) / (m - r)."""
    r = np.arange(1, m - k + 2, dtype=np.float64)
    ratios = (m - r[:-1] - k + 1) / (m - r[:-1])
    p = np.concatenate([[k / m], (k / m) * np.cumprod(ratios)])
    return float(np.sum(p / (offset + r)))


def exact_text_mrr(texts: list[str], locks: list[str], n_queries: int) -> tuple[float, float]:
    """What string equality alone retrieves, as a REAL ranked MRR comparable
    to same_lock_mrr, plus its hit rate. Returns (mrr, hit_rate).

    The string-equality retriever scores every other pool row 1 if its encoder
    text is byte-identical to the query's, else 0, and ranks by that score.
    Tie policy: ties are broken uniformly at random, and the reported
    reciprocal rank is its exact expectation over that tie-break (no sampling).
    So identical-text rows of a DIFFERENT lock compete with the same-lock
    match inside the score-1 block, and a query with no identical same-lock
    text gets the expected reciprocal rank of its first same-lock row inside
    the score-0 block, after every score-1 row. The earlier version was a
    binary any-identical-same-lock indicator (now `hit_rate`), which ignored
    different-lock identical texts and was not an MRR, so subtracting it from
    the model's MRR was not a margin over a baseline (review of #10)."""
    text_ids = np.unique(np.asarray(texts, dtype=object), return_inverse=True)[1]
    lock_ids = np.unique(np.asarray(locks, dtype=object), return_inverse=True)[1]
    n = len(texts)
    rr, hits = [], []
    for i in range(n_queries):
        other = np.ones(n, dtype=bool)
        other[i] = False
        eq = other & (text_ids == text_ids[i])
        same = other & (lock_ids == lock_ids[i])
        m_eq, k_eq = int(eq.sum()), int((eq & same).sum())
        hits.append(1.0 if k_eq else 0.0)
        if k_eq:
            rr.append(expected_rr_first_of(m_eq, k_eq))
        else:
            rest = other & ~eq
            k_rest = int((rest & same).sum())
            rr.append(expected_rr_first_of(int(rest.sum()), k_rest, offset=m_eq) if k_rest else 0.0)
    return float(np.mean(rr)), float(np.mean(hits))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="sealed S2 train run_id")
    ap.add_argument("--groups", type=int, default=2000, help="same-lock groups to sample")
    ap.add_argument("--distractors", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--split", default="test", help="held-out split to evaluate on")
    ap.add_argument("--expect-split-rule", default=None, choices=sorted(splits.ASSIGNERS),
                    help="the split rule the run trained under; must agree with its record or pin")
    ap.add_argument("--cache-report", type=Path, default=REPO_ROOT / "corpora" / "cache" / "cache-report.json",
                    help="the training cache's report; its per-split counts must match (required)")
    ap.add_argument("--split-manifest", type=Path, default=None,
                    help="optional: a split-manifest.json to cross-check against the resolved split")
    args = ap.parse_args()
    src_state = ledger.preflight_source()  # refuse a dirty tree before compute; start_run checks it did not change since

    train_rec = ledger.citable(args.run)
    rows = list(iter_rows("declarations", verify=False))
    # the split the checkpoint trained against, by evidence; refuses rather than guessing (review of #10)
    binding = split_binding.resolve(train_rec, rows, expect_rule=args.expect_split_rule)
    cache_counts = check_cache_report(args.cache_report, binding.assignment)
    manifest_file_blake3 = check_manifest_file(args.split_manifest, binding) if args.split_manifest else None
    assign = binding.assignment

    ckpt = REPO_ROOT / "runs" / "ckpts" / f"{args.run}.safetensors"
    tok = Tokenizer.from_file(str(REPO_ROOT / "runs" / "tokenizer" / "lean-unigram-32k.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_encoder(ckpt, tok.get_vocab_size(), device)

    rng = random.Random(args.seed)
    members, distractors, pool_stats = heldout_pool(rows, assign, args.groups, args.distractors, rng, args.split)
    groups = pool_stats["groups"]
    pool = members + distractors
    texts = [encoder_text(r) for r in pool]
    locks = [r.lock for r in pool]
    modules = [r.module for r in pool]
    # the identities behind every number, sealed: which split, how it is tied
    # to the checkpoint (train_record_cid, or pinned_rule_reconstructed for a
    # run sealed before #10), and which statements were asked and offered
    pool_stats.update(binding.disclosure())
    pool_stats["cache_report_counts"] = cache_counts
    pool_stats["split_manifest_file_blake3"] = manifest_file_blake3
    pool_stats["query_names_cid"] = ledger.ket_put_json([r.name for r in members])
    pool_stats["pool_names_cid"] = ledger.ket_put_json([r.name for r in pool])

    t0 = time.monotonic()
    Z = embed(enc, tok, texts, device)
    sims = Z @ Z.T
    sims.fill_diagonal_(-1.0)

    # positive control: rank of the nearest OTHER same-lock statement
    mrr, hit1 = same_lock_mrr(sims, locks, len(members))

    # baselines on the SAME pool: string equality, and an untrained encoder of the same shape
    exact_mrr, exact_hit = exact_text_mrr(texts, locks, len(members))
    torch.manual_seed(args.seed)
    untrained = Encoder(EncoderConfig(vocab_size=tok.get_vocab_size())).to(device).eval()
    Zu = embed(untrained, tok, texts, device)
    sims_u = Zu @ Zu.T
    sims_u.fill_diagonal_(-1.0)
    untrained_mrr, _ = same_lock_mrr(sims_u, locks, len(members))

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
        "queries": len(members), "pool": len(pool), "groups": groups,
        "heldout": pool_stats,
        # both MRRs over the same pool and queries; exact-text ties broken uniformly at random (exact expectation)
        "same_lock_mrr_exact_text_baseline": round(exact_mrr, 4),
        "exact_text_hit_rate": round(exact_hit, 4),
        "exact_text_tie_policy": "uniform random tie-break, exact expected reciprocal rank",
        "same_lock_mrr_untrained_encoder": round(untrained_mrr, 4),
        "same_lock_mrr_margin_over_exact_text": round(mrr - exact_mrr, 4),
        "neg_cross_module_top1_rate": round(neg_rate, 4),
        "masked_acc_model": round(model_acc, 4), "masked_acc_unigram_baseline": round(unigram_acc, 4),
        "masked_acc_margin_pts": round(100 * (model_acc - unigram_acc), 1),
        "mrr_gate": MRR_GATE, "pass_mrr": mrr >= MRR_GATE,
        "pass_margin": (model_acc - unigram_acc) >= 0.25,
    }
    calib = {"pass": metrics["pass_mrr"] and neg_rate < 0.05 and metrics["pass_margin"],
             "controls": [
                 {"id": "same-lock-retrieval-heldout", "expect": f"MRR>={MRR_GATE} on the {args.split} split", "observed": mrr, "ok": metrics["pass_mrr"]},
                 {"id": "exact-text-baseline", "expect": "reported; the margin is the learned part", "observed": exact_mrr, "ok": True},
                 {"id": "untrained-encoder-baseline", "expect": "reported", "observed": untrained_mrr, "ok": True},
                 {"id": "cross-module-negative", "expect": "top1<0.05", "observed": neg_rate, "ok": neg_rate < 0.05},
                 {"id": "beats-unigram-by-25pts", "expect": ">=25", "observed": metrics["masked_acc_margin_pts"], "ok": metrics["pass_margin"]},
             ]}

    rec = ledger.RunRecord(
        run_id=f"s2-eval-{args.run}", kind="eval", stage="S2", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=train_rec.inputs.corpus_cids,
                                tokenizer_cid=train_rec.inputs.tokenizer_cid,
                                config_cid=train_rec.inputs.config_cid,
                                init_checkpoint_cid=train_rec.checkpoint_cids[-1],
                                code_git=ledger.git_head(), mathlib_pin=train_rec.inputs.mathlib_pin,
                                split_manifest_cid=binding.certificate_cid),
        seed=args.seed, hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    ledger.start_run(rec, preflight=src_state)
    rec.calibration_record_cid = ledger.ket_put_json(calib)
    ledger.finish_run(rec, metrics, started_monotonic=t0,
                      status="completed" if calib["pass"] else "calibration_failed")
    print(json.dumps({**metrics, "calibration_pass": calib["pass"],
                      "metrics_cid": rec.metrics_cid}, indent=1))
    return 0 if calib["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
