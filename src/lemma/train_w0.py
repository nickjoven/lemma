"""Stage W0 — the world model: f(embed(goal_before), tactic) -> embed(goal_after).

The predictor is trained against an EMA target encoder (both warm-started
from the sealed S2 encoder) with an in-batch InfoNCE objective, so the
training signal IS the gate: next-state retrieval among candidates. Two
auxiliary heads on the same predicted state:
  outcome  closed | open | error | budget          (CE, per-class support reported)
  cost     log(heartbeats) regression, CENSORED at heartbeat_cap: a budget row
           contributes -log P(cost >= cap) under the predicted Gaussian (Tobit),
           never a value.
Transitions inherit the demonstrandum's module split; the gate (evaluate)
is next-state MRR on module-held-out single-step transitions >= the better
of the offset and lexical baselines + 0.10, permutation control, cost head
scored on uncensored rows and calibrated on censored ones. "JEPA" is not a
word this repo uses until this gate is passed.

    uv run python -u -m lemma.train_w0 train --config configs/w0_world.yaml
    uv run python -u -m lemma.train_w0 evaluate --run <w0 run_id>
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer

from . import ledger
from .data import splits
from .data.corpus import iter_goals, iter_rows, iter_transitions
from .models.encoder import Encoder, EncoderConfig
from .train_s3 import load_s2_encoder, tokenize
from .w0_baselines import mrr, offsets_by_tactic, single_step_rows

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTCOMES = ["closed", "open", "error", "budget"]


class WorldModel(nn.Module):
    def __init__(self, enc: Encoder, tactics: list[str], d_tac: int = 64):
        super().__init__()
        self.enc = enc
        self.tactics = list(tactics)
        d = enc.cfg.d_model
        self.tac = nn.Embedding(len(tactics) + 1, d_tac)      # last index = unknown tactic
        self.pred = nn.Sequential(nn.Linear(d + d_tac, d), nn.GELU(), nn.Linear(d, d))
        self.head_out = nn.Linear(d, len(OUTCOMES))
        self.head_cost = nn.Linear(d, 2)                       # mean, log-sigma of log(heartbeats)

    def tac_idx(self, names: list[str], device) -> torch.Tensor:
        return torch.tensor([self.tactics.index(n) if n in self.tactics else len(self.tactics) for n in names], device=device)

    def forward(self, ids, tac_names):
        z = self.enc.pool(ids)
        e = self.tac(self.tac_idx(tac_names, ids.device))
        z_hat = self.pred(torch.cat([z, e], -1))
        return z_hat, self.head_out(z_hat), self.head_cost(z_hat)


def tobit_nll(mu, log_sigma, y, censored):
    """Gaussian NLL for observed log-costs; survival term -log P(Y >= y) for
    censored rows (y = log cap there)."""
    sigma = log_sigma.exp().clamp_min(1e-3)
    z = (y - mu) / sigma
    nll_obs = 0.5 * z ** 2 + log_sigma + 0.5 * math.log(2 * math.pi)
    surv = -torch.log(0.5 * torch.erfc(z / math.sqrt(2)) + 1e-9)
    return torch.where(censored, surv, nll_obs).mean()


def load_data(cfg: dict):
    goals = {g.lock: g.readable for g in iter_goals(cfg.get("goals_corpus", "goals"))}
    rows = [t for t in iter_transitions(cfg.get("transitions_corpus", "transitions")) if t.kind == "rung"]
    assign = splits.assign(list(iter_rows("declarations", verify=False)))
    sp = splits.assign_transitions(rows, assign)
    data = {s: [] for s in splits.SPLITS}
    for i, t in enumerate(rows):
        s = sp.get(i)
        if s is None or t.goal_before not in goals:
            continue
        after = t.goal_after[0] if (isinstance(t.goal_after, list) and len(t.goal_after) == 1 and t.goal_after[0] in goals) else None
        data[s].append({"before": goals[t.goal_before], "after": goals[after] if after else None,
                        "single": after is not None and after != t.goal_before,
                        "tactic": t.tactic, "outcome": OUTCOMES.index(t.outcome) if t.outcome in OUTCOMES else OUTCOMES.index("error"),
                        "logcost": math.log(max(t.heartbeats, 1)) if not t.censored else math.log(max(t.heartbeat_cap, 1)),
                        "censored": t.censored, "before_lock": t.goal_before, "after_lock": after})
    tactics = sorted({r["tactic"] for r in data["train"]})
    return data, tactics


@torch.no_grad()
def embed_texts(enc, tok, texts, max_seq, device, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        ids = tokenize(tok, texts[i:i + bs], max_seq, enc.cfg.pad_id).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(enc.pool(ids).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, enc.cfg.d_model))


@torch.no_grad()
def predict_states(model, tok, rows, max_seq, device, bs=128):
    model.eval()
    zs, outs, costs = [], [], []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids = tokenize(tok, [r["before"] for r in chunk], max_seq, model.enc.cfg.pad_id).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            z, o, c = model(ids, [r["tactic"] for r in chunk])
        zs.append(z.float().cpu()); outs.append(o.float().cpu()); costs.append(c.float().cpu())
    model.train()
    return torch.cat(zs).numpy(), torch.cat(outs).numpy(), torch.cat(costs).numpy()


def train(args) -> int:
    cfg = yaml.safe_load(args.config.read_text())
    config_cid = ledger.ket_put(args.config.read_bytes())
    run_id = args.run_id or f"w0-world-{time.strftime('%Y%m%d-%H%M%S')}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 1337))
    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    s2_rec = ledger.citable(cfg["s2_run"])
    enc = load_s2_encoder(REPO_ROOT / "runs" / "ckpts" / f"{cfg['s2_run']}.safetensors", tok.get_vocab_size(), cfg["max_seq"])
    data, tactics = load_data(cfg)
    train_rows, val_rows = data["train"], data["val"]
    train_single = [r for r in train_rows if r["single"]]
    val_single = [r for r in val_rows if r["single"]]
    if len(train_single) < 100 or len(val_single) < 20:
        raise SystemExit(f"W0: too few single-step transitions (train {len(train_single)}, val {len(val_single)})")
    model = WorldModel(enc, tactics).to(device)
    target = copy.deepcopy(model.enc).to(device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    ema = float(cfg.get("ema", 0.99))
    pad = enc.cfg.pad_id
    print(f"W0 train rows {len(train_rows)} (single-step {len(train_single)}) val {len(val_rows)} (single {len(val_single)}) "
          f"test {len(data['test'])} | tactics {tactics} | outcomes {dict(Counter(OUTCOMES[r['outcome']] for r in train_rows))}", flush=True)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": float(cfg.get("lr_encoder", 3e-5))},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("enc.")], "lr": float(cfg.get("lr_heads", 5e-4))},
    ], weight_decay=0.01)
    rec = ledger.start_run(ledger.RunRecord(
        run_id=run_id, kind="train", stage="W0", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=[*s2_rec.inputs.corpus_cids, *cfg.get("transitions_corpus_cids", [])],
                                tokenizer_cid=s2_rec.inputs.tokenizer_cid, config_cid=config_cid,
                                init_checkpoint_cid=s2_rec.checkpoint_cids[-1], code_git=ledger.git_head(),
                                mathlib_pin=s2_rec.inputs.mathlib_pin),
        seed=cfg.get("seed", 1337), hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__}))
    t0 = time.monotonic()
    rng = np.random.default_rng(cfg.get("seed", 1337))
    bs, steps, val_every = cfg.get("batch_size", 64), cfg.get("steps", 4000), cfg.get("val_every", 250)
    temp = float(cfg.get("temperature", 0.07))
    ckpt_dir = REPO_ROOT / "runs" / "ckpts"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_id}-best.safetensors"
    best, best_step, losses = -1.0, -1, []

    def validate(step):
        nonlocal best, best_step
        z_hat, _, _ = predict_states(model, tok, val_single, cfg["max_seq"], device)
        z_after = embed_texts(target, tok, [r["after"] for r in val_single], cfg["max_seq"], device)
        m = mrr(z_hat, z_after, np.arange(len(val_single)))
        if m > best:
            best, best_step = m, step
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(best_path))
        print(f"step {step}: val next-state MRR {m:.4f}{' (best, saved)' if best_step == step else ''}", flush=True)

    model.train()
    for step in range(steps):
        if step > 0 and step % val_every == 0:
            validate(step)
        # half the batch single-step (retrieval), half any rung row (outcome/cost)
        b1 = [train_single[i] for i in rng.integers(0, len(train_single), size=bs // 2)]
        b2 = [train_rows[i] for i in rng.integers(0, len(train_rows), size=bs // 2)]
        batch = b1 + b2
        ids = tokenize(tok, [r["before"] for r in batch], cfg["max_seq"], pad).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            z_hat, o, c = model(ids, [r["tactic"] for r in batch])
            with torch.no_grad():
                ids_a = tokenize(tok, [r["after"] for r in b1], cfg["max_seq"], pad).to(device)
                z_a = target.pool(ids_a)
            zh = F.normalize(z_hat[:len(b1)].float(), dim=-1); za = F.normalize(z_a.float(), dim=-1)
            logits = zh @ za.T / temp
            loss_nce = F.cross_entropy(logits, torch.arange(len(b1), device=device))
            y_out = torch.tensor([r["outcome"] for r in batch], device=device)
            loss_out = F.cross_entropy(o.float(), y_out)
            y_cost = torch.tensor([r["logcost"] for r in batch], device=device, dtype=torch.float32)
            cens = torch.tensor([r["censored"] for r in batch], device=device)
            loss_cost = tobit_nll(c[:, 0].float(), c[:, 1].float(), y_cost, cens)
            loss = loss_nce + 0.5 * loss_out + 0.2 * loss_cost
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pt, ps in zip(target.parameters(), model.enc.parameters()):
                pt.mul_(ema).add_(ps.detach(), alpha=1 - ema)
        losses.append((loss.item(), loss_nce.item(), loss_out.item(), loss_cost.item()))
        if step % 100 == 0:
            l = np.mean(losses[-100:], axis=0)
            print(f"step {step}: loss {l[0]:.4f} (nce {l[1]:.4f} out {l[2]:.4f} cost {l[3]:.4f})", flush=True)
    validate(steps)
    ckpt = ckpt_dir / f"{run_id}.safetensors"; best_path.replace(ckpt)
    tgt_path = ckpt_dir / f"{run_id}-target.safetensors"
    save_file({k: v.contiguous() for k, v in target.state_dict().items()}, str(tgt_path))
    cid = ledger.ket_put(ckpt.read_bytes()); rec.checkpoint_cids.append(cid)
    tcid = ledger.ket_put(tgt_path.read_bytes()); rec.checkpoint_cids.append(tcid)
    gi = REPO_ROOT / ".gitignore"
    for c_ in (cid, tcid):
        line = f".ket/cas/{c_}"
        if line not in gi.read_text().splitlines():
            with open(gi, "a") as f: f.write(f"{line}\n")
    (REPO_ROOT / "runs" / f"{run_id}-tactics.json").write_text(json.dumps(tactics))
    ledger.finish_run(rec, metrics={"steps": steps, "final_loss": float(np.mean([l[0] for l in losses[-100:]])),
                                    "train_rows": len(train_rows), "train_single_step": len(train_single),
                                    "val_mrr_best": round(best, 4), "best_step": best_step, "tactics": tactics},
                      started_monotonic=t0)
    print(f"run {run_id} sealed; metrics_cid {rec.metrics_cid}", flush=True)
    return 0


def evaluate(args) -> int:
    cfg = yaml.safe_load(args.config.read_text())
    rng = random.Random(args.seed)
    train_rec = ledger.citable(args.run)
    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tactics = json.loads((REPO_ROOT / "runs" / f"{args.run}-tactics.json").read_text())
    ecfg = EncoderConfig(vocab_size=tok.get_vocab_size(), max_seq=cfg["max_seq"])
    model = WorldModel(Encoder(ecfg), tactics)
    model.load_state_dict(load_file(str(REPO_ROOT / "runs" / "ckpts" / f"{args.run}.safetensors")))
    model = model.to(device).eval()
    target = Encoder(ecfg); target.load_state_dict(load_file(str(REPO_ROOT / "runs" / "ckpts" / f"{args.run}-target.safetensors")))
    target = target.to(device).eval()
    s2 = load_s2_encoder(REPO_ROOT / "runs" / "ckpts" / f"{cfg['s2_run']}.safetensors", tok.get_vocab_size(), cfg["max_seq"]).to(device).eval()

    data, _ = load_data(cfg)
    test = data["test"]
    single = [r for r in test if r["single"]]
    if len(single) > args.max_candidates:
        single = rng.sample(single, args.max_candidates)
    t0 = time.monotonic()
    tgt = np.arange(len(single))
    z_hat, _, _ = predict_states(model, tok, single, cfg["max_seq"], device)
    z_after = embed_texts(target, tok, [r["after"] for r in single], cfg["max_seq"], device)
    m_w0 = mrr(z_hat, z_after, tgt)
    # baselines on the SAME rows, under the sealed S2 encoder (as w0_baselines)
    zb_s2 = embed_texts(s2, tok, [r["before"] for r in single], cfg["max_seq"], device)
    za_s2 = embed_texts(s2, tok, [r["after"] for r in single], cfg["max_seq"], device)
    tr_single = [r for r in data["train"] if r["single"]]
    if len(tr_single) > args.max_candidates:
        tr_single = rng.sample(tr_single, args.max_candidates)
    off = offsets_by_tactic(embed_texts(s2, tok, [r["before"] for r in tr_single], cfg["max_seq"], device),
                            embed_texts(s2, tok, [r["after"] for r in tr_single], cfg["max_seq"], device),
                            [r["tactic"] for r in tr_single]) if tr_single else {}
    pred_off = np.stack([zb_s2[i] + off.get(r["tactic"], {"mu": np.zeros(zb_s2.shape[1])})["mu"] for i, r in enumerate(single)])
    m_id, m_off = mrr(zb_s2, za_s2, tgt), mrr(pred_off, za_s2, tgt)
    vec = TfidfVectorizer(token_pattern=r"[^\s]+", min_df=1)
    X = vec.fit_transform([r["before"] for r in single] + [r["after"] for r in single])
    m_lex = mrr(X[:len(single)].toarray(), X[len(single):].toarray(), tgt)
    p = list(range(len(single))); rng.shuffle(p)
    m_perm = mrr(z_hat, z_after, np.array(p))
    per_tac = {}
    for tac in sorted({r["tactic"] for r in single}):
        idx = [i for i, r in enumerate(single) if r["tactic"] == tac]
        if len(idx) < 5:
            continue
        w0_t, off_t = mrr(z_hat[idx], z_after, np.array(idx)), mrr(pred_off[idx], za_s2, np.array(idx))
        per_tac[tac] = {"n": len(idx), "mrr_w0": round(w0_t, 4), "mrr_offset": round(off_t, 4),
                        "translation": (w0_t - off_t) < args.translation_margin}
    # outcome + cost heads on every rung row of the test split
    _, o, c = predict_states(model, tok, test, cfg["max_seq"], device)
    y_out = np.array([r["outcome"] for r in test]); pred_out = o.argmax(1)
    support = dict(Counter(OUTCOMES[i] for i in y_out))
    from sklearn.metrics import f1_score
    f1_out = float(f1_score(y_out, pred_out, average="macro"))
    cens = np.array([r["censored"] for r in test]); y_cost = np.array([r["logcost"] for r in test])
    mae_unc = float(np.abs(c[~cens, 0] - y_cost[~cens]).mean()) if (~cens).any() else None
    cens_cal = float((c[cens, 0] >= y_cost[cens]).mean()) if cens.any() else None
    bar = max(m_off, m_lex) + 0.10
    metrics = {"test_single_step": len(single), "test_rows": len(test), "outcome_support": support,
               "mrr_w0": round(m_w0, 4), "mrr_identity": round(m_id, 4), "mrr_offset": round(m_off, 4),
               "mrr_lexical": round(m_lex, 4), "mrr_w0_permuted": round(m_perm, 4), "gate_bar": round(bar, 4),
               "pass_retrieval": m_w0 >= bar, "per_tactic": per_tac,
               "translation_tactics": [t for t, v in per_tac.items() if v["translation"]],
               "outcome_macro_f1": round(f1_out, 4), "cost_mae_log_uncensored": round(mae_unc, 4) if mae_unc is not None else None,
               "cost_censored_rows": int(cens.sum()), "cost_censored_pred_ge_cap_frac": round(cens_cal, 4) if cens_cal is not None else None}
    calib = {"pass": metrics["pass_retrieval"] and m_perm < 0.2,
             "controls": [{"id": "next-state-mrr", "expect": f">={bar:.3f} (max(offset, lexical)+0.10)", "observed": m_w0, "ok": metrics["pass_retrieval"]},
                          {"id": "permutation-sanity", "expect": "<0.2", "observed": m_perm, "ok": m_perm < 0.2},
                          {"id": "outcome-support", "expect": "reported per class", "observed": support, "ok": True},
                          {"id": "cost-censoring", "expect": "censored rows scored as survival, not values", "observed": cens_cal, "ok": True}]}
    rec = ledger.RunRecord(run_id=f"w0-eval-{args.run}", kind="eval", stage="W0", started=ledger.now_iso(),
                           inputs=ledger.RunInputs(corpus_cids=train_rec.inputs.corpus_cids, tokenizer_cid=train_rec.inputs.tokenizer_cid,
                                                   config_cid=train_rec.inputs.config_cid, init_checkpoint_cid=train_rec.checkpoint_cids[0],
                                                   code_git=ledger.git_head(), mathlib_pin=train_rec.inputs.mathlib_pin),
                           seed=args.seed, hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    ledger.start_run(rec)
    rec.calibration_record_cid = ledger.ket_put_json(calib)
    ledger.finish_run(rec, metrics, started_monotonic=t0, status="completed" if calib["pass"] else "calibration_failed")
    print(json.dumps({**metrics, "calibration_pass": calib["pass"], "metrics_cid": rec.metrics_cid}, indent=1))
    return 0 if calib["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("train"); a.add_argument("--config", type=Path, required=True); a.add_argument("--run-id", default=None)
    b = sub.add_parser("evaluate"); b.add_argument("--run", required=True)
    b.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "w0_world.yaml")
    b.add_argument("--max-candidates", type=int, default=2000); b.add_argument("--seed", type=int, default=1337)
    b.add_argument("--translation-margin", type=float, default=0.03)
    args = ap.parse_args()
    return train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
