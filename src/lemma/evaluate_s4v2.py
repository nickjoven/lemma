"""S4 v2 eval on module-held-out mutants with prover-relative labels.

  verdict     macro-F1 over the decided class set (GATE >= 0.80) and accuracy
  baseline    bag-of-tokens logistic regression on the same split; the model
              must beat it by >= 10 F1 points (GATE)
  ood         macro-F1 on the held-out operator class alone, if one was
              excluded from training (reported separately from IID)
  per-class   precision/recall per class, and the support of each — a macro
              number over a class with a handful of rows is flagged, not hidden
  permutation sanity: shuffled labels -> F1 collapses

Every number is ledgered (metrics_cid) or it does not exist.

    uv run python -m lemma.evaluate_s4v2 --run <s4v2 run_id>
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from tokenizers import Tokenizer

from . import ledger
from .models.encoder import Encoder, EncoderConfig
from .train_s4 import macro_f1
from .train_s4v2 import VerdictModelV2, load_labelled, predict

REPO_ROOT = Path(__file__).resolve().parents[2]
F1_GATE, MARGIN_GATE = 0.80, 0.10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "s4v2_verdict.yaml")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rng = random.Random(args.seed)

    train_rec = ledger.citable(args.run)
    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, classes, audit = load_labelled(cfg)
    model = VerdictModelV2(Encoder(EncoderConfig(vocab_size=tok.get_vocab_size(), max_seq=cfg["max_seq"])), len(classes))
    model.load_state_dict(load_file(str(REPO_ROOT / "runs" / "ckpts" / f"{args.run}.safetensors")))
    model = model.to(device).eval()

    train, test = data["train"], data["test"]
    ood = cfg.get("ood_operator")
    test_iid = [t for t in test if t[2] != ood] if ood else test
    test_ood = [t for t in test if t[2] == ood] if ood else []
    t0 = time.monotonic()

    p = predict(model, tok, test_iid, cfg["max_seq"], device)
    y = [t[1] for t in test_iid]
    f1, acc = macro_f1(y, p), float(np.mean(np.array(y) == p))
    prec, rec_, f1c, sup = precision_recall_fscore_support(y, p, labels=list(range(len(classes))), zero_division=0)
    per_class = {c: {"precision": round(float(prec[i]), 4), "recall": round(float(rec_[i]), 4),
                     "f1": round(float(f1c[i]), 4), "support": int(sup[i])} for i, c in enumerate(classes)}

    vec = CountVectorizer(tokenizer=lambda s: [str(i) for i in tok.encode(s).ids], token_pattern=None, min_df=2)
    Xtr = vec.fit_transform([t[0] for t in train]); Xte = vec.transform([t[0] for t in test_iid])
    base = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, [t[1] for t in train])
    f1_base = macro_f1(y, base.predict(Xte))

    f1_ood = float("nan")
    if test_ood:
        f1_ood = macro_f1([t[1] for t in test_ood], predict(model, tok, test_ood, cfg["max_seq"], device))

    yp = list(y); rng.shuffle(yp)
    f1_perm = macro_f1(yp, p)
    thin = [c for c, v in per_class.items() if v["support"] < 30]

    metrics = {
        "classes": classes, "test_rows_iid": len(test_iid), "test_rows_ood": len(test_ood), "ood_operator": ood,
        "test_class_counts": dict(Counter(t[3] for t in test_iid)),
        "macro_f1": round(f1, 4), "accuracy": round(acc, 4), "per_class": per_class,
        "macro_f1_bag_of_tokens": round(f1_base, 4), "margin_pts": round(100 * (f1 - f1_base), 1),
        "macro_f1_ood": round(f1_ood, 4) if f1_ood == f1_ood else None,
        "permutation_macro_f1": round(f1_perm, 4),
        "f1_gate": F1_GATE, "pass_f1": f1 >= F1_GATE, "pass_margin": (f1 - f1_base) >= MARGIN_GATE,
        "thin_classes": thin, "label_audit": audit,
        "label_scope": f"Phase-3b gate_verdict, prover-relative (prover filter: {cfg.get('prover')})",
    }
    calib = {"pass": metrics["pass_f1"] and metrics["pass_margin"] and f1_perm < 0.6,
             "controls": [
                 {"id": "macro-f1", "expect": f">={F1_GATE}", "observed": f1, "ok": metrics["pass_f1"]},
                 {"id": "beats-bag-of-tokens", "expect": f">={MARGIN_GATE}", "observed": f1 - f1_base, "ok": metrics["pass_margin"]},
                 {"id": "permutation-sanity", "expect": "<0.6", "observed": f1_perm, "ok": f1_perm < 0.6},
                 {"id": "ood-operator", "expect": "reported", "observed": f1_ood, "ok": True},
                 {"id": "class-support", "expect": "every class >= 30 test rows (else flagged)", "observed": thin, "ok": True},
             ]}
    rec = ledger.RunRecord(
        run_id=f"s4v2-eval-{args.run}", kind="eval", stage="S4", started=ledger.now_iso(),
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
