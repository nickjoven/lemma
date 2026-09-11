"""S4 v1 eval on module-held-out mutants, with the plan's controls.

  elaborates  macro-F1 (GATE >= 0.80) and accuracy on the test split
  baseline    bag-of-tokens logistic regression on the same split; the model
              must beat it by >= 10 F1 points (GATE)
  op_class    macro-F1 of operator identification (reported)
  ood         elaborates macro-F1 on the held-out operator class alone, if
              one was excluded from training — generalization to an unseen
              mutation type, reported separately from the IID number
  permutation sanity: shuffled labels -> F1 collapses

Every number is ledgered (metrics_cid) or it does not exist.

    uv run python -m lemma.evaluate_s4 --run <s4 run_id>
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tokenizers import Tokenizer

from . import ledger
from .models.encoder import Encoder, EncoderConfig
from .train_s4 import OP_CLASSES, VerdictModel, macro_f1, mutants_by_split, predict

REPO_ROOT = Path(__file__).resolve().parents[2]
F1_GATE, MARGIN_GATE = 0.80, 0.10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "s4_verdict.yaml")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rng = random.Random(args.seed)

    train_rec = ledger.citable(args.run)
    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VerdictModel(Encoder(EncoderConfig(vocab_size=tok.get_vocab_size(), max_seq=cfg["max_seq"])))
    model.load_state_dict(load_file(str(REPO_ROOT / "runs" / "ckpts" / f"{args.run}.safetensors")))
    model = model.to(device).eval()

    data = mutants_by_split(cfg.get("ood_operator"))
    train, test = data["train"], data["test"]
    ood = cfg.get("ood_operator")
    test_iid = [t for t in test if t[3] != ood] if ood else test
    test_ood = [t for t in test if t[3] == ood] if ood else []
    t0 = time.monotonic()

    pe, po = predict(model, tok, test_iid, cfg["max_seq"], device)
    ye, yo = [t[1] for t in test_iid], [t[2] for t in test_iid]
    f1_elab, acc_elab = macro_f1(ye, pe), float(np.mean(np.array(ye) == pe))
    f1_op = macro_f1(yo, po)

    # bag-of-tokens logistic baseline on the same split
    vec = CountVectorizer(tokenizer=lambda s: [str(i) for i in tok.encode(s).ids], token_pattern=None, min_df=2)
    Xtr = vec.fit_transform([t[0] for t in train]); Xte = vec.transform([t[0] for t in test_iid])
    base = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, [t[1] for t in train])
    f1_base = macro_f1(ye, base.predict(Xte))

    f1_ood = float("nan")
    if test_ood:
        pe_o, _ = predict(model, tok, test_ood, cfg["max_seq"], device)
        f1_ood = macro_f1([t[1] for t in test_ood], pe_o)

    yp = list(ye); rng.shuffle(yp)
    f1_perm = macro_f1(yp, pe)

    metrics = {
        "test_mutants_iid": len(test_iid), "test_mutants_ood": len(test_ood), "ood_operator": ood,
        "elab_macro_f1": round(f1_elab, 4), "elab_accuracy": round(acc_elab, 4),
        "elab_macro_f1_bag_of_tokens": round(f1_base, 4), "elab_margin_pts": round(100 * (f1_elab - f1_base), 1),
        "op_macro_f1": round(f1_op, 4), "elab_macro_f1_ood": round(f1_ood, 4) if f1_ood == f1_ood else None,
        "permutation_macro_f1": round(f1_perm, 4),
        "f1_gate": F1_GATE, "pass_f1": f1_elab >= F1_GATE,
        "pass_margin": (f1_elab - f1_base) >= MARGIN_GATE,
        "label_scope": "Phase-3a: elaborates + operator; gate_verdict for statement mutants pending Phase-3b",
    }
    calib = {"pass": metrics["pass_f1"] and metrics["pass_margin"] and f1_perm < 0.6,
             "controls": [
                 {"id": "elab-macro-f1", "expect": f">={F1_GATE}", "observed": f1_elab, "ok": metrics["pass_f1"]},
                 {"id": "beats-bag-of-tokens", "expect": f">={MARGIN_GATE}", "observed": f1_elab - f1_base, "ok": metrics["pass_margin"]},
                 {"id": "permutation-sanity", "expect": "<0.6", "observed": f1_perm, "ok": f1_perm < 0.6},
                 {"id": "ood-operator", "expect": "reported", "observed": f1_ood, "ok": True},
             ]}
    rec = ledger.RunRecord(
        run_id=f"s4-eval-{args.run}", kind="eval", stage="S4", started=ledger.now_iso(),
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
