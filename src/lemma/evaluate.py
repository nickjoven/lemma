"""CID-emitting eval runner with calibration gating.

Usage pattern (every stage):
    rec = start_run(...)                      # inputs are CIDs, or no run
    ok = run_calibration(controls)            # known-answer cases first
    metrics = compute(...)                    # the actual eval
    finish_run(rec, metrics, status="completed" if ok else "calibration_failed")

Calibration controls are small known-answer cases; each stage's evaluator
(evaluate_s2.py .. evaluate_s5.py) embeds its own and seals the results with
the run. A failing control makes the run's training metrics uncitable,
mirroring quod's rule that a failing positive control means the gates are
wrong. `load_controls` also accepts YAML control files under calibration/;
none are checked in.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from . import ledger

CALIBRATION_DIR = Path(__file__).resolve().parents[2] / "calibration"


def load_controls() -> list[dict]:
    controls = []
    for p in sorted(CALIBRATION_DIR.glob("*.yml")):
        doc = yaml.safe_load(p.read_text())
        for c in doc.get("controls", []):
            c["source"] = p.name
            controls.append(c)
    return controls


def run_calibration(check_fn, controls: list[dict]) -> tuple[bool, dict]:
    """check_fn(control) -> observed value; compared against control['expect'].

    Returns (all_pass, record). The record is ket-put by the caller via
    finish_run's calibration_record_cid.
    """
    results = []
    for c in controls:
        observed = check_fn(c)
        ok = observed == c["expect"]
        results.append({"id": c["id"], "expect": c["expect"], "observed": observed, "ok": ok})
    all_ok = all(r["ok"] for r in results)
    return all_ok, {"pass": all_ok, "controls": results}


def seal_calibration(record: dict) -> str:
    return ledger.ket_put_json(record)
