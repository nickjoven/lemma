"""The only code path that makes a run citable.

Doctrine (mirrors quod / SEMANTICS.md): a metric without a metrics_cid does
not exist. `finish_run` refuses to record a completed run whose metrics were
not first `ket put` into the project store; everything else in the repo calls
through here, never around it.

The ket store is the repo's own `.ket/` (federation model: provenance travels
with the code). All ket access is via the CLI so the pinned binary is the one
observable in `PATH`, the same one whose sha lands in the run record.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .schemas import RunRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
KET_HOME = Path(os.environ.get("LEMMA_KET_HOME", REPO_ROOT / ".ket"))
LEDGER = REPO_ROOT / "runs" / "ledger.jsonl"


class LedgerError(RuntimeError):
    pass


def _ket(*args: str, stdin: str | bytes | None = None) -> str:
    r = subprocess.run(
        ["ket", "--home", str(KET_HOME), *args],
        input=stdin.encode() if isinstance(stdin, str) else stdin,
        capture_output=True,
    )
    if r.returncode != 0:
        raise LedgerError(f"ket {' '.join(args)} failed: {r.stderr.decode().strip()}")
    return r.stdout.decode().strip()


def ket_put(data: str | bytes) -> str:
    """Store bytes, return their CID. Loud on failure — never a silent None."""
    return _ket("put", "-", stdin=data)


def ket_put_json(obj) -> str:
    return ket_put(json.dumps(obj, sort_keys=True, indent=1))


def ket_get(cid: str) -> bytes:
    r = subprocess.run(
        ["ket", "--home", str(KET_HOME), "get", cid], capture_output=True
    )
    if r.returncode != 0:
        raise LedgerError(f"ket get {cid} failed: {r.stderr.decode().strip()}")
    return r.stdout


def git_head() -> str:
    r = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return r.stdout.strip() if r.returncode == 0 else "no-git"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(record: RunRecord) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(record.model_dump_json() + "\n")


def start_run(record: RunRecord) -> RunRecord:
    """Record the run's intent before any compute. Inputs must already be CIDs."""
    if not record.inputs.config_cid:
        raise LedgerError("a run without a config_cid is not startable")
    record.started = record.started or now_iso()
    append(record)
    return record


def finish_run(record: RunRecord, metrics: dict, *, started_monotonic: float | None = None,
               status: str = "completed") -> RunRecord:
    """Seal the run: metrics go to ket first, then the ledger line is appended.

    A completed run without metrics is a contradiction and is refused.
    """
    if status == "completed" and not metrics:
        raise LedgerError("refusing to complete a run with no metrics")
    if metrics:
        record.metrics_cid = ket_put_json(
            {"run_id": record.run_id, "metrics": metrics}
        )
    if started_monotonic is not None:
        record.wall_clock_s = round(time.monotonic() - started_monotonic, 1)
    record.status = status
    append(record)
    # One DAG node per sealed run so lineage is walkable from the store alone.
    _ket(
        "dag", "create", f"run {record.run_id}: {record.stage} {record.status}",
        "--kind", "score", "--agent", "lemma",
    )
    return record


def read_ledger() -> list[RunRecord]:
    if not LEDGER.exists():
        return []
    return [RunRecord.model_validate_json(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def citable(run_id: str) -> RunRecord:
    """Return the sealed record for run_id iff its metrics resolve in the store."""
    sealed = [r for r in read_ledger() if r.run_id == run_id and r.status == "completed"]
    if not sealed:
        raise LedgerError(f"run {run_id} has no completed ledger line")
    rec = sealed[-1]
    if not rec.metrics_cid:
        raise LedgerError(f"run {run_id} completed without a metrics_cid — not citable")
    ket_get(rec.metrics_cid)  # raises if the evidence does not resolve
    return rec
