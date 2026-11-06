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

import blake3

from .schemas import RunInputs, RunRecord  # re-exported for callers (train.py uses ledger.RunInputs)

__all__ = ["RunInputs", "RunRecord"]

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


def ket_get_verified(cid: str) -> bytes:
    """Read a blob and check that the bytes actually returned hash to `cid`
    (a CID is the BLAKE3 hex of the content). Verifying the bytes in hand
    rather than a separate `ket verify` closes the verify-then-read race and
    catches a blob edited in place under its own name (lemma #7)."""
    data = ket_get(cid)
    got = blake3.blake3(data).hexdigest()
    if got != cid:
        raise LedgerError(f"evidence {cid} is corrupted: its bytes hash to {got}")
    return data


def git_head() -> str:
    r = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return r.stdout.strip() if r.returncode == 0 else "no-git"


# The paths whose bytes decide what a run computed. A commit hash names them
# only if the working tree matched the commit; lemma #8 found a sealed run whose
# recorded commit did not contain the evaluator that produced it.
SOURCE_PATHS = ("src", "configs", "pyproject.toml", "uv.lock")


class NotACheckoutError(LedgerError):
    pass


def source_state(repo: Path | None = None) -> dict:
    """{"head": <commit>, "dirty": bool, "changed": [paths]} for the source
    paths: `changed` lists every tracked file that differs from HEAD and every
    untracked, non-ignored file (git status --porcelain, untracked=all).

    Outside a git checkout there is no commit to name and no diff to seal, so
    this refuses (NotACheckoutError) instead of returning a placeholder: the
    earlier `{"<not a git checkout>": null}` snapshot was a CID that recovered
    nothing, recorded as if it were provenance (review of #10)."""
    repo = repo or REPO_ROOT
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    if head.returncode != 0:
        raise NotACheckoutError(
            f"{repo} is not a git checkout with a commit: the source that runs cannot be named "
            "or sealed, so no run is recorded from here")
    st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all",
                         "--", *SOURCE_PATHS], capture_output=True, text=True, check=True)
    changed = sorted(_porcelain_path(line) for line in st.stdout.splitlines() if line.strip())
    return {"head": head.stdout.strip(), "dirty": bool(changed), "changed": changed}


def source_snapshot(changed: list[str], repo: Path | None = None) -> str:
    """Seal the exact bytes of every changed/untracked source file: each file
    goes to the store under its own CID and the {path: cid} manifest's CID is
    returned, so the code that ran is recoverable from the record alone."""
    repo = repo or REPO_ROOT
    base = source_state(repo)["head"]   # refuses outside a checkout: a snapshot needs a base commit
    files = {}
    for rel in changed:
        path = repo / rel
        if path.is_file():
            files[rel] = ket_put(path.read_bytes())
        else:
            files[rel] = None  # deleted relative to HEAD
    return ket_put_json({"base_commit": base, "files": files})


def _porcelain_path(line: str) -> str:
    """The on-disk path of one `git status --porcelain` line: a rename or copy
    reads `XY old -> new` (the new path is the one whose bytes matter), and a
    path with unusual characters is C-quoted."""
    p = line[3:]
    if " -> " in p:
        p = p.split(" -> ", 1)[1]
    if len(p) >= 2 and p[0] == p[-1] == '"':
        p = p[1:-1].encode("latin-1").decode("unicode_escape").encode("latin-1").decode("utf-8")
    return p


class DirtySourceError(LedgerError):
    pass


def preflight_source(dirty_source: str | None = None) -> dict:
    """start_run's source-provenance refusal, callable BEFORE any compute: a
    dirty tree under the reject policy is refused here instead of after the GPU
    work and the evidence puts.

    Returns the source state it saw (with the snapshot CID of the changed
    bytes under the snapshot policy). Pass it to start_run(preflight=...): an
    evaluator calls start_run only after its compute, and start_run then
    refuses if the source changed in between, so a tree edited mid-run is not
    recorded as the clean commit it was at the end (review of #10)."""
    state = source_state()
    policy = dirty_source or os.environ.get("LEMMA_DIRTY_SOURCE", "reject")
    if state["dirty"] and policy != "snapshot":
        raise DirtySourceError(
            "source differs from the checked-out commit; commit first, or seal the exact bytes with "
            f"LEMMA_DIRTY_SOURCE=snapshot: {', '.join(state['changed'])}")
    if state["dirty"]:
        state = {**state, "snapshot_cid": source_snapshot(state["changed"])}
    return state


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(record: RunRecord) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(record.model_dump_json() + "\n")


def start_run(record: RunRecord, *, dirty_source: str | None = None,
              preflight: dict | None = None) -> RunRecord:
    """Record the run's intent before any compute. Inputs must already be CIDs.

    Provenance (lemma #8): `inputs.code_git` names a commit, so the source
    paths must match that commit. If they do not, the run is refused, unless
    `dirty_source="snapshot"` (or LEMMA_DIRTY_SOURCE=snapshot) asks for the
    exact bytes of the changed files to be sealed into the store first; the
    record then carries `source_snapshot_cid` and `code_git_dirty=True` so a
    reader can tell an exact commit from a commit-plus-snapshot.

    Scope: only SOURCE_PATHS are checked and sealed. Gitignored files,
    scripts/, and the `ket` binary are outside it (the ket sha is recorded
    separately where a caller records it).

    `code_git` must be the checked-out HEAD exactly; "no-git" is refused (it
    used to be accepted even when a HEAD existed), and so is a run started
    outside a git checkout. `preflight` is the state preflight_source returned
    before the compute; if given, the source must not have changed since.
    """
    if not record.inputs.config_cid:
        raise LedgerError("a run without a config_cid is not startable")
    state = source_state()
    if record.inputs.code_git != state["head"]:
        raise LedgerError(f"inputs.code_git {record.inputs.code_git} is not the checked-out HEAD {state['head']}")
    if state["dirty"]:
        policy = dirty_source or os.environ.get("LEMMA_DIRTY_SOURCE", "reject")
        if policy != "snapshot":
            raise DirtySourceError(
                "source differs from the recorded commit; commit first, or seal the exact bytes with "
                f"LEMMA_DIRTY_SOURCE=snapshot: {', '.join(state['changed'])}")
        state = {**state, "snapshot_cid": source_snapshot(state["changed"])}
    if preflight is not None:
        keys = ("head", "dirty", "changed", "snapshot_cid")
        if any(preflight.get(k) != state.get(k) for k in keys):
            raise DirtySourceError(
                "source changed between preflight and start_run (during the compute): "
                f"before {({k: preflight.get(k) for k in keys})}, now {({k: state.get(k) for k in keys})}")
    if state["dirty"]:
        record.inputs.source_snapshot_cid = state["snapshot_cid"]
        record.inputs.code_git_dirty = True
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
    """Return the sealed record for run_id iff its metrics resolve in the store,
    hash to their CID, and name this run (lemma #7)."""
    sealed = [r for r in read_ledger() if r.run_id == run_id and r.status == "completed"]
    if not sealed:
        raise LedgerError(f"run {run_id} has no completed ledger line")
    rec = sealed[-1]
    if not rec.metrics_cid:
        raise LedgerError(f"run {run_id} completed without a metrics_cid — not citable")
    data = ket_get_verified(rec.metrics_cid)  # bytes must resolve AND hash to the CID
    try:
        envelope = json.loads(data)
    except ValueError as e:
        raise LedgerError(f"run {run_id}: metrics {rec.metrics_cid} is not JSON: {e}") from e
    if not isinstance(envelope, dict) or "metrics" not in envelope:
        raise LedgerError(f"run {run_id}: metrics {rec.metrics_cid} is not a metrics envelope")
    if envelope.get("run_id") != run_id:
        raise LedgerError(
            f"run {run_id}: metrics {rec.metrics_cid} belongs to run {envelope.get('run_id')!r}")
    return rec
