"""S0 doctrine tests: a metric without a metrics_cid does not exist; cited bytes
must hash to their CID and name their run (lemma #7); a run records the source
that actually ran (lemma #8)."""

import json
import subprocess

import blake3
import pytest

from lemma import ledger
from conftest import git_repo
from lemma.schemas import RunInputs, RunRecord


def record(run_id="t-0"):
    return RunRecord(
        run_id=run_id,
        kind="eval",
        stage="S0",
        started=ledger.now_iso(),
        inputs=RunInputs(
            corpus_cids=[], config_cid="deadbeef", code_git=ledger.git_head(),
            mathlib_pin="crouzeix:8f9d9cff",
        ),
    )


def test_completed_run_without_metrics_is_refused():
    with pytest.raises(ledger.LedgerError):
        ledger.finish_run(record(), metrics={})


def test_config_cid_required_to_start():
    rec = record()
    rec.inputs.config_cid = ""
    with pytest.raises(ledger.LedgerError):
        ledger.start_run(rec)


def test_metrics_round_trip_through_store(tmp_path, monkeypatch):
    import subprocess

    home = tmp_path / ".ket"
    subprocess.run(["ket", "--home", str(home), "init"], check=True, capture_output=True)
    monkeypatch.setattr(ledger, "KET_HOME", home)
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")

    rec = ledger.finish_run(record("t-roundtrip"), metrics={"acc": 0.5})
    assert rec.metrics_cid
    sealed = ledger.citable("t-roundtrip")
    assert sealed.metrics_cid == rec.metrics_cid
    assert b'"acc"' in ledger.ket_get(rec.metrics_cid)


def test_uncompleted_run_is_not_citable(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")
    ledger.start_run(record("t-open"))
    with pytest.raises(ledger.LedgerError):
        ledger.citable("t-open")


def ket_store(tmp_path, monkeypatch):
    home = tmp_path / ".ket"
    subprocess.run(["ket", "--home", str(home), "init"], check=True, capture_output=True)
    monkeypatch.setattr(ledger, "KET_HOME", home)
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")
    return home


def test_tampered_metrics_blob_is_not_citable(tmp_path, monkeypatch):
    home = ket_store(tmp_path, monkeypatch)
    rec = ledger.finish_run(record("t-tamper"), metrics={"accuracy": 0.5})
    blob = home / "cas" / rec.metrics_cid
    assert blob.exists()
    blob.write_bytes(json.dumps({"run_id": "someone-else", "metrics": {"accuracy": 1.0}}).encode())
    with pytest.raises(ledger.LedgerError, match="corrupted"):
        ledger.citable("t-tamper")


def test_metrics_envelope_must_name_the_run(tmp_path, monkeypatch):
    ket_store(tmp_path, monkeypatch)
    rec = ledger.finish_run(record("t-mine"), metrics={"accuracy": 0.5})
    # a well-formed envelope for ANOTHER run, stored under its own honest CID
    other = json.dumps({"run_id": "t-other", "metrics": {"accuracy": 0.9}}, sort_keys=True, indent=1)
    other_cid = ledger.ket_put(other)
    assert other_cid == blake3.blake3(other.encode()).hexdigest()
    rec.metrics_cid = other_cid
    ledger.append(rec)
    with pytest.raises(ledger.LedgerError, match="belongs to run"):
        ledger.citable("t-mine")


def test_malformed_metrics_blob_is_not_citable(tmp_path, monkeypatch):
    ket_store(tmp_path, monkeypatch)
    rec = ledger.finish_run(record("t-json"), metrics={"accuracy": 0.5})
    rec.metrics_cid = ledger.ket_put("not json at all")
    ledger.append(rec)
    with pytest.raises(ledger.LedgerError, match="not JSON"):
        ledger.citable("t-json")


def test_clean_source_starts_and_records_head(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")
    rec = ledger.start_run(record("t-clean"))
    assert rec.inputs.code_git == ledger.source_state()["head"]
    assert rec.inputs.code_git_dirty is None and rec.inputs.source_snapshot_cid is None


def test_modified_evaluator_is_refused_before_a_run(tmp_path, monkeypatch, clean_repo):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")
    (clean_repo / "src" / "evaluator.py").write_text("print('v2')\n")
    with pytest.raises(ledger.DirtySourceError, match="src/evaluator.py"):
        ledger.start_run(record("t-dirty"))


def test_untracked_evaluator_is_refused_before_a_run(tmp_path, monkeypatch, clean_repo):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")
    (clean_repo / "src" / "evaluate_new.py").write_text("print('new')\n")
    with pytest.raises(ledger.DirtySourceError, match="src/evaluate_new.py"):
        ledger.start_run(record("t-untracked"))


def test_code_git_must_be_the_checked_out_head(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")
    rec = record("t-stale")
    rec.inputs.code_git = "0" * 40
    with pytest.raises(ledger.LedgerError, match="not the checked-out HEAD"):
        ledger.start_run(rec)


def test_snapshot_policy_seals_the_exact_changed_bytes(tmp_path, monkeypatch, clean_repo):
    ket_store(tmp_path, monkeypatch)
    (clean_repo / "src" / "evaluator.py").write_text("print('v2')\n")
    (clean_repo / "src" / "evaluate_new.py").write_text("print('new')\n")
    rec = ledger.start_run(record("t-snap"), dirty_source="snapshot")
    assert rec.inputs.code_git_dirty is True
    manifest = json.loads(ledger.ket_get_verified(rec.inputs.source_snapshot_cid))
    assert manifest["base_commit"] == rec.inputs.code_git
    assert set(manifest["files"]) == {"src/evaluator.py", "src/evaluate_new.py"}
    assert ledger.ket_get_verified(manifest["files"]["src/evaluator.py"]) == b"print('v2')\n"
    assert ledger.ket_get_verified(manifest["files"]["src/evaluate_new.py"]) == b"print('new')\n"
