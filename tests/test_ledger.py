"""S0 doctrine tests: a metric without a metrics_cid does not exist."""

import pytest

from lemma import ledger
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
