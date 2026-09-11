"""Release verifier: a downloaded file is accepted iff a sealed ledger record cites its BLAKE3."""

import blake3

from lemma import ledger, verify_release
from lemma.schemas import RunInputs, RunRecord


def _record(run_id, ckpt_cid, tok_cid="tok-cid", status="completed"):
    return RunRecord(
        run_id=run_id, kind="train", stage="S2", started=ledger.now_iso(), status=status,
        inputs=RunInputs(corpus_cids=[], config_cid="cfg", code_git="g",
                         mathlib_pin="crouzeix:8f9d9cff", tokenizer_cid=tok_cid),
        checkpoint_cids=[ckpt_cid],
    )


def _setup(tmp_path, monkeypatch, data=b"weights", **kw):
    f = tmp_path / "asset.safetensors"
    f.write_bytes(data)
    cid = blake3.blake3(data).hexdigest()
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")
    ledger.append(_record("s2-test", cid, **kw))
    return f, cid


def test_cited_checkpoint_matches(tmp_path, monkeypatch, capsys):
    f, cid = _setup(tmp_path, monkeypatch)
    assert verify_release.main([str(f)]) == 0
    out = capsys.readouterr().out
    assert "s2-test" in out and f"the ledger says {cid}; the file hashes to {cid}" in out


def test_uncited_file_is_rejected(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    other = tmp_path / "other.safetensors"
    other.write_bytes(b"not the weights")
    assert verify_release.main([str(other)]) == 1


def test_run_filter_and_uncompleted_record(tmp_path, monkeypatch):
    f, cid = _setup(tmp_path, monkeypatch)
    assert verify_release.main([str(f), "--run", "s2-test"]) == 0
    assert verify_release.main([str(f), "--run", "some-other-run"]) == 1
    # a running (unsealed) record does not vouch for a file
    data = b"other weights"
    f2 = tmp_path / "unsealed.safetensors"
    f2.write_bytes(data)
    ledger.append(_record("s2-open", blake3.blake3(data).hexdigest(), status="running"))
    assert verify_release.main([str(f2)]) == 1


def test_tokenizer_cid_matches(tmp_path, monkeypatch):
    data = b'{"version": "1.0"}'
    tok = tmp_path / "tok.json"
    tok.write_bytes(data)
    cid = blake3.blake3(data).hexdigest()
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "ledger.jsonl")
    ledger.append(_record("s2-tok", "ckpt-cid", tok_cid=cid))
    assert verify_release.main([str(tok)]) == 0


def test_missing_path_is_usage_error(tmp_path):
    assert verify_release.main([str(tmp_path / "nope")]) == 2
