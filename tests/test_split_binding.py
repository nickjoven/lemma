"""Review of #10, findings 1, 2 and 7: an evaluator (or a downstream trainer)
takes the split a checkpoint trained against, by evidence, never today's
rule; a run with no recorded split needs a pinned rule; the recorded split is
a small certificate, not the 15-20 MB manifest."""

import json
import subprocess
from pathlib import Path

import pytest

from lemma import ledger
from lemma.data import split_binding, splits
from lemma.schemas import DeclarationRow, RunInputs, RunRecord

REAL_REPO = Path(__file__).resolve().parents[1]


def decl(name, module, lock):
    return DeclarationRow(name=name, module=module, kind="theorem", canonical_type=f"T[{name}]", lock=lock)


def rule_change_rows():
    """Rows whose split differs between v1 and v2 (the reviewer's repro): the
    component key A hashes to test, module B to train, and B's row is read
    first, so under v1 the shared lock follows B into train."""
    mods = [f"Mathlib.X{i}" for i in range(400)]
    te = [m for m in mods if splits.split_of_area(m) == "test"]
    tr = [m for m in mods if splits.split_of_area(m) == "train"]
    a = min(te)
    b = next(m for m in tr if m > a)
    rows = [decl("b_first", b, "L"), decl("a_dup", a, "L"), decl("b_other", b, "own"),
            decl("a2", a, "L2"), decl("a3", a, "L2")]
    for i in range(40):
        rows.append(decl(f"x{i}", f"Mathlib.Y{i}", f"y{i}"))
    return rows


def run(run_id="s2-legacy", split_manifest_cid=None):
    return RunRecord(run_id=run_id, kind="train", stage="S2", started="t",
                     inputs=RunInputs(corpus_cids=[], config_cid="c", code_git="g", mathlib_pin="p",
                                      split_manifest_cid=split_manifest_cid))


def pins(tmp_path, entries):
    path = tmp_path / "split_pins.yml"
    path.write_text(json.dumps({"runs": entries}))   # JSON is YAML
    return path


def test_v1_and_v2_disagree_on_the_fixture():
    rows = rule_change_rows()
    v1, v2 = splits.assign_v1(rows), splits.assign(rows)
    assert v1["a_dup"] == "train" and v2["a_dup"] == "test"
    assert splits.assign_under(splits.RULE_V1, rows) == v1
    assert splits.rule_id("module-area blake3, lock-follows-first, mutant-inherits-parent") == splits.RULE_V1
    with pytest.raises(ValueError, match="unknown split rule"):
        splits.rule_id("fixture")


def test_assign_v1_is_the_pre_10_assign_byte_for_byte():
    # the legacy function must be the one sealed runs trained under (main's splits.py)
    src = subprocess.run(["git", "-C", str(REAL_REPO), "show", "main:src/lemma/data/splits.py"],
                         capture_output=True, text=True)
    if src.returncode != 0:
        pytest.skip("main is not available in this checkout")
    ns: dict = {}
    exec(compile(src.stdout.replace("from ..schemas", "from lemma.schemas"), "main_splits", "exec"), ns)
    rows = rule_change_rows()
    for order in (rows, rows[::-1]):
        assert splits.assign_v1(order) == ns["assign"](order)


def test_unpinned_run_without_a_recorded_split_is_refused(tmp_path, ket_store):
    with pytest.raises(split_binding.SplitBindingError, match="no split-rule pin"):
        split_binding.resolve(run(), rule_change_rows(), pins_path=pins(tmp_path, {}))


def test_pinned_legacy_run_gets_its_old_split_not_todays(tmp_path, ket_store):
    rows = rule_change_rows()
    b = split_binding.resolve(run(), rows, pins_path=pins(tmp_path, {"s2-legacy": {"split_rule": splits.RULE_V1}}))
    assert b.assignment == splits.assign_v1(rows) != splits.assign(rows)
    assert b.assignment["a_dup"] == "train"           # trained on: never in its test split
    assert b.binding == "pinned_rule_reconstructed" and b.rule_id == splits.RULE_V1
    cert = json.loads(ledger.ket_get_verified(b.certificate_cid))
    assert splits.certificate_mismatch(cert, splits.assign_v1(rows)) is None


def test_explicit_rule_must_agree_with_the_pin(tmp_path, ket_store):
    path = pins(tmp_path, {"s2-legacy": {"split_rule": splits.RULE_V1}})
    with pytest.raises(split_binding.SplitBindingError, match="contradicts the pin"):
        split_binding.resolve(run(), rule_change_rows(), expect_rule=splits.RULE_V2, pins_path=path)
    b = split_binding.resolve(run(), rule_change_rows(), expect_rule=splits.RULE_V1, pins_path=pins(tmp_path, {}))
    assert b.pin_source == "explicit"


def test_recorded_certificate_is_verified_against_the_rebuild(tmp_path, ket_store):
    rows = rule_change_rows()
    cid = split_binding.put_certificate(splits.certificate(splits.assign(rows), splits.RULE_V2))
    b = split_binding.resolve(run(split_manifest_cid=cid), rows, pins_path=pins(tmp_path, {}))
    assert b.binding == "train_record_cid" and b.assignment == splits.assign(rows)
    with pytest.raises(split_binding.SplitBindingError, match="trained under"):
        split_binding.resolve(run(split_manifest_cid=cid), rows, expect_rule=splits.RULE_V1)
    # a corpus that changed since training cannot reproduce the recorded split
    with pytest.raises(split_binding.SplitBindingError, match="not the one it trained against"):
        split_binding.resolve(run(split_manifest_cid=cid), rows + [decl("new", "Mathlib.New", "n")])


def test_certificate_claiming_the_wrong_rule_is_refused(ket_store):
    rows = rule_change_rows()
    forged = splits.certificate(splits.assign_v1(rows), splits.RULE_V2)   # v1 split labelled v2
    cid = split_binding.put_certificate(forged)
    with pytest.raises(split_binding.SplitBindingError, match="not the one it trained against"):
        split_binding.resolve(run(split_manifest_cid=cid), rows)


def test_a_full_manifest_is_not_a_certificate(ket_store):
    rows = rule_change_rows()
    cid = ledger.ket_put_json(splits.manifest(rows))
    with pytest.raises(split_binding.SplitBindingError, match="not a lemma-split-cert"):
        split_binding.resolve(run(split_manifest_cid=cid), rows)


def test_certificate_is_small_and_order_free():
    rows = rule_change_rows()
    a = splits.assign(rows)
    c1 = splits.certificate(a, splits.RULE_V2)
    c2 = splits.certificate(dict(reversed(list(a.items()))), splits.RULE_V2)
    assert c1 == c2 and len(json.dumps(c1)) < 600


def test_component_guard_refuses_a_giant_component():
    rows = [decl(f"h{i}", f"Mathlib.Hub{i}", "hub") for i in range(5)]
    rows += [decl(f"s{i}", f"Mathlib.Solo{i}", f"s{i}") for i in range(20)]
    with pytest.raises(ValueError, match="L-12"):
        splits.component_guard(rows)
    assert splits.component_guard(rows, max_fraction=0.5)["largest_component_rows"] == 5


def test_real_pins_name_v1_and_match_code_history():
    entries = split_binding.load_pins()
    sealed = {json.loads(l)["run_id"] for l in (REAL_REPO / "runs" / "ledger.jsonl").read_text().splitlines()
              if l.strip() and json.loads(l)["kind"] == "train" and json.loads(l)["status"] == "completed"}
    assert sealed <= set(entries), f"sealed train runs without a pin: {sorted(sealed - set(entries))}"
    v1_blobs = {"524a2b447f383761d70f120dc673c05b662adb6e", "b0d106ca5b607aed8c52655927cf61cf850a1bf5"}
    for run_id, pin in entries.items():
        assert pin["split_rule"] == splits.RULE_V1 and pin["splits_py_blob"] in v1_blobs
        got = subprocess.run(["git", "-C", str(REAL_REPO), "rev-parse", f"{pin['code_git']}:src/lemma/data/splits.py"],
                             capture_output=True, text=True)
        if got.returncode == 0:
            assert got.stdout.strip() == pin["splits_py_blob"], run_id


def _cache(tmp_path, rows, rule=splits.RULE_V2, counts=None, rule_id=True):
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    man = splits.manifest(rows, rule)
    if not rule_id:
        man.pop("rule_id")
    (cache / "split-manifest.json").write_text(json.dumps(man))
    counts = counts or man["counts"]
    (cache / "cache-report.json").write_text(json.dumps({"splits": {s: {"sequences": counts.get(s, 0)} for s in splits.SPLITS}}))
    return cache


def test_train_records_a_certificate_that_resolves(tmp_path, ket_store):
    from lemma.train import record_cache_split
    rows = rule_change_rows()
    cid = record_cache_split(_cache(tmp_path, rows))
    b = split_binding.resolve(run(split_manifest_cid=cid), rows)
    assert b.assignment == splits.assign(rows)
    assert len(ledger.ket_get(cid)) < 600     # what goes into the git-tracked store


def test_train_refuses_a_cache_it_cannot_bind(tmp_path, ket_store):
    from lemma.train import record_cache_split
    rows = rule_change_rows()
    with pytest.raises(SystemExit, match="no rule_id"):
        record_cache_split(_cache(tmp_path / "a", rows, rule_id=False))
    with pytest.raises(SystemExit, match="differ from the cache report"):
        record_cache_split(_cache(tmp_path / "b", rows, counts={"train": 1}))
    bare = tmp_path / "c" / "cache"
    bare.mkdir(parents=True)
    (bare / "split-manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="cache-report.json is missing"):
        record_cache_split(bare)
