"""Review of #10, findings 1, 2 and 7: an evaluator (or a downstream trainer)
takes the split a checkpoint trained against, by evidence, never today's
rule; a run with no recorded split needs a pinned rule; the recorded split is
a small certificate, not the 15-20 MB manifest."""

import ast
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from lemma import ledger
from lemma.data import split_binding, splits
from lemma.schemas import DeclarationRow, RunInputs, RunRecord

from corpus_fixture import SOURCE, write_corpus

REAL_REPO = Path(__file__).resolve().parents[1]
# the identity of an in-memory corpus, for tests that hand resolve() rows directly
IDENT = {"corpus": "declarations", "source_manifest_cid": SOURCE, "shards": 1, "shards_blake3": "fixture-shards"}
V1_PIN = {"split_rule": splits.RULE_V1, "declarations_shards_blake3": "fixture-shards"}


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
                     inputs=RunInputs(corpus_cids=[SOURCE], config_cid="c", code_git="g", mathlib_pin="p",
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


# the splits.py blobs every sealed run's code_git holds (configs/split_pins.yml):
# 524a2b44 is the pre-#10 file (c283a57..bfc9556), b0d106ca only adds assign_transitions
V1_BLOBS = ("524a2b447f383761d70f120dc673c05b662adb6e", "b0d106ca5b607aed8c52655927cf61cf850a1bf5")


def git_object(spec: str) -> str:
    """`git cat-file -p spec` in this repo. Missing evidence FAILS in a full
    clone; only a shallow clone (which cannot hold history) skips."""
    got = subprocess.run(["git", "-C", str(REAL_REPO), "cat-file", "-p", spec], capture_output=True, text=True)
    if got.returncode == 0:
        return got.stdout
    shallow = subprocess.run(["git", "-C", str(REAL_REPO), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True).stdout.strip()
    if shallow == "true":
        pytest.skip(f"{spec} is not in this shallow clone (CI fetches full history)")
    pytest.fail(f"{spec} is missing from a full clone: the pin evidence it names cannot be checked")


def _body(tree, name):
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) else fn.body
    return [ast.dump(n) for n in body]


def _const(tree, name):
    return next(ast.dump(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


@pytest.mark.parametrize("blob", V1_BLOBS)
def test_assign_v1_is_the_pinned_blobs_assign_ast_equal_and_on_fixtures(blob):
    """splits.assign_v1 is the assign() of the pinned blobs: the same function
    body (AST-equal, docstring aside), the same split_of_area / split_key /
    SPLITS / WEIGHTS it depends on, and the same output on fixtures in several
    orders. Pinned to the blobs, not to `main`, whose assign() is the v2 rule
    after #10 merges."""
    old_src = git_object(blob)
    old, new = ast.parse(old_src), ast.parse(Path(splits.__file__).read_text())
    assert _body(old, "assign") == _body(new, "assign_v1")
    for name in ("split_of_area", "split_key"):
        assert _body(old, name) == _body(new, name), name
    for name in ("SPLITS", "WEIGHTS"):
        assert _const(old, name) == _const(new, name), name
    ns: dict = {}
    exec(compile(old_src.replace("from ..schemas", "from lemma.schemas"), f"splits@{blob[:8]}", "exec"), ns)
    first, second = swap_rows()
    for order in (rule_change_rows(), rule_change_rows()[::-1], first + second, second + first):
        assert splits.assign_v1(order) == ns["assign"](order)


def test_certificate_detects_equal_counts_with_different_names():
    """The per-split name digests, not only the counts, must match: two
    assignments with identical counts that differ by a swap are told apart."""
    rows = rule_change_rows()
    a = splits.assign(rows)
    t = next(n for n, v in a.items() if v == "test")
    r = next(n for n, v in a.items() if v == "train")
    b = {**a, t: "train", r: "test"}
    assert counts(a) == counts(b) and a != b
    why = splits.certificate_mismatch(splits.certificate(a, splits.RULE_V2), b)
    assert why is not None and "name list" in why


def test_unpinned_run_without_a_recorded_split_is_refused(tmp_path, ket_store):
    with pytest.raises(split_binding.SplitBindingError, match="no split-rule pin"):
        split_binding.resolve(run(), corpus=IDENT, rows=rule_change_rows(), pins_path=pins(tmp_path, {}))


def test_pinned_legacy_run_gets_its_old_split_not_todays(tmp_path, ket_store):
    rows = rule_change_rows()
    b = split_binding.resolve(run(), corpus=IDENT, rows=rows, pins_path=pins(tmp_path, {"s2-legacy": V1_PIN}))
    assert b.assignment == splits.assign_v1(rows) != splits.assign(rows)
    assert b.assignment["a_dup"] == "train"           # trained on: never in its test split
    assert b.binding == "pinned_rule_reconstructed" and b.rule_id == splits.RULE_V1
    cert = json.loads(ledger.ket_get_verified(b.certificate_cid))
    assert splits.certificate_mismatch(cert, splits.assign_v1(rows)) is None


def test_explicit_rule_must_agree_with_the_pin(tmp_path, ket_store):
    path = pins(tmp_path, {"s2-legacy": V1_PIN})
    with pytest.raises(split_binding.SplitBindingError, match="contradicts the pin"):
        split_binding.resolve(run(), corpus=IDENT, rows=rule_change_rows(), expect_rule=splits.RULE_V2, pins_path=path)
    b = split_binding.resolve(run(), corpus=IDENT, rows=rule_change_rows(), expect_rule=splits.RULE_V2, pins_path=pins(tmp_path, {}))
    assert b.pin_source == "explicit"
    # v1 depends on row order, which an explicit rule cannot pin
    with pytest.raises(split_binding.SplitBindingError, match="depends on row order"):
        split_binding.resolve(run(), corpus=IDENT, rows=rule_change_rows(), expect_rule=splits.RULE_V1,
                              pins_path=pins(tmp_path, {}))


def test_recorded_certificate_is_verified_against_the_rebuild(tmp_path, ket_store):
    rows = rule_change_rows()
    cid = split_binding.put_certificate(splits.certificate(splits.assign(rows), splits.RULE_V2))
    b = split_binding.resolve(run(split_manifest_cid=cid), corpus=IDENT, rows=rows, pins_path=pins(tmp_path, {}))
    assert b.binding == "train_record_cid" and b.assignment == splits.assign(rows)
    with pytest.raises(split_binding.SplitBindingError, match="trained under"):
        split_binding.resolve(run(split_manifest_cid=cid), corpus=IDENT, rows=rows, expect_rule=splits.RULE_V1)
    # a corpus that changed since training cannot reproduce the recorded split
    with pytest.raises(split_binding.SplitBindingError, match="not the one it trained against"):
        split_binding.resolve(run(split_manifest_cid=cid), corpus=IDENT, rows=rows + [decl("new", "Mathlib.New", "n")])


def test_certificate_claiming_the_wrong_rule_is_refused(ket_store):
    rows = rule_change_rows()
    forged = splits.certificate(splits.assign_v1(rows), splits.RULE_V2)   # v1 split labelled v2
    cid = split_binding.put_certificate(forged)
    with pytest.raises(split_binding.SplitBindingError, match="not the one it trained against"):
        split_binding.resolve(run(split_manifest_cid=cid), corpus=IDENT, rows=rows)


def test_a_full_manifest_is_not_a_certificate(ket_store):
    rows = rule_change_rows()
    cid = ledger.ket_put_json(splits.manifest(rows))
    with pytest.raises(split_binding.SplitBindingError, match="not a lemma-split-cert"):
        split_binding.resolve(run(split_manifest_cid=cid), corpus=IDENT, rows=rows)


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
    """Every pin's evidence is re-checked against history: its code_git is the
    code_git its train record sealed, the splits.py blob at that commit is the
    pinned v1 blob, and the shard order and corpus source at that commit are
    the pinned ones. A commit missing from a full clone fails."""
    from lemma.data import corpus
    entries = split_binding.load_pins()
    lines = [json.loads(l) for l in (REAL_REPO / "runs" / "ledger.jsonl").read_text().splitlines() if l.strip()]
    trained = {r["run_id"]: r for r in lines if r["kind"] == "train" and r["status"] == "completed"}
    assert set(trained) <= set(entries), f"sealed train runs without a pin: {sorted(set(trained) - set(entries))}"
    assert set(entries) <= set(trained), f"pins for runs with no completed train record: {sorted(set(entries) - set(trained))}"
    for run_id, pin in entries.items():
        rec = trained[run_id]
        assert pin["split_rule"] == splits.RULE_V1 and pin["splits_py_blob"] in V1_BLOBS, run_id
        assert pin["code_git"] == rec["inputs"]["code_git"], f"{run_id}: pin code_git is not the sealed one"
        blob = subprocess.run(["git", "-C", str(REAL_REPO), "rev-parse", f"{pin['code_git']}:src/lemma/data/splits.py"],
                              capture_output=True, text=True).stdout.strip()
        if not blob:
            git_object(pin["code_git"])     # fails in a full clone, skips in a shallow one
        assert blob == pin["splits_py_blob"], run_id
        man = yaml.safe_load(git_object(f"{pin['code_git']}:corpora/MANIFEST.yml"))
        assert corpus.shard_order_blake3(man["corpora"]["declarations"]) == pin["declarations_shards_blake3"], run_id
        assert man["source"]["manifest_cid"] == rec["inputs"]["corpus_cids"][0], run_id


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
    b = split_binding.resolve(run(split_manifest_cid=cid), corpus=IDENT, rows=rows)
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


# review of #10, second round, finding 1: v1 depends on row order, so a pinned
# v1 rebuild is only the trained split over the recorded corpus in the order
# the build read it. These go through resolve_corpus, as every entrypoint does.

def swap_rows():
    """Two shards whose order flips which side of two shared locks is seen
    first, so v1 moves four rows and the per-split counts stay identical."""
    mods = [f"Mathlib.Z{i}" for i in range(400)]
    tr = next(m for m in mods if splits.split_of_area(m) == "train")
    te = next(m for m in mods if splits.split_of_area(m) == "test")
    first = [decl("t1", tr, "L1"), decl("s2", te, "L2")]
    second = [decl("s1", te, "L1"), decl("t2", tr, "L2")]
    pad = [decl(f"p{i}", f"Mathlib.P{i}", f"p{i}") for i in range(20)]
    return first + pad, second


def counts(a):
    return {s: sum(v == s for v in a.values()) for s in splits.SPLITS}


def test_pinned_v1_over_the_recorded_corpus_resolves(tmp_path, monkeypatch, ket_store):
    rows = rule_change_rows()
    ident = write_corpus(tmp_path / "corpora", monkeypatch, [rows[:3], rows[3:]])
    pin = {"split_rule": splits.RULE_V1, "declarations_shards_blake3": ident["shards_blake3"]}
    monkeypatch.setattr(split_binding, "PINS_PATH", pins(tmp_path, {"s2-legacy": pin}))
    got, b = split_binding.resolve_corpus(run())
    assert [r.name for r in got] == [r.name for r in rows]
    assert b.assignment == splits.assign_v1(rows) and b.binding == "pinned_rule_reconstructed"


def test_reversed_row_order_under_a_pinned_v1_run_is_refused(tmp_path, monkeypatch, ket_store):
    rows = rule_change_rows()
    assert splits.assign_v1(rows[::-1]) != splits.assign_v1(rows), "fixture: reversing must change the v1 split"
    ident = write_corpus(tmp_path / "a", monkeypatch, [rows])
    pin = {"split_rule": splits.RULE_V1, "declarations_shards_blake3": ident["shards_blake3"]}
    monkeypatch.setattr(split_binding, "PINS_PATH", pins(tmp_path, {"s2-legacy": pin}))
    write_corpus(tmp_path / "b", monkeypatch, [rows[::-1]])     # same rows, same source CID, reversed
    with pytest.raises(split_binding.SplitBindingError, match="shard order"):
        split_binding.resolve_corpus(run())


def test_swapped_lock_order_corpus_with_identical_counts_is_refused(tmp_path, monkeypatch, ket_store):
    first, second = swap_rows()
    a, b = splits.assign_v1(first + second), splits.assign_v1(second + first)
    assert a != b and counts(a) == counts(b), "fixture: a cache report's counts cannot tell these apart"
    ident = write_corpus(tmp_path / "a", monkeypatch, [first, second])
    pin = {"split_rule": splits.RULE_V1, "declarations_shards_blake3": ident["shards_blake3"]}
    monkeypatch.setattr(split_binding, "PINS_PATH", pins(tmp_path, {"s2-legacy": pin}))
    assert split_binding.resolve_corpus(run())[1].assignment == a
    write_corpus(tmp_path / "b", monkeypatch, [first, second], order=[1, 0])   # every shard still verifies
    with pytest.raises(split_binding.SplitBindingError, match="shard order"):
        split_binding.resolve_corpus(run())


def test_a_corpus_other_than_the_recorded_one_is_refused(tmp_path, monkeypatch, ket_store):
    rows = rule_change_rows()
    cid = split_binding.put_certificate(splits.certificate(splits.assign(rows), splits.RULE_V2))
    write_corpus(tmp_path / "c", monkeypatch, [rows], source="another-quod-run")
    monkeypatch.setattr(split_binding, "PINS_PATH", pins(tmp_path, {}))
    for r in (run(split_manifest_cid=cid), run()):
        with pytest.raises(split_binding.SplitBindingError, match="different corpus"):
            split_binding.resolve_corpus(r)


def test_a_shard_that_does_not_verify_is_refused(tmp_path, monkeypatch, ket_store):
    from lemma.data.corpus import CorpusError
    rows = rule_change_rows()
    ident = write_corpus(tmp_path / "d", monkeypatch, [rows])
    pin = {"split_rule": splits.RULE_V1, "declarations_shards_blake3": ident["shards_blake3"]}
    monkeypatch.setattr(split_binding, "PINS_PATH", pins(tmp_path, {"s2-legacy": pin}))
    shard = tmp_path / "d" / "declarations" / "declarations-0000.jsonl"
    shard.write_text("".join(r.model_dump_json() + "\n" for r in rows[::-1]))   # edited under its old CID
    with pytest.raises(CorpusError, match="content hash"):
        split_binding.resolve_corpus(run())
