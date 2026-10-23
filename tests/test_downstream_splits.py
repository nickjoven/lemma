"""Review of #10, finding 2: S3/S4/S4v2/S5/W0 evaluators (and the trainers
that build on S2) used to call splits.assign() at run time, so the #6 rule
change silently moved rows a sealed checkpoint trained on into its test
split. They now take the split the run trained against (split_binding)."""

import ast
import json
from pathlib import Path

import pytest

from lemma.data import corpus, split_binding, splits
from lemma.schemas import MutantRow, RunInputs, RunRecord, TransitionRow

from test_split_binding import decl, rule_change_rows

SRC = Path(__file__).resolve().parents[1] / "src" / "lemma"


def legacy(run_id="legacy-run"):
    return RunRecord(run_id=run_id, kind="train", stage="S3", started="t",
                     inputs=RunInputs(corpus_cids=[], config_cid="c", code_git="g", mathlib_pin="p"))


@pytest.fixture
def corpus_rows(monkeypatch, tmp_path, ket_store):
    rows = rule_change_rows()
    for r in rows:
        r.docstring = f"doc of {r.name}"
    monkeypatch.setattr(corpus, "iter_rows", lambda *a, **k: iter(rows))
    pins = tmp_path / "pins.yml"
    pins.write_text(json.dumps({"runs": {"legacy-run": {"split_rule": splits.RULE_V1}}}))
    monkeypatch.setattr(split_binding, "PINS_PATH", pins)
    return rows


def moved(rows):
    """names that v2 puts in test but v1 (the rule sealed runs trained under) does not"""
    v1, v2 = splits.assign_v1(rows), splits.assign(rows)
    out = {n for n in v2 if v2[n] == "test" and v1[n] != "test"}
    assert out, "fixture must contain a row the rule change moves into test"
    return out


def test_no_entrypoint_recomputes_the_split():
    """Only build_cache (which writes a new split) and splits itself may call
    an assigner; everything else goes through split_binding."""
    allowed = {"data/splits.py", "data/build_cache.py", "data/split_binding.py"}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "splits"
                    and node.func.attr in {"assign", "assign_v1", "assign_under"}):
                pytest.fail(f"{rel}:{node.lineno} calls splits.{node.func.attr}")


def test_unpinned_legacy_run_is_refused_downstream(corpus_rows):
    from lemma.train_s3 import gloss_pairs
    with pytest.raises(split_binding.SplitBindingError, match="no split-rule pin"):
        gloss_pairs("test", legacy("some-unpinned-run"))


def test_s3_gloss_test_split_is_the_trained_one(corpus_rows):
    from lemma.train_s3 import gloss_pairs
    test, binding = gloss_pairs("test", legacy())
    names = {d[len("doc of "):] for d, _, _ in test}
    assert binding.rule_id == splits.RULE_V1
    assert names == {n for n, s in splits.assign_v1(corpus_rows).items() if s == "test"}
    assert not names & moved(corpus_rows)


def test_s4_mutants_inherit_the_trained_split(corpus_rows, monkeypatch):
    from lemma import train_s4
    muts = [MutantRow(parent_name=r.name, operator="swap_lit_01", mutated_type=f"M[{r.name}]", lock=f"m{r.name}",
                      lock_changed=True, elaborates=True) for r in corpus_rows]
    monkeypatch.setattr(train_s4, "iter_mutants", lambda *a, **k: iter(muts))
    data, binding = train_s4.mutants_by_split(None, legacy())
    test_parents = {t[0][2:-1] for t in data["test"]}
    assert test_parents == {n for n, s in splits.assign_v1(corpus_rows).items() if s == "test"}
    assert not test_parents & moved(corpus_rows)


def test_s4v2_labels_inherit_the_trained_split(corpus_rows, monkeypatch):
    from lemma import train_s4v2
    labelled = [{"text": r.name, "parent": r.name, "op_class": "swap_lit", "label3": "proven", "prover": "p"}
                for r in corpus_rows]
    monkeypatch.setattr(train_s4v2, "iter_attempts", lambda *a, **k: iter([]))
    monkeypatch.setattr(train_s4v2, "iter_mutants", lambda *a, **k: iter([]))
    monkeypatch.setattr(train_s4v2, "join_mutant_attempts", lambda m, a: (labelled, {}))
    data, _, audit, binding = train_s4v2.load_labelled({}, legacy())
    test = {t[0] for t in data["test"]}
    assert test == {n for n, s in splits.assign_v1(corpus_rows).items() if s == "test"}
    assert audit["split"]["split_manifest_binding"] == "pinned_rule_reconstructed"


def test_s5_pairs_and_distractors_use_the_trained_split(corpus_rows, monkeypatch):
    from lemma import train_s5
    monkeypatch.setattr(train_s5, "iter_iff_pairs", lambda: iter([]))
    by_split, rows, binding = train_s5.equivalence_pairs_for(legacy())
    assert binding.assignment == splits.assign_v1(corpus_rows)
    # the dedup pair of lock L: v1 put both members in train, so it is a train pair
    assert any(p[3] == "dedup" and {p[0], p[1]} == {"T[b_first]", "T[a_dup]"} for p in by_split["train"])
    assert not any("a_dup" in p[0] + p[1] for p in by_split["test"])


def test_w0_transitions_inherit_the_trained_split(corpus_rows, monkeypatch):
    from lemma import train_w0
    from lemma.schemas import GoalRow
    trans = [TransitionRow(attempt_id=f"a{i}", demonstrandum=r.name, prover="p", pos=1, kind="rung", tactic="simp",
                           goal_before=f"g{i}", goal_after=[f"h{i}"], outcome="open", heartbeats=10,
                           heartbeat_cap=100, attempt_outcome="x") for i, r in enumerate(corpus_rows)]
    goals = [GoalRow(lock=f"{p}{i}", readable=f"{p}{i}") for i in range(len(corpus_rows)) for p in "gh"]
    monkeypatch.setattr(train_w0, "iter_transitions", lambda *a, **k: iter(trans))
    monkeypatch.setattr(train_w0, "iter_goals", lambda *a, **k: iter(goals))
    data, _, binding = train_w0.load_data({}, legacy())
    test_dem = {corpus_rows[int(r["before"][1:])].name for r in data["test"]}
    assert test_dem == {n for n, s in splits.assign_v1(corpus_rows).items() if s == "test"}
    assert not test_dem & moved(corpus_rows)


def test_new_runs_record_the_certificate_they_inherit(corpus_rows):
    """A downstream trainer records its init run's split; evaluating it later
    resolves by that CID, not by a pin."""
    from lemma.train_s3 import gloss_pairs
    _, at_train = gloss_pairs("train", legacy())
    s3 = legacy("s3-new")
    s3.inputs.split_manifest_cid = at_train.certificate_cid
    _, at_eval = gloss_pairs("test", s3)
    assert at_eval.binding == "train_record_cid" and at_eval.assignment == at_train.assignment


@pytest.mark.parametrize("fname", ["train_s3.py", "train_s4.py", "train_s4v2.py", "train_s5.py", "train_w0.py",
                                   "w0_baselines.py", "evaluate_s2.py", "evaluate_s3.py", "evaluate_s4.py",
                                   "evaluate_s4v2.py", "evaluate_s5.py", "train.py"])
def test_every_run_record_names_its_split(fname):
    tree = ast.parse((SRC / fname).read_text())
    inputs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "RunInputs"]
    assert inputs and all(any(k.arg == "split_manifest_cid" for k in n.keywords) for n in inputs), fname
