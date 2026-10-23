"""Review of #10, finding 2: S3/S4/S4v2/S5/W0 evaluators (and the trainers
that build on S2) used to call splits.assign() at run time, so the #6 rule
change silently moved rows a sealed checkpoint trained on into its test
split. They now take the split the run trained against (split_binding)."""

import ast
import json
from pathlib import Path

import pytest

from lemma.data import split_binding, splits
from lemma.schemas import MutantRow, RunInputs, RunRecord, TransitionRow

from corpus_fixture import SOURCE, write_corpus
from test_split_binding import rule_change_rows

SRC = Path(__file__).resolve().parents[1] / "src" / "lemma"


def legacy(run_id="legacy-run"):
    return RunRecord(run_id=run_id, kind="train", stage="S3", started="t",
                     inputs=RunInputs(corpus_cids=[SOURCE], config_cid="c", code_git="g", mathlib_pin="p"))


@pytest.fixture
def corpus_rows(monkeypatch, tmp_path, ket_store):
    rows = rule_change_rows()
    for r in rows:
        r.docstring = f"doc of {r.name}"
    ident = write_corpus(tmp_path / "corpora", monkeypatch, [rows[:3], rows[3:]])
    pins = tmp_path / "pins.yml"
    pins.write_text(json.dumps({"runs": {"legacy-run": {"split_rule": splits.RULE_V1,
                                                        "declarations_shards_blake3": ident["shards_blake3"]}}}))
    monkeypatch.setattr(split_binding, "PINS_PATH", pins)
    return rows


def moved(rows):
    """names that v2 puts in test but v1 (the rule sealed runs trained under) does not"""
    v1, v2 = splits.assign_v1(rows), splits.assign(rows)
    out = {n for n in v2 if v2[n] == "test" and v1[n] != "test"}
    assert out, "fixture must contain a row the rule change moves into test"
    return out


ASSIGNERS = {"assign", "assign_v1", "assign_under", "split_of_area", "_components", "ASSIGNERS"}


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def split_violations(tree) -> list[str]:
    """Any route to an assigner outside splits/build_cache/split_binding:
    splits.<assigner> under any alias of the splits module, a from-import
    of an assigner, getattr by string, or split_binding.resolve() called
    directly (it trusts its caller that the rows are the recorded corpus;
    entrypoints use resolve_corpus)."""
    splits_aliases, binding_aliases, errs = {"splits"}, {"split_binding"}, []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mod = getattr(n, "module", None) or ""
            for a in n.names:
                if a.name == "splits" or a.name.endswith(".splits"):
                    splits_aliases.add(a.asname or a.name)
                if a.name == "split_binding" or a.name.endswith(".split_binding"):
                    binding_aliases.add(a.asname or a.name)
                if isinstance(n, ast.ImportFrom) and mod.split(".")[-1] == "splits" and (
                        a.name in ASSIGNERS or a.name == "*"):
                    errs.append(f"{n.lineno}: from {mod} import {a.name}")
                if isinstance(n, ast.ImportFrom) and mod.split(".")[-1] == "split_binding" and a.name in ("resolve", "*"):
                    errs.append(f"{n.lineno}: from {mod} import {a.name}")
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in ASSIGNERS - {"ASSIGNERS"} and _dotted(n.value) in splits_aliases:
            errs.append(f"{n.lineno}: {_dotted(n)}")
        # the ASSIGNERS table may be listed (argparse choices), not indexed or walked
        if (isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(n.value, ast.Attribute)
                and n.value.attr == "ASSIGNERS" and _dotted(n.value.value) in splits_aliases):
            errs.append(f"{n.lineno}: {_dotted(n.value)} indexed")
        if isinstance(n, ast.Attribute) and n.attr == "resolve" and _dotted(n.value) in binding_aliases:
            errs.append(f"{n.lineno}: {_dotted(n)} (use resolve_corpus)")
        if isinstance(n, ast.Constant) and n.value in ASSIGNERS - {"assign"}:
            errs.append(f"{n.lineno}: {n.value!r} by string")
        if (isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr" and len(n.args) > 1
                and isinstance(n.args[1], ast.Constant) and n.args[1].value in ASSIGNERS | {"resolve"}):
            errs.append(f"{n.lineno}: getattr(..., {n.args[1].value!r})")
    return errs


def test_no_entrypoint_recomputes_the_split():
    """Only build_cache (which writes a new split) and splits itself may call
    an assigner; everything else goes through split_binding.resolve_corpus."""
    allowed = {"data/splits.py", "data/build_cache.py", "data/split_binding.py"}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel not in allowed:
            assert split_violations(ast.parse(path.read_text())) == [], rel


@pytest.mark.parametrize("snippet", [
    "from .data.splits import assign\nassign(rows)\n",
    "from .data.splits import assign as a\na(rows)\n",
    "from .data import splits as sp\nsp.assign_v1(rows)\n",
    "import lemma.data.splits\nlemma.data.splits.assign_under(r, rows)\n",
    "from .data import splits\nsplits.ASSIGNERS[r](rows)\n",
    "from .data import splits\ngetattr(splits, 'assign')(rows)\n",
    "from .data import splits\nsplits.ASSIGNERS.get(r)(rows)\n",
    "from .data import split_binding\nsplit_binding.resolve(run, rows, corpus={})\n",
    "from .data.split_binding import resolve\n",
])
def test_routes_around_split_binding_are_caught(snippet):
    assert split_violations(ast.parse(snippet))


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


def test_new_runs_record_the_certificate_they_inherit_and_its_weakest_link(corpus_rows):
    """A downstream trainer records its init run's split; evaluating it later
    resolves by that CID, not by a pin, but the binding it reports is the
    weakest link of the chain: the init run's split was only reconstructed
    under a pin, so the downstream eval says so and names the pinned run
    (second review of #10: it used to report train_record_cid)."""
    from lemma import ledger
    from lemma.train_s3 import gloss_pairs
    _, at_train = gloss_pairs("train", legacy())
    assert at_train.binding == "pinned_rule_reconstructed" and at_train.pinned_run == "legacy-run"
    s3 = legacy("s3-new")
    s3.inputs.split_manifest_cid = at_train.certificate_cid
    _, at_eval = gloss_pairs("test", s3)
    assert at_eval.assignment == at_train.assignment and at_eval.certificate_cid == at_train.certificate_cid
    assert at_eval.binding == "pinned_rule_reconstructed"
    assert at_eval.disclosure()["split_pinned_run"] == "legacy-run"
    assert at_eval.pin_source == at_train.pin_source == "configs/pins.yml"
    # the provenance is in the certificate's own bytes, so it survives any number of generations
    cert = json.loads(ledger.ket_get_verified(at_train.certificate_cid))
    assert cert["provenance"]["binding"] == "pinned_rule_reconstructed"
    s4 = legacy("s4-on-s3")
    s4.inputs.split_manifest_cid = at_eval.certificate_cid
    assert gloss_pairs("test", s4)[1].binding == "pinned_rule_reconstructed"


def test_a_certificate_with_unknown_provenance_is_refused(corpus_rows):
    from lemma.train_s3 import gloss_pairs
    cert = splits.certificate(splits.assign(corpus_rows), splits.RULE_V2)
    cert["provenance"] = {"binding": "train_record_cid"}     # a provenance claiming the strong binding
    run = legacy("forged")
    run.inputs.split_manifest_cid = split_binding.put_certificate(cert)
    with pytest.raises(split_binding.SplitBindingError, match="unknown provenance"):
        gloss_pairs("test", run)


@pytest.mark.parametrize("fname", ["train_s3.py", "train_s4.py", "train_s4v2.py", "train_s5.py", "train_w0.py",
                                   "w0_baselines.py", "evaluate_s2.py", "evaluate_s3.py", "evaluate_s4.py",
                                   "evaluate_s4v2.py", "evaluate_s5.py"])
def test_every_split_binding_run_records_the_disclosure(fname):
    """The binding kind and pin source go into the sealed metrics of every run
    that resolves a split, trainers included."""
    tree = ast.parse((SRC / fname).read_text())
    assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "disclosure" for n in ast.walk(tree)), fname


@pytest.mark.parametrize("fname", ["train_s3.py", "train_s4.py", "train_s4v2.py", "train_s5.py", "train_w0.py",
                                   "w0_baselines.py", "evaluate_s2.py", "evaluate_s3.py", "evaluate_s4.py",
                                   "evaluate_s4v2.py", "evaluate_s5.py", "train.py"])
def test_every_run_record_names_its_split(fname):
    tree = ast.parse((SRC / fname).read_text())
    inputs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "RunInputs"]
    assert inputs and all(any(k.arg == "split_manifest_cid" for k in n.keywords) for n in inputs), fname


def test_s5_iff_pairs_land_with_their_modules_declarations():
    """Finding 3: in a multi-module component an Iff pair used its module's
    own hash split while that module's declarations used the component's."""
    from lemma.train_s5 import equivalence_pairs_by_split
    rows = rule_change_rows()
    assign = splits.assign(rows)
    b_module = next(r.module for r in rows if r.name == "b_other")
    assert splits.split_of_area(b_module) == "train" and assign["b_other"] == "test"   # the leak's precondition
    iff = [{"lhs": "P", "rhs": "Q", "module": b_module}, {"lhs": "R", "rhs": "S", "module": "Mathlib.NotInCorpus"}]
    out = equivalence_pairs_by_split(rows, assign, iff, splits.RULE_V2)
    where = {s for s, ps in out.items() for p in ps if p[3] == "iff"}
    assert where == {"test"}, "the Iff pair must share its module's declarations' split"
    assert sum(p[3] == "iff" for ps in out.values() for p in ps) == 1, "a module with no declarations is excluded"
    # a v1 run is reproduced as it trained (module hash), not silently re-split
    v1 = equivalence_pairs_by_split(rows, splits.assign_v1(rows), iff[:1], splits.RULE_V1)
    assert [p[3] for p in v1["train"] if p[3] == "iff"] == ["iff"]


def _called(node) -> set[str]:
    return {getattr(c.func, "attr", getattr(c.func, "id", None)) for c in ast.walk(node) if isinstance(c, ast.Call)}


def test_every_entrypoint_resolves_its_split_before_loading_weights():
    """Second review of #10, finding 7: an unpinned or mismatched run must be
    refused before any checkpoint is read. The split resolvers are found, not
    listed: resolve_corpus and every function that calls it. Trainers too:
    a trainer must refuse an unbindable init run before reading its weights."""
    fns = [fn for p in SRC.rglob("*.py") for fn in ast.walk(ast.parse(p.read_text()))
           if isinstance(fn, ast.FunctionDef)]
    # resolve_corpus and the helpers that call it directly (gloss_pairs, mutants_by_split, ...)
    resolvers = {"resolve_corpus"} | {fn.name for fn in fns if "resolve_corpus" in _called(fn)}
    loads = {"load_file", "load_state_dict", "load_s2_encoder", "load_encoder"}
    checked = 0
    for fn in fns:
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
        res = [c.lineno for c in calls if getattr(c.func, "attr", getattr(c.func, "id", None)) in resolvers]
        lds = [c.lineno for c in calls if getattr(c.func, "attr", getattr(c.func, "id", None)) in loads]
        if res and lds:
            checked += 1
            assert min(res) < min(lds), f"{fn.name}: loads weights (line {min(lds)}) before resolving its split"
    assert checked >= 10, f"only {checked} entrypoints found; the discovery is broken"
