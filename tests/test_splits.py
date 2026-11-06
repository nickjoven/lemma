"""Leakage rules: module unity, shared locks never cross a split (both at once,
for any row order: lemma #6), mutant-inherits-parent."""

from lemma.data import splits
from lemma.schemas import DeclarationRow, MutantRow


def decl(name, module, lock, deps=()):
    return DeclarationRow(
        name=name, module=module, kind="theorem",
        canonical_type=f"T[{name}]", lock=lock, proof_consts=list(deps),
    )


def test_same_module_same_split():
    # all declarations in ONE module land in one split (same-file leakage guard)
    rows = [decl(f"d{i}", "Mathlib.Topology.Basic", f"l{i}") for i in range(20)]
    got = splits.assign(rows)
    assert len(set(got.values())) == 1


def test_many_modules_populate_all_splits():
    # module-level (not area-level) granularity fills train/val/test, not lumpy
    rows = [decl(f"d{i}", f"Mathlib.Area.Mod{i}", f"l{i}") for i in range(3000)]
    got = splits.assign(rows)
    assert set(got.values()) == set(splits.SPLITS)


def test_shared_lock_never_crosses_split():
    rows = [decl("a", "Mathlib.Order.Basic", "samelock"),
            decl("b", "Mathlib.CategoryTheory.Limits", "samelock")]
    got = splits.assign(rows)
    assert got["a"] == got["b"]


def test_mutant_inherits_parent_split():
    rows = [decl("parent", "Mathlib.Algebra.Group", "lk")]
    ds = splits.assign(rows)
    muts = [MutantRow(parent_name="parent", operator="hyp_del", mutated_type="T'",
                      lock="lk2", lock_changed=True),
            MutantRow(parent_name="unknown", operator="hyp_del", mutated_type="T''",
                      lock="lk3", lock_changed=True)]
    got = splits.assign_mutants(muts, ds)
    assert got[0] == ds["parent"]
    assert 1 not in got  # unknown parent: excluded, never guessed


def test_assignment_is_deterministic():
    rows = [decl(f"d{i}", f"Mathlib.Area{i}.M", f"l{i}") for i in range(50)]
    assert splits.assign(rows) == splits.assign(list(rows))


def test_manifest_discloses_dep_overlap():
    rows = [decl(f"d{i}", f"Mathlib.Area{i}.M", f"l{i}", deps=["Nat.succ"]) for i in range(50)]
    m = splits.manifest(rows)
    assert "train_test_dep_jaccard" in m
    assert set(m["counts"]) <= {"train", "val", "test"}


def _both_invariants(rows, got):
    by_module, by_lock = {}, {}
    for r in rows:
        assert by_module.setdefault(r.module, got[r.name]) == got[r.name], r.module
        assert by_lock.setdefault(r.lock, got[r.name]) == got[r.name], r.lock


def test_issue6_reproducer_keeps_module_and_lock_unity_in_any_order():
    rows = [decl("a", "Mathlib.Review.M0", "shared"),
            decl("b", "Mathlib.Review.M60", "shared"),
            decl("c", "Mathlib.Review.M60", "unique")]
    forward, backward = splits.assign(rows), splits.assign(rows[::-1])
    assert forward == backward
    _both_invariants(rows, forward)


def test_transitive_shared_locks_join_three_modules_for_any_permutation():
    import itertools
    rows = [decl("a1", "Mathlib.A", "l-ab"), decl("b1", "Mathlib.B", "l-ab"),
            decl("b2", "Mathlib.B", "l-bc"), decl("c1", "Mathlib.C", "l-bc"),
            decl("c2", "Mathlib.C", "l-c"), decl("d1", "Mathlib.D", "l-d")]
    reference = splits.assign(rows)
    assert len({reference[n] for n in ("a1", "b1", "b2", "c1", "c2")}) == 1
    for perm in itertools.permutations(rows):
        got = splits.assign(list(perm))
        assert got == reference
        _both_invariants(rows, got)


def test_unjoined_modules_keep_their_own_hash_split():
    # a module that shares no lock with another module is assigned exactly as before
    rows = [decl(f"d{i}", f"Mathlib.Solo{i}", f"l{i}") for i in range(200)]
    got = splits.assign(rows)
    assert all(got[f"d{i}"] == splits.split_of_area(f"Mathlib.Solo{i}") for i in range(200))


def test_manifest_reports_component_sizes():
    rows = [decl("a", "Mathlib.A", "shared"), decl("b", "Mathlib.B", "shared"),
            decl("c", "Mathlib.C", "own")]
    m = splits.manifest(rows)
    assert m["components"] == {
        "modules": 3, "components": 2, "multi_module_components": 1,
        "modules_in_multi_module_components": 2, "largest_component_modules": 2,
        "largest_component_rows": 2,
    }


# --- RULE_V3 (OPEN.yml L-12, owner decision 2026-11-05): only theorem locks join modules ---

def row(name, module, lock, kind):
    return DeclarationRow(name=name, module=module, kind=kind,
                          canonical_type=f"T[{name}]", lock=lock, proof_consts=[])


def _modules_in_different_splits(n=400):
    """two module names whose own hashes land in different splits"""
    mods = [f"Mathlib.M{i}" for i in range(n)]
    by = {}
    for m in mods:
        by.setdefault(splits.split_of_area(m), m)
    return by["train"], by["test"]


def test_v3_signature_lock_does_not_join_modules():
    a, b = _modules_in_different_splits()
    rows = [row("f", a, "sig", "def"), row("g", b, "sig", "def")]
    got = splits.assign(rows)
    assert got["f"] == "train" and got["g"] == "test"          # each module keeps its own split
    assert splits.assign_v2(rows)["f"] == splits.assign_v2(rows)["g"]   # v2 joined them


def test_v3_theorem_lock_still_joins_modules():
    a, b = _modules_in_different_splits()
    rows = [row("t1", a, "stmt", "theorem"), row("t2", b, "stmt", "theorem")]
    got = splits.assign(rows)
    assert got["t1"] == got["t2"]


def test_v3_mixed_kind_lock_does_not_join():
    # a lock carried by a theorem and a def is a coincidence of types, not a duplicate statement
    a, b = _modules_in_different_splits()
    rows = [row("t", a, "mixed", "theorem"), row("d", b, "mixed", "def")]
    got = splits.assign(rows)
    assert got["t"] != got["d"]
    assert splits.joining_locks(rows, splits.RULE_V3) == set()


def test_v3_is_current_and_discloses_crossing_signatures():
    a, b = _modules_in_different_splits()
    rows = [row("f", a, "sig", "def"), row("g", b, "sig", "def"), row("t", a, "s1", "theorem")]
    assert splits.CURRENT_RULE == splits.RULE_V3
    man = splits.manifest(rows)
    assert man["rule_id"] == splits.RULE_V3
    assert man["signature_locks_across_splits"] == {"locks": 1, "rows_outside_train": 1}
    assert splits.rule_id(man["rule"]) == splits.RULE_V3


def test_v3_hub_signature_cannot_build_a_giant_component():
    # one parser-descriptor-like signature shared by 50 modules: v2 makes one component, v3 keeps 50
    rows = [row(f"p{i}", f"Mathlib.P{i}", "Lean.ParserDescr", "def") for i in range(50)]
    rows += [row(f"t{i}", f"Mathlib.P{i}", f"stmt{i}", "theorem") for i in range(50)]
    assert splits.component_stats(rows, splits.RULE_V2)["largest_component_modules"] == 50
    assert splits.component_stats(rows, splits.RULE_V3)["largest_component_modules"] == 1
    splits.component_guard(rows, splits.RULE_V3)
