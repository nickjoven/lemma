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
    }
