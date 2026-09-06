"""Leakage rules: module-area unity, lock-follows-first, mutant-inherits-parent."""

from lemma.data import splits
from lemma.schemas import DeclarationRow, MutantRow


def decl(name, module, lock, deps=()):
    return DeclarationRow(
        name=name, module=module, kind="theorem",
        canonical_type=f"T[{name}]", lock=lock, proof_consts=list(deps),
    )


def test_same_area_same_split():
    rows = [decl(f"d{i}", f"Mathlib.Topology.Sub{i}", f"l{i}") for i in range(20)]
    got = splits.assign(rows)
    assert len(set(got.values())) == 1


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
