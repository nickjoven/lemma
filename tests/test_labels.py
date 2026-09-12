"""S4 v2 label doctrine: prover-relative three-way verdicts, lock-verified joins,
and the class-support collapse rule."""

from lemma.data.labels import collapse_rule, join_mutant_attempts
from lemma.schemas import AttemptRow, MutantRow


def mut(parent, op, lock, elab=True):
    return MutantRow(parent_name=parent, operator=op, mutated_type=f"T[{parent}/{op}]",
                     lock=lock, lock_changed=True, elaborates=elab, module="Mathlib.X.Y")


def att(parent, op, lock, verdict, outcome="accepted", prover="ladder-A"):
    return AttemptRow(demonstrandum=parent, prover=prover, outcome=outcome, verdict=verdict,
                      mutant_operator=op, mutant_lock=lock)


def test_three_way_is_prover_relative_and_never_refuted():
    a = att("p", "hyp_del_1", "L", verdict=None, outcome="no_proof_found")
    assert a.gate_verdict == "no_proof_found"
    assert "refut" not in a.gate_verdict
    assert att("p", "o", "L", "accepted").gate_verdict == "proven"
    assert att("p", "o", "L", "rejected: self-proof").gate_verdict == "rejected"


def test_join_requires_lock_equality_and_elaborated_mutant():
    mutants = [mut("p", "swap_eq_ne", "L1"), mut("p", "hyp_del_2", "L2", elab=False)]
    attempts = [att("p", "swap_eq_ne", "L1", "accepted"),
                att("p", "swap_eq_ne", "WRONG", "accepted"),      # rebuilt mutant != corpus mutant
                att("p", "hyp_del_2", "L2", "accepted")]          # ill-formed mutant: not a demonstrandum
    rows, audit = join_mutant_attempts(mutants, attempts)
    assert len(rows) == 1 and rows[0]["label3"] == "proven" and rows[0]["prover"] == "ladder-A"
    assert audit["lock_mismatch"] == 1 and audit["no_corpus_mutant"] == 1


def test_negated_attempts_are_not_mutant_labels():
    rows, _ = join_mutant_attempts([mut("p", "o", "L")],
                                   [AttemptRow(demonstrandum="p", prover="x", outcome="no_proof_found",
                                               mutant_operator="o", mutant_lock="L", negated=True)])
    assert rows == []


def test_collapse_rule_by_support():
    rows = [{"label3": "proven"}] * 40 + [{"label3": "no_proof_found"}] * 40 + [{"label3": "rejected"}] * 3
    classes, support = collapse_rule(rows, min_support=30)
    assert classes == ["proven", "not_proven"] and support["rejected"] == 3
    assert {r["label"] for r in rows} == {"proven", "not_proven"}
    rows2 = [{"label3": "rejected"}] * 50 + [{"label3": "proven"}] * 50
    classes2, _ = collapse_rule(rows2, min_support=30)
    assert classes2 == ["proven", "no_proof_found", "rejected"]
