"""Transition corpus doctrine: censored budget rows, progress definition,
split inheritance from the demonstrandum, dedup attempts are proven."""

from lemma.data.splits import assign_transitions
from lemma.schemas import AttemptRow, TransitionRow


def tr(**kw):
    base = dict(attempt_id="r:1", demonstrandum="p", prover="ladder-A", pos=0, kind="rung", tactic="simp",
                goal_before="G0", goal_after="error", outcome="error", heartbeats=5, heartbeat_cap=10,
                attempt_outcome="no_proof_found")
    base.update(kw)
    return TransitionRow(**base)


def test_budget_rows_are_censored_not_costs():
    t = tr(outcome="budget", err_class="budget", censored=True, heartbeats=10)
    assert t.censored and t.heartbeats <= t.heartbeat_cap and not t.progressed


def test_progress_definition():
    assert tr(outcome="closed", goal_after="closed").progressed
    assert tr(outcome="open", goal_after=["G1"]).progressed
    assert not tr(outcome="open", goal_after=["G0"]).progressed      # intros: same proposition
    assert not tr(outcome="error").progressed


def test_transitions_inherit_demonstrandum_split():
    rows = [tr(demonstrandum="a"), tr(demonstrandum="b"), tr(demonstrandum="zzz")]
    assert assign_transitions(rows, {"a": "train", "b": "test"}) == {0: "train", 1: "test"}


def test_dedup_attempt_is_proven_and_source_defaults():
    a = AttemptRow(demonstrandum="p", prover="ladder-A", outcome="accepted", verdict="accepted",
                   mutant_operator="swap_and_or", mutant_lock="L", selfproof_ok=False, dedup=True, dedup_of=["q"])
    assert a.gate_verdict == "proven" and a.source == "ladder" and a.predictor_cid is None
