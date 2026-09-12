"""W0 pre-training gates on synthetic rows: displacement rows, censored energy,
offset baseline beats identity when the tactic is a translation."""

import numpy as np

from lemma.schemas import TransitionRow
from lemma.w0_baselines import energy_targets, mrr, offsets_by_tactic, single_step_rows


def tr(**kw):
    base = dict(attempt_id="r:1", demonstrandum="p", prover="ladder-A", pos=0, kind="rung", tactic="simp",
                goal_before="G0", goal_after=["G1"], outcome="open", heartbeats=5, heartbeat_cap=10,
                attempt_outcome="no_proof_found")
    base.update(kw)
    return TransitionRow(**base)


def test_single_step_rows_exclude_identity_multi_and_closed():
    rows = [tr(), tr(goal_after=["G0"]), tr(goal_after=["G1", "G2"]), tr(outcome="closed", goal_after="closed"), tr(kind="intros", goal_after=["G0"])]
    assert len(single_step_rows(rows)) == 1


def test_energy_targets_censor_budget_and_count_only_rungs():
    rows = [tr(outcome="closed", goal_after="closed", heartbeats=3), tr(outcome="error", goal_after="error"),
            tr(outcome="budget", goal_after="error", censored=True, heartbeats=10), tr(kind="intros", goal_after=["G0"])]
    e = energy_targets(rows)["G0"]
    assert e["rungs"] == 3 and e["closed"] == 1 and e["censored"] == 1
    assert abs(e["closure_rate"] - 1 / 3) < 1e-4 and abs(e["energy"] + np.log(1 / 3)) < 1e-3
    assert e["cost_mean"] == 4.0                      # the censored 10 is not a cost
    assert energy_targets([tr(outcome="error", goal_after="error")])["G0"]["energy"] is None


def test_offset_baseline_beats_identity_on_a_translation():
    rng = np.random.default_rng(0)
    zb = rng.normal(size=(50, 8)); mu = np.array([3.0] + [0] * 7)
    za = zb + mu + rng.normal(scale=0.05, size=zb.shape)
    off = offsets_by_tactic(zb, za, ["t"] * 50)["t"]
    assert off["n"] == 50 and off["spread"] < 0.1 and abs(off["mu_norm"] - 3.0) < 0.2
    tgt = np.arange(50)
    assert mrr(zb + off["mu"], za, tgt) > mrr(zb, za, tgt)
