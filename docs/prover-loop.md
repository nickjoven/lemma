# Prover-in-the-loop — step 0 (intake) for review

Status: **design for review, nothing implemented.** This PR is the Phase-0
INTAKE artifact under the four-voice protocol: intent, scope, boundary, exit
conditions, rollback, and the Catalyst queue. Merge = approval to execute
Phase 1 as scoped here; comments change the scope before any code is written.

## Why this, why now

Every remaining item in the ML track is blocked on the same missing capability:
something that *attempts proofs* and hands each attempt to quod's gates.

| Blocked item | What it needs from the loop |
|---|---|
| S4 real target (`gate_verdict`) — OPEN.yml L-2, L-6 | attempt a proof of each mutated statement; the gate's verdict is the label |
| S5 "real test" (≥10 of top-100 proposals discharged) — L-5 | attempt `A ↔ B` for each reranked proposal |
| JEPA proper (plan row 5) and frontier estimation (row 6) | the *attempt ledger* is the on-policy data the plan names |

S4 v1 showed why the current labels are not enough: `elaborates` is lexically
shallow (bag-of-tokens 0.880 vs model 0.941; unseen-operator F1 0.305). The
question "does the mutated claim still hold?" can only be answered by a proof
attempt verified by the gates.

## Intent

A harness that, given a **demonstrandum** (a Lean statement at the crouzeix
pin, identified by its lock), produces zero or more **attempts**, each of which
is verified *only* by quod's existing gates (lock match, axiom triple,
lean4checker) and recorded as a content-addressed **attempt** node. The harness
never declares a verdict; the gates compute it. An attempt that finds no proof
is recorded as `no_proof_found`, which is **not** `refuted` — absence of a
proof is not evidence of falsity, and the ledger must never let the two be
confused.

## Scope (what changes)

1. **quod** gains the attempt harness (it must run inside the pinned Lean env
   and call the gates as they are):
   - `scripts/AttemptWalk.lean` — for each demonstrandum, elaborate the
     statement, run a **bounded tactic ladder**, emit the proof term on success.
   - `scripts/attempt.py` — driver (CorpusWalk pattern: explicit-handle
     output, START markers, per-attempt heartbeat cap, watchdog resume); runs
     the gates on every success; writes `attempts.jsonl` + an append-only
     manifest; ket-puts every proof term and gate output.
2. **lemma** gains `corpus.iter_attempts` and the attempt-derived labels:
   `gate_verdict` on mutants (S4 v2) and discharged/undischarged on proposals
   (S5 real test).
3. An **attempt ledger** entry per attempt (append-only, one line):
   `demonstrandum_lock | prover | prover_config_cid | tactic_script_cid |
   outcome | gate: lock_ok axioms_ok checker_ok | verdict | heartbeats | wall_s`.

## Boundary (what does NOT change)

- No gate is modified. The gates stay quod's calibrated `lock.py`,
  `axiom_gate.py`, `refute_check.py`, lean4checker. If a positive control
  fails, the *harness* is wrong, never the gate (HANDOFF.md rule).
- No existing corpus, tokenizer, or checkpoint is touched or re-pinned.
- No status in quod's claim ledger is written by the harness; attempts are a
  separate ledger. Promoting an attempt to a claim status is a later, gated step.

## The decision this PR asks for: which prover

| Tier | Prover | Pros | Cons | Cost |
|---|---|---|---|---|
| A | **Automation ladder** in-process: `rfl`, `decide`, `simp`, `omega`, `exact?`/`apply?`, `aesop`, each under a heartbeat cap | zero external artifacts; deterministic; cheap; honest baseline | low coverage on real Mathlib statements (most need nontrivial proofs) | ~1–2 days |
| B | **LLM prover** (Claude API) proposing Lean proof scripts, *verified by the gates* | far higher coverage; the proof is checked, so provenance holds regardless of the prover | an external model in the loop; API cost; prover identity/version must be recorded per attempt; it may have seen Mathlib (irrelevant to *verification*, relevant to interpreting "found a proof") | ~1 day + API |
| C | **lean-dojo tactic search** (bounded best-first over tactic states) | is the row-5 substrate; yields state/tactic/next-state traces | heaviest to build; RAM (one Lean env at a time) | ~3–5 days |

**Recommendation: A first, then B as an escalation on A's failures, C deferred
to the JEPA stage.** A gives an honest floor and a positive-control set for
free; B is where the coverage comes from. The plan's doctrine excluded external
*weights inside lemma's models*; a prover whose every output is gate-verified is
a search heuristic, not a trusted artifact — but this is a doctrine call and is
explicitly put to the reviewer here. If B is declined, S4 v2 and the S5 real
test will report automation-tier coverage only, which will be low.

## Exit conditions (Phase 1, falsifiable)

1. **Positive controls:** 200 Mathlib theorems with proofs stripped; the ladder
   must reprove ≥ P% (P to be set by tier: A alone ~20%, A+B ≥ 70%). A control
   the gates accept but the harness mislabels = harness bug.
2. **Negative controls:** 0 accepted attempts on statements known ill-typed
   (`elaborates=false` mutants) or on `sorry`/`axiom` injections — the gates
   must reject every one; any acceptance halts the run.
3. **S4 v2 labels:** `gate_verdict` ∈ {proven, no_proof_found, rejected:*} for
   ≥ 5,000 statement mutants (larger mutant sample, `sample_mod` 46 → ~10).
4. **S5 real test:** top-100 *reranked* proposals (L-5 fix first) each attempted
   as `A ↔ B`; report discharged count against the plan's bar of ≥ 10.
5. Every number above lands only as a sealed ledger record with a metrics_cid.

## Rollback

Pure addition: new scripts, new ledger, new corpora. Reverting = deleting them;
no existing artifact or pin changes.

## Risks

- **Verdict semantics:** the single biggest failure mode is treating
  `no_proof_found` as `refuted`. The attempt schema makes them distinct values
  and the S4 v2 label set is 3-way, never binary.
- **Coverage:** tier A alone may prove too little to train on; hence B.
- **Cost/RAM:** one Lean process at a time (15 GB box); per-attempt heartbeat
  cap; the watchdog/resume machinery from `corpus_extract.py` is reused as-is.
- **Prover leakage into interpretation:** an LLM "reproving" a Mathlib theorem
  it has memorized still yields a *valid* gate-verified proof; it is only the
  *difficulty* signal that is contaminated. The ledger records prover identity so
  downstream stages can condition on it.

## Catalyst queue (not part of this scope)

- The attempt ledger is the JEPA's on-policy data (row 5) and enables frontier
  estimation (row 6) — the moment attempts exist, both become plannable.
- A per-prover *calibration* stage (prover claims vs gate outcomes) generalizes
  the S-stage control pattern to provers themselves.
- Reranked S5 proposals that the loop *refutes* (a proof of `¬(A ↔ B)`) are a
  new negative corpus nobody has today.

## Open questions for the reviewer

1. Tier B (LLM prover) — acceptable under the doctrine, given gate verification?
2. Positive-control pass rate P for the tier you approve.
3. Per-attempt budget (heartbeats / wall seconds) and total attempt budget.
4. Mutant sample size for S4 v2 (proposal: `sample_mod` 10 → ~25k parents).
