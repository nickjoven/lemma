# Prover-in-the-loop — lemma (consumer) side, step 0, revision 2

Status: **design for review, nothing implemented.** The harness itself —
`AttemptWalk.lean`, `attempt.py`, the attempt ledger, the prover-tier
decision, prover calibration controls — lives in quod and is reviewed there:
https://github.com/nickjoven/quod/pull/2. This PR scopes only what lemma does
with attempts. Revision 2 records the review's four items and replaces the
open questions with recorded choices.

## Scope (lemma)

1. `corpus.iter_attempts` — verified reader for quod's `attempts.jsonl`
   (shards + manifest CID in `corpora/MANIFEST.yml`, like declarations and
   mutants). Every row carries the prover id and `prover_config_cid`.

2. **S4 v2 — the real target, stated prover-relatively.** The label is
   quod's per-mutant `gate_verdict`, and it is **prover-relative**:
   `no_proof_found` means *the named prover, within its recorded budget, found
   nothing* — not "unknown". S4 v2 is therefore described, trained, and
   reported as **predicting the harness outcome under a named prover and
   budget**; the prover id from the attempt record is available as an input
   feature and as a hold-out axis (e.g. train on tier-A outcomes, test on
   tier-B, or vice versa), so a model cannot silently learn "what prover X can
   prove" and present it as "what is true". Without this framing L-6 would
   recur in a new form.
   - **Classes and support.** `rejected:*` is expected to be nearly empty: a
     ladder emits kernel-accepted terms, so a gate rejection means an axiom
     leak from a tactic or a checker failure, both rare. Macro-F1 over three
     classes with one at near-zero support is not a meaningful gate. Rule:
     report per-class support; **if `rejected` support is small, collapse to
     `proven` vs `not-proven`** (and drop `rejected` from the macro average).
     The reported gate is over the classes that have support.
   - Same controls as v1: module-held-out mutants, bag-of-tokens margin
     ≥ 10 pts, held-out-operator OOD, permutation. Sample: the existing 8,187
     statement mutants (quod decision 4).

3. **S5 real test.** Rerank proposals first (OPEN.yml L-5), then consume
   quod's `A ↔ B` attempt outcomes.
   - **Reranker.** The token-Jaccard threshold is chosen on the **known-Iff
     validation pairs** — the value that drops at most a few percent of true
     equivalences — so it is not tuned on the proposals it filters. The
     reranked list is reported in **two buckets, cross-namespace and
     same-namespace**, rather than penalizing and hiding the latter.
   - The discharged count of the top-100 (per bucket) is reported against the
     bar of 10; each proposal's outcome is prover-tagged like any attempt.

4. **S3 exact eval (L-2), with its caveat.** The docstring × mutant AUC treats
   verdict-carrying mutants as *wrong statements*. Under three-way semantics
   only mutants that are `rejected`, **or whose negation was proven**,
   qualify as wrong. **`no_proof_found` mutants are excluded from the
   negative side** — they are not evidence of falsity. L-2 closes only with an
   AUC computed on that negative set.

5. Ledger: every consumer metric sealed with a metrics_cid, as today, with
   the prover id(s) it was computed over recorded in the metrics.

## Boundary

No changes to quod's gates, corpora, tokenizer, or existing checkpoints. No
training runs start until the quod harness has passed its positive, prover-
negative, and gate-wiring control sets (quod PR #2, exit conditions 1–4).

## Exit conditions

- S4 v2: macro-F1 ≥ 0.80 over the classes with support (three-way, or
  proven-vs-not if `rejected` support is small), on module-held-out mutants,
  beating bag-of-tokens by ≥ 10 pts, with the OOD-operator number and a
  prover hold-out number reported separately; per-class support reported.
- S5: discharged count of the reranked top-100, per bucket, against 10.
- L-2 closed with a sealed AUC whose negatives are `rejected` or
  negation-proven mutants only.

## Recorded choices (replacing the former open questions)

1. Reranker: threshold from known-Iff validation pairs (≤ a few % true-pair
   loss); two-bucket reporting, no hidden penalty.
2. `rejected:*` sub-reasons: collapsed; `rejected` itself collapsed into
   `not-proven` when its support is small.
