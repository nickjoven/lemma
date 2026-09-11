# Prover-in-the-loop — lemma (consumer) side, step 0 for review

Status: **design for review, nothing implemented.** The harness itself —
`AttemptWalk.lean`, `attempt.py`, the attempt ledger, the prover-tier
decision, calibration controls for provers — lives in quod and is reviewed
there: https://github.com/nickjoven/quod/pull/2. This PR scopes only what lemma does with attempts.

## Scope (lemma)

1. `corpus.iter_attempts` — verified reader for quod's `attempts.jsonl`
   (shards + manifest CID in `corpora/MANIFEST.yml`, like declarations and
   mutants).
2. **S4 v2** — the real target. Label = `gate_verdict` per statement mutant,
   **three-way**: `proven` / `no_proof_found` / `rejected:*`. Never
   binary; `no_proof_found` is not `refuted`. Same controls as v1
   (module-held-out, bag-of-tokens margin ≥ 10 pts, held-out operator OOD,
   permutation), on a larger mutant sample.
3. **S5 real test** — rerank proposals first (OPEN.yml L-5: lexical-overlap
   and same-namespace penalties, distinct locks), then consume quod's
   `A ↔ B` attempt outcomes; report discharged count vs the bar of 10.
4. **S3 exact eval** (L-2) — docstring × gate-verified mutant AUC becomes
   computable once mutants carry verdicts.
5. Ledger: every consumer metric sealed with a metrics_cid, as today.

## Boundary

No changes to quod's gates, corpora, tokenizer, or existing checkpoints. No
training runs start until the quod harness has passed its positive and
negative control sets.

## Exit conditions

- S4 v2 macro-F1 over the three-way verdict ≥ 0.80 on module-held-out
  mutants, beating bag-of-tokens by ≥ 10 pts, with the OOD-operator number
  reported separately.
- S5: discharged count of the reranked top-100 reported against 10.
- L-2 closed with a sealed AUC.

## Open questions

1. Reranker weights for L-5 (token-Jaccard threshold; namespace penalty).
2. Whether S4 v2 should also predict `rejected:*` sub-reasons or collapse them.
