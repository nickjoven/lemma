# lemma

Self-supervised encoders over gate-verified Lean/Mathlib corpora. quod
(~/code/quod) is the label generator; this repo never computes a label itself.

## Doctrine

- **A metric without a metrics_cid does not exist.** `src/lemma/ledger.py` is
  the only code path that writes metrics; it `ket put`s them into this repo's
  `.ket/` and appends `runs/ledger.jsonl` (append-only). Cite runs via
  `ledger.citable(run_id)` — it verifies the evidence resolves.
- **Corpora arrive by CID.** Shards live in `corpora/` (gitignored) and are
  verified against `corpora/MANIFEST.yml` (tracked) before parsing. The
  producing quod run's manifest CID is the corpus version.
- **Splits are leakage-audited** (`src/lemma/data/splits.py`): assignment by
  top-level Mathlib area, locks never cross splits, mutants inherit their
  parent's split. `(mathlib_pin, tokenizer_cid)` is stamped on every record;
  metrics are never compared across different pairs.
- **Calibration before citation**: known-answer controls in `calibration/`
  run before/after every training run; a failing control marks the run
  `calibration_failed` and its metrics uncitable.

## Environment

uv-managed, python 3.12 pinned, `torch==2.9.0` from the cu128 index (RTX
4070). `uv sync --frozen` reproduces it; `uv.lock` gets a CID in run records.
Run everything as `uv run ...`. No transformers/datasets/lightning by design.

## Stages

S0 harness → S1 tokenizer (Unigram 24k, byte-fallback, frozen+CID'd) →
S2 33M SDPA encoder, span-MLM → S3 gloss-fidelity → S4 verdict predictor →
S5 equivalence discovery → (horizon) proof-state JEPA on lean-dojo traces.
Per-stage success criteria live in the plan and in each stage's config.

## ket

Own substrate at `.ket/` (federation model). `ket --home .ket <cmd>`; the
ledger uses the `ket` on PATH (~/.local/bin/ket, sha in run records).
