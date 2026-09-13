# lemma

[N. Joven](https://github.com/nickjoven) — 2026 — [ORCID 0009-0008-0679-0812](https://orcid.org/0009-0008-0679-0812) — CC0 1.0

Self-supervised encoders over a gate-verified Lean 4 / Mathlib corpus. The
corpus and its labels come from [quod](https://github.com/nickjoven/quod);
this repository never computes a label itself. Every metric here exists only
as a content-addressed record: the ledger refuses to seal a run whose metrics
were not first stored in the repository's own evidence store, and a run is
citable only if that record still resolves. The model is a 33M-parameter
statement encoder written from scratch (SDPA attention, RoPE, pre-norm; no
`transformers`, `datasets`, or `lightning`), trained on leakage-audited
splits. Each staged experiment carries a falsifiable gate, and the ledger
records the failures as failures: S3 missed its gate twice, and the first
world-model run missed its retrieval gate by 1.2 points before the second
passed. The world model (stage W0) is a predictor from a goal embedding and a
tactic to the next goal embedding, trained on tactic-level proof-state
transitions that quod's attempts harness emits; its retrieval gate is passed
and its cost head is not. That is the first step of a proof-state JEPA, and
the name is used here only for what has passed a gate.

## Results

One row per sealed final run (smoke runs omitted). Every number below is read
from the metrics record named in its row, which lives in the tracked
[`.ket/`](.ket/) store; the ledger line is in [`runs/ledger.jsonl`](runs/ledger.jsonl).

| run | stage | measures | number | gate | ledger status | metrics_cid | wall clock |
|---|---|---|---|---|---|---|---|
| `s2-mlm-20260909-141736` | S2 train | span-MLM masked accuracy | 0.9317 | — | completed | `a7f3354d` | 27.2 h |
| `s2-eval-s2-mlm-20260909-141736` | S2 eval | same-lock retrieval MRR; masked acc margin over a unigram baseline | MRR 0.9921; +65.9 pts | MRR ≥ 0.95 | completed (pass) | `088bf529` | 25 s |
| `s3-gloss-20260910-174141` | S3 v1 train | contrastive docstring ↔ statement | — | — | completed | `78ff7b26` | 7.6 h |
| `s3-eval-s3-gloss-20260910-174141` | S3 v1 eval | same-module hard-drift AUC; retrieval MRR | AUC 0.767; MRR 0.398 | AUC ≥ 0.90 | **calibration_failed** | `ca34a4b7` | 6 s |
| `s3-gloss-20260911-012141` | S3 v2 train | v1 + module-grouped hard negatives, early stopping | — | — | completed | `c2c05884` | 1.8 h |
| `s3-eval-s3-gloss-20260911-012141` | S3 v2 eval | same-module hard-drift AUC; retrieval MRR | AUC 0.8074; MRR 0.4001 | AUC ≥ 0.90 | **calibration_failed** | `3d8a926c` | 6 s |
| `s4-verdict-20260911-152400` | S4 v1 train | `elaborates` + operator heads on Phase-3a mutant labels | — | — | completed | `40a2b7f7` | 6 min |
| `s4-eval-s4-verdict-20260911-152400` | S4 v1 eval | `elaborates` macro-F1; margin over bag-of-tokens; held-out operator | F1 0.9411; baseline 0.8797 (+6.1 pts); OOD `swap_lit_01` 0.3051 | F1 ≥ 0.80 (pass); margin ≥ 10 pts (fail) | **calibration_failed** | `3690e8f2` | 2 s |
| `s5-equiv-20260911-125959` | S5 train | symmetric InfoNCE over Iff sides and same-lock pairs | — | — | completed | `bfa9c24a` | 2.4 h |
| `s5-eval-s5-equiv-20260911-125959` | S5 eval | known-Iff retrieval MRR; dedup MRR; 100 proposals | MRR 0.5635; dedup 0.9743 | MRR ≥ 0.5 | completed (pass) | `1a0ed42c` | 26 s |
| `s4v2-verdict-20260911-223439` | S4 v2 train | three-way prover-relative `gate_verdict` on quod attempt labels | — | — | completed | `da31a16a` | 7.5 min |
| `s4v2-eval-s4v2-verdict-20260911-223439` | S4 v2 eval | macro-F1 over classes with support; margin over bag-of-tokens | F1 0.707; +18.1 pts | F1 ≥ 0.80 (fail); margin ≥ 10 (pass) | **calibration_failed** | `a9c99344` | 1 s |
| `w0-baselines-20260912-124021` | W0 baselines | identity / per-tactic offset / lexical next-state MRR on held-out transitions | 0.304 / 0.589 / 0.456 | — | completed | `d0f7a039` | 25 s |
| `w0-world-20260912-012400` | W0 v1 train | next-state predictor on terminal tier-A transitions | — | — | completed | `71ff9102` | 9.7 min |
| `w0-eval-w0-world-20260912-012400` | W0 v1 eval | next-state MRR vs offset + 0.10 | 0.883 vs bar 0.895 | bar 0.895 | **calibration_failed** | `4e493983` | 6 s |
| `w0-world-20260912-124023` | W0 v2 train | same config; corpus now includes stepping-prover trajectories | — | — | completed | `3caa6e8f` | 9.7 min |
| `w0-eval-w0-world-20260912-124023` | W0 v2 eval | next-state MRR vs offset + 0.10; margin over lexical; permutation | 0.733 vs bar 0.689; +27.8; permuted 0.015 | bar 0.689 | completed (pass) | `d991f7e1` | 7 s |

- **S3 failed twice** (`OPEN.yml` L-1). The statement tower is strong (S2 MRR
  0.992); the gloss tower is limited by a Lean-only tokenizer that
  byte-fallbacks English docstrings. Recorded citation limit: useful for gross
  prose/statement mismatch only (easy-drift AUC 0.977), not for same-module
  fidelity.
- **S4 passed its F1 gate and failed its baseline-margin control**
  (`OPEN.yml` L-6). A bag-of-tokens logistic regression reaches 0.880, and the
  model collapses on an unseen operator. Recorded citation limit:
  "operator-conditional `elaborates` on this mutant distribution", never a
  verdict predictor.
- **S5 passed its gate**, but the top of its proposal list is boilerplate
  near-duplicates inside one namespace (`OPEN.yml` L-5); proposals are not
  submitted to the gates until reranked.

## What is here

| piece | path | what it does |
|---|---|---|
| Ledger | [`src/lemma/ledger.py`](src/lemma/ledger.py) | The only code path that writes metrics. `finish_run` refuses a completed run without a `metrics_cid`; `citable(run_id)` returns a record only if its metrics still resolve in the store. |
| Splits | [`src/lemma/data/splits.py`](src/lemma/data/splits.py) | Deterministic module-level assignment (BLAKE3 of the module name, no RNG); a lock never crosses a split; a mutant inherits its parent's split; the manifest discloses train/test dependency Jaccard. |
| Corpus reader | [`src/lemma/data/corpus.py`](src/lemma/data/corpus.py) | Verifies every shard's BLAKE3 against [`corpora/MANIFEST.yml`](corpora/MANIFEST.yml) before parsing; a mismatch is an error. |
| Encoder | [`src/lemma/models/encoder.py`](src/lemma/models/encoder.py), [`heads.py`](src/lemma/models/heads.py) | 33M params: 8 layers, d=512, 8 heads, RoPE, pre-norm, SDPA; mean-pooled statement embedding. |
| Objectives | [`src/lemma/objectives/`](src/lemma/objectives/) | Span-MLM ([`mlm.py`](src/lemma/objectives/mlm.py)) and symmetric InfoNCE ([`contrastive.py`](src/lemma/objectives/contrastive.py)). |
| Tokenizer | [`src/lemma/tokenizer/train.py`](src/lemma/tokenizer/train.py), [`runs/tokenizer/`](runs/tokenizer/) | Unigram with byte fallback, trained once and frozen by CID. |
| Stages | [`train.py`](src/lemma/train.py) (S2), [`train_s3.py`](src/lemma/train_s3.py), [`train_s4.py`](src/lemma/train_s4.py), [`train_s4v2.py`](src/lemma/train_s4v2.py), [`train_s5.py`](src/lemma/train_s5.py) and the matching `evaluate_*.py` | Each trainer seals its run through the ledger; each evaluator embeds its own known-answer controls and marks the run `calibration_failed` if any fails. |
| World model | [`w0_baselines.py`](src/lemma/w0_baselines.py), [`train_w0.py`](src/lemma/train_w0.py) | Baselines first (identity, per-tactic offset, lexical, permutation), sealed; then a predictor from goal embedding and tactic to next-goal embedding with outcome and cost heads, gated against the baselines. |
| Release verifier | [`src/lemma/verify_release.py`](src/lemma/verify_release.py) | Hashes a downloaded checkpoint or tokenizer and checks it against the ledger. |
| Run record | [`runs/ledger.jsonl`](runs/ledger.jsonl) | Append-only, one line per start and per seal. |
| Corpus contract | [`corpora/MANIFEST.yml`](corpora/MANIFEST.yml) | quod run ids, manifest CIDs, and per-shard CIDs for the declarations and mutants corpora. |
| Problem ledger | [`OPEN.yml`](OPEN.yml) | Every known problem with severity and status. |
| Evidence store | [`.ket/`](.ket/) | Tracked content-addressed store ([ket](https://github.com/nickjoven/ket)) holding metrics records, configs, the tokenizer, and calibration records. Checkpoints are excluded by CID (see below). |

## Reproduce

Environment: Python 3.12, `torch==2.9.0` from the cu128 index, an RTX 4070
class GPU. Everything runs as `uv run …`.

```sh
uv sync --frozen
uv run pytest                      # 37 tests
```

Corpus: obtain quod's shards for runs `full-20260908b` (declarations,
296,187 records) and `mutants-20260911` (19,007 mutants) and place them under
`corpora/` at the paths in `MANIFEST.yml`. The reader verifies each shard's
BLAKE3 against the manifest before parsing. quod regenerates the shards with
its corpus extractor in about eight hours; publishing them as release assets
is planned.

```sh
uv run python -m lemma.tokenizer.train                      # S1 (already frozen; new run = new CID)
uv run python -u -m lemma.train      --config configs/s2_mlm.yaml
uv run python    -m lemma.evaluate_s2 --run <s2 run_id>
uv run python -u -m lemma.train_s3   --config configs/s3_gloss.yaml
uv run python    -m lemma.evaluate_s3 --run <s3 run_id>
uv run python -u -m lemma.train_s4   --config configs/s4_verdict.yaml
uv run python    -m lemma.evaluate_s4 --run <s4 run_id>
uv run python -u -m lemma.train_s5   --config configs/s5_equiv.yaml
uv run python    -m lemma.evaluate_s5 --run <s5 run_id>
```

Citation check, which raises unless the sealed metrics resolve:

```sh
uv run python -c "from lemma import ledger; print(ledger.citable('s2-mlm-20260909-141736'))"
```

## Checkpoints and evidence

The seven sealed checkpoints (roughly 130 MB each) are excluded from git by
CID. The ledger records each `checkpoint_cid`, which is the raw BLAKE3 of the
file bytes, the same check the corpus reader applies to shards. They are to
be published as GitHub release assets named by CID, all seven including the
failed-gate runs, with no model-card claims: they exist so the numbers above
can be re-run. To check a download:

```sh
uv run python -m lemma.verify_release <downloaded file>
# or, without Python: b3sum <downloaded file>  and compare to the table
```

| run | checkpoint_cid | ledger status | citation limit |
|---|---|---|---|
| `s2-mlm-20260909-141736` | `b2a83ed2054546d4407d28227e44782f1f50ac4631fa51f15f82d0b624776796` | completed; eval pass | statement encoder; same-lock MRR 0.992 on held-out modules |
| `s3-gloss-20260910-174141` | `28338bdf987ad720959f2c075363203f15c62f5b0c4dcc9f937f009b6be45850` | completed; eval calibration_failed | gross prose/statement mismatch only (L-1) |
| `s3-gloss-20260911-012141` | `176ef79d921bc0caca58f06ff36cb0ef79f9e80e38ec1538bc35ce64f88c3de5` | completed; eval calibration_failed | gross prose/statement mismatch only (L-1) |
| `s4-verdict-20260911-152400` | `16a64f5b1272c584384ccb1a70b7b9e57613bc09384f8ecb0cabff705e041d97` | completed; eval calibration_failed | operator-conditional `elaborates` on this mutant distribution (L-6) |
| `s5-equiv-20260911-125959` | `04cd648b14dedc89da84ac5976b3ce92f4ab775711d8e671622f6af6c3ada587` | completed; eval pass | known-Iff retrieval; proposals unreranked (L-5) |
| `w0-world-20260912-012400` | `63f915b15447abd18b02920f04b0fa5cd7619362e2716d32dbd3a8e41f7e8358` | completed; eval calibration_failed | W0 v1: trained on terminal tier-A transitions, 93% on mutant goals (L-7) |
| `w0-world-20260912-124023` | `56b3e715b33fcb894856705885c58dd03da3941b639718401a0b31254688605a` | completed; eval pass (retrieval) | W0 v2: next-state retrieval only; cost head uncalibrated on censored rows (L-8) |

The tokenizer is tracked at
[`runs/tokenizer/lean-unigram-32k.json`](runs/tokenizer/lean-unigram-32k.json),
CID `3b33be812ca5991f454b242b7d6fc05512f582384f7d23792ef64a0944a620d4`. It
has about 14.9k effective pieces despite the `32k` in the filename
(`OPEN.yml` L-4). Every run above was trained and evaluated under the pair
(`mathlib_pin = crouzeix:v4.28.0:8f9d9cff`, this tokenizer CID); the ledger
never compares metrics across a different pair.

## Decisions

Each row is a design decision I made, the alternative it displaced, and the
rationale as recorded in this repository.

| decision | alternative rejected | objective rationale (as recorded) | where |
|---|---|---|---|
| A metric without a `metrics_cid` does not exist; one code path writes metrics | Metrics in logs, notebooks, or commit messages | `finish_run` refuses a completed run with no metrics; `citable` re-resolves the CID before returning a record, so a number that cannot be fetched cannot be cited | [`src/lemma/ledger.py`](src/lemma/ledger.py); `CLAUDE.md` |
| Provenance travels with the code: the evidence store is tracked | A store outside the repository | The metrics, configs, tokenizer, and calibration records that the ledger cites are in `.ket/`; only the checkpoints are excluded, each by its own CID line in `.gitignore` | `CLAUDE.md` ("federation model"); [`.gitignore`](.gitignore) |
| Split unit is the full module path | Top-level Mathlib area | Area-level splitting gave about 50 units; a 90/5/5 split came out lumpy and val was empty. Module-level keeps same-file declarations together and yields thousands of units | [`src/lemma/data/splits.py`](src/lemma/data/splits.py) (`split_key`) |
| A lock never crosses a split; a mutant inherits its parent's split; unknown parents are excluded, never guessed | Independent assignment per row | Same theorem under two names in different splits is a leak; a mutant of a train theorem in test is a leak; the rules are tested | [`src/lemma/data/splits.py`](src/lemma/data/splits.py); [`tests/test_splits.py`](tests/test_splits.py) |
| Train/test dependency overlap is disclosed in the split manifest | Hide it or claim independence | Cross-module proof-constant overlap is real; the manifest reports `train_test_dep_jaccard` rather than asserting none | [`src/lemma/data/splits.py`](src/lemma/data/splits.py) (`manifest`) |
| `readable_pp` is the encoder input; `pp.all` stays the lock identity | Feed the canonical `pp.all` string | `readable_pp` tokenizes to p50 = 88 / p99 = 309 tokens with 0.17% over 512, so seq 512 is comfortable; the `pp.all` 1024/batch-48 stopgap was retired | [`configs/s2_mlm.yaml`](configs/s2_mlm.yaml); commit `c283a57` |
| Corpus shards are verified against the manifest before parsing | Trust files on disk | A shard whose BLAKE3 differs from its manifest CID is an error, not a warning; the producing quod run's manifest CID is the corpus version | [`src/lemma/data/corpus.py`](src/lemma/data/corpus.py); [`corpora/MANIFEST.yml`](corpora/MANIFEST.yml) |
| No `transformers`, `datasets`, or `lightning` | Off-the-shelf model and trainer stacks | Recorded as "by design": the encoder, objectives, cache, and training loop are small enough to read in full, and SDPA dispatches to a fused kernel on sm89 without a flash-attn build | `CLAUDE.md`; [`pyproject.toml`](pyproject.toml); [`src/lemma/models/encoder.py`](src/lemma/models/encoder.py) |
| Tokenizer trained once and frozen by CID; a retrain is a new artifact | Overwrite in place | Token ids never shift under a Mathlib pin change; unseen lexemes degrade to bytes | [`src/lemma/tokenizer/train.py`](src/lemma/tokenizer/train.py) |
| A failing known-answer control marks the run `calibration_failed` and its metrics uncitable | Report the metric with a caveat | Mirrors quod's rule that a failing positive control means the gates are wrong; the status is a distinct ledger value | `CLAUDE.md`; [`src/lemma/schemas.py`](src/lemma/schemas.py) (`RunRecord.status`) |
| S3: record the failure and move on; do not re-pin the tokenizer mid-stream | Retrain a gloss-capable tokenizer now | A new tokenizer pin changes the corpus contract for every downstream stage; decided 2026-09-11 to revisit when a re-pin is planned | [`OPEN.yml`](OPEN.yml) L-1 |
| S4 trains on statement mutants only | Include proof-side (`sorry`/axiom injection) records | Proof-side records leave the statement unchanged and carried no mutation signal | commit `ce5b4ed`; [`src/lemma/train_s4.py`](src/lemma/train_s4.py) |
| S4 is cited only as operator-conditional `elaborates` on this distribution | Call it a verdict predictor | It beats bag-of-tokens by 6.1 points against a 10-point gate and scores 0.305 on an unseen operator; the substantive target (`gate_verdict`) needs proof attempts | [`OPEN.yml`](OPEN.yml) L-6 |
| S5 proposals are not submitted to the gates until reranked | Submit the raw top-100 by cosine | The top of the list is score-1.000 near-duplicates in one namespace; a discharged count on that list would confirm trivialities | [`OPEN.yml`](OPEN.yml) L-5 |
| The world model must beat two cheap baselines before it is trained on: state plus a learned per-tactic offset, and a lexical baseline, by 10 points each | Train the predictor and report its score | The offset baseline alone reaches 0.59 on next-state retrieval; a predictor that does not clear it by a margin has learned nothing the geometry did not already hold. v1 failed this by 1.2 points and was recorded as a failure | [`OPEN.yml`](OPEN.yml) L-7; W0 baseline runs in the ledger |
| Budget-exhausted transitions are censored observations, not infinite cost | Average cap hits as failures | Treating cap hits as failures biases every cost estimate low on hard states; the cost head carries a survival term and its calibration on censored rows is its own gate | [`OPEN.yml`](OPEN.yml) L-8 |
| "JEPA" is used only once a world-model gate has passed | Name the horizon after the architecture from the start | The word implies a predictor over transitions; until W0 v2 the repository had only static judgments about statements | [`OPEN.yml`](OPEN.yml) L-7; this README |

## Open

[`OPEN.yml`](OPEN.yml) lists every known problem with an id, a severity, and
a status; a known gap that is not listed there is a defect of the ledger.

- **L-1** S3's gloss tower is representation-limited (hard-drift AUC 0.807 < 0.90); fix requires a tokenizer re-pin or an external prose encoder, both deferred.
- **L-2** S3's exact drift eval (docstring × gate-verified mutant) is deferred until statement mutants carry verdicts.
- **L-5** S5 proposals are dominated by boilerplate near-duplicates; a lexical-overlap and same-namespace reranker is needed before gate submission.
- **L-6** S4 v1 barely beats a bag-of-tokens baseline and collapses on an unseen operator; S4 v2 on prover-relative labels clears the baseline margin (+18.1) but not the absolute F1 gate, over a 7-row class.
- **L-7** Tier-A transitions were terminal, so the first world-model gate had no test set; resolved for retrieval by quod's stepping prover, which produced multi-step trajectories. Recorded in full, including the failed v1.
- **L-8** The W0 cost head predicts a cost at or above the cap for 1.6% of the rows that actually hit it; a discrete cap-hit head or a reweighted survival term is the cheap next run.

The prover-in-the-loop that feeds L-2, L-5, L-6, and W0 lives in quod (its
attempts harness); this repository consumes its transition and verdict
corpora by CID and never runs a prover itself.

## Authorship

The code and documentation were written with Claude (Claude Code) as a
co-author; the `Co-Authored-By` trailers are kept on every commit. The design
decisions are mine and are the table above.

## License

[CC0 1.0 Universal](LICENSE). To the extent possible under law, the author
has waived all copyright and related rights to this work.
