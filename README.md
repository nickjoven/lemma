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
splits. Five staged experiments each carry a falsifiable gate. Two of them
failed their gate, and the ledger records them as failures. The horizon,
labeled as such, is a proof-state JEPA over lean-dojo traces.

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
| Stages | [`train.py`](src/lemma/train.py) (S2), [`train_s3.py`](src/lemma/train_s3.py), [`train_s4.py`](src/lemma/train_s4.py), [`train_s5.py`](src/lemma/train_s5.py) and [`evaluate_s2.py`](src/lemma/evaluate_s2.py) … [`evaluate_s5.py`](src/lemma/evaluate_s5.py) | Each trainer seals its run through the ledger; each evaluator embeds its own known-answer controls and marks the run `calibration_failed` if any fails. |
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
uv run pytest                      # 22 tests
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

The five sealed checkpoints (132–134 MB each) are excluded from git by CID.
The ledger records each `checkpoint_cid`, which is the raw BLAKE3 of the file
bytes, the same check the corpus reader applies to shards. They are to be
published as GitHub release assets named by CID, all five including the
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

## Open

[`OPEN.yml`](OPEN.yml) lists every known problem with an id, a severity, and
a status; a known gap that is not listed there is a defect of the ledger.

- **L-1** S3's gloss tower is representation-limited (hard-drift AUC 0.807 < 0.90); fix requires a tokenizer re-pin or an external prose encoder, both deferred.
- **L-2** S3's exact drift eval (docstring × gate-verified mutant) is deferred until statement mutants carry verdicts.
- **L-5** S5 proposals are dominated by boilerplate near-duplicates; a lexical-overlap and same-namespace reranker is needed before gate submission.
- **L-6** S4 v1 barely beats a bag-of-tokens baseline and collapses on an unseen operator; the real target is a three-way `gate_verdict`.

The next unlock for L-2, L-5, and L-6 is a prover-in-the-loop that attempts
proofs of mutated statements and proposed equivalences and lets quod's gates
compute every verdict. Its design is under review as a design-only PR;
nothing in this repository depends on it yet.

## Authorship

The code and documentation were written with Claude (Claude Code) as a
co-author; the `Co-Authored-By` trailers are kept on every commit. The design
decisions are mine and are the table above.

## License

[CC0 1.0 Universal](LICENSE). To the extent possible under law, the author
has waived all copyright and related rights to this work.
