# AGENTS.md — operating rules for agents working in lemma

Fleet-wide rules (ports, processes, shell hygiene, secrets) live in
[homeserv/AGENTS.md](https://github.com/nickjoven/homeserv/blob/main/AGENTS.md)
and apply here. lemma adds:

- **A metric without a `metrics_cid` does not exist.** `src/lemma/ledger.py`
  is the only writer; cite runs through `ledger.citable(run_id)`.
- **Corpora arrive by CID** (`corpora/MANIFEST.yml`, wired with
  `scripts/wire_quod_run.py`); this repo never computes a label itself.
- **Splits are leakage-audited:** by top-level area, locks never cross
  splits, mutants and transitions inherit their parent's split; control
  sets never enter training.
- **Gates are falsifiable and recorded either way:** a failed gate is an
  `OPEN.yml` entry with CIDs and a decided-or-pending fix, not a deleted run.
- **One GPU job at a time on the 4070**; training and serving do not overlap
  on one card. Long runs use `python -u`, rolling checkpoints, and a log;
  check a process's age before touching it.
- **"JEPA" names only what has passed a gate** (W0 retrieval gate passed;
  cost head has not, L-8).
