"""S4 v2 labels: join mutants to attempts, prover-relatively (lemma PR #1 §2).

For each statement mutant M = (parent, operator) and each prover, the label
is the attempt's three-way `gate_verdict`: proven | no_proof_found | rejected.
An attempt counts for M only if its rebuilt mutant has M's lock (the
nomination "this attempt is for M" verified by lock equality — quod's
`mutant_lock` field). Records are tagged with the prover so a model can use
it as a feature or as a hold-out axis; nothing here ever maps
`no_proof_found` to `refuted`.

`collapse_rule`: report per-class support; if `rejected` support is below
`min_support`, collapse to proven-vs-not (two classes) — macro-F1 over a
class with ~zero support is not a meaningful gate.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from ..schemas import AttemptRow, MutantRow

CLASSES3 = ("proven", "no_proof_found", "rejected")


def join_mutant_attempts(mutants: Iterable[MutantRow], attempts: Iterable[AttemptRow]
                         ) -> tuple[list[dict], dict]:
    """-> (labelled rows, audit). Each row: text, parent, operator, prover,
    prover_config_cid, label3. Attempts whose mutant_lock != the corpus lock are
    dropped and counted in audit['lock_mismatch'] — they are not evidence about M."""
    by_key: dict[tuple[str, str], MutantRow] = {}
    for m in mutants:
        if m.elaborates:  # ill-formed mutants are not demonstranda
            by_key[(m.parent_name, m.operator)] = m
    rows, audit = [], Counter()
    for a in attempts:
        if not a.mutant_operator or a.negated:
            continue
        m = by_key.get((a.demonstrandum, a.mutant_operator))
        if m is None:
            audit["no_corpus_mutant"] += 1
            continue
        if a.mutant_lock != m.lock:
            audit["lock_mismatch"] += 1
            continue
        rows.append({"text": m.mutated_type, "parent": m.parent_name, "operator": m.operator,
                     "op_class": m.op_class, "prover": a.prover, "prover_config_cid": a.prover_config_cid,
                     "label3": a.gate_verdict, "module": m.module})
        audit[f"label:{a.gate_verdict}"] += 1
    return rows, dict(audit)


def collapse_rule(rows: list[dict], min_support: int = 30) -> tuple[list[str], dict]:
    """Decide the class set from support. Returns (classes, support)."""
    support = Counter(r["label3"] for r in rows)
    if support.get("rejected", 0) < min_support:
        for r in rows:
            r["label"] = "proven" if r["label3"] == "proven" else "not_proven"
        return ["proven", "not_proven"], dict(support)
    for r in rows:
        r["label"] = r["label3"]
    return list(CLASSES3), dict(support)
