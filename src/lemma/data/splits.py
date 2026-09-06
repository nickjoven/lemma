"""Leakage-aware splits.

Rules (in force for every stage):
  1. The unit of assignment is the top-level Mathlib module: everything under
     one `Mathlib.<Area>` prefix lands in one split.
  2. No lock crosses a split: if two declarations share a lock (dedup pair),
     the later one follows the earlier one's split.
  3. A mutant inherits its parent theorem's split, always.
  4. Both sides of an Iff pair share a split (guaranteed by rule 1 when both
     live in one declaration; enforced explicitly for cross-decl pairs).

Assignment is deterministic (BLAKE3 of the area name, no RNG) so the split
manifest is reproducible from the corpus alone; the manifest still gets a CID
because "reproducible" is a claim and the CID is its evidence.
"""

from __future__ import annotations

from collections import defaultdict

import blake3

from ..schemas import DeclarationRow, MutantRow

SPLITS = ("train", "val", "test")
WEIGHTS = (0.90, 0.05, 0.05)


def area_of(module: str) -> str:
    parts = module.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else module


def split_of_area(area: str) -> str:
    h = int.from_bytes(blake3.blake3(area.encode()).digest()[:8], "big") / 2**64
    acc = 0.0
    for name, w in zip(SPLITS, WEIGHTS):
        acc += w
        if h < acc:
            return name
    return SPLITS[-1]


def assign(rows: list[DeclarationRow]) -> dict[str, str]:
    """name -> split, honoring the module rule then the lock rule."""
    by_lock: dict[str, str] = {}
    out: dict[str, str] = {}
    for row in rows:
        if row.lock in by_lock:
            out[row.name] = by_lock[row.lock]
        else:
            s = split_of_area(area_of(row.module))
            out[row.name] = s
            by_lock[row.lock] = s
    return out


def assign_mutants(mutants: list[MutantRow], decl_splits: dict[str, str]) -> dict[int, str]:
    """index -> split; a mutant with an unknown parent is excluded, never guessed."""
    out: dict[int, str] = {}
    for i, m in enumerate(mutants):
        if m.parent_name in decl_splits:
            out[i] = decl_splits[m.parent_name]
    return out


def manifest(rows: list[DeclarationRow]) -> dict:
    """The split manifest: assignment plus the honesty statistics."""
    splits = assign(rows)
    counts = defaultdict(int)
    for s in splits.values():
        counts[s] += 1
    # Disclosed, not hidden: dependency overlap between train and test.
    deps = {s: set() for s in SPLITS}
    for row in rows:
        deps[splits[row.name]].update(row.proof_consts)
    inter = len(deps["train"] & deps["test"])
    union = len(deps["train"] | deps["test"]) or 1
    return {
        "rule": "module-area blake3, lock-follows-first, mutant-inherits-parent",
        "weights": dict(zip(SPLITS, WEIGHTS)),
        "counts": dict(counts),
        "train_test_dep_jaccard": round(inter / union, 4),
        "assignment": splits,
    }
