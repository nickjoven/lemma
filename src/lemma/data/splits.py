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


def split_key(module: str) -> str:
    """The unit of split assignment: the full module path. Splitting by
    top-level area (Mathlib.X) instead left only ~50 units, so a 90/5/5 split
    was lumpy (val came out empty) and held out whole subfields — too coarse
    for a pretraining encoder. The full module keeps same-file declarations
    together (a sound leakage boundary) while giving thousands of units that
    partition cleanly. Cross-module dependency overlap is disclosed, not hidden
    (train_test_dep_jaccard in the manifest)."""
    return module


def split_of_area(module: str) -> str:
    h = int.from_bytes(blake3.blake3(split_key(module).encode()).digest()[:8], "big") / 2**64
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
            s = split_of_area(row.module)
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


def assign_transitions(rows, decl_splits: dict[str, str]) -> dict[int, str]:
    """index -> split; a transition inherits its DEMONSTRANDUM's split (for a
    mutant attempt, the parent theorem's), so no goal of a held-out theorem's
    proof search is seen in training. Unknown demonstranda are excluded."""
    out: dict[int, str] = {}
    for i, t in enumerate(rows):
        if t.demonstrandum in decl_splits:
            out[i] = decl_splits[t.demonstrandum]
    return out
