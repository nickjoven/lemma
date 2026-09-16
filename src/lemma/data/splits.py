"""Leakage-aware splits.

Rules (in force for every stage):
  1. The unit of assignment is the top-level Mathlib module: everything under
     one `Mathlib.<Area>` prefix lands in one split.
  2. No lock crosses a split: two declarations that share a lock (a dedup
     pair) land in one split. Since rule 1 also holds, the modules they live
     in must land in one split too, so the real unit of assignment is a
     CONNECTED COMPONENT of modules joined by shared locks (lemma #6).
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


def _components(rows: list[DeclarationRow]) -> dict[str, str]:
    """module -> component key, where a component is a maximal set of modules
    joined (transitively) by shared locks. The key is the lexicographically
    smallest module in the component, so it depends only on the SET of rows,
    never on their order (lemma #6: the previous first-seen-lock rule let one
    declaration leave its module's split and made the result order-dependent).
    Union-find over module names."""
    parent: dict[str, str] = {}

    def find(m: str) -> str:
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # keep the smaller name as root so the root IS the canonical key
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    modules_of_lock: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        parent.setdefault(row.module, row.module)
        modules_of_lock[row.lock].add(row.module)
    for mods in modules_of_lock.values():
        first = min(mods)
        for m in mods:
            union(first, m)
    return {m: find(m) for m in parent}


def assign(rows: list[DeclarationRow]) -> dict[str, str]:
    """name -> split. Every module lands whole in one split (rule 1), and
    every shared lock stays inside one split (rule 2), because the split is
    chosen per connected component of modules joined by shared locks and a
    module that shares no lock with another module is its own component, so
    its split is exactly split_of_area(module) as before."""
    comp = _components(rows)
    return {row.name: split_of_area(comp[row.module]) for row in rows}


def component_stats(rows: list[DeclarationRow]) -> dict:
    """Disclosed with the manifest: how much the shared-lock joins coarsened
    the module partition. A large component is a warning that the split may
    be lumpy or that a family of duplicated statements spans many files."""
    comp = _components(rows)
    sizes: dict[str, int] = defaultdict(int)
    for m, key in comp.items():
        sizes[key] += 1
    multi = {k: n for k, n in sizes.items() if n > 1}
    return {
        "modules": len(comp),
        "components": len(sizes),
        "multi_module_components": len(multi),
        "modules_in_multi_module_components": sum(multi.values()),
        "largest_component_modules": max(sizes.values()) if sizes else 0,
    }


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
        "rule": "module blake3 over connected components joined by shared locks, mutant-inherits-parent",
        "weights": dict(zip(SPLITS, WEIGHTS)),
        "counts": dict(counts),
        "components": component_stats(rows),
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
