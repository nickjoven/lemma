"""Leakage-aware splits.

Rules (in force for every stage):
  1. The unit of assignment is the full module path: every declaration of
     one module lands in one split (split_key says why not the area).
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

Rules are VERSIONED, because sealed checkpoints were trained under an older
one and an evaluator must reproduce the split a checkpoint trained against,
never today's (review of #10):
  RULE_V1  module blake3, a row follows the first-seen split of its lock
           (assign_v1). Every run sealed before #10 trained under it. It is
           order-dependent and can split a module (#6); kept only so those
           runs can be evaluated against their own split.
  RULE_V2  module blake3 over connected components (assign). Current rule;
           its giant component is an open owner decision (OPEN.yml L-12).
A run records which rule and which assignment by `certificate`: the rule id
plus a BLAKE3 digest of each split's sorted name list. It is small enough
for the git-tracked store (the full name->split manifest is ~15-20 MB on the
real corpus) and lets an evaluator verify a reconstruction exactly.
"""

from __future__ import annotations

from collections import defaultdict

import blake3

from ..schemas import DeclarationRow, MutantRow

SPLITS = ("train", "val", "test")
WEIGHTS = (0.90, 0.05, 0.05)

RULE_V1 = "v1-module-lock-follows-first"
RULE_V2 = "v2-module-lock-components"
CURRENT_RULE = RULE_V2
# the free-text `rule` strings older manifests carry, mapped to their rule id
RULE_DESCRIPTIONS = {
    "module-area blake3, lock-follows-first, mutant-inherits-parent": RULE_V1,
    "module blake3 over connected components joined by shared locks, mutant-inherits-parent": RULE_V2,
}
CERT_FORMAT = "lemma-split-cert/1"
# L-12 guard: a component holding more than this fraction of the rows makes a
# RULE_V2 manifest refuse to build until the owner decides L-12. A guard, not
# the decision: it only stops the rule from being used silently.
GIANT_COMPONENT_MAX_FRACTION = 0.10


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


def assign_v1(rows: list[DeclarationRow]) -> dict[str, str]:
    """RULE_V1, byte-for-byte the pre-#10 assign(): name -> split, the module
    hash unless the row's lock was already seen, in which case it follows the
    first-seen row. ORDER-DEPENDENT: reproduces a v1 split only over rows in
    the order the original build read them (iter_rows over the same shards).
    Not for new manifests; see the module docstring."""
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


ASSIGNERS = {RULE_V1: assign_v1, RULE_V2: assign}


def rule_id(rule: str) -> str:
    """A rule id from an id or an older manifest's description; unknown is an error."""
    if rule in ASSIGNERS:
        return rule
    if rule in RULE_DESCRIPTIONS:
        return RULE_DESCRIPTIONS[rule]
    raise ValueError(f"unknown split rule {rule!r}; known: {sorted(ASSIGNERS)}")


def assign_under(rule: str, rows: list[DeclarationRow]) -> dict[str, str]:
    return ASSIGNERS[rule_id(rule)](rows)


def certificate(assignment: dict[str, str], rule: str) -> dict:
    """The compact, verifiable record of a split: rule id, per-split counts,
    and the BLAKE3 of each split's sorted, newline-joined name list."""
    names = {s: sorted(n for n, v in assignment.items() if v == s) for s in SPLITS}
    unknown = set(assignment.values()) - set(SPLITS)
    if unknown:
        raise ValueError(f"assignment uses unknown splits {sorted(unknown)}")
    return {"format": CERT_FORMAT, "rule_id": rule_id(rule),
            "counts": {s: len(v) for s, v in names.items()},
            "names_blake3": {s: blake3.blake3("\n".join(v).encode()).hexdigest() for s, v in names.items()}}


def certificate_mismatch(cert: dict, assignment: dict[str, str]) -> str | None:
    """None if `assignment` is exactly the split `cert` records, else why not."""
    if cert.get("format") != CERT_FORMAT:
        return f"not a split certificate (format {cert.get('format')!r}, want {CERT_FORMAT})"
    got = certificate(assignment, cert["rule_id"])
    if got["counts"] != cert["counts"]:
        return f"per-split counts {got['counts']} differ from the certificate's {cert['counts']}"
    bad = [s for s in SPLITS if got["names_blake3"][s] != cert["names_blake3"][s]]
    if bad:
        return f"the {', '.join(bad)} name list(s) differ from the certificate's"
    return None


def component_guard(rows: list[DeclarationRow], max_fraction: float = GIANT_COMPONENT_MAX_FRACTION) -> dict:
    """L-12: refuse a RULE_V2 split whose largest component holds more than
    max_fraction of the rows. Returns the component stats when it passes."""
    stats = component_stats(rows)
    frac = stats["largest_component_rows"] / max(len(rows), 1)
    if frac > max_fraction:
        raise ValueError(
            f"the largest connected component holds {stats['largest_component_rows']} of {len(rows)} rows "
            f"({frac:.1%} > {max_fraction:.0%}): OPEN.yml L-12 is an open owner decision (accept, exclude hub "
            "locks from the join, or keep the component out of val/test); this split rule is not used until it is decided")
    return stats


def component_stats(rows: list[DeclarationRow]) -> dict:
    """Disclosed with the manifest: how much the shared-lock joins coarsened
    the module partition. A large component is a warning that the split may
    be lumpy or that a family of duplicated statements spans many files."""
    comp = _components(rows)
    sizes: dict[str, int] = defaultdict(int)
    for key in comp.values():
        sizes[key] += 1
    row_sizes: dict[str, int] = defaultdict(int)
    for row in rows:
        row_sizes[comp[row.module]] += 1
    multi = {k: n for k, n in sizes.items() if n > 1}
    return {
        "modules": len(comp),
        "components": len(sizes),
        "multi_module_components": len(multi),
        "modules_in_multi_module_components": sum(multi.values()),
        "largest_component_modules": max(sizes.values()) if sizes else 0,
        "largest_component_rows": max(row_sizes.values()) if row_sizes else 0,
    }


def assign_mutants(mutants: list[MutantRow], decl_splits: dict[str, str]) -> dict[int, str]:
    """index -> split; a mutant with an unknown parent is excluded, never guessed."""
    out: dict[int, str] = {}
    for i, m in enumerate(mutants):
        if m.parent_name in decl_splits:
            out[i] = decl_splits[m.parent_name]
    return out


def manifest(rows: list[DeclarationRow], rule: str = CURRENT_RULE) -> dict:
    """The split manifest: assignment plus the honesty statistics."""
    rule = rule_id(rule)
    splits = assign_under(rule, rows)
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
        "rule": next(d for d, r in RULE_DESCRIPTIONS.items() if r == rule),
        "rule_id": rule,
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
