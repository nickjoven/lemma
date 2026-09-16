"""lemma #5: the S2 same-lock evaluation must be held-out. Only test-split
groups are queried, only test-split rows are candidates, and a lock that
also occurs outside the split is an error, not a silent leak."""

import random

import pytest
import torch

from lemma.data import splits
from lemma.data.corpus import CorpusError
from lemma.evaluate_s2 import exact_text_mrr, heldout_pool, same_lock_mrr
from lemma.schemas import DeclarationRow


def decl(name, module, lock, text=None):
    return DeclarationRow(name=name, module=module, kind="theorem",
                          canonical_type=text or f"T[{name}]", lock=lock)


def fixture_rows():
    rows = []
    # three duplicate groups, each inside one module family; plus singletons
    for i in range(60):
        rows.append(decl(f"a{i}", f"Mathlib.Fam{i}.A", f"dup{i}", text=f"stmt{i}"))
        rows.append(decl(f"b{i}", f"Mathlib.Fam{i}.A", f"dup{i}", text=f"stmt{i}"))
        rows.append(decl(f"s{i}", f"Mathlib.Fam{i}.B", f"solo{i}"))
    return rows


def test_only_test_split_groups_and_candidates_are_used():
    rows = fixture_rows()
    assign = splits.assign(rows)
    by_split = {s: {r.name for r in rows if assign[r.name] == s} for s in splits.SPLITS}
    assert by_split["test"], "fixture must put something in test"
    members, distractors, stats = heldout_pool(rows, assign, 1000, 1000, random.Random(0))
    assert members and {r.name for r in members} <= by_split["test"]
    assert {r.name for r in distractors} <= by_split["test"]
    assert not ({r.name for r in members} & {r.name for r in distractors})
    assert stats["split"] == "test" and stats["queries"] == len(members)
    # every query has a same-lock partner inside the pool
    locks = [r.lock for r in members + distractors]
    for r in members:
        assert locks.count(r.lock) >= 2


def test_leaked_lock_is_an_error_not_a_silent_pool():
    rows = fixture_rows()
    assign = splits.assign(rows)
    test_group = next(r for r in rows if assign[r.name] == "test" and r.lock.startswith("dup"))
    # forge a training row that shares a test lock (splits.assign never produces this)
    rows.append(decl("leak", "Mathlib.Leak.X", test_group.lock))
    assign["leak"] = "train"
    with pytest.raises(CorpusError, match="outside the test split"):
        heldout_pool(rows, assign, 1000, 1000, random.Random(0))


def test_pool_is_independent_of_row_order():
    rows = fixture_rows()
    assign = splits.assign(rows)
    m1, d1, _ = heldout_pool(rows, assign, 5, 10, random.Random(7))
    m2, d2, _ = heldout_pool(rows[::-1], assign, 5, 10, random.Random(7))
    assert [r.name for r in m1] == [r.name for r in m2]
    assert [r.name for r in d1] == [r.name for r in d2]


def test_mrr_helpers():
    locks = ["x", "x", "y"]
    sims = torch.tensor([[-1.0, 0.9, 0.1], [0.9, -1.0, 0.1], [0.5, 0.2, -1.0]])
    mrr, hit1 = same_lock_mrr(sims, locks, 2)
    assert mrr == 1.0 and hit1 == 1.0
    assert exact_text_mrr(["s", "s", "t"], locks, 2) == 1.0
    assert exact_text_mrr(["s", "s2", "t"], locks, 2) == 0.0
