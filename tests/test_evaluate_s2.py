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
    assert exact_text_mrr(["s", "s", "t"], locks, 2) == (1.0, 1.0)
    # no identical text: the same-lock row is found in the score-0 block, rank 1 or 2 at random
    mrr0, hit0 = exact_text_mrr(["s", "s2", "t"], locks, 2)
    assert hit0 == 0.0 and mrr0 == pytest.approx(0.75)


def _brute_force_exact_text_mrr(texts, locks, n_queries):
    """Average the reciprocal rank over EVERY tie-break order."""
    import itertools
    out = []
    for i in range(n_queries):
        cands = [j for j in range(len(texts)) if j != i]
        rrs = []
        for order in itertools.permutations(cands):
            ranked = sorted(order, key=lambda j: texts[j] != texts[i])   # stable: score-1 block first
            rrs.append(next(1 / (r + 1) for r, j in enumerate(ranked) if locks[j] == locks[i]))
        out.append(sum(rrs) / len(rrs))
    return sum(out) / len(out)


def test_exact_text_mrr_is_a_ranked_mrr_and_counts_different_lock_twins():
    # query 0 has a same-lock identical twin (1) AND two different-lock rows with the same text (2, 3):
    # string equality cannot tell them apart, so its expected rank is not 1
    texts = ["s", "s", "s", "s", "u", "v"]
    locks = ["x", "x", "y", "z", "w", "x"]
    mrr, hit = exact_text_mrr(texts, locks, 2)
    assert hit == 1.0
    assert mrr < 1.0, "the binary indicator called this 1.0; a ranked MRR must not"
    assert mrr == pytest.approx(_brute_force_exact_text_mrr(texts, locks, 2))
    # a query whose only same-lock rows have different text ranks behind every identical-text row
    texts2 = ["s", "t", "s", "s", "u"]
    locks2 = ["x", "x", "y", "z", "w"]
    assert exact_text_mrr(texts2, locks2, 1)[0] == pytest.approx(_brute_force_exact_text_mrr(texts2, locks2, 1))


def test_duplicate_groups_available_is_counted_before_truncation():
    rows = fixture_rows()
    assign = splits.assign(rows)
    in_test = [r for r in rows if assign[r.name] == "test"]
    available = len({r.lock for r in in_test if sum(1 for x in in_test if x.lock == r.lock) >= 2})
    assert available >= 1, "fixture must put a duplicate group in test"
    _, _, stats = heldout_pool(rows, assign, 1, 10, random.Random(0))
    assert stats["groups"] == 1 and stats["duplicate_groups_available"] == available


def _manifest(tmp_path, rows, assign, counts=None):
    import json

    import blake3
    path = tmp_path / "split-manifest.json"
    data = json.dumps({"rule": "fixture", "assignment": assign}).encode()
    path.write_bytes(data)
    counts = counts or {s: sum(1 for v in assign.values() if v == s) for s in splits.SPLITS}
    (tmp_path / "cache-report.json").write_text(json.dumps({"splits": {s: {"sequences": n} for s, n in counts.items()}}))
    return path, blake3.blake3(data).hexdigest()


def test_trained_split_takes_the_manifest_the_checkpoint_trained_against(tmp_path):
    from lemma.evaluate_s2 import trained_split
    rows = fixture_rows()
    assign = splits.assign(rows)
    path, cid = _manifest(tmp_path, rows, assign)
    got, got_cid, rule = trained_split(path, None, rows)
    assert got == assign and got_cid == cid and rule == "fixture"
    assert trained_split(path, cid, rows)[1] == cid


def test_trained_split_refuses_a_manifest_the_train_record_does_not_name(tmp_path):
    from lemma.evaluate_s2 import trained_split
    rows = fixture_rows()
    path, _ = _manifest(tmp_path, rows, splits.assign(rows))
    with pytest.raises(CorpusError, match="train run records"):
        trained_split(path, "0" * 64, rows)


def test_trained_split_refuses_a_manifest_that_does_not_match_its_cache(tmp_path):
    from lemma.evaluate_s2 import trained_split
    rows = fixture_rows()
    assign = splits.assign(rows)
    path, _ = _manifest(tmp_path, rows, assign, counts={"train": 1, "val": 1, "test": 1})
    with pytest.raises(CorpusError, match="differ from the training cache"):
        trained_split(path, None, rows)


def test_trained_split_refuses_a_manifest_that_does_not_cover_the_corpus(tmp_path):
    from lemma.evaluate_s2 import trained_split
    rows = fixture_rows()
    assign = splits.assign(rows)
    path, _ = _manifest(tmp_path, rows, assign)
    rows.append(decl("new", "Mathlib.New.A", "newlock"))
    with pytest.raises(CorpusError, match="not in the split manifest"):
        trained_split(path, None, rows)


def test_module_split_across_splits_is_disclosed():
    # a pre-#6 manifest can put one module's rows in two splits while keeping
    # every lock inside one split; the pool stats must say so
    rows = fixture_rows()
    assign = splits.assign(rows)
    test_mod = next(r.module for r in rows if assign[r.name] == "test")
    _, _, clean = heldout_pool(rows, assign, 1000, 1000, random.Random(0))
    assert clean["modules_shared_with_other_splits"] == 0
    rows.append(decl("same-module-train", test_mod, "own-lock"))
    assign["same-module-train"] = "train"
    _, _, stats = heldout_pool(rows, assign, 1000, 1000, random.Random(0))
    assert stats["modules_shared_with_other_splits"] == 1
    assert stats["modules_in_split"] == clean["modules_in_split"]
