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


# review of #10, finding 1: the split is resolved by evidence, the cache report is required,
# and main() wires the preflight, the binding and the recorded CID

def _report(path, assign, counts=None):
    import json
    counts = counts or {s: sum(1 for v in assign.values() if v == s) for s in splits.SPLITS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"splits": {s: {"sequences": n} for s, n in counts.items()}}))
    return path


def test_missing_cache_report_is_an_error_not_a_skipped_check(tmp_path):
    from lemma.evaluate_s2 import check_cache_report
    rows = fixture_rows()
    with pytest.raises(CorpusError, match="missing"):
        check_cache_report(tmp_path / "cache-report.json", splits.assign(rows))


def test_cache_report_counts_must_match_the_resolved_split(tmp_path):
    from lemma.evaluate_s2 import check_cache_report
    assign = splits.assign(fixture_rows())
    assert check_cache_report(_report(tmp_path / "r.json", assign), assign)
    with pytest.raises(CorpusError, match="differ from the training cache"):
        check_cache_report(_report(tmp_path / "bad.json", assign, {"train": 1, "val": 1, "test": 1}), assign)


def test_manifest_file_must_name_the_resolved_rule_and_hold_its_assignment(tmp_path):
    import json

    from lemma.data.split_binding import Binding
    from lemma.evaluate_s2 import check_manifest_file
    from test_split_binding import rule_change_rows
    rows = rule_change_rows()
    legacy = Binding(splits.assign_v1(rows), splits.RULE_V1, "cid", "pinned_rule_reconstructed")
    # build_cache now overwrites the mutable manifest with a v2 one: refused for a v1 run
    (tmp_path / "v2.json").write_text(json.dumps(splits.manifest(rows)))
    with pytest.raises(CorpusError, match="built under v2"):
        check_manifest_file(tmp_path / "v2.json", legacy)
    # an older manifest that carries only the free-text rule is accepted iff it is the same split
    old = {"rule": "module-area blake3, lock-follows-first, mutant-inherits-parent",
           "assignment": splits.assign_v1(rows)}
    (tmp_path / "v1.json").write_text(json.dumps(old))
    assert len(check_manifest_file(tmp_path / "v1.json", legacy)) == 64
    old["assignment"] = dict(old["assignment"], a_dup="test")
    (tmp_path / "v1-edited.json").write_text(json.dumps(old))
    with pytest.raises(CorpusError, match="disagrees with the resolved split on 1 rows"):
        check_manifest_file(tmp_path / "v1-edited.json", legacy)


class _Harness:
    """Drive evaluate_s2.main() on CPU: a sealed fixture train run in a temp
    ket store, a tiny word-level tokenizer and encoder, fixture corpus rows,
    and a temp repo root holding the cache report and the train-cache ids."""

    def __init__(self, tmp_path, monkeypatch, rows, *, split_manifest_cid=None, pins=None):
        import json
        import sys

        import numpy as np
        from tokenizers import Tokenizer, models, pre_tokenizers

        from lemma import evaluate_s2, ledger
        from lemma.data import split_binding
        from lemma.models.encoder import Encoder, EncoderConfig
        from lemma.schemas import RunInputs, RunRecord

        self.ledger, self.rows = ledger, rows
        texts = sorted({tok for r in rows for tok in r.canonical_type.split()})
        vocab = {"<pad>": 0, "<unk>": 1, "<mask>": 2, "<s>": 3, **{t: i + 4 for i, t in enumerate(texts)}}
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        monkeypatch.setattr(evaluate_s2.Tokenizer, "from_file", staticmethod(lambda _p: tok))

        def small(vocab_size, **kw):
            return EncoderConfig(vocab_size=vocab_size, max_seq=16, d_model=16, n_layers=1, n_heads=2, d_ff=32)
        monkeypatch.setattr(evaluate_s2, "EncoderConfig", small)
        self.loaded = []

        def load_encoder(ckpt, vocab_size, device):
            self.loaded.append(ckpt)
            torch.manual_seed(1)
            return Encoder(small(vocab_size)).eval()
        monkeypatch.setattr(evaluate_s2, "load_encoder", load_encoder)
        monkeypatch.setattr(evaluate_s2, "iter_rows", lambda *a, **k: iter(rows))
        root = tmp_path / "evalroot"
        (root / "corpora" / "cache").mkdir(parents=True)
        np.save(root / "corpora" / "cache" / "declarations-train.ids.npy", np.arange(4, 4 + len(texts)) % len(vocab))
        monkeypatch.setattr(evaluate_s2, "REPO_ROOT", root)
        self.report = root / "corpora" / "cache" / "cache-report.json"
        self.pins = tmp_path / "pins.yml"
        self.pins.write_text(json.dumps({"runs": pins or {}}))
        monkeypatch.setattr(split_binding, "PINS_PATH", self.pins)

        self.train = RunRecord(run_id="s2-fixture", kind="train", stage="S2", started=ledger.now_iso(),
                               inputs=RunInputs(corpus_cids=["corpus"], config_cid="cfg", code_git=ledger.git_head(),
                                                mathlib_pin="pin", split_manifest_cid=split_manifest_cid))
        self.train.checkpoint_cids.append(ledger.ket_put(b"fixture checkpoint"))
        ledger.finish_run(self.train, {"masked_acc": 0.9})
        self.preflights = []
        real = ledger.preflight_source

        def spy(*a, **k):
            self.preflights.append(len(self.loaded))     # how many checkpoints were loaded before it
            return real(*a, **k)
        monkeypatch.setattr(ledger, "preflight_source", spy)
        monkeypatch.setattr(sys, "argv", ["evaluate_s2", "--run", "s2-fixture", "--groups", "50",
                                          "--distractors", "50"])

    def run(self):
        from lemma import evaluate_s2
        return evaluate_s2.main()

    def eval_record(self):
        import json
        recs = [r for r in self.ledger.read_ledger() if r.run_id == "s2-eval-s2-fixture" and r.status != "running"]
        assert len(recs) == 1
        return recs[0], json.loads(self.ledger.ket_get(recs[0].metrics_cid))["metrics"]


def test_main_refuses_an_unpinned_legacy_run_before_loading_anything(tmp_path, monkeypatch, ket_store):
    from lemma.data.split_binding import SplitBindingError
    h = _Harness(tmp_path, monkeypatch, fixture_rows())
    _report(h.report, splits.assign(fixture_rows()))
    with pytest.raises(SplitBindingError, match="no split-rule pin"):
        h.run()
    assert h.loaded == [] and h.preflights == [0]


def test_main_on_a_pinned_legacy_run_records_the_v1_binding(tmp_path, monkeypatch, ket_store):
    import json
    rows = fixture_rows()
    h = _Harness(tmp_path, monkeypatch, rows, pins={"s2-fixture": {"split_rule": splits.RULE_V1}})
    _report(h.report, splits.assign_v1(rows))
    h.run()
    rec, metrics = h.eval_record()
    assert h.preflights == [0], "preflight must run once, before the checkpoint is loaded"
    held = metrics["heldout"]
    assert held["split_manifest_binding"] == "pinned_rule_reconstructed"
    assert held["split_rule"] == splits.RULE_V1 and held["split_pin_source"] == "configs/pins.yml"
    assert rec.inputs.split_manifest_cid == held["split_manifest_cid"]
    cert = json.loads(ket_store.joinpath("cas", rec.inputs.split_manifest_cid).read_bytes())
    assert splits.certificate_mismatch(cert, splits.assign_v1(rows)) is None
    assert "exact_text_hit_rate" in metrics
    assert rec.inputs.code_git == h.ledger.git_head() and rec.inputs.code_git_dirty is None


def test_main_refuses_a_missing_cache_report(tmp_path, monkeypatch, ket_store):
    h = _Harness(tmp_path, monkeypatch, fixture_rows(), pins={"s2-fixture": {"split_rule": splits.RULE_V1}})
    with pytest.raises(CorpusError, match="cache-report.json is missing"):
        h.run()
    assert h.loaded == []


def test_main_binds_by_the_recorded_certificate_and_rejects_a_rebuilt_cache(tmp_path, monkeypatch, ket_store):
    from lemma.data import split_binding
    rows = fixture_rows()
    cid = split_binding.put_certificate(splits.certificate(splits.assign(rows), splits.RULE_V2))
    h = _Harness(tmp_path, monkeypatch, rows, split_manifest_cid=cid)
    _report(h.report, splits.assign(rows), {"train": 1, "val": 2, "test": 3})   # a cache rebuilt since
    with pytest.raises(CorpusError, match="differ from the training cache"):
        h.run()
    _report(h.report, splits.assign(rows))
    h.run()
    rec, metrics = h.eval_record()
    assert metrics["heldout"]["split_manifest_binding"] == "train_record_cid"
    assert rec.inputs.split_manifest_cid == cid


def test_main_refuses_a_dirty_tree_before_loading_anything(tmp_path, monkeypatch, ket_store, clean_repo):
    from lemma import ledger
    h = _Harness(tmp_path, monkeypatch, fixture_rows(), pins={"s2-fixture": {"split_rule": splits.RULE_V1}})
    _report(h.report, splits.assign_v1(fixture_rows()))
    (clean_repo / "src" / "evaluator.py").write_text("print('edited')\n")
    with pytest.raises(ledger.DirtySourceError):
        h.run()
    assert h.loaded == []


def test_cross_module_duplicate_group_is_held_out_whole():
    """Review of #10 test gap: a duplicate group whose members live in two
    modules (one component under the current rule). The group and both
    modules must be in one split, the group must be queryable there, and
    under the old rule the same corpus splits a module, which the pool
    disclosure must report."""
    from test_split_binding import rule_change_rows
    rows = rule_change_rows()
    assign = splits.assign(rows)
    group = [r for r in rows if r.lock == "L"]
    assert len({r.module for r in group}) == 2 and {assign[r.name] for r in group} == {"test"}
    members, _, stats = heldout_pool(rows, assign, 1000, 1000, random.Random(0))
    assert {r.name for r in group} <= {r.name for r in members}
    assert stats["modules_shared_with_other_splits"] == 0
    # v1 keeps lock L in one split (train) but leaves module A's other rows in test
    v1 = splits.assign_v1(rows)
    _, _, legacy = heldout_pool(rows, v1, 1000, 1000, random.Random(0))
    assert legacy["modules_shared_with_other_splits"] >= 1
