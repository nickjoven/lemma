"""S4 v2 data path: split inheritance, OOD hold-out, class indices from the
collapse rule — on the pure function, no corpus or GPU needed."""

from lemma.data.labels import collapse_rule
from lemma.train_s4v2 import labelled_by_split


def rows(n_proven, n_npf, n_rej, parent="p", op="swap_eq_ne"):
    mk = lambda lab: {"text": f"T[{parent}/{op}/{lab}]", "parent": parent, "operator": op,
                      "op_class": op, "prover": "ladder-A", "label3": lab}
    return [mk("proven")] * n_proven + [mk("no_proof_found")] * n_npf + [mk("rejected")] * n_rej


def test_mutants_inherit_parent_split_and_ood_is_test_only():
    r = rows(2, 2, 0, parent="a") + rows(1, 1, 0, parent="b", op="swap_lit_01") + rows(1, 0, 0, parent="zzz")
    classes, _ = collapse_rule(r, min_support=30)
    assert classes == ["proven", "not_proven"]
    data = labelled_by_split(r, {"a": "train", "b": "train"}, classes, ood_operator="swap_lit_01")
    assert len(data["train"]) == 4 and data["val"] == [] and data["test"] == []   # b: ood dropped from train; zzz: unknown parent
    data2 = labelled_by_split(r, {"a": "train", "b": "test"}, classes, ood_operator="swap_lit_01")
    assert len(data2["test"]) == 2 and data2["test"][0][2] == "swap_lit_01"
    assert {t[1] for t in data["train"]} == {0, 1} and data["train"][0][3] == "proven"


def test_three_classes_when_rejected_has_support():
    r = rows(40, 40, 40)
    classes, support = collapse_rule(r, min_support=30)
    assert classes == ["proven", "no_proof_found", "rejected"] and support["rejected"] == 40
    data = labelled_by_split(r, {"p": "val"}, classes, None)
    assert sorted({t[1] for t in data["val"]}) == [0, 1, 2]
