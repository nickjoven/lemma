"""Review of #10: every entrypoint that records a run must check the source
tree before its compute (ledger.preflight_source) and hand that state to
start_run, so a tree edited during the compute is refused instead of being
recorded as the clean commit it was at the end."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "lemma"
ENTRYPOINTS = ["train.py", "train_s3.py", "train_s4.py", "train_s4v2.py", "train_s5.py", "train_w0.py",
               "w0_baselines.py", "evaluate_s2.py", "evaluate_s3.py", "evaluate_s4.py", "evaluate_s4v2.py",
               "evaluate_s5.py"]


def _is_call(node, name):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == name)


@pytest.mark.parametrize("fname", ENTRYPOINTS)
def test_start_run_is_preceded_by_preflight_and_bound_to_it(fname):
    tree = ast.parse((SRC / fname).read_text())
    seen = 0
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        starts = [c for c in calls if _is_call(c, "start_run")]
        if not starts:
            continue
        seen += 1
        pre = [c for c in calls if _is_call(c, "preflight_source")]
        assert pre, f"{fname}:{fn.name} records a run without preflight_source"
        assert min(p.lineno for p in pre) < min(s.lineno for s in starts)
        for s in starts:
            assert any(k.arg == "preflight" for k in s.keywords), \
                f"{fname}:{fn.name} line {s.lineno}: start_run without preflight="
    assert seen, f"{fname} has no start_run call"


@pytest.mark.parametrize("fname", ENTRYPOINTS)
def test_preflight_runs_before_corpus_and_checkpoint_loads(fname):
    tree = ast.parse((SRC / fname).read_text())
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        pre = [c.lineno for c in calls if _is_call(c, "preflight_source")]
        if not pre:
            continue
        loads = [c.lineno for c in calls
                 if isinstance(c.func, (ast.Name, ast.Attribute))
                 and getattr(c.func, "id", getattr(c.func, "attr", "")) in
                 {"iter_rows", "load_file", "load_s2_encoder", "load_encoder", "citable", "from_file"}]
        assert not loads or min(pre) < min(loads), f"{fname}:{fn.name}: preflight after a load"
