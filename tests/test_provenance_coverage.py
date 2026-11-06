"""Review of #10: every entrypoint that records a run must check the source
tree before its compute (ledger.preflight_source) and hand that state to
start_run, so a tree edited during the compute is refused instead of being
recorded as the clean commit it was at the end.

Second review: the entrypoints are DISCOVERED (every module that refers to
start_run or finish_run), the ledger API may only be called directly as
ledger.start_run(...) / ledger.finish_run(...) (an alias or a from-import
would hide a call from these checks, so it fails), and preflight= must be
the variable the same function bound from ledger.preflight_source(), not
None or a constant.

These are lint-style guards against ACCIDENTAL regressions (a new
entrypoint that forgets preflight, an aliased or from-imported ledger call,
preflight=None), not a proof against deliberate bypass: the third review of
#10 got past them by mutating the preflight state after binding it, wrapping
load_file in a helper one level down, reaching the API through importlib or a
concatenated getattr string, and re-reading a corpus with raw open(). The
runtime check is start_run comparing the tree with the preflight state it
is handed (which the caller could alter); these tests only catch the
straightforward ways of forgetting to call or wire it."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "lemma"
API = {"start_run", "finish_run"}
# the entrypoints known when this test was written; discovery must find at least these
KNOWN = {"train.py", "train_s3.py", "train_s4.py", "train_s4v2.py", "train_s5.py", "train_w0.py",
         "w0_baselines.py", "evaluate_s2.py", "evaluate_s3.py", "evaluate_s4.py", "evaluate_s4v2.py",
         "evaluate_s5.py"}


def _refers_to_api(tree) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in API:
            return True
        if isinstance(n, ast.Name) and n.id in API:
            return True
        if isinstance(n, (ast.ImportFrom, ast.Import)) and any(a.name in API or a.name == "*" for a in n.names):
            return True
        if isinstance(n, ast.Constant) and n.value in API:
            return True
    return False


def _discover() -> list[str]:
    out = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel != "ledger.py" and _refers_to_api(ast.parse(path.read_text())):
            out.append(rel)
    return out


ENTRYPOINTS = _discover()


def test_entrypoints_are_discovered():
    assert KNOWN <= set(ENTRYPOINTS), f"discovery missed {sorted(KNOWN - set(ENTRYPOINTS))}"


def _is_call(node, name):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == name and isinstance(node.func.value, ast.Name) and node.func.value.id == "ledger")


def api_violations(tree) -> list[str]:
    direct = {id(n.func) for n in ast.walk(tree) if _is_call(n, "start_run") or _is_call(n, "finish_run")}
    errs = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in API and id(n) not in direct:
            errs.append(f"{n.lineno}: {n.attr} used other than as a direct ledger.{n.attr}(...) call")
        if isinstance(n, ast.Name) and n.id in API:
            errs.append(f"{n.lineno}: bare {n.id} (an imported or aliased ledger call)")
        if isinstance(n, ast.ImportFrom) and any(a.name in API or a.name == "*" for a in n.names):
            errs.append(f"{n.lineno}: imports the ledger API by name")
        if isinstance(n, (ast.Import, ast.ImportFrom)) and any(
                a.name.split(".")[-1] == "ledger" and a.asname not in (None, "ledger") for a in n.names):
            errs.append(f"{n.lineno}: imports ledger under another name")
        if isinstance(n, ast.Constant) and n.value in API:
            errs.append(f"{n.lineno}: refers to {n.value} by string (getattr)")
        if isinstance(n, (ast.Assign, ast.NamedExpr)) and isinstance(n.value, ast.Name) and n.value.id == "ledger":
            errs.append(f"{n.lineno}: aliases the ledger module")
    return errs


def _bound_from_preflight(fn, name: str, before: int) -> bool:
    """`name` is assigned in `fn` exactly once, from ledger.preflight_source(...), before line `before`."""
    binds = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.For, ast.With)):
            targets = [getattr(n, "target", None)] + [i.optional_vars for i in getattr(n, "items", [])]
        else:
            continue
        if any(isinstance(x, ast.Name) and x.id == name for t in targets if t is not None for x in ast.walk(t)):
            binds.append(n)
    args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    return (name not in args and len(binds) == 1 and isinstance(binds[0], ast.Assign)
            and len(binds[0].targets) == 1 and isinstance(binds[0].targets[0], ast.Name)
            and _is_call(binds[0].value, "preflight_source") and binds[0].lineno < before)


def preflight_violations(tree) -> tuple[int, list[str]]:
    seen, errs = 0, []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        starts = [c for c in calls if _is_call(c, "start_run")]
        if not starts:
            continue
        seen += 1
        pre = [c for c in calls if _is_call(c, "preflight_source")]
        if not pre or min(p.lineno for p in pre) > min(s.lineno for s in starts):
            errs.append(f"{fn.name}: start_run without an earlier preflight_source")
        for s in starts:
            kw = [k for k in s.keywords if k.arg == "preflight"]
            if not kw:
                errs.append(f"{fn.name} line {s.lineno}: start_run without preflight=")
            elif not (isinstance(kw[0].value, ast.Name) and _bound_from_preflight(fn, kw[0].value.id, s.lineno)):
                errs.append(f"{fn.name} line {s.lineno}: preflight= is not the state preflight_source returned")
    return seen, errs


@pytest.mark.parametrize("fname", ENTRYPOINTS)
def test_the_ledger_api_is_only_called_directly(fname):
    assert api_violations(ast.parse((SRC / fname).read_text())) == [], fname


@pytest.mark.parametrize("fname", ENTRYPOINTS)
def test_start_run_is_preceded_by_preflight_and_bound_to_it(fname):
    seen, errs = preflight_violations(ast.parse((SRC / fname).read_text()))
    assert errs == [], fname
    assert seen, f"{fname} refers to the ledger API but has no start_run call"


@pytest.mark.parametrize("snippet", [
    "from .ledger import start_run\ndef main():\n    s = ledger.preflight_source()\n    start_run(r, preflight=s)\n",
    "from . import ledger\nsr = ledger.start_run\ndef main():\n    s = ledger.preflight_source()\n    sr(r, preflight=s)\n",
    "from . import ledger as L\ndef main():\n    s = L.preflight_source()\n    L.start_run(r, preflight=s)\n",
    "def main():\n    getattr(ledger, 'finish_run')(r, {})\n",
])
def test_aliased_ledger_calls_are_caught(snippet):
    assert api_violations(ast.parse(snippet))


@pytest.mark.parametrize("snippet", [
    "def main():\n    s = ledger.preflight_source()\n    ledger.start_run(r, preflight=None)\n",
    "def main():\n    s = ledger.preflight_source()\n    ledger.start_run(r, preflight={})\n",
    "def main():\n    s = ledger.preflight_source()\n    s = {}\n    ledger.start_run(r, preflight=s)\n",
    "def main(s):\n    ledger.preflight_source()\n    ledger.start_run(r, preflight=s)\n",
    "def main():\n    ledger.start_run(r, preflight=s)\n    s = ledger.preflight_source()\n",
])
def test_a_preflight_that_is_not_the_preflight_state_is_caught(snippet):
    assert preflight_violations(ast.parse(snippet))[1]


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
                 {"iter_rows", "read_declarations", "resolve_corpus", "load_file", "load_s2_encoder",
                  "load_encoder", "citable", "from_file"}]
        assert not loads or min(pre) < min(loads), f"{fname}:{fn.name}: preflight after a load"
