#!/usr/bin/env python3
"""Tests for commit_bypass.py (DUCK 38cg). Run: python3 test_commit_bypass.py"""
import ast, asyncio, copy, glob, inspect, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commit_bypass as B  # noqa: E402

FAILS = []
FLAG = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def cell(src):
    return {"cell_type": "code", "source": src.splitlines(True), "outputs": [], "execution_count": None, "metadata": {}}


def make_nb(true_sub):
    return {"cells": [
        {"cell_type": "markdown", "source": "# x"},
        cell("from pathlib import Path\nTRUE_SUBMISSION = %r\nWORKING_DIR = Path(OUT)\nlog = []\n" % true_sub),
        cell(B.LAUNCHER + "\nimport time\nserver = 'up'\nlog.append('launch')\ndef helper():\n    return server\n"),
        cell("bm = {'games': 0}\nlog.append('setup')\n"),
        cell(B.CENSUS + " ---\nlog.append('census:' + helper())\n"),
        cell("print('Starting benchmark...')\nasync def run():\n    log.append('bench')\n    return 3\n"
             "if TRUE_SUBMISSION:\n    bm['games'] = await run()\nelse:\n    bm['games'] = -1\n"),
        cell("log.append('report')\n"),
    ]}


def run_nb(nb, out):
    """Execute code cells like a notebook kernel with top-level await."""
    g = {"OUT": out}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        code = compile(B._src(c), "<cell>", "exec", flags=FLAG)
        r = eval(code, g)
        if inspect.iscoroutine(r):
            asyncio.run(r)
    return g


# Kaggle images ship pandas; the test machine may not, so a stand-in records the placeholder write.
import types
if "pandas" not in sys.modules:
    try:
        import pandas  # noqa: F401
    except ImportError:
        _pd = types.ModuleType("pandas")

        class _DF:
            def __init__(self, rows, columns):
                self.rows, self.columns = rows, columns

            def to_parquet(self, path, index=False):
                Path(path).write_text(json.dumps({"columns": self.columns, "rows": self.rows}))
        _pd.DataFrame = _DF
        sys.modules["pandas"] = _pd

tmp = tempfile.mkdtemp()
# reference: untransformed TRUE run
ref = run_nb(make_nb(True), tmp)
for true_sub in (True, False):
    nb = make_nb(true_sub)
    done = B.apply(nb)
    g = run_nb(nb, tmp)
    if true_sub:
        check("scored rerun path identical: same log", g["log"] == ref["log"] == ["launch", "setup", "census:up", "bench", "report"])
        check("scored rerun path: bench result kept", g["bm"]["games"] == 3)
        check("scored rerun path: launcher names visible", g["server"] == "up" and g["helper"]() == "up")
    else:
        check("commit path skips launcher, census, bench, report", g["log"] == ["setup"])
        check("commit path writes placeholder parquet", (Path(tmp) / "submission.parquet").exists())
    check("actions reported (%s)" % true_sub, [a for _, a in done] == ["launcher exec-wrapped", "census exec-wrapped",
                                                                    "bench indented under TRUE_SUBMISSION",
                                                                    "post-bench exec-wrapped"])
nb = make_nb(True)
B.apply(nb)
try:
    B.apply(nb)
    check("double apply refused", False)
except ValueError:
    check("double apply refused", True)
bad = make_nb(True)
bad["cells"][5] = cell("print('Starting benchmark...')\nx = '''a\nb'''\n")
try:
    B.apply(bad)
    check("triple-quoted bench refused", False)
except ValueError:
    check("triple-quoted bench refused", True)
nolaunch = make_nb(True)
del nolaunch["cells"][2]
try:
    B.apply(nolaunch)
    check("missing launcher refused", False)
except ValueError:
    check("missing launcher refused", True)
magic = make_nb(True)
magic["cells"][4] = cell(B.CENSUS + " ---\n!nvidia-smi\n")
try:
    B.apply(magic)
    check("shell escape in a wrapped cell refused", False)
except ValueError:
    check("shell escape in a wrapped cell refused", True)
early = make_nb(True)
early["cells"][1] = cell("OUT2 = 1\n")
try:
    B.apply(early)
    check("TRUE_SUBMISSION must be defined before launcher", False)
except ValueError:
    check("TRUE_SUBMISSION must be defined before launcher", True)
src_nb = os.path.join(tmp, "in.ipynb")
json.dump(make_nb(False), open(src_nb, "w"))
check("cli rc 0", B.main([src_nb, os.path.join(tmp, "out.ipynb")]) == 0 and
      B.MARK in json.dumps(json.load(open(os.path.join(tmp, "out.ipynb")))))

# --- the real M2 notebook (skipped when absent) ---
real = glob.glob(os.path.expanduser("~/m2/kaggle-franzen-fork/*.ipynb"))
if real:
    nb = json.load(open(real[0], encoding="utf-8"))
    orig = copy.deepcopy(nb)
    done = B.apply(nb)
    check("real: launcher, census, bench and report wrapped", len(done) == 4)
    ok = True
    for c in nb["cells"]:
        if c["cell_type"] == "code" and B._src(c).startswith(B.MARK):   # untouched cells may hold IPython syntax
            try:
                compile(B._src(c), "<c>", "exec", flags=FLAG)
            except SyntaxError:
                ok = False
    check("real: every transformed code cell compiles (top-level await allowed)", ok)
    # the exec-wrapped sources are the original texts, unchanged
    origs = {B._src(c) for c in orig["cells"] if c["cell_type"] == "code"}
    emb = []
    for c in nb["cells"]:
        s = B._src(c)
        if s.startswith(B.MARK) and "exec(compile(" in s:
            node = next(n for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "compile")
            emb.append(ast.literal_eval(node.args[0]))
    check("real: exec-wrapped texts are byte-identical originals", emb and all(e in origs for e in emb))
    bench_new = next(B._src(c) for c in nb["cells"] if B._src(c).startswith(B.MARK) and "if TRUE_SUBMISSION:\n    print('Starting" in B._src(c))
    body = "".join(ln[4:] if ln.startswith("    ") else ln for ln in bench_new.split("if TRUE_SUBMISSION:\n", 1)[1].split("else:\n    import pandas as _fi939_pd")[0].splitlines(True))
    orig_bench = next(s for s in origs if s.startswith(B.BENCH))
    check("real: indented bench body is the original text", body.rstrip("\n") == orig_bench.rstrip("\n"))
else:
    print("SKIP real-notebook checks")

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
