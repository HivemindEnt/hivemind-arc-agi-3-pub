#!/usr/bin/env python3
"""Tests for trackrecord.py and the build_arm_cell 'trackrecord' arm (DUCK 38bv). Run: python3 test_trackrecord.py"""
import ast, os, sys, types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trackrecord as tr  # noqa: E402
try:  # the arm builder lives in the private repo; the public copy runs the trackrecord checks without it
    import build_arm_cell as bac  # noqa: E402
except ImportError:
    bac = None

FAILS = []
REAL = os.environ.get("FI939_M2_SOLVER", os.path.expanduser(
    "~/m2/arc-agi-3-solution/ARC3-Inference/inference/framework/solver.py"))


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


FAKE_SRC = '''
NEW_NOOP_WORDING = True
def _action_verdict(item):
    if item.get("game_over"):
        return "GAME OVER (attempt ended immediately after this action)"
    if item.get("level_completed"):
        return "LEVEL COMPLETED"
    if item.get("gameplay_changed") is False:
        return "NO-OP (executed, no change inside the board area)"
    if item.get("gameplay_changed") is True:
        return "effect"
    return "executed"
def _build_action_echo(displays, payloads, stop_reason):
    return "[action] %s -> %s" % (displays[0], _action_verdict(payloads[0]))
class _HarnessGameSession:
    def __init__(self, script):
        self.script = list(script)
    def _execute_action(self, action, *, batch_index=0, batch_size=1, automatic=False):
        p = dict(self.script.pop(0))
        p.setdefault("executed", True)
        p["automatic"] = automatic
        return p
'''


def fake_solver():
    m = types.ModuleType("fake_solver")
    exec(FAKE_SRC, m.__dict__)
    return m


def P(name, changed, level=1, disp=None, **kw):
    d = {"action_name": name, "gameplay_changed": changed, "level": level, "action_display": disp or name}
    d.update(kw)
    return d


sv = fake_solver()
msg = tr.install(sv)
check("install returns description", "TRACKRECORD" in msg)
s = sv._HarnessGameSession([P("ACTION1", True, disp="UP"), P("ACTION1", False, disp="UP"),
                            P("ACTION1", False, disp="UP"), P("ACTION2", False, disp="DOWN")])
r1 = s._execute_action("a")
r2 = s._execute_action("a")
check("tally after effect + no-op", r1[tr.TAG] == (1, 1) and r2[tr.TAG] == (2, 1))
e2 = sv._build_action_echo(["UP"], [r2], None)
check("NO-OP echo carries record",
      e2 == "[action] UP -> NO-OP (executed, no change inside the board area) [UP on this level: 1 of 2 tries changed the board]")
check("effect verdict unchanged", sv._build_action_echo(["UP"], [r1], None) == "[action] UP -> effect")
r3 = s._execute_action("a")
r4 = s._execute_action("a")
check("controls tallied separately", r3[tr.TAG] == (3, 1) and r4[tr.TAG] == (1, 0))
check("0 of n is still reported", sv._action_verdict(r4).endswith("[DOWN on this level: 0 of 1 tries changed the board]"))

s2 = sv._HarnessGameSession([P("ACTION1", False, level=2, disp="UP"), P("ACTION1", False, disp="UP")])
check("per session and per level", s2._execute_action("a")[tr.TAG] == (1, 0) and s2._execute_action("a")[tr.TAG] == (1, 0))

s3 = sv._HarnessGameSession([P("ACTION1", False), P("RESET", True), P("ACTION1", True, game_over=True),
                             P("ACTION1", True, level_completed=True), P("ACTION1", None),
                             P("ACTION1", False, executed=False)])
a = s3._execute_action("a", automatic=True)
b, c, d, e, f = (s3._execute_action("a") for _ in range(5))
check("automatic / RESET / terminal / unknown / unexecuted not tallied",
      all(tr.TAG not in x for x in (a, b, c, d, e, f)) and "_fi939_tally" not in s3.__dict__)
check("game over verdict unchanged", sv._action_verdict(c).startswith("GAME OVER") and "[" not in sv._action_verdict(c))

s4 = sv._HarnessGameSession([P("ACTION6", False, disp="click(10,12)"), P("ACTION6", True, disp="click(3,4)")])
x = s4._execute_action("a")
check("clicks pooled and labelled", sv._action_verdict(x).endswith("[clicks on this level: 0 of 1 tries changed the board]"))
check("clicks tally", s4._execute_action("a")[tr.TAG] == (2, 1))

try:
    tr.install(sv)
    check("double install refused", False)
except RuntimeError:
    check("double install refused", True)
bad = types.ModuleType("bad")
try:
    tr.install(bad)
    check("missing session refused", False)
except RuntimeError:
    check("missing session refused", True)


sv5 = fake_solver()
sv5._HarnessGameSession._execute_action = lambda self, *a, **k: "not a dict"
tr.install(sv5)
check("non-dict payload passes through", sv5._HarnessGameSession([])._execute_action("a") == "not a dict")
check("verdict on item without tag unchanged", sv._action_verdict({"gameplay_changed": False}) ==
      "NO-OP (executed, no change inside the board area)")

# --- the arm cell runs against a module registered as inference.framework.solver ---
if bac is None:
    print("SKIP arm-cell checks (build_arm_cell.py not present)")
else:
    cell = bac.cell_for("trackrecord")
    compile(cell, "<cell>", "exec")
    pkg = types.ModuleType("inference"); fw = types.ModuleType("inference.framework"); fs = fake_solver()
    pkg.framework = fw; fw.solver = fs
    saved = {k: sys.modules.get(k) for k in ("inference", "inference.framework", "inference.framework.solver")}
    sys.modules.update({"inference": pkg, "inference.framework": fw, "inference.framework.solver": fs})
    try:
        exec(cell, {})
        ss = fs._HarnessGameSession([P("ACTION5", False, disp="SPACE")])
        check("cell installs on the solver module",
              fs._action_verdict(ss._execute_action("a")).endswith("[SPACE on this level: 0 of 1 tries changed the board]"))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    check("cell embeds current trackrecord source", repr(bac.trackrecord_source()) in cell)
    check("arm listed in docstring", "trackrecord" in bac.__doc__)

# --- shape of the real M2 solver (skipped when the source is not on this machine) ---
if os.path.exists(REAL):
    src = open(REAL, encoding="utf-8").read()
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    sess = classes.get("_HarnessGameSession")
    meths = {n.name for n in sess.body if isinstance(n, ast.FunctionDef)} if sess else set()
    check("real: _HarnessGameSession._execute_action exists", "_execute_action" in meths)
    check("real: _action_verdict is a module function", "_action_verdict" in funcs)
    uses = lambda fn: any(isinstance(n, ast.Name) and n.id == "_action_verdict" for n in ast.walk(funcs[fn]))
    check("real: echo and trace look _action_verdict up by global name",
          uses("_build_action_echo") and uses("_build_action_trace"))
    step = next(n for n in sess.body if isinstance(n, ast.FunctionDef) and n.name == "step_env")
    seg = ast.get_source_segment(src, step)
    check("real: step_env appends the payload _execute_action returned",
          "payload = self._execute_action(" in seg and "executed_payloads.append(payload)" in seg)
    # run the REAL verdict/echo/trace functions through the patch
    m = types.ModuleType("real_subset")
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in
                                     ("_action_verdict", "_build_action_echo", "_build_action_trace"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("_BLOCKED_VERDICTS",) for t in n.targets))]
    m.__dict__["NEW_NOOP_WORDING"] = True
    exec("from __future__ import annotations\n" + "\n\n".join(ast.get_source_segment(src, n) for n in keep), m.__dict__)
    m._HarnessGameSession = fake_solver()._HarnessGameSession
    tr.install(m)
    rs = m._HarnessGameSession([P("ACTION3", True, disp="LEFT"), P("ACTION3", False, disp="LEFT")])
    p1, p2 = rs._execute_action("a"), rs._execute_action("a")
    check("real echo carries record", m._build_action_echo(["LEFT"], [p2], None).endswith(
        "[LEFT on this level: 1 of 2 tries changed the board]"))
    trace = m._build_action_trace(["LEFT", "LEFT"], [p1, p2])
    check("real batch trace carries record on the NO-OP line only",
          trace[1].endswith("1 of 2 tries changed the board]") and "[" not in trace[0].split("->", 1)[1])
else:
    print("SKIP real-solver checks (no %s)" % REAL)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
