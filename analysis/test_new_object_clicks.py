#!/usr/bin/env python3
"""Tests for new_object_clicks.py (DUCK 38bz). Run: python3 test_new_object_clicks.py"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import new_object_clicks as N  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def board(objs, n=8):
    b = [[0] * n for _ in range(n)]
    for col, cells in objs:
        for r, c in cells:
            b[r][c] = col
    return b


A = (3, [(1, 1), (1, 2)])              # colour 3 horizontal domino
A2 = (3, [(5, 5), (5, 6)])             # same type elsewhere
B = (4, [(4, 1), (5, 1), (6, 1)])      # colour 4 vertical bar: appears on level 2
where, comps = N.components(board([A, B]))
check("two components, background ignored", len(comps) == 2)
check("singletons ignored", len(N.components(board([(5, [(0, 0)])]))[1]) == 0)
check("type is position-free", N.types_of(board([A])) == N.types_of(board([A2])))
check("shape matters", N.types_of(board([A])) != N.types_of(board([(3, [(1, 1), (2, 1)])])))


def row(kind, level, b, disp=None):
    r = {"type": kind, "board": b}
    if kind == "action":
        r.update(level=level, action_display=disp or "UP")
    return r


L1 = board([A])
L2 = board([A2, B])
rows = [row("initial", None, L1), row("action", 1, L1), row("action", 2, L2),          # completing action -> L2 start
        row("action", 2, L2, "MOUSE(row=5, col=5)"),     # click on old type
        row("action", 2, L2, "MOUSE(row=5, col=1)"),     # click on appeared bar
        row("action", 2, L2, "MOUSE(row=0, col=7)"),     # click on background
        row("action", 2, L2, "LEFT")]
res = N.analyse(rows)
check("one level >= 2 analysed", len(res) == 1 and res[0]["level"] == 2 and res[0]["appeared_types"] == 1)
r = res[0]
check("three clicks counted", r["clicks"] == 3)
check("one appeared hit", r["hits"] == 1)
check("expected = appeared area share per click (3/5 x 3)", abs(r["exp"] - 3 * 3 / 5) < 1e-9)
check("first appeared click is the 2nd of 4 level actions (index 1 -> 0.25)", abs(r["first"] - 1 / 4) < 1e-9)
check("level without appeared types skipped",
      N.analyse([row("initial", None, L1), row("action", 2, L1), row("action", 2, L1, "MOUSE(row=1, col=1)")]) == [])
never = N.analyse(rows[:4])
check("never clicked an appeared object -> first None", never[0]["first"] is None and never[0]["hits"] == 0)
s = N.summarize(res + never, "g")
check("summary", "levels=2 clicks=4 appeared-object clicks=1" in s and "never in 1/2 levels" in s)
check("summary empty", "levels=0" in N.summarize([], "g"))

d = tempfile.mkdtemp()
with open(os.path.join(d, "g1-ab_p0_events.jsonl"), "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
    f.write("not json\n")
with open(os.path.join(d, "zz-ab_p0_events.jsonl"), "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
out = os.path.join(d, "o.json")
check("main rc 0 + json", N.main([d, "--json", out]) == 0 and len(json.load(open(out))) == 2)
asc = os.path.join(d, "asc")
os.makedirs(asc)
with open(os.path.join(asc, "g2-ab_p0_events.jsonl"), "w") as f:
    for x in rows:
        y = {k: v for k, v in x.items() if k != "board"}
        y["board_ascii"] = "\n".join("".join(".ABCDEFG"[v] for v in line) for line in x["board"])
        f.write(json.dumps(y) + "\n")
ra = N.run([asc])["g2"]
check("board_ascii fallback gives the same result", ra[0]["hits"] == 1 and ra[0]["clicks"] == 3)
check("games filter", set(N.run([d], {"g1"})) == {"g1"})
check("main rc 3 on empty", N.main([tempfile.mkdtemp()]) == 3)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
