#!/usr/bin/env python3
"""Tests for make_fig2.py. Run: python3 test_make_fig2.py (draw test skips without matplotlib)."""
import json, os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_fig2 as F  # noqa: E402

FAILS = []


def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c:
        FAILS.append(n)


def write(d, game, p, levels):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "%s-0a1b2c3d_p%d_events.jsonl" % (game, p)), "w") as f:
        for L in range(1, levels + 1):
            f.write(json.dumps({"type": "action", "action_name": "ACTION1", "score": L}) + "\n")


tmp = tempfile.mkdtemp()
try:
    A, B = os.path.join(tmp, "a"), os.path.join(tmp, "b")
    for p, (x, y) in enumerate([(2, 3), (4, 5), (2, 3), (4, 5)]):
        write(A, "g1", p, x); write(B, "g1", p, y)
        write(A, "g2", p, 3); write(B, "g2", p, 3)
    d = F.compute([("base", A), ("arm", B)], 16)
    ys = [m for _, m in d["curve"]]
    check("curve decreasing", all(ys[i] > ys[i + 1] for i in range(len(ys) - 1)))
    check("curve covers 2..16", [n for n, _ in d["curve"]] == list(range(2, 17, 2)))
    check("effect is +1 level of 6 = +16.7%", abs(d["effects"][0][1] - 100.0 / 6) < 1e-6)
    check("variances sorted, g1 first", d["variances"][0][0] == "g1" and d["variances"][-1][1] == 0.0)
    check("passes_for finds first n under pct", F.passes_for([(2, 30.0), (4, 20.0), (6, 10.0)], 15) == 6)
    check("passes_for none", F.passes_for([(2, 30.0)], 5) is None)
    check("usage rc 2", F.main(["base=" + A]) == 2 and F.main(["nobase", "x=" + B]) == 2)
    os.makedirs(os.path.join(tmp, "empty"))
    check("no traces rc 4", F.main(["base=" + A, "x=" + os.path.join(tmp, "empty")]) == 4)
    try:
        import matplotlib  # noqa: F401
        out = os.path.join(tmp, "fig", "f2")
        check("draw writes png+svg", F.main(["base=" + A, "arm=" + B, "--out", out, "--max-passes", "8"]) == 0
              and os.path.getsize(out + ".png") > 1000 and os.path.exists(out + ".svg"))
    except ImportError:
        print("skip draw: no matplotlib")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
