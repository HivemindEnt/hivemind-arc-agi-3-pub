#!/usr/bin/env python3
"""Tests for nb_diff.py (DUCK 38cf). Run: python3 test_nb_diff.py"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb_diff as N  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


setup = "import os\nos.environ['ARC3_LEVEL_INVENTORY'] = '0'\nos.environ.setdefault('ARC3_DRAIN', '58000')\n"
bench = "print('Starting benchmark...')\nrun_all()\n"
patch = "PATCH = '''\n" + "\n".join("line %d" % i for i in range(50)) + "\n'''\napply(PATCH)\n"
a = [setup, patch, bench]
patch2 = patch.replace("line 7\n", "line 7 changed\n")
setup2 = setup.replace("'0'", "'1'") + "os.environ['ARC3_NEW_KNOB'] = 'on'\n"
b = [setup2, patch2, "# new cell\nos.environ['ARC3_TAIL_FADE'] = '1'\n", bench]
r = N.compare(a, b)
check("knob value change found", r["knobs"]["ARC3_LEVEL_INVENTORY"] == ("'0'", "'1'"))
check("new knobs found", r["knobs"]["ARC3_NEW_KNOB"] == (None, "'on'") and r["knobs"]["ARC3_TAIL_FADE"] == (None, "'1'"))
check("unchanged knob not reported", "ARC3_DRAIN" not in r["knobs"])
check("added cell found", r["added"] == [2])
check("nothing removed", r["removed"] == [])
ch = {(i, j): (p, m) for i, j, p, m, _ in r["changed"]}
check("patch cell changed +1 -1", ch.get((1, 1)) == (1, 1))
check("setup cell changed", (0, 0) in ch)
check("identical cell not in changed", (2, 3) not in ch)
txt = N.render(r, a, b, show_diff=True, max_lines=5)
check("render lists everything", "CELLS base=3 other=4" in txt and "KNOB ARC3_NEW_KNOB" in txt and "ADDED other#2" in txt
      and "diff truncated" in txt)
check("dict-style ARC3 knobs", N.knobs(["cfg = {'ARC3_X': '2', 'other': 1}"]) == {"ARC3_X": "'2'"})
check("export-style knobs", N.knobs(["%env ARC3_Y=3\n"]) == {"ARC3_Y": "3"})
r2 = N.compare(a, a[:2])
check("removed cell found", r2["removed"] == [2] and r2["added"] == [])

d = tempfile.mkdtemp()
def nb(cells, name):
    p = os.path.join(d, name)
    json.dump({"cells": [{"cell_type": "markdown", "source": "# t"}] + [{"cell_type": "code", "source": c.splitlines(True)}
                                                                      for c in cells]}, open(p, "w"))
    return p
pa, pb = nb(a, "a.ipynb"), nb(b, "b.ipynb")
check("code_cells skips markdown and joins list sources", N.code_cells(pa) == a)
check("main rc 0", N.main([pa, pb, "--diff"]) == 0)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
