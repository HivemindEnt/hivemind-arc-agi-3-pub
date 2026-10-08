#!/usr/bin/env python3
"""Tests for trackrecord_premise.py (DUCK 38cb). Run: python3 test_trackrecord_premise.py"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trackrecord_premise as T  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


tl = [(0, 1, "UP", False), (1, 1, "UP", True), (2, 1, "UP", False), (3, 1, "SPACE", False),
      (4, 2, "UP", False), (5, 2, "UP", True), (6, 2, "UP", False)]
r = T.analyse(tl)
check("counts actions and no-ops", r["actions"] == 7 and r["noops"] == 5)
check("informative = no-op after an earlier effect of that control on the level", r["informative"] == 2)
check("record resets per level (step 4 not informative)", r["shown"] == [1 / 3, 1 / 3])
check("no-op before any effect not informative", T.analyse([(0, 1, "UP", False)])["informative"] == 0)
s = T.summarize([r, T.analyse([(0, 1, "UP", True)])], "g")
check("summary", "passes=2 actions=8 no-ops=5" in s and ": 2 (40% of no-ops), 1.0 per pass" in s and "0.33" in s)
check("summary empty", "passes=0" in T.summarize([], "g"))

d = tempfile.mkdtemp()
blank = "\n".join("." * 12 for _ in range(12))
mark = blank[:13 * 6 + 6] + "X" + blank[13 * 6 + 7:]
with open(os.path.join(d, "g1-ab_p0_events.jsonl"), "w") as f:
    f.write(json.dumps({"type": "initial", "board_ascii": blank}) + "\n")
    for i, b in enumerate([blank, mark, mark, blank, blank]):
        f.write(json.dumps({"type": "action", "action_num": i, "level": 1, "action_name": "ACTION1",
                            "board_ascii": b}) + "\n")
check("main rc 0", T.main([d, "--per-game"]) == 0)
check("run groups by game", list(T.run([d])) == ["g1"])
check("main rc 3 on empty", T.main([tempfile.mkdtemp()]) == 3)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
