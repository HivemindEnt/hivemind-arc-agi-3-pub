#!/usr/bin/env python3
"""Tests for replay_tests.py (DUCK 38bw). Run: python3 test_replay_tests.py"""
import json, math, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_tests as rt  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


check("h2 extremes", rt.h2(0.5) == 1.0 and rt.h2(0.0) == 0.0 and rt.h2(1.0) == 0.0)
check("h2 symmetric", abs(rt.h2(0.2) - rt.h2(0.8)) < 1e-12)

ann = rt.annotate([(0, 1, "UP", True), (1, 1, "UP", True), (2, 1, "UP", False), (3, 2, "UP", True)])
check("first try is maximally uncertain", ann[0][4] == 1.0 and abs(ann[0][5] - 1.0) < 1e-12)
check("laplace after one effect: p=2/3", abs(ann[1][4] - rt.h2(2 / 3)) < 1e-12)
check("surprise of a no-op after two effects: p=3/4 -> -log2(1/4)=2", abs(ann[2][5] - 2.0) < 1e-12)
check("record is per level", ann[3][4] == 1.0)


def msgs_at(step, level, rc):
    return [("user", "Current state: step %d, level %d" % (step, level), ""), ("assistant", "", rc)]


a = rt.announcements(msgs_at(4, 1, "Let me test whether UP moves the block. I'll check if SPACE does anything."))
check("two announcements", [x["control"] for x in a] == ["UP", "SPACE"])
check("no control, no announcement", rt.announcements(msgs_at(4, 1, "Let me think about the goal.")) == [])
check("dedup per level/step/control",
      len(rt.announcements(msgs_at(4, 1, "Let me test UP. Let me verify UP again."))) == 1)
check("arrows map to DIRS", rt.announcements(msgs_at(4, 1, "Let me try the arrow keys"))[0]["control"] == "DIRS")
check("plain mention is not a test", rt.announcements(msgs_at(4, 1, "UP moved the block left.")) == [])

tl = [(0, 1, "UP", True), (1, 1, "UP", True), (4, 1, "LEFT", True), (5, 1, "UP", False), (6, 1, "SPACE", True)]
rows, ann = rt.analyse(msgs_at(4, 1, "Let me test whether UP still works."), tl, k=5)
r = rows[0]
check("executed within k", r["executed"] and r["effect"] is False)
check("H at execution uses record before it (2/2 effects -> p=3/4)", abs(r["H"] - rt.h2(0.75)) < 1e-12)
rows2, _ = rt.analyse(msgs_at(4, 1, "Let me test SPACE."), tl, k=2)
check("not executed when outside k", rows2[0]["executed"] is False and rows2[0]["H"] is None)
rows3, _ = rt.analyse(msgs_at(4, 2, "Let me test UP."), tl, k=5)
check("other level does not count", rows3[0]["executed"] is False)

s = rt.summarize(rows, ann)
check("summary fields", "announcements=1" in s and "executed=1 (100%)" in s and "H(all)=" in s)
check("tries_before recorded", r["tries_before"] == 2)
st = rt.strata(rows, ann)
check("strata lines", st.count("prior tries") == 4)
check("strata put the test in its band", "prior tries 1-2: tests n=1" in st and "prior tries 0: tests n=0" in st)
check("summary on empty", "announcements=0" in rt.summarize([], []))

d = tempfile.mkdtemp()
blank = "\n".join("." * 12 for _ in range(12))
mark = blank[:13 * 6 + 6] + "X" + blank[13 * 6 + 7:]
with open(os.path.join(d, "g1-ab_p0_events.jsonl"), "w") as f:
    f.write(json.dumps({"type": "initial", "board_ascii": blank}) + "\n")
    boards = [mark, mark, blank, blank, mark]
    for i, b in enumerate(boards):
        f.write(json.dumps({"type": "action", "action_num": i, "level": 1, "action_name": "ACTION1",
                            "board_ascii": b}) + "\n")
with open(os.path.join(d, "g1-ab_p0_requests.jsonl"), "w") as f:
    f.write(json.dumps({"messages": [{"role": "system", "content": "s"},
                                     {"role": "user", "content": "Current state: step 1, level 1"},
                                     {"role": "assistant", "content": "", "reasoning_content": "Let me test UP."}]}) + "\n")
out = os.path.join(d, "o.json")
check("main rc 0 + json", rt.main([d, "--json", out, "--per-game"]) == 0 and json.load(open(out))[0]["game"] == "g1")
check("main rc 3 on empty", rt.main([tempfile.mkdtemp()]) == 3)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
