#!/usr/bin/env python3
"""Tests for replay_revise.py (DUCK 38bx). Run: python3 test_replay_revise.py"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_revise as rv  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


m = rv.mentions("SPACE is a no-op. Then UP moved the block left.")
check("mentions in order", [(k, c) for k, c, _ in m] == [("NEG", "SPACE"), ("POS", "UP")])
check("negated positive counts as negative", rv.mentions("UP moved nothing, it does nothing")[0][0] == "NEG")
check("no control, no mention", rv.mentions("The block moved left.") == [])
check("DIRS matches a single arrow", rv._match("UP", ("UP", "DOWN", "LEFT", "RIGHT")) and rv._match("DIRS", ("UP",)))
check("different control no match", not rv._match("SPACE", ("UP",)))


def turn(step, level, rc):
    return [("user", "Current state: step %d, level %d" % (step, level), ""), ("assistant", "", rc)]


tl = [(3, 1, "SPACE", False), (5, 1, "SPACE", True), (6, 1, "UP", True)]
msgs = turn(4, 1, "SPACE is a no-op.") + turn(6, 1, "Thinking about the goal.") + turn(7, 1, "SPACE toggled the gate!")
r = rv.analyse(msgs, tl)
check("one durable statement", len(r) == 1 and r[0]["control"] == "SPACE")
check("contradiction found at step 5", r[0]["contradicted"] and r[0]["contra_step"] == 5)
check("REVISED after 2 turns", r[0]["outcome"] == "REVISED" and r[0]["turns_to"] == 2)

msgs2 = turn(4, 1, "SPACE is a no-op.") + turn(7, 1, "SPACE does nothing here, as I said.")
check("REASSERTED", rv.analyse(msgs2, tl)[0]["outcome"] == "REASSERTED")
msgs3 = turn(4, 1, "SPACE is a no-op.") + turn(7, 1, "Next, UP.") + turn(9, 2, "SPACE moved the bar.")
check("SILENT when the level ends first", rv.analyse(msgs3, tl)[0]["outcome"] == "SILENT")
tl_no = [(3, 1, "SPACE", False), (5, 1, "SPACE", False)]
r4 = rv.analyse(msgs, tl_no)
check("no contradiction -> no outcome", not r4[0]["contradicted"] and r4[0]["outcome"] is None)
check("hypotheticals excluded by default", rv.analyse(turn(4, 1, "SPACE would be a no-op if blocked."), tl) == [])
check("classes option includes them",
      len(rv.analyse(turn(4, 1, "SPACE would be a no-op if blocked."), tl, ("HYPOTHETICAL",))) == 1)
msgs5 = turn(4, 1, "SPACE is a no-op.") + turn(5, 1, "SPACE toggled the gate!")
check("a turn before the contradiction does not count", rv.analyse(msgs5, tl)[0]["outcome"] == "SILENT")

s = rv.summarize(rv.analyse(msgs, tl) + rv.analyse(msgs2, tl) + r4)
check("summary counts", "statements=3 contradicted on the same level=2" in s and "REVISED 1 (50%)" in s
      and "REASSERTED 1 (50%)" in s and "median turns to revision 2" in s)
check("summary empty", "statements=0" in rv.summarize([]))

d = tempfile.mkdtemp()
blank = "\n".join("." * 12 for _ in range(12))
mark = blank[:13 * 6 + 6] + "X" + blank[13 * 6 + 7:]
with open(os.path.join(d, "g1-ab_p0_events.jsonl"), "w") as f:
    f.write(json.dumps({"type": "initial", "board_ascii": blank}) + "\n")
    for i, b in enumerate([blank, blank, mark, mark, blank]):
        f.write(json.dumps({"type": "action", "action_num": i, "level": 1, "action_name": "ACTION5",
                            "board_ascii": b}) + "\n")
hist = [{"role": "system", "content": "s"},
        {"role": "user", "content": "Current state: step 1, level 1"},
        {"role": "assistant", "content": "", "reasoning_content": "SPACE is a no-op."},
        {"role": "user", "content": "Current state: step 4, level 1"},
        {"role": "assistant", "content": "", "reasoning_content": "SPACE changed the board."}]
with open(os.path.join(d, "g1-ab_p0_requests.jsonl"), "w") as f:
    f.write(json.dumps({"messages": hist}) + "\n")
out = os.path.join(d, "o.json")
check("main rc 0", rv.main([d, "--json", out]) == 0)
check("main end to end REVISED", json.load(open(out))[0]["outcome"] == "REVISED")
check("main rc 3 on empty", rv.main([tempfile.mkdtemp()]) == 3)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
