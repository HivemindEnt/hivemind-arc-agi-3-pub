#!/usr/bin/env python3
"""Tests for loss_decompose.py (DUCK 38cd). Run: python3 test_loss_decompose.py"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loss_decompose as L  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


tmp = tempfile.mkdtemp()
env = os.path.join(tmp, "env")
os.makedirs(os.path.join(env, "ab12", "v1"))
json.dump({"game_id": "ab12-0011aabb", "baseline_actions": [4, 4, 4]}, open(os.path.join(env, "ab12", "v1", "metadata.json"), "w"))
run = os.path.join(tmp, "run")
os.makedirs(run)


def write(p, scores):
    with open(os.path.join(run, "ab12-0011aabb_p%d_events.jsonl" % p), "w") as f:
        for s in scores:
            f.write(json.dumps({"type": "action", "action_name": "ACTION1", "score": s}) + "\n")


write(0, [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2])   # L1 in 4 actions (1.0), L2 in 7 actions ((4/7)^2), stuck at L3
write(1, [0, 0, 0, 0, 0, 0, 0, 0])             # stuck at L1
base = L.D.load_baselines(env)
r0 = L.score_pass(os.path.join(run, "ab12-0011aabb_p0_events.jsonl"), base)
check("pass parsed", r0["game"] == "ab12" and r0["pass"] == 0 and r0["levels"] == 2)
check("compl = (1+2)/6", abs(r0["compl"] - 3 / 6) < 1e-9)
check("official = (1*1 + 2*(4/7)^2)/6", abs(r0["official"] - (1 + 2 * (4 / 7) ** 2) / 6) < 1e-9)
check("stuck at L3", r0["stuck"] == 3)
check("unparseable name skipped", L.score_pass(os.path.join(run, "notes.jsonl"), base) is None)
rows = L.run([run], env)
s = L.summarize(rows)
check("two passes, one game", s["games"] == 1 and s["passes"] == 2)
check("game mean over passes", abs(s["official"] - r0["official"] / 2) < 1e-9 and abs(s["compl"] - 0.25) < 1e-9)
check("split adds up", abs(s["unfinished"] + s["inefficiency"] + s["official"] - 1.0) < 1e-9)
check("stuck counts", s["stuck"]["L1"] == 1 and s["stuck"]["L3+"] == 1)
txt = L.render(s, "X")
check("render", "unfinished levels" in txt and "first uncleared level" in txt and "L1 1 (50%)" in txt)
check("main rc 0", L.main([run, "--env-dir", env, "--per-game"]) == 0)
check("main rc 3 without baselines", L.main([run, "--env-dir", os.path.join(tmp, "none")]) == 3)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
