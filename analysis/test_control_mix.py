#!/usr/bin/env python3
"""Tests for control_mix.py. Run: python3 test_control_mix.py"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control_mix as C

OK, F = 0, []


def check(n, c):
    global OK
    if c:
        OK += 1
    else:
        F.append(n); print("FAIL", n)


def a(name, done=False):
    return {"type": "action", "action_name": name, "level_completed": done}


acts = [a("ACTION6"), a("ACTION6"), a("ACTION1", True), a("RESET"), a("ACTION6"), a("ACTION6"), a("ACTION6")]
s = C.segments(acts)
check("two segments", [x["level"] for x in s] == [1, 2] and [x["won"] for x in s] == [True, False])
check("reset ignored", s[1]["fams"] == ["click", "click", "click"])
check("second family index", C.second_family_at(["click", "click", "move"]) == 2)
check("single family -> None", C.second_family_at(["click", "click"]) is None)
rows = C.analyse({("g1", "p0"): acts})
r2 = [r for r in rows if r["level"] == 2][0]
check("untried family on L2", r2["untried"] == ["move"] and r2["used"] == ["click"])
r1 = [r for r in rows if r["level"] == 1][0]
check("L1 used both", r1["untried"] == [] and r1["second_at"] == 2)
rows2 = [{"game": "g", "pass": "p0", "level": 2, "won": False, "n": 12, "used": ["click"], "untried": ["move"], "second_at": None},
         {"game": "g", "pass": "p1", "level": 2, "won": False, "n": 12, "used": ["click", "move"], "untried": [], "second_at": 5},
         {"game": "g", "pass": "p2", "level": 2, "won": True, "n": 3, "used": ["move"], "untried": ["click"], "second_at": None}]
sm = C.summarise(rows2)
check("short segment excluded", "L2+ won" not in sm)
check("failed group stats", sm["L2+ failed"]["segments"] == 2 and abs(sm["L2+ failed"]["share_with_untried_family"] - 0.5) < 1e-9
      and abs(sm["L2+ failed"]["share_single_family"] - 0.5) < 1e-9 and sm["L2+ failed"]["median_actions_before_2nd_family"] == 5)
with tempfile.TemporaryDirectory() as d:
    run = os.path.join(d, "runA")
    os.makedirs(os.path.join(run, "artifacts"))
    for sub in ("", "artifacts"):
        with open(os.path.join(run, sub, "zz09-x_p0_events.jsonl"), "w") as f:
            for e in acts + [{"type": "analysis"}]:
                f.write(json.dumps(e) + "\n")
            f.write("garbage\n")
    runs = C.load([run])
    check("load dedupes artifacts copy", len(runs) == 1 and len(list(runs.values())[0]) == 7)
    out = os.path.join(d, "o.json")
    cp = subprocess.run([sys.executable, os.path.join(HERE, "control_mix.py"), run, "--json", out], capture_output=True, text=True)
    check("cli rc 0 + json", cp.returncode == 0 and len(json.load(open(out))["rows"]) == 2)
    check("cli usage rc 2", subprocess.run([sys.executable, os.path.join(HERE, "control_mix.py")], capture_output=True, text=True).returncode == 2)
check("last_family_at never tried -> 1.0", C.last_family_at(["click"] * 4, {"click", "move"}) == 1.0)
check("last_family_at position", C.last_family_at(["click", "click", "click", "move"], {"click", "move"}) == 0.75)
check("rows carry last_family_at", r2["last_family_at"] == 1.0 and abs(r1["last_family_at"] - 2 / 3) < 1e-3)
wrows = [{"game": "run:aa01-x", "level": 2, "won": True, "n": 10, "game_families": 2, "last_family_at": 0.2},
         {"game": "run:aa01-x", "level": 2, "won": False, "n": 10, "game_families": 2, "last_family_at": 0.9},
         {"game": "run:bb02-x", "level": 2, "won": False, "n": 10, "game_families": 1, "last_family_at": 0.0},
         {"game": "run:aa01-x", "level": 1, "won": False, "n": 10, "game_families": 2, "last_family_at": 0.5}]
wg = C.within_game(wrows)
check("within_game filters L1 and 1-family games", wg == {"aa01": {"won": [0.2], "failed": [0.9]}})
wins = [a("ACTION6"), a("ACTION6", True), a("ACTION6"), dict(a("ACTION6"), state="WIN")]
sw = C.segments(wins)
check("WIN action closes the final level as won (38bg)", [x["won"] for x in sw] == [True, True] and len(sw) == 2)
print("control_mix tests: %d passed, %d failed" % (OK, len(F)))
sys.exit(1 if F else 0)
