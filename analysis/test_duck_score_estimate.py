#!/usr/bin/env python3
"""Tests for duck_score_estimate.py (FI-939). Stdlib only. Exit 0 = all pass."""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import duck_score_estimate as S  # noqa: E402

F, OK = [], 0


def check(n, c):
    global OK
    if c:
        OK += 1
    else:
        F.append(n); print("FAIL:", n)


def close(a, b, eps=1e-9):
    return abs(a - b) < eps


# level_score: official formula (arc_agi scorecard.add_level): (base/ai)^2, capped at 1.15, 0 for no actions
check("equal actions -> 1.0", close(S.level_score(10, 10), 1.0))
check("2x actions -> 0.25", close(S.level_score(10, 20), 0.25))
check("10x actions -> 0.01", close(S.level_score(10, 100), 0.01))
check("faster than human capped at 1.15", close(S.level_score(10, 5), 1.15))
check("zero ai actions -> 0", S.level_score(10, 0) == 0.0)

# game_score: weighted by 1-based level over ALL levels
check("all 3 levels perfect -> 1.0", close(S.game_score([5, 5, 5], {1: 1.0, 2: 1.0, 3: 1.0}), 1.0))
check("first 2 of 3 perfect -> 3/6", close(S.game_score([5, 5, 5], {1: 1.0, 2: 1.0}), 0.5))
check("docs example 4 of 5 -> 10/15", close(S.game_score([1] * 5, {1: 1, 2: 1, 3: 1, 4: 1}), 10 / 15.0))
check("level beyond baseline ignored", close(S.game_score([5], {1: 1.0, 2: 1.0}), 1.0))


def act(score, name="ACTION1", go=False):
    return {"type": "action", "score": score, "action_name": name, "game_over": go}


# level_actions: cumulative (all attempts, RESET included) vs final attempt only
ev = [act(0), act(0), act(0, go=True), act(0, "RESET"), act(0), act(1),   # L1: 6 cum, 2 last
      {"type": "analysis"}, act(1), act(2)]                                 # L2: 2 cum, 2 last
lv = S.level_actions(ev)
check("two levels found", [x[0] for x in lv] == [1, 2])
check("L1 cum counts every action incl RESET", lv[0][1] == 6)
check("L1 last = final attempt only", lv[0][2] == 2)
check("L2 counts restart after completion", lv[1][1] == 2 and lv[1][2] == 2)
check("analysis events ignored", S.level_actions([{"type": "analysis", "score": 1}]) == [])
check("double-level jump recorded", [x[0] for x in S.level_actions([act(0), act(2)])] == [1, 2])

# analyze + CLI on a synthetic run dir + env dir
d = tempfile.mkdtemp()
env = os.path.join(d, "env", "zz09", "abc123")
os.makedirs(env)
json.dump({"game_id": "zz09-abc123", "baseline_actions": [2, 4]}, open(os.path.join(env, "metadata.json"), "w"))
run = os.path.join(d, "run")
os.makedirs(run)
with open(os.path.join(run, "zz09-abc123_p0_events.jsonl"), "w") as f:
    for e in ev:
        f.write(json.dumps(e) + "\n")
with open(os.path.join(run, "qq01-ffff_p0_events.jsonl"), "w") as f:
    f.write(json.dumps(act(0)) + "\n")
bl = S.load_baselines(os.path.join(d, "env"))
check("baseline by full id", bl.get("zz09-abc123") == [2, 4])
check("baseline by short id", bl.get("zz09") == [2, 4])
r = S.analyze(os.path.join(run, "zz09-abc123_p0_events.jsonl"), bl)
# L1 cum: (2/6)^2 = 0.1111, L2: (4/2)^2 capped 1.15; game_cum = (1*0.1111 + 2*1.15)/3
check("game_cum formula", close(r["game_cum"], (1 * (2 / 6.0) ** 2 + 2 * 1.15) / 3.0))
check("game_last capped at completed-weight share (scorer max_score)", close(r["game_last"], 1.0))
check("game cap: 1.15 levels cannot exceed completed share", close(S.game_score([5, 5], {1: 1.15}), 1 / 3.0))
check("game cap: zero-score level adds no cap weight", close(S.game_score([5, 5], {1: 0.0, 2: 0.5}), 1 / 3.0))
check("game_compl = all completed at 1.0", close(r["game_compl"], 1.0))
check("no baseline -> excluded", S.analyze(os.path.join(run, "qq01-ffff_p0_events.jsonl"), bl)["baseline"] is None)
out = os.path.join(d, "o.json")
cp = subprocess.run([sys.executable, os.path.join(HERE, "duck_score_estimate.py"), run, "--env-dir", os.path.join(d, "env"),
                     "--levels", "--json", out], capture_output=True, text=True)
check("cli rc 0", cp.returncode == 0)
check("cli TOTAL over scored games only", "TOTAL over 1 games" in cp.stdout)
check("cli marks excluded game", "no baseline; excluded" in cp.stdout)
check("cli json", os.path.exists(out) and len(json.load(open(out))) == 2)
cp2 = subprocess.run([sys.executable, os.path.join(HERE, "duck_score_estimate.py"), run, "--env-dir", os.path.join(d, "none")],
                     capture_output=True, text=True)
check("cli no baselines rc 3", cp2.returncode == 3)
cp3 = subprocess.run([sys.executable, os.path.join(HERE, "duck_score_estimate.py"), run, "qq01", "--env-dir", os.path.join(d, "env")],
                     capture_output=True, text=True)
check("cli game filter", "qq01" in cp3.stdout and "zz09" not in cp3.stdout)
check("cli usage rc 2", subprocess.run([sys.executable, os.path.join(HERE, "duck_score_estimate.py")],
                                       capture_output=True, text=True).returncode == 2)
# EX-HIGHVAR line (2026-09-27): mean without ft09, printed before TOTAL; absent when nothing is excluded
check("no EX-HIGHVAR line when no high-variance game", "EX-HIGHVAR" not in cp.stdout)
check("TOTAL stays the last line", cp2.returncode == 3 or cp.stdout.strip().splitlines()[-1].startswith("TOTAL"))
fenv = os.path.join(d, "env", "ft09", "d00d")
os.makedirs(fenv)
json.dump({"game_id": "ft09-d00d", "baseline_actions": [1]}, open(os.path.join(fenv, "metadata.json"), "w"))
with open(os.path.join(run, "ft09-d00d_p0_events.jsonl"), "w") as f:
    f.write(json.dumps(act(1)) + "\n")   # L1 in 1 action = 1.0
cp4 = subprocess.run([sys.executable, os.path.join(HERE, "duck_score_estimate.py"), run, "--env-dir", os.path.join(d, "env")],
                     capture_output=True, text=True)
lines4 = cp4.stdout.strip().splitlines()
check("EX-HIGHVAR printed when ft09 present", any(l.startswith("EX-HIGHVAR (excl ft09) over 1 games") for l in lines4))
check("TOTAL last with EX-HIGHVAR present", lines4[-1].startswith("TOTAL over 2 games"))
exl = [l for l in lines4 if l.startswith("EX-HIGHVAR")]
zz = 100 * (1 * (2 / 6.0) ** 2 + 2 * 1.15) / 3.0
check("EX-HIGHVAR value = zz09 only", exl and ("OFFICIAL(score_cum)=%.3f%%" % zz) in exl[0])
cp5 = subprocess.run([sys.executable, os.path.join(HERE, "duck_score_estimate.py"), run, "ft09", "--env-dir", os.path.join(d, "env")],
                     capture_output=True, text=True)
check("no EX-HIGHVAR when only ft09 selected", "EX-HIGHVAR" not in cp5.stdout and "TOTAL over 1 games" in cp5.stdout)
# MAX-INFLUENCE (2026-09-28)
mk = lambda g, v: {"game": g + "-x", "game_cum": v}
gi, sh, wo = S.max_influence([mk("aa01", 0.5), mk("bb02", 0.1), mk("cc03", 0.0)])
check("max influence picks the dominant game", gi == "aa01")
check("max influence share", abs(sh - 100 * 0.5 / 0.6) < 1e-9)
check("max influence mean without it", abs(wo - 5.0) < 1e-9)
gi2, _, _ = S.max_influence([mk("aa01", 0.0), mk("bb02", 0.0), mk("cc03", 0.0)])
check("all-zero run does not crash", gi2 in ("aa01", "bb02", "cc03"))
gi3, _, wo3 = S.max_influence([mk("aa01", 0.2), mk("bb02", 0.2), mk("cc03", 0.8), mk("dd04", 0.2)])
check("outlier high game is the influence", gi3 == "cc03" and abs(wo3 - 20.0) < 1e-9)
check("no MAX-INFLUENCE line with < 3 scored games", "MAX-INFLUENCE" not in cp4.stdout)
check("TOTAL still last", cp4.stdout.strip().splitlines()[-1].startswith("TOTAL"))
# duplicate copies under artifacts/ count once (2026-10-08)
import shutil
os.makedirs(os.path.join(run, "artifacts"), exist_ok=True)
shutil.copy(os.path.join(run, "zz09-abc123_p0_events.jsonl"), os.path.join(run, "artifacts", "zz09-abc123_p0_events.jsonl"))
cp6 = subprocess.run([sys.executable, os.path.join(HERE, "duck_score_estimate.py"), run, "--env-dir", os.path.join(d, "env")],
                     capture_output=True, text=True)
check("duplicate artifacts/ copy not double-counted", cp6.stdout.strip().splitlines()[-1].startswith("TOTAL over 2 games"))
dd = S.dedupe_by_game([os.path.join("a", "artifacts", "x_p0_events.jsonl"), os.path.join("a", "x_p0_events.jsonl"),
                       os.path.join("a", "artifacts", "y_p0_events.jsonl")])
check("dedupe keeps shallowest", dd == {"x_p0_events.jsonl": os.path.join("a", "x_p0_events.jsonl"),
                                        "y_p0_events.jsonl": os.path.join("a", "artifacts", "y_p0_events.jsonl")})
print("duck_score_estimate tests: %d passed, %d failed" % (OK, len(F)))
sys.exit(1 if F else 0)
