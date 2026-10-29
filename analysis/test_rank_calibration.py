#!/usr/bin/env python3
"""Tests for rank_calibration.py (FI-939). Hermetic: synthetic submissions + synthetic FINAL logs. Exit 0 = pass."""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rank_calibration as R  # noqa: E402

F, OK = [], 0


def check(n, c):
    global OK
    if c:
        OK += 1
    else:
        F.append(n); print("FAIL:", n)


def log(scores, ended="2026-10-04 13:10:01"):
    lines = ["benchmark: duck-harness-kaggle", "games:     %d" % len(scores), "passes:    1",
             "ended:     %s" % ended, "mean score:    %.2f" % (sum(scores.values()) / len(scores)), "",
             "per-game (mean across passes):"]
    for g, s in scores.items():
        lines.append("  %s-0c556536: score=%.2f, levels=%.1f/8, actions=10, tokens=10" % (g, s, s / 10))
    return "\n".join(lines) + "\n"


def sub(kernel, sid, score, status="SubmissionStatus.COMPLETE"):
    return {"date": "2026-10-04 01:05:00", "description": kernel, "public_score": "" if score is None else str(score),
            "status": status, "url": "https://www.kaggle.com/code/hivemindadmin/%s?scriptVersionId=%s" % (kernel, sid)}


RECS = [sub("arm-a", "1", 25.5), sub("arm-a", "1", 25.6), sub("arm-b", "2", 28.0), sub("arm-c", "3", 22.0),
        sub("arm-d", "4", 30.0), sub("arm-b", "5", 27.0), sub("arm-e", "6", None, "SubmissionStatus.PENDING"),
        sub("arm-z", "7", 10.0)]
d = tempfile.mkdtemp()
cache = os.path.join(d, "cache")
for slug, mean in (("arm-a", 40.0), ("arm-b", 45.0), ("arm-c", 35.0), ("arm-d", 30.0), ("arm-e", 50.0)):
    os.makedirs(os.path.join(cache, slug))
    open(os.path.join(cache, slug, slug + ".log"), "w").write(log({"g1": mean - 5, "g2": mean + 5}))
os.makedirs(os.path.join(cache, "arm-x"))
open(os.path.join(cache, "arm-x", "other-name.log"), "w").write(log({"g1": 1.0}))
inprog = os.path.join(cache, "arm-c-inprog")
os.makedirs(inprog)
open(os.path.join(inprog, "arm-c-inprog.log"), "w").write(log({"g1": 1.0}, ended="in progress"))

h = R.hidden_by_kernel(RECS)
check("hidden pools all versions of a slug", abs(h["arm-b"][0] - 27.5) < 1e-9 and h["arm-b"][1] == 2)
check("pending submission not counted", "arm-e" not in h)
o = R.offline_by_kernel(cache)
check("offline means from FINAL logs", o == {"arm-a": 40.0, "arm-b": 45.0, "arm-c": 35.0, "arm-d": 30.0, "arm-e": 50.0})
check("log not named after its dir ignored; in-progress ignored", "arm-x" not in o and "arm-c-inprog" not in o)
check("levels metric", abs(R.offline_by_kernel(cache, "levels")["arm-a"] - 4.0) < 1e-9)
ext = os.path.join(d, "old.log"); open(ext, "w").write(log({"g1": 9.0}))
check("extra log joins an older-era kernel", R.offline_by_kernel(cache, extra={"arm-z": ext})["arm-z"] == 9.0)
check("extra log that does not exist is skipped", "arm-q" not in R.offline_by_kernel(cache, extra={"arm-q": ext + "x"}))

check("ranks with ties", R.ranks([3, 1, 3, 2]) == [3.5, 1.0, 3.5, 2.0])
check("spearman perfect", abs(R.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9)
check("spearman reversed", abs(R.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9)
check("spearman constant -> None", R.spearman([1, 1, 1], [1, 2, 3]) is None)

lines = R.run(RECS, cache)
res = lines[-1]
# hidden: d 30 > b 27.5 > a 25.55 > c 22 ; offline: b 45 > a 40 > c 35 > d 30
check("result line counts pairs", res.startswith("CALIB_RESULT agree=3 disagree=3 ties=0 of 6 pairs (noise 1.48)"))
check("rho printed for n>=4", "rho=-0.20" in res and "n=4 kernels" in res)
check("disagreement listed for arm-d", any(l.strip().startswith("DISAGREE") and "arm-d" in l for l in lines))
check("hidden-only kernel reported as missing offline", any("no offline FINAL" in l and "arm-z" in l for l in lines))
check("noise turns small hidden gaps into ties", "ties=1" in R.run(RECS, cache, noise=2.0)[-1])
check("prefix filter + insufficient", R.run(RECS, cache, only_prefix="arm-a")[-1].startswith("CALIB_INSUFFICIENT 1"))

f = os.path.join(d, "subs.json"); open(f, "w").write("Warning: old cli\n" + json.dumps(RECS))
o2 = subprocess.run([sys.executable, os.path.join(HERE, "rank_calibration.py"), "--from-file", f, "--cache", cache,
                     "--log", "arm-z=" + ext], capture_output=True, text=True)
check("CLI with --log joins 5 kernels", o2.returncode == 0 and "n=5 kernels" in o2.stdout.strip().splitlines()[-1])
o3 = subprocess.run([sys.executable, os.path.join(HERE, "rank_calibration.py"), "--from-file", os.path.join(d, "none")],
                    capture_output=True, text=True)
check("CLI unreadable file -> CALIB_ERROR exit 3", o3.returncode == 3 and o3.stdout.startswith("CALIB_ERROR"))

print("rank_calibration tests: %d passed, %d failed" % (OK, len(F)))
sys.exit(1 if F else 0)
