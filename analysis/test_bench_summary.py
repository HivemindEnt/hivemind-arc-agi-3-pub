#!/usr/bin/env python3
"""Tests for bench_summary.py. Exit 0 = all pass. No network."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_summary as B  # noqa: E402

FAIL, OK = [], 0


def check(n, c):
    global OK
    if c:
        OK += 1
    else:
        FAIL.append(n); print("FAIL:", n)


def block(ended, mean, games):
    s = ("benchmark: duck-harness-kaggle\nsolver:    duck-harness\ngames:     %d\npasses:    1\nruns:      %d (won: 0)\n"
         "started:   2026-10-03 14:53:14\nended:     %s\nmean score:    %s\nmedian score:  8.62\ntotal actions: 512\n"
         "total tokens:  1\n\nper-game (mean across passes):\n") % (len(games), len(games), ended, mean)
    for g, sc, lv, mx, a in games:
        s += "  %s: score=%s, levels=%s/%d, actions=%d, tokens=1\n" % (g, sc, lv, mx, a)
    return s


G1 = [("ar25-0c556536", "8.33", "2.0", 8, 49), ("ft09-0d8bbf25", "47.62", "4.0", 6, 46)]
G2 = [("ar25-0c556536", "41.67", "5.0", 8, 153), ("ft09-0d8bbf25", "47.62", "4.0", 6, 97)]
mid = block("in progress", "12.65", G1)
fin = block("2026-10-03 15:18:20", "43.59", G2)
text = "noise\n" + mid + "[sys] RAM\n" + mid + fin + "\ndiagnostics: x\n"

s = B.summaries(text)
check("three summaries found", len(s) == 3)
check("in-progress summaries are not complete", [x["complete"] for x in s] == [False, False, True])
f = B.final_summary(text)
check("final summary is the timestamped one (the 12.65 trap)", f["fields"]["mean score"] == "43.59")
check("final per-game parsed", [g["levels"] for g in f["games"]] == [5.0, 4.0] and f["games"][0]["max_levels"] == 8)
check("only in-progress -> None", B.final_summary("x\n" + mid) is None)

# Kaggle JSON stream form, split across chunks mid-line
chunks = [text[i:i + 37] for i in range(0, len(text), 37)]
stream = json.dumps([{"stream_name": "stdout", "time": i, "data": c} for i, c in enumerate(chunks)])
check("JSON stream is reassembled", B.final_summary(B.log_text(stream))["fields"]["mean score"] == "43.59")
check("plain text passes through", B.log_text("abc") == "abc")
check("bad JSON falls back to text", B.log_text("[not json") == "[not json")

# truncated per-game line (a mid-run print cut off) stops the per-game list instead of mis-parsing
trunc = block("2026-10-03 15:18:20", "43.59", G2).replace("  ft09-0d8bbf25: score=47.62, levels=4.0/6, actions=97, tokens=1\n",
                                                          "  ft09-0d8bbf25\n")
check("truncated game line ends the list", [g["game"] for g in B.final_summary(trunc)["games"]] == ["ar25-0c556536"])

# compare
rows = B.compare(f, {"runA": {"ar25": 1, "ft09": 4}, "runB": {"ar25": 2}})
check("compare rows keyed by game prefix", rows[0] == {"game": "ar25", "bench": 5.0, "runA": 1, "runB": 2})
check("compare: missing game -> None", rows[1]["runB"] is None)

# main exit codes
d = tempfile.mkdtemp()
p = os.path.join(d, "log.txt"); open(p, "w").write(text)
check("main final -> 0", B.main([p]) == 0)
p2 = os.path.join(d, "mid.txt"); open(p2, "w").write(mid)
check("main in-progress only -> 3", B.main([p2]) == 3)
p3 = os.path.join(d, "none.txt"); open(p3, "w").write("nothing here\n")
check("main no summary -> 4", B.main([p3]) == 4)
check("main usage -> 2", B.main([]) == 2)
check("main json -> 0", B.main([p, "--json"]) == 0)

# progression: budget-bound or not
G0 = [("ar25-0c556536", "8.33", "1.0", 8, 20), ("ft09-0d8bbf25", "47.62", "4.0", 6, 46)]
pr = B.progression(B.summaries(block("in progress", "5", G0) + mid + fin))
check("progression uses every snapshot in order", pr["snapshots"] == 3 and pr["total_levels"] == [5.0, 6.0, 9.0])
check("progression last interval: ar25 climbing, ft09 active (more actions, no level)",
      pr["last_interval"] == {"climbing": 1, "active": 1, "stopped": 0} and pr["last_interval_levels"] == 3.0)
check("progression per-game level history", pr["per_game"][0] == {"game": "ar25", "levels": [1.0, 2.0, 5.0],
                                                                   "state": "climbing"})
G3 = [("ar25-0c556536", "41.67", "5.0", 8, 153), ("ft09-0d8bbf25", "47.62", "4.0", 6, 97)]
st = B.progression(B.summaries(fin + block("2026-10-03 16:00:00", "43.59", G3)))
check("progression: no new actions -> stopped", st["last_interval"]["stopped"] == 2)
check("progression needs two snapshots", B.progression(B.summaries(fin)) is None)

print("bench_summary tests: %d passed, %d failed" % (OK, len(FAIL)))
sys.exit(1 if FAIL else 0)
