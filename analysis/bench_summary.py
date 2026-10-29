#!/usr/bin/env python3
"""FI-939 38b: read the FINAL benchmark summary from a Kaggle kernel log and compare per-game levels with our runs.

Why: on 2026-10-03 the M2 fork's commit run was reported as "mean 12.65" because a first-match grep for
"mean score:" picked up one of the periodic in-progress summaries the TAAF/Duck benchmark prints while games are
still running (`ended:     in progress`). The real result, in the last summary whose `ended:` is a timestamp, was
43.59. The error went into a doc commit and a Slack readout before it was caught. This module parses summaries
structurally and only trusts a COMPLETE one (ended = timestamp). It refuses to report when the run never finished.

  bench_summary.py KERNEL_LOG [--compare RUN_DIR ...] [--json]
KERNEL_LOG: the JSON stream Kaggle returns for `kernels output` (list of {stream_name,time,data}) or plain text.
RUN_DIR: one of our recorded runs (*_p0_events.jsonl); per-game levels via duck_scorecard.load_run.
Also prints PROGRESSION: levels per snapshot and, for the last interval, how many games were still gaining
levels (BUDGET-BOUND when at least half were).
Exit: 0 final summary found | 3 only in-progress summaries | 4 no summary | 2 usage
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HDR = re.compile(r"^benchmark:\s+\S+\s*$", re.M)
FIELD = re.compile(r"^(games|passes|runs|started|ended|mean score|median score|total actions|total tokens|total wallclock):\s*(.*?)\s*$", re.M)
GAME = re.compile(r"^\s+([a-z0-9]{2,8}-[0-9a-f]{6,})(?::\s*score=([\d.]+),\s*levels=([\d.]+)/(\d+),\s*actions=(\d+))?", re.M)
TS = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def log_text(raw):
    """Kaggle kernel log JSON stream -> concatenated text; plain text passes through."""
    s = raw.lstrip()
    if s.startswith("["):
        try:
            return "".join(e.get("data", "") for e in json.loads(s))
        except (ValueError, AttributeError):
            pass
    return raw


def summaries(text):
    """All summary blocks in order: dicts with fields + per-game list + complete flag."""
    starts = [m.start() for m in HDR.finditer(text)]
    out = []
    for k, a in enumerate(starts):
        b = starts[k + 1] if k + 1 < len(starts) else len(text)
        block = text[a:b]
        f = {m.group(1): m.group(2) for m in FIELD.finditer(block)}
        if "mean score" not in f:
            continue
        pg_at = block.find("per-game")
        games = []
        if pg_at >= 0:
            for m in GAME.finditer(block[pg_at:]):
                if m.group(2) is None:
                    break   # truncated line (mid-run print cut off)
                games.append({"game": m.group(1), "score": float(m.group(2)), "levels": float(m.group(3)),
                              "max_levels": int(m.group(4)), "actions": int(m.group(5))})
        out.append({"fields": f, "games": games, "complete": bool(TS.match(f.get("ended", "")))})
    return out


def final_summary(text):
    """The LAST complete summary, or None."""
    done = [s for s in summaries(text) if s["complete"]]
    return done[-1] if done else None


def compare(final, run_levels):
    """final: summary dict; run_levels: {name: {game_prefix: levels}} -> rows over the summary's games."""
    rows = []
    for g in final["games"]:
        pre = g["game"].split("-")[0]
        row = {"game": pre, "bench": g["levels"]}
        for name, lv in run_levels.items():
            row[name] = lv.get(pre)
        rows.append(row)
    return rows


def levels_of_run(run_dir):
    import duck_scorecard as D
    return {g.split("-")[0]: r["levels"] for g, r in D.load_run(os.path.expanduser(run_dir)).items()}


def progression(allsum):
    """Is the run budget-bound? Uses every summary (in-progress ones included) in log order.
    Per game: levels and actions at each snapshot. In the LAST interval (second-to-last summary -> last), a game is
    'climbing' if it gained levels, 'active' if it took actions but gained none, 'stopped' if it took no actions
    (its own budget or the game ended earlier). Many climbing games at the end means more time or tokens would still
    buy levels: the score is limited by the budget, not by the agent running out of ideas."""
    snaps = [x for x in allsum if x["games"]]
    if len(snaps) < 2:
        return None
    a, b = snaps[-2], snaps[-1]
    prev = {g["game"]: g for g in a["games"]}
    per, state = [], {"climbing": 0, "active": 0, "stopped": 0}
    for g in b["games"]:
        p = prev.get(g["game"])
        if p is None:
            continue
        st = "climbing" if g["levels"] > p["levels"] else ("active" if g["actions"] > p["actions"] else "stopped")
        state[st] += 1
        per.append({"game": g["game"].split("-")[0], "levels": [x_lv(x, g["game"]) for x in snaps], "state": st})
    tot = [sum(g["levels"] for g in x["games"]) for x in snaps]
    wall = [x["fields"].get("total wallclock", "") for x in snaps]
    return {"snapshots": len(snaps), "total_levels": tot, "wallclock": wall, "last_interval": state,
            "last_interval_levels": tot[-1] - tot[-2], "per_game": per}


def x_lv(summary, game):
    for g in summary["games"]:
        if g["game"] == game:
            return g["levels"]
    return None


def main(argv):
    as_json = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    if not argv:
        print(__doc__); return 2
    runs = []
    if "--compare" in argv:
        i = argv.index("--compare")
        runs, argv = argv[i + 1:], argv[:i]
    if len(argv) != 1:
        print(__doc__); return 2
    text = log_text(open(os.path.expanduser(argv[0]), encoding="utf-8", errors="replace").read())
    allsum = summaries(text)
    if not allsum:
        print("NO_SUMMARY"); return 4
    fin = final_summary(text)
    if fin is None:
        print("IN_PROGRESS_ONLY: %d summaries, none with a timestamp 'ended:' (last mean %s)" % (
            len(allsum), allsum[-1]["fields"].get("mean score")))
        return 3
    f = fin["fields"]
    lv = sum(g["levels"] for g in fin["games"]); mx = sum(g["max_levels"] for g in fin["games"])
    res = {"summaries": len(allsum), "ended": f.get("ended"), "games": len(fin["games"]), "mean": f.get("mean score"),
           "median": f.get("median score"), "actions": f.get("total actions"), "levels": lv, "max_levels": mx}
    rows = compare(fin, {os.path.basename(os.path.normpath(r)): levels_of_run(r) for r in runs}) if runs else []
    prog = progression(allsum)
    if as_json:
        print(json.dumps({"final": res, "per_game": fin["games"], "compare": rows, "progression": prog}, indent=1))
        return 0
    print("FINAL ended=%s games=%d mean=%s median=%s actions=%s levels=%g/%d (summaries in log: %d)" % (
        res["ended"], res["games"], res["mean"], res["median"], res["actions"], lv, mx, len(allsum)))
    if prog:
        li = prog["last_interval"]
        print("PROGRESSION levels by snapshot %s (wallclock %s); last interval +%g levels: climbing=%d active=%d "
              "stopped=%d%s" % (prog["total_levels"], prog["wallclock"], prog["last_interval_levels"], li["climbing"],
                                li["active"], li["stopped"],
                                " -> BUDGET-BOUND" if li["climbing"] * 2 >= sum(li.values()) else ""))
    if rows:
        names = [os.path.basename(os.path.normpath(r)) for r in runs]
        print("game    bench  " + "  ".join("%14s" % n[:14] for n in names))
        tot = {n: 0 for n in names}; btot = 0
        for r in rows:
            btot += r["bench"]
            cells = []
            for n in names:
                v = r[n]
                cells.append("%14s" % ("-" if v is None else "%g" % v))
                tot[n] += v or 0
            print("%-6s %6g  %s" % (r["game"], r["bench"], "  ".join(cells)))
        print("TOTAL  %6g  %s" % (btot, "  ".join("%14g" % tot[n] for n in names)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
