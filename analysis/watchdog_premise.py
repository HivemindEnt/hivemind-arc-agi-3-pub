#!/usr/bin/env python3
"""FI-939 DUCK 38ci: premise check for a stuck-level watchdog arm (replay, no GPU).

The top public fork (38cf) adds a watchdog: after 18/30/45/60 model turns on one level without clearing it, the
prompt tells the model to change tactics and, from 45 on, to call RESET. Before building our own version, this
asks whether the traces leave room for it:
  1. Does M2 ever RESET by choice? A RESET right after GAME_OVER is forced (the game needs it); any other RESET
     is voluntary. If voluntary RESETs are near zero, the nudge is new behaviour, not a re-weighting.
  2. How often is a level still cleared after T actions (or turns) without a clear? If late clears are common, a
     forced RESET at T can destroy progress; if rare, little is at risk.
  3. What share of all clears happen after T, and (with baselines) what those late clears score. A clear after
     many actions scores (human/agent)^2, so it is worth little; the level's weight in later levels is the real
     stake, which is why clearing matters more than efficiency (38cd).
  4. Restarts after GAME_OVER are the closest natural experiment to a RESET: how often is a level cleared after
     its first restart, compared with levels that reach the same action count without one?

Level of an action = the level BEFORE it executes (the event that completes level L is logged with level L+1 and
level_completed true). Turns = distinct analysis_step values on the level.

Usage: watchdog_premise.py DIR [DIR ...] [--thresholds 18,30,45,60] [--env-dir DIR] [--per-game]
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NAME = re.compile(r"^(?P<g>[a-z0-9]+)(?:-[0-9a-f]+)?_p(?P<p>\d+)_events\.jsonl$")


def attempts(events):
    """[{level, actions, turns, cleared, resets_vol, resets_forced, restarts, clear_at, first_restart_at}] per level."""
    out, cur, level, prev_go = [], None, None, False
    for r in events:
        t = r.get("type")
        if t == "initial":
            level = r.get("level", level) if r.get("level") is not None else (level or 1)
            continue
        if t != "action" or r.get("action_num") is None:
            continue
        if level is None:
            level = 1
        if cur is None or cur["level"] != level:
            cur = {"level": level, "actions": 0, "turns": set(), "cleared": False, "resets_vol": 0,
                   "resets_forced": 0, "restarts": 0, "clear_at": None, "first_restart_at": None}
            out.append(cur)
        cur["actions"] += 1
        if r.get("analysis_step") is not None:
            cur["turns"].add(r.get("analysis_step"))
        name = str(r.get("action_name") or r.get("action_display") or "").split("(")[0].strip().upper()
        if name == "RESET":
            if prev_go:
                cur["resets_forced"] += 1
            else:
                cur["resets_vol"] += 1
        if r.get("game_over"):
            cur["restarts"] += 1
            if cur["first_restart_at"] is None:
                cur["first_restart_at"] = cur["actions"]
        prev_go = bool(r.get("game_over"))
        if r.get("level_completed"):
            cur["cleared"] = True
            cur["clear_at"] = cur["actions"]
            cur = None
        if r.get("level") is not None:
            level = int(r["level"])
    for a in out:
        a["turns"] = len(a["turns"])
    return out


def load(path):
    ev = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
    return ev


def run(paths):
    per = defaultdict(list)    # game -> [attempt]
    passes = 0
    for p in paths:
        for f in sorted(glob.glob(os.path.join(p, "*_events.jsonl"))):
            m = NAME.match(os.path.basename(f))
            if not m:
                continue
            passes += 1
            for a in attempts(load(f)):
                a["game"] = m.group("g")
                per[a["game"]].append(a)
    return per, passes


def _late_score(rows, T, key, baselines):
    """Mean official level score of clears that happened after T (actions-or-turns key), using baselines."""
    if not baselines:
        return None
    s = []
    for a in rows:
        b = baselines.get(a["game"])
        if a["cleared"] and a[key] > T and b and a["level"] <= len(b):
            s.append(min((b[a["level"] - 1] / float(a["actions"])) ** 2, 1.15))
    return (sum(s) / len(s), len(s)) if s else None


def summarize(rows, thresholds, baselines=None):
    won = [a for a in rows if a["cleared"]]
    res = {"attempts": len(rows), "cleared": len(won),
           "resets_vol": sum(a["resets_vol"] for a in rows), "resets_forced": sum(a["resets_forced"] for a in rows),
           "restarts": sum(a["restarts"] for a in rows), "by": {}}
    for key in ("actions", "turns"):
        for T in thresholds:
            # still open after T units: unfinished with more than T units, or cleared on a later unit
            reach = [a for a in rows if a[key] > T]
            late = [a for a in reach if a["cleared"]]
            res["by"][(key, T)] = {"reached": len(reach), "cleared_after": len(late),
                                  "share_of_clears": (len(late) / len(won)) if won else None,
                                  "late_score": _late_score(rows, T, key, baselines)}
    # restart natural experiment: levels with >= 1 GAME_OVER restart vs same-length levels without
    rs = [a for a in rows if a["restarts"]]
    res["restart_levels"] = len(rs)
    res["restart_cleared_after"] = sum(1 for a in rs if a["cleared"] and a["clear_at"] > a["first_restart_at"])
    matched = []
    for a in rs:
        k = a["first_restart_at"]
        pool = [b for b in rows if not b["restarts"] and (b["actions"] > k)]
        if pool:
            matched.append(sum(1 for b in pool if b["cleared"]) / float(len(pool)))
    res["restart_matched_clear"] = (sum(matched) / len(matched)) if matched else None
    return res


def render(res, label, thresholds):
    out = ["%s attempts=%d cleared=%d | RESET voluntary=%d forced(after GAME_OVER)=%d | GAME_OVER restarts=%d"
           % (label, res["attempts"], res["cleared"], res["resets_vol"], res["resets_forced"], res["restarts"])]
    for key in ("actions", "turns"):
        parts = []
        for T in thresholds:
            b = res["by"][(key, T)]
            pct = ("%.0f%%" % (100.0 * b["cleared_after"] / b["reached"])) if b["reached"] else "-"
            sh = ("%.0f%%" % (100.0 * b["share_of_clears"])) if b["share_of_clears"] is not None else "-"
            ls = (" score %.2f" % b["late_score"][0]) if b["late_score"] else ""
            parts.append("T=%d open %d, cleared later %d (%s), %s of all clears%s"
                         % (T, b["reached"], b["cleared_after"], pct, sh, ls))
        out.append("  by %-7s " % key + " | ".join(parts))
    if res["restart_levels"]:
        m = res["restart_matched_clear"]
        out.append("  restart levels %d: cleared after the first restart %d (%.0f%%); levels without a restart still "
                   "open at the same action count clear %s"
                   % (res["restart_levels"], res["restart_cleared_after"],
                      100.0 * res["restart_cleared_after"] / res["restart_levels"],
                      ("%.0f%%" % (100.0 * m)) if m is not None else "-"))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--thresholds", default="18,30,45,60")
    ap.add_argument("--env-dir", default=None)
    ap.add_argument("--per-game", action="store_true")
    a = ap.parse_args(argv)
    th = [int(x) for x in a.thresholds.split(",") if x.strip()]
    baselines = None
    try:
        import duck_score_estimate as D
        d = a.env_dir or D.ENV_DIR
        baselines = D.load_baselines(d) if os.path.isdir(d) else None
    except ImportError:
        baselines = None
    per, passes = run(a.paths)
    rows = [x for g in per for x in per[g]]
    print("WATCHDOG_PREMISE passes=%d from %s%s" % (passes, ", ".join(a.paths),
                                                    "" if baselines else " (no baselines: scores omitted)"))
    print(render(summarize(rows, th, baselines), "ALL", th))
    if a.per_game:
        for g in sorted(per):
            print(render(summarize(per[g], th, baselines), g, th))
    return 0 if rows else 3


if __name__ == "__main__":
    sys.exit(main())
