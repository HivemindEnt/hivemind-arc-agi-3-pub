#!/usr/bin/env python3
"""FI-939 DUCK 38cb: premise check for the queued trackrecord arm (replay, no GPU).

trackrecord (38bv) appends "[X on this level: k of n tries changed the board]" to every NO-OP verdict. That only
tells the model something new when k >= 1: the control has worked before on this level, so this no-op is about the
board, not the control. With k = 0 it repeats what the model already saw. So the arm's room is the share of no-op
actions whose control already had an effect earlier on the same level, and how often that happens per pass.

Effect / no-op use replay_claims.events_timeline (interior change outside the level's learned HUD mask). M2's own
verdict uses its interior gameplay_changed flag, so counts here approximate the verdicts the model would see.
Clicks are pooled, as in the arm.

Usage: trackrecord_premise.py DIR [DIR ...] [--per-game]
"""
import argparse, glob, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_claims import events_timeline  # noqa: E402


def analyse(timeline):
    """Per pass: no-op count, no-ops whose control had >= 1 earlier effect on the level, and the k/n shown."""
    rec = defaultdict(lambda: [0, 0])
    noops = informative = 0
    shown = []
    for _, lv, c, eff in timeline:
        t = rec[(lv, c)]
        t[0] += 1
        t[1] += 1 if eff else 0
        if not eff:
            noops += 1
            if t[1] >= 1:
                informative += 1
                shown.append(t[1] / t[0])
    return {"actions": len(timeline), "noops": noops, "informative": informative, "shown": shown}


def summarize(rows, label):
    a = sum(r["actions"] for r in rows)
    n = sum(r["noops"] for r in rows)
    i = sum(r["informative"] for r in rows)
    sh = sorted(x for r in rows for x in r["shown"])
    med = sh[len(sh) // 2] if sh else None
    passes = len(rows)
    return ("%-6s passes=%d actions=%d no-ops=%d (%s) | no-ops whose control had worked earlier on the level: %d (%s of "
            "no-ops), %.1f per pass | median share of tries that worked, as shown: %s"
            % (label, passes, a, n, ("%.1f%%" % (100.0 * n / a)) if a else "-", i,
               ("%.0f%%" % (100.0 * i / n)) if n else "-", (i / passes) if passes else 0.0,
               ("%.2f" % med) if med is not None else "-"))


def run(paths):
    per = defaultdict(list)
    for p in paths:
        for f in sorted(glob.glob(os.path.join(p, "*_events.jsonl"))):
            per[os.path.basename(f).split("-")[0]].append(analyse(events_timeline(f)))
    return per


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--per-game", action="store_true")
    a = ap.parse_args(argv)
    per = run(a.paths)
    rows = [r for g in per for r in per[g]]
    print("TRACKRECORD_PREMISE from %s" % ", ".join(a.paths))
    print(summarize(rows, "ALL"))
    if a.per_game:
        for g in sorted(per):
            print(summarize(per[g], g))
    return 0 if rows else 3


if __name__ == "__main__":
    sys.exit(main())
