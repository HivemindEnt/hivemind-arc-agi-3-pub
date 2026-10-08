#!/usr/bin/env python3
"""FI-939: control-family coverage per level from environment event logs (paper table T4, DUCK 38bc).

Why: Figure 1 shows one lf52 pass spending ~45 clicks on level 2 before trying the movement keys. This measures the
pattern across all games and passes, from ground truth only (events.jsonl). For each (game, pass, level) segment:
  - families used on the level (move ACTION1-4 / act ACTION5 / click ACTION6 / undo ACTION7);
  - families the GAME has (used anywhere in that game, any pass, any level) but this level never tried;
  - actions before the level's second distinct family appeared (None if only one family was used);
  - last_family_at: position (fraction of the level's actions) at which the LAST of the game's families was
    first tried on this level (1.0 if one was never tried).
Then it compares won and failed segments, split L1 vs L2+, and (WITHIN) per game: only games with >= 2 families,
L2+ segments with >= 10 actions, comparing mean last_family_at of won vs failed segments of the same game.
That within-game view matters: a pooled comparison mixes games that need one family with games that need several.
RESET is ignored (it is not a control).
Usage: control_mix.py TRACE_DIR [TRACE_DIR ...] [--json OUT]
"""
import glob, json, os, re, sys
from collections import defaultdict

FAM = {"ACTION1": "move", "ACTION2": "move", "ACTION3": "move", "ACTION4": "move",
       "ACTION5": "act", "ACTION6": "click", "ACTION7": "undo"}


def segments(acts):
    """Split one pass's action events into level segments: [{level, won, fams:[...]}] in order."""
    segs, cur, lvl = [], [], 1
    for e in acts:
        f = FAM.get(str(e.get("action_name") or "").upper())
        if f:
            cur.append(f)
        # FI-939 2026-10-08 (38bg): the game-winning action has level_completed False and state WIN, so the
        # final level of a won game was counted as a FAILED segment (base run: 369 vs 402 events-score levels).
        if e.get("level_completed") or str(e.get("state") or "").upper() == "WIN":
            segs.append({"level": lvl, "won": True, "fams": cur})
            cur, lvl = [], lvl + 1
    if cur:
        segs.append({"level": lvl, "won": False, "fams": cur})
    return segs


def second_family_at(fams):
    seen = []
    for i, f in enumerate(fams):
        if f not in seen:
            seen.append(f)
            if len(seen) == 2:
                return i
    return None


def last_family_at(fams, game_families):
    n = len(fams)
    if not n:
        return 1.0
    return max((fams.index(f) / n if f in fams else 1.0) for f in game_families) if game_families else 0.0


def within_game(rows, min_actions=10):
    """{game: {won: [..], failed: [..]}} of last_family_at over L2+ segments, games with >= 2 families."""
    out = defaultdict(lambda: {"won": [], "failed": []})
    for r in rows:
        if r["level"] < 2 or r["n"] < min_actions or r["game_families"] < 2:
            continue
        out[r["game"].split(":")[-1].split("-")[0]]["won" if r["won"] else "failed"].append(r["last_family_at"])
    return dict(out)


def load(dirs):
    """{(game, pass): [action events]} from *_pN_events.jsonl, one file per basename (shallowest)."""
    out = {}
    for d in dirs:
        paths = sorted(glob.glob(os.path.join(d, "**", "*_p*_events.jsonl"), recursive=True),
                       key=lambda x: (x.count(os.sep), x))
        for p in paths:
            m = re.match(r"(.+?)_(p\d+)_events\.jsonl$", os.path.basename(p))
            if not m:
                continue
            key = (os.path.basename(d.rstrip("/")) + ":" + m.group(1), m.group(2))
            if key in out:
                continue
            acts = []
            with open(p) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get("type") == "action":
                        acts.append(e)
            out[key] = acts
    return out


def analyse(runs):
    game_fams = defaultdict(set)
    segs = []
    for (game, pas), acts in runs.items():
        for s in segments(acts):
            s.update(game=game, pas=pas)
            segs.append(s)
            game_fams[game].update(s["fams"])
    rows = []
    for s in segs:
        used = set(s["fams"])
        rows.append({"game": s["game"], "pass": s["pas"], "level": s["level"], "won": s["won"], "n": len(s["fams"]),
                     "used": sorted(used), "untried": sorted(game_fams[s["game"]] - used),
                     "second_at": second_family_at(s["fams"]),
                     "game_families": len(game_fams[s["game"]]),
                     "last_family_at": round(last_family_at(s["fams"], game_fams[s["game"]]), 4)})
    return rows


def summarise(rows, min_actions=10):
    """Groups (L1|L2+, won|failed) over segments with >= min_actions actions (short segments say little)."""
    out = {}
    for lv in ("L1", "L2+"):
        for won in (True, False):
            g = [r for r in rows if (r["level"] == 1) == (lv == "L1") and r["won"] == won and r["n"] >= min_actions]
            if not g:
                continue
            k = "%s %s" % (lv, "won" if won else "failed")
            multi = [r for r in g if len(set(r["used"])) >= 1]
            sec = [r["second_at"] for r in g if r["second_at"] is not None]
            out[k] = {"segments": len(g),
                      "share_with_untried_family": sum(1 for r in g if r["untried"]) / len(g),
                      "mean_families_used": sum(len(r["used"]) for r in multi) / len(multi),
                      "share_single_family": sum(1 for r in g if len(r["used"]) == 1) / len(g),
                      "median_actions_before_2nd_family": sorted(sec)[len(sec) // 2] if sec else None}
    return out


def main(argv):
    out_json, dirs = None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--json":
            out_json = argv[i + 1]; i += 2
        else:
            dirs.append(argv[i]); i += 1
    if not dirs:
        print(__doc__); return 2
    rows = analyse(load(dirs))
    summ = summarise(rows)
    print("%-12s %8s %22s %18s %16s %22s" % ("group", "segments", "has_untried_family", "families_used", "single_family", "median_acts_to_2nd_fam"))
    for k, v in summ.items():
        print("%-12s %8d %21.0f%% %18.2f %15.0f%% %22s" % (k, v["segments"], 100 * v["share_with_untried_family"],
              v["mean_families_used"], 100 * v["share_single_family"], v["median_actions_before_2nd_family"]))
    wg = within_game(rows)
    mean = lambda xs: sum(xs) / len(xs)
    print("WITHIN-GAME last_family_at (L2+, >=10 actions, games with >=2 families): game nW nF won failed")
    both = []
    for g in sorted(wg):
        w, f = wg[g]["won"], wg[g]["failed"]
        print("  %-6s %3d %3d %6s %6s" % (g, len(w), len(f), "%.2f" % mean(w) if w else "-", "%.2f" % mean(f) if f else "-"))
        if w and f:
            both.append((mean(w), mean(f)))
    if both:
        print("WITHIN-GAME over %d games with both: mean won=%.2f failed=%.2f; failed later in %d/%d games" % (
            len(both), mean([x for x, _ in both]), mean([y for _, y in both]), sum(y > x for x, y in both), len(both)))
    if out_json:
        json.dump({"rows": rows, "summary": summ, "within_game": wg}, open(out_json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
