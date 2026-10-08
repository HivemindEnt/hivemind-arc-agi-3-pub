#!/usr/bin/env python3
"""FI-939 DUCK 38cd: where does the official score go - unfinished levels or inefficiency on finished ones?

The official game score weights level L by L and scores each completed level min((human/agent actions)^2, 1.15),
capped at the completed levels' weight share. So 1 - score splits exactly into:
  unfinished    1 - compl            (compl = the score if every completed level scored 1.0)
  inefficiency  compl - official     (completed levels scored below 1.0 because of extra actions)
Efficiency arms can only recover the second part; level-clearing arms the first.

The tool also reports where runs get stuck: the first uncleared level per game-pass. Because later levels carry
more weight, getting past level k unlocks the weight of every level after it.

Uses duck_score_estimate's scoring (level_actions, level_score, game_score) on every pass, not just p0.

Usage: loss_decompose.py RUN_DIR [RUN_DIR ...] [--env-dir DIR] [--per-game]
"""
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duck_score_estimate as D  # noqa: E402

NAME = re.compile(r"^(?P<g>[a-z0-9]+(?:-[0-9a-f]+)?)_p(?P<p>\d+)_events\.jsonl$")


def score_pass(path, baselines):
    m = NAME.match(os.path.basename(path))
    if not m:
        return None
    game = m.group("g")
    base = baselines.get(game) or baselines.get(game.split("-")[0])
    if not base:
        return None
    events = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    lv = D.level_actions(events)
    cum = {L: D.level_score(base[L - 1], c) for L, c, _ in lv if L <= len(base)}
    full = {L: 1.0 for L, _, _ in lv if L <= len(base)}
    done = sorted(cum)
    stuck = next((L for L in range(1, len(base) + 1) if L not in cum), None)   # None = all levels cleared
    return {"game": game.split("-")[0], "pass": int(m.group("p")), "n_levels": len(base),
            "official": D.game_score(base, cum), "compl": D.game_score(base, full), "levels": len(done),
            "stuck": stuck}


def summarize(rows):
    """Mean over passes within a game, then over games (the competition averages games)."""
    by = defaultdict(list)
    for r in rows:
        by[r["game"]].append(r)
    g_off = {g: sum(r["official"] for r in v) / len(v) for g, v in by.items()}
    g_cmp = {g: sum(r["compl"] for r in v) / len(v) for g, v in by.items()}
    n = len(by)
    off = sum(g_off.values()) / n if n else 0.0
    cmp_ = sum(g_cmp.values()) / n if n else 0.0
    stuck = Counter(("all cleared" if r["stuck"] is None else ("L1" if r["stuck"] == 1 else "L2" if r["stuck"] == 2
                     else "L3+")) for r in rows)
    return {"games": n, "passes": len(rows), "official": off, "compl": cmp_, "unfinished": 1 - cmp_,
            "inefficiency": cmp_ - off, "stuck": stuck, "per_game": {g: (g_off[g], g_cmp[g], len(by[g])) for g in by}}


def render(s, label):
    lost = 1 - s["official"]
    st = s["stuck"]
    tot = sum(st.values()) or 1
    lines = ["%s: %d games, %d game-passes | official %.1f%% | lost %.1f points: unfinished levels %.1f (%.0f%% of the "
             "loss), inefficiency on finished levels %.1f (%.0f%%)"
             % (label, s["games"], s["passes"], 100 * s["official"], 100 * lost, 100 * s["unfinished"],
                100 * s["unfinished"] / lost if lost else 0, 100 * s["inefficiency"],
                100 * s["inefficiency"] / lost if lost else 0),
             "  first uncleared level per game-pass: " + ", ".join(
                 "%s %d (%.0f%%)" % (k, st[k], 100.0 * st[k] / tot) for k in ("L1", "L2", "L3+", "all cleared"))]
    return "\n".join(lines)


def run(paths, env_dir):
    baselines = D.load_baselines(env_dir)
    rows = []
    for p in paths:
        for f in sorted(glob.glob(os.path.join(p, "*_events.jsonl"))):
            r = score_pass(f, baselines)
            if r:
                rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--env-dir", default=D.ENV_DIR)
    ap.add_argument("--per-game", action="store_true")
    a = ap.parse_args(argv)
    rows = run(a.paths, a.env_dir)
    if not rows:
        print("LOSS_DECOMPOSE no scorable passes (check --env-dir)")
        return 3
    s = summarize(rows)
    print(render(s, "LOSS_DECOMPOSE " + ", ".join(os.path.basename(p.rstrip("/")) for p in a.paths)))
    if a.per_game:
        for g, (o, c, n) in sorted(s["per_game"].items(), key=lambda kv: kv[1][0]):
            print("  %-5s passes=%d official %5.1f%%  unfinished %5.1f  inefficiency %4.1f" % (
                g, n, 100 * o, 100 * (1 - c), 100 * (c - o)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
