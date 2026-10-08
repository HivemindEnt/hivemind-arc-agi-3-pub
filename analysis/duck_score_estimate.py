#!/usr/bin/env python3
"""FI-939: estimate the OFFICIAL ARC-AGI-3 score of a recorded run (offline, stdlib only).

Methodology (docs.arcprize.org/methodology.md, read 2026-09-26):
  level_score = min((human_baseline_actions / ai_actions) ** 2, LEVEL_CAP)   for each COMPLETED level
  game_score  = sum(L * level_score_L over completed L) / sum(L over all levels)   (L = 1-based level number)
  total       = mean game_score over games (unplayed / uncompleted levels contribute 0)
Baselines: baseline_actions in each game's environment_files/<game>/<ver>/metadata.json.

The docs do not say how failed attempts count, but the arc_agi package's own scorer (duck-venv
arc_agi/scorecard.py, read 2026-09-26) does: level actions = running action total at the level-up minus the
total at the previous level-up, and every RESET adds 1 (inc_reset_count). So:
  cum   every action taken while on the level, all attempts, RESET included  = the OFFICIAL count
  last  actions of the final, successful attempt only                         (what a perfect-retry agent would get)
Per-level score is capped at 1.15 and the game score at the weight share of completed levels (min(score,
max_score) in EnvironmentScoreCalculator.to_score), both mirrored here. Assumes the Kaggle gateway runs the
same scorer version as the local package.

Also reported: `compl` = the game score if every completed level scored 1.0, which separates what is lost to
action inefficiency from what is lost to levels never completed.
Usage: duck_score_estimate.py <run_dir> [game_substr ...] [--env-dir DIR] [--json out.json] [--levels]
"""
import glob, json, os, sys

LEVEL_CAP = 1.15
# competition environment_files (metadata.json per game); override with FI939_ENV_DIR
ENV_DIR = os.environ.get("FI939_ENV_DIR", "/mnt/c/arc-agi-3/arc-agi-3-benchmarking/environment_files")


HIGH_VARIANCE_GAMES = ("ft09",)


def load_baselines(env_dir):
    """{game_id ('ft09-0d8bbf25') and short id ('ft09'): [baseline actions per level]}."""
    out = {}
    for p in glob.glob(os.path.join(env_dir, "*", "*", "metadata.json")):
        try:
            m = json.load(open(p, encoding="utf-8-sig"))
        except Exception:
            continue
        b = m.get("baseline_actions")
        gid = m.get("game_id")
        if not gid or not isinstance(b, list) or not b:
            continue
        out[gid] = [int(x) for x in b]
        out.setdefault(gid.split("-")[0], [int(x) for x in b])
    return out


def level_score(base, ai, cap=LEVEL_CAP):
    if not ai or ai <= 0 or not base:
        return 0.0
    return min((float(base) / float(ai)) ** 2, cap)


def game_score(baselines, completed):
    """completed: {level_number(1-based): level_score}. Weighted by level number over ALL levels."""
    n = len(baselines)
    den = n * (n + 1) / 2.0
    if not den:
        return 0.0
    raw = sum(L * s for L, s in completed.items() if 1 <= L <= n) / den
    cap = sum(L for L, s in completed.items() if 1 <= L <= n and s > 0) / den   # scorer: min(score, max_score)
    return min(raw, cap)


def level_actions(events):
    """Walk action events. Returns [(level_number, cum_actions, last_attempt_actions)] per completed level."""
    done, score, cum, last = [], 0, 0, 0
    for e in events:
        if e.get("type") != "action":
            continue
        cum += 1
        if str(e.get("action_name") or "").upper() == "RESET":
            last = 0          # a RESET starts a new attempt; it is not part of the successful attempt
        else:
            last += 1
        sc = e.get("score")
        if isinstance(sc, (int, float)) and int(sc) > score:
            for lv in range(score + 1, int(sc) + 1):   # normally one level at a time
                done.append((lv, cum, last))
                cum = last = 0
            score = int(sc)
        elif e.get("game_over"):
            last = 0          # the attempt died; the next attempt starts fresh (the RESET, if any, is not counted in last)
    return done


def analyze(path, baselines):
    game = os.path.basename(path).split("_p0_events")[0]
    base = baselines.get(game) or baselines.get(game.split("-")[0])
    events = []
    with open(path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    lv = level_actions(events)
    n_act = sum(1 for e in events if e.get("type") == "action")
    if not base:
        return {"game": game, "baseline": None, "levels": len(lv), "actions": n_act}
    cum = {L: level_score(base[L - 1], c) for L, c, _ in lv if L <= len(base)}
    last = {L: level_score(base[L - 1], a) for L, _, a in lv if L <= len(base)}
    full = {L: 1.0 for L, _, _ in lv if L <= len(base)}
    return {"game": game, "baseline": base, "n_levels": len(base), "levels": len(lv), "actions": n_act,
            "per_level": [{"level": L, "base": base[L - 1] if L <= len(base) else None, "cum": c, "last": a,
                           "score_cum": round(cum.get(L, 0.0), 4), "score_last": round(last.get(L, 0.0), 4)}
                          for L, c, a in lv],
            "game_cum": game_score(base, cum), "game_last": game_score(base, last), "game_compl": game_score(base, full)}


def max_influence(scored, key="game_cum"):
    """(game, share of the summed score in %, mean without that game in %) for the game whose removal changes the
    mean most. Ties go to the larger score."""
    tot = sum(r[key] for r in scored)
    n = len(scored)
    mean = tot / n
    best = max(scored, key=lambda r: (abs((tot - r[key]) / (n - 1) - mean), r[key]))
    share = 100.0 * best[key] / tot if tot else 0.0
    return best["game"].split("-")[0], share, 100.0 * (tot - best[key]) / (n - 1)


def dedupe_by_game(paths):
    """{basename: path}, keeping the shallowest path for each basename (ties: lexical order)."""
    out = {}
    for pth in sorted(paths, key=lambda x: (x.count(os.sep), x)):
        out.setdefault(os.path.basename(pth), pth)
    return dict(sorted(out.items()))


def main(argv):
    env_dir, out_json, show_levels, args = ENV_DIR, None, False, []
    i = 0
    while i < len(argv):
        if argv[i] == "--env-dir":
            env_dir = argv[i + 1]; i += 2
        elif argv[i] == "--json":
            out_json = argv[i + 1]; i += 2
        elif argv[i] == "--levels":
            show_levels = True; i += 1
        else:
            args.append(argv[i]); i += 1
    if not args:
        print(__doc__); return 2
    bl = load_baselines(env_dir)
    if not bl:
        print("NO_BASELINES: no metadata.json with baseline_actions under %s" % env_dir); return 3
    subs = args[1:]
    paths = sorted(glob.glob(os.path.join(args[0], "**", "*_p0_events.jsonl"), recursive=True))
    # FI-939 2026-10-08: M2 trace pulls hold the same game at the top level AND under artifacts/ (byte-identical),
    # so the recursive glob counted 6 games twice (31 "games" for a 25-game run). One path per game: shallowest.
    paths = list(dedupe_by_game(paths).values())
    res = [analyze(p, bl) for p in paths if not subs or any(x in os.path.basename(p) for x in subs)]
    scored = [r for r in res if r.get("baseline")]
    print("%-15s %6s %7s %8s %9s %9s %9s" % ("game", "levels", "actions", "n_lvls", "score_cum", "score_last", "compl"))
    for r in res:
        if not r.get("baseline"):
            print("%-15s %6d %7d  (no baseline; excluded)" % (r["game"][:15], r["levels"], r["actions"])); continue
        print("%-15s %6d %7d %8d %8.2f%% %8.2f%% %8.2f%%" % (r["game"][:15], r["levels"], r["actions"], r["n_levels"],
              100 * r["game_cum"], 100 * r["game_last"], 100 * r["game_compl"]))
        if show_levels:
            for d in r["per_level"]:
                print("      L%-2d base=%-4s ai_cum=%-4d ai_last=%-4d score_cum=%.4f score_last=%.4f" % (
                    d["level"], d["base"], d["cum"], d["last"], d["score_cum"], d["score_last"]))
    if scored:
        m = lambda k: 100.0 * sum(r[k] for r in scored) / len(scored)
        # FI-939 2026-09-27: ft09 alone swings the small-sample mean ~2x (sonpham fork + FN arms), so report the
        # mean without the high-variance games next to the headline. Printed BEFORE the TOTAL line so callers
        # that read `tail -n 1` keep getting TOTAL.
        stable = [r for r in scored if not any(r["game"].startswith(g) for g in HIGH_VARIANCE_GAMES)]
        if stable and len(stable) < len(scored):
            ms = lambda k: 100.0 * sum(r[k] for r in stable) / len(stable)
            print("EX-HIGHVAR (excl %s) over %d games: OFFICIAL(score_cum)=%.3f%%  perfect-retry(score_last)=%.3f%%" % (
                ",".join(HIGH_VARIANCE_GAMES), len(stable), ms("game_cum"), ms("game_last")))
        # FI-939 2026-09-28: generalizes EX-HIGHVAR for wide (25-game) runs, where the dominant game need not be
        # ft09: name the single game whose removal moves the mean most, and the mean without it.
        if len(scored) >= 3:
            g_top, share, m_wo = max_influence(scored)
            print("MAX-INFLUENCE %s: %.1f%% of the total; OFFICIAL(score_cum) without it over %d games=%.3f%%" % (
                g_top, share, len(scored) - 1, m_wo))
        print("TOTAL over %d games: OFFICIAL(score_cum)=%.3f%%  perfect-retry(score_last)=%.3f%%  if-every-completed-level-were-1.0=%.3f%%" % (
            len(scored), m("game_cum"), m("game_last"), m("game_compl")))
    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
