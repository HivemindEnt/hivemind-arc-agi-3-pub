#!/usr/bin/env python3
"""FI-939: run-to-run noise and statistical power for loadsim arms (25 games x N passes), from per-pass traces.

Why (2026-10-08, DUCK 38bg): unanchor (+15 levels, +4%) and D' (+9) could not be told apart from noise, and every
arm so far is one loadsim run of 4 passes compared with one base run. Before spending the next week's GPU quota
we need to know, from data we already have, how big an effect a 4-pass loadsim CAN detect, and how many passes an
arm would need to resolve an effect of the size we are seeing.

Method:
  - Value per (arm, game, pass) from <game>-<hash>_p<N>_events.jsonl (deduped by basename, shallowest path):
      levels  = levels completed in that pass                    (default)
      l2plus  = levels >= 2 completed (what level-transition arms target)
      score   = official per-game estimate in percent (duck_score_estimate method)
  - Pass-to-pass noise per game: pooled WITHIN-arm variance s2_g (deviations from each arm's own game mean, pooled
    over arms). Pooling assumes the arms do not change a game's noise; it is the only way to get more than 3 degrees
    of freedom per game from 4-pass runs.
  - Comparing two arms' totals (sum over games of the per-game pass mean), with na and nb passes:
      SD_diff = sqrt(sum_g s2_g * (1/na + 1/nb))
    The normal approximation is used for the z, the two-sided p, and the 80%-power minimum detectable effect
    MDE = (1.96 + 0.8416) * SD_diff. Passes needed per arm (equal n) to detect an effect d: n = 2 * 2.8016^2 * S / d^2,
    S = sum_g s2_g.
Caveats: levels are discrete and some games have zero variance; a game's noise may differ between arms; pass-level
values within one run share a kernel and a server and may not be independent. Read the numbers as an estimate of
order of magnitude, not a test of record.

Usage: loadsim_power.py label=DIR [label=DIR ...] [--metric levels|l2plus|score] [--passes 4,8,12,16]
                        [--env-dir DIR] [--json out.json]
  The first arm is the base; every other arm is compared with it.
Exit: 0 ok | 2 usage | 3 fewer than 2 passes in some arm | 4 an arm has no traces
"""
import glob, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import duck_score_estimate as S  # noqa: E402

Z_ALPHA, Z_POWER = 1.959964, 0.841621
K = Z_ALPHA + Z_POWER
NAME_RE = re.compile(r"^(?P<g>[a-z0-9]+)-[0-9a-f]+_p(?P<p>\d+)_events\.jsonl$")


def read_events(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def value(events, metric, base=None):
    lv = S.level_actions(events)
    if metric == "levels":
        return float(len(lv))
    if metric == "l2plus":
        return float(sum(1 for L, _, _ in lv if L >= 2))
    if metric == "score":
        if not base:
            return None
        cum = {L: S.level_score(base[L - 1], c) for L, c, _ in lv if L <= len(base)}
        return 100.0 * S.game_score(base, cum)
    raise ValueError(metric)


def load_arm(path, metric, baselines=None):
    """{game: {pass: value}} for one arm directory."""
    paths = [p for p in glob.glob(os.path.join(path, "**", "*_events.jsonl"), recursive=True)
             if NAME_RE.match(os.path.basename(p))]
    out = {}
    for name, p in S.dedupe_by_game(paths).items():
        m = NAME_RE.match(name)
        g, ps = m.group("g"), int(m.group("p"))
        v = value(read_events(p), metric, (baselines or {}).get(g))
        if v is not None:
            out.setdefault(g, {})[ps] = v
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def pooled_var(arms, games):
    """{game: pooled within-arm variance} over the given games; arms: [{game: {pass: v}}]."""
    out = {}
    for g in games:
        ss, df = 0.0, 0
        for a in arms:
            xs = list(a[g].values())
            if len(xs) < 2:
                continue
            m = mean(xs)
            ss += sum((x - m) ** 2 for x in xs)
            df += len(xs) - 1
        out[g] = ss / df if df else 0.0
    return out


def sd_diff(s2, na, nb):
    return math.sqrt(sum(s2.values()) * (1.0 / na + 1.0 / nb))


def p_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def passes_needed(s2, d):
    """Equal passes per arm to detect an effect d (in total units) at alpha .05 two-sided, power .8."""
    if not d:
        return None
    return math.ceil(2.0 * K * K * sum(s2.values()) / (d * d))


def analyze(specs, metric="levels", passes=(4, 8, 12, 16), baselines=None):
    arms = [(lab, load_arm(d, metric, baselines)) for lab, d in specs]
    for lab, a in arms:
        if not a:
            return {"error": "no traces", "arm": lab}
    games = sorted(set.intersection(*[set(a) for _, a in arms]))
    nmin = {lab: min(len(a[g]) for g in games) for lab, a in arms} if games else {}
    for lab, n in nmin.items():
        if n < 2:
            return {"error": "fewer than 2 passes", "arm": lab}
    s2 = pooled_var([a for _, a in arms], games)
    totals = {lab: sum(mean(list(a[g].values())) for g in games) for lab, a in arms}
    base_lab = arms[0][0]
    sum_tot = totals[base_lab] or 1.0
    noisiest = sorted(s2.items(), key=lambda kv: -kv[1])[:5]
    S2 = sum(s2.values())
    mde = [{"passes": n, "sd_diff": math.sqrt(S2 * 2.0 / n), "mde80": K * math.sqrt(S2 * 2.0 / n),
            "mde80_pct": 100.0 * K * math.sqrt(S2 * 2.0 / n) / sum_tot} for n in passes]
    comps = []
    for lab, a in arms[1:]:
        na, nb = nmin[base_lab], nmin[lab]
        sd = sd_diff(s2, na, nb)
        d = totals[lab] - totals[base_lab]
        z = d / sd if sd else 0.0
        comps.append({"arm": lab, "diff": d, "diff_pct": 100.0 * d / sum_tot, "sd_diff": sd, "z": z,
                      "p": p_two_sided(z) if sd else 1.0, "passes_needed": passes_needed(s2, d)})
    return {"metric": metric, "games": len(games), "passes": nmin, "totals": totals, "sum_var": S2,
            "noisiest": noisiest, "mde": mde, "compare": comps, "base": base_lab}


def render(r):
    lines = ["metric=%s games=%d base=%s passes=%s" % (r["metric"], r["games"], r["base"],
             ",".join("%s:%d" % kv for kv in r["passes"].items()))]
    lines.append("TOTALS (sum over games of per-game pass mean): " +
                 "  ".join("%s=%.2f" % kv for kv in r["totals"].items()))
    lines.append("NOISE: sum of pooled per-game pass variance = %.3f; noisiest games: %s" % (
        r["sum_var"], ", ".join("%s %.2f" % kv for kv in r["noisiest"])))
    for m in r["mde"]:
        lines.append("MDE80 at %2d passes per arm: SD_diff=%.2f  MDE=%.2f (%.1f%% of base total)" % (
            m["passes"], m["sd_diff"], m["mde80"], m["mde80_pct"]))
    for c in r["compare"]:
        lines.append("COMPARE %s - %s: diff=%+.2f (%+.1f%%)  SD=%.2f  z=%+.2f  p=%.3f  passes/arm for 80%% power: %s" % (
            c["arm"], r["base"], c["diff"], c["diff_pct"], c["sd_diff"], c["z"], c["p"],
            c["passes_needed"] if c["passes_needed"] is not None else "n/a"))
    return "\n".join(lines)


def main(argv):
    metric, passes, env_dir, out_json, specs = "levels", (4, 8, 12, 16), S.ENV_DIR, None, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--metric" and i + 1 < len(argv):
            metric = argv[i + 1]; i += 2
        elif a == "--passes" and i + 1 < len(argv):
            try:
                passes = tuple(int(x) for x in argv[i + 1].split(",") if x)
            except ValueError:
                print(__doc__); return 2
            i += 2
        elif a == "--env-dir" and i + 1 < len(argv):
            env_dir = argv[i + 1]; i += 2
        elif a == "--json" and i + 1 < len(argv):
            out_json = argv[i + 1]; i += 2
        elif "=" in a and not a.startswith("--"):
            lab, d = a.split("=", 1); specs.append((lab, d)); i += 1
        else:
            print(__doc__); return 2
    if len(specs) < 2 or metric not in ("levels", "l2plus", "score") or not passes or min(passes) < 1:
        print(__doc__); return 2
    bl = S.load_baselines(env_dir) if metric == "score" else None
    r = analyze(specs, metric, passes, bl)
    if r.get("error") == "no traces":
        print("NO_TRACES: %s" % r["arm"]); return 4
    if r.get("error"):
        print("TOO_FEW_PASSES: %s" % r["arm"]); return 3
    print(render(r))
    if out_json:
        with open(out_json, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
