#!/usr/bin/env python3
"""FI-939: apply the pre-registered LoRA slot rule to an HF 25-game screen, mechanically.

Rule (pre-registered before the G6 screen ran): a candidate adapter earns a Kaggle slot only if
  (1) held-out games: sum of per-game mean levels is NOT below base (or, with --min-heldout-gain-pct X,
      at least X percent above base), AND
  (2) all games: total is not significantly worse than base, tested with a one-sided paired sign-flip
      permutation test over per-game mean differences (p_worse < alpha => significantly worse).
Per-game means are used because pass counts can differ per game (hard stops, olen gate).
Levels come from ab_metrics.game_from_events, the same completion rule the fleet's A/B readouts use.

usage: fi939_gate.py --base hf-base --cand hf-lora-g6 --heldout bp35,cd82,... [--traces DIR] [--json]
--base/--cand also take comma-separated run dirs (e.g. hf-lora-g12,hf-lora-g12r): their passes are POOLED per game
before the per-game means, so replicate screens are read as one larger sample (2026-10-22, G12 replicate).
exit 0 = PASS, 1 = FAIL, 2 = cannot evaluate (missing games / data).
"""
import argparse, glob, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EV = re.compile(r"([a-z0-9]{4})-[0-9a-f]{8}_p(\d+)_events\.jsonl$")


def per_game(run_dir, levels_fn=None):
    if levels_fn is None:
        from ab_metrics import game_from_events
        # 'levels' includes a won game's last level (levels_completed misses it); see ab_metrics, 2026-10-29
        levels_fn = lambda p: game_from_events(p)["levels"]
    g = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "artifacts", "*_events.jsonl"))):
        m = EV.match(os.path.basename(p))
        if m:
            g.setdefault(m.group(1), []).append(levels_fn(p))
    return g


def pooled_per_game(traces, names, per_game_fn=None):
    """Merge the per-game pass lists of several run dirs (comma-separated names) into one sample."""
    per_game_fn = per_game_fn or per_game
    out = {}
    for n in [x.strip() for x in names.split(",") if x.strip()]:
        for g, v in per_game_fn(os.path.join(traces, n)).items():
            out.setdefault(g, []).extend(v)
    return out


def sign_flip_p_worse(diffs, n_perm=200000, seed=939):
    """One-sided p that mean(cand-base) is this low or lower by chance (H0: symmetric around 0)."""
    obs = sum(diffs)
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    if n <= 20:  # exact
        cnt = 0
        for mask in range(1 << n):
            s = sum(-d if (mask >> i) & 1 else d for i, d in enumerate(nz))
            cnt += s <= obs + 1e-12
        return cnt / (1 << n)
    rng = random.Random(seed)
    cnt = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in nz)
        cnt += s <= obs + 1e-12
    return (cnt + 1) / (n_perm + 1)


def evaluate(base, cand, heldout, alpha=0.05, min_games=20, min_heldout_gain_pct=0.0):
    games = sorted(set(base) & set(cand))
    missing_held = [k for k in heldout if k not in games]
    out = {"games": len(games), "heldout": list(heldout), "missing_heldout": missing_held, "alpha": alpha,
           "min_heldout_gain_pct": min_heldout_gain_pct}
    if len(games) < min_games or missing_held:
        out["verdict"] = "CANNOT_EVALUATE"
        return out
    mb = {k: sum(base[k]) / len(base[k]) for k in games}
    mc = {k: sum(cand[k]) / len(cand[k]) for k in games}
    tb, tc = sum(mb.values()), sum(mc.values())
    hb, hc = sum(mb[k] for k in heldout), sum(mc[k] for k in heldout)
    diffs = [mc[k] - mb[k] for k in games]
    p = sign_flip_p_worse(diffs)
    out.update({
        "total_base": round(tb, 3), "total_cand": round(tc, 3), "total_pct": round(100 * (tc - tb) / tb, 2) if tb else None,
        "heldout_base": round(hb, 3), "heldout_cand": round(hc, 3),
        "heldout_pct": round(100 * (hc - hb) / hb, 2) if hb else None,
        "up_same_down": [sum(mc[k] > mb[k] for k in games), sum(mc[k] == mb[k] for k in games),
                         sum(mc[k] < mb[k] for k in games)],
        "p_worse": round(p, 4),
        "rule_heldout_not_worse": (hc >= hb) if min_heldout_gain_pct <= 0 else
                                  (hb > 0 and 100 * (hc - hb) / hb >= min_heldout_gain_pct),
        "rule_total_not_sig_worse": p >= alpha,
    })
    out["verdict"] = "PASS" if out["rule_heldout_not_worse"] and out["rule_total_not_sig_worse"] else "FAIL"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--traces", default=os.path.expanduser(r"~\m2\traces") if os.name == "nt" else os.path.expanduser("~/m2/traces"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-heldout-gain-pct", type=float, default=0.0,
                    help="held-out leg requires cand >= base by this many percent (0 = not worse). Added 2026-10-20 after "
                         "G5 scored 24.12 on Kaggle vs base mean 26.77 while the screen total said +0.3%%")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    held = [h.strip() for h in a.heldout.split(",") if h.strip()]
    r = evaluate(pooled_per_game(a.traces, a.base), pooled_per_game(a.traces, a.cand), held, a.alpha,
                 min_heldout_gain_pct=a.min_heldout_gain_pct)
    r.update({"base": a.base, "cand": a.cand})
    if a.json:
        print(json.dumps(r))
    else:
        for k, v in r.items():
            print("%-26s %s" % (k, v))
    sys.exit({"PASS": 0, "FAIL": 1}.get(r["verdict"], 2))


if __name__ == "__main__":
    main()
