#!/usr/bin/env python3
"""FI-939: does the offline HF screen predict the hidden leaderboard?

Why (2026-10-28): every arm is now screened offline (25 public games x 4 passes, ~US$24) before it may use one of the
few daily hidden submissions, and the screen decides which arms reach a slot (DUCK 38ex). That rule is only as good as
the screen's link to the hidden score, and nothing measured it: rank_calibration.py answers the same question for the
older commit-run logs, not for the HF screens. The paper needs the answer too ("does the screen predict hidden?"), and
the 10-28 deep140 draw was kept as a pre-registered test of exactly this (DUCK 38ez).

Joins, per kernel slug (screen_kernel_map.json: slug -> the HF trace dirs screened for that exact config):
  hidden: mean of all scored submissions of the slug (experiment_ledger grouping), pooled over the slugs listed in the
          map's hidden_pool for that slug; 0.00 draws (failed runs) are dropped unless --keep-zero
  screen: official-metric public score (paper_public_eval), the mean over the mapped replicate dirs
Reports each kernel against the base kernel (hidden delta vs screen delta and whether the signs agree, with hidden
deltas inside --noise counted as ties), pairwise order agreement over all joined kernels, Spearman rho when n >= 4,
and kernels screened but without a usable hidden score (pending, or every draw failed), each with its screen delta:
the prediction on record.
Final line (cmd-safe): SCREEN_CALIB vs-base agree=A disagree=D ties=T; pairs agree=.. disagree=.. ties=.., noise N; rho=.. n=..
         or SCREEN_CALIB_INSUFFICIENT <why>.
Usage: screen_calibration.py [--from-file submissions.json] [--map screen_kernel_map.json] [--traces DIR]
                             [--env-dir DIR] [--noise 1.48]
Read-only: lists submissions (or reads a saved list) and reads traces; it never submits.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rank_calibration as R  # noqa: E402

MAP_FILE = os.path.join(HERE, "screen_kernel_map.json")
DEFAULT_TRACES = os.path.expanduser(r"~\m2\traces") if os.name == "nt" else os.path.expanduser("~/m2/traces")


def load_map(path=MAP_FILE, with_pools=False):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    kernels = {k: list(v) for k, v in d["kernels"].items()}
    base = d["base"]
    if base not in kernels:
        raise ValueError("base kernel %s is not in the map" % base)
    pools = {k: list(v) for k, v in (d.get("hidden_pool") or {}).items()}
    return (base, kernels, pools) if with_pools else (base, kernels)


def hidden_scores(records, L=None):
    """{slug: [complete scores]} over every scored submission (experiment_ledger grouping)."""
    if L is None:
        import experiment_ledger as L
    out = {}
    for g in L.group([L.normalize(r) for r in records]):
        if g["kernel"] and g["n"]:
            out.setdefault(g["kernel"], []).extend(g["scores"])
    return out


def pool_hidden(scores, pools, drop_zero=True):
    """Pure. {slug: (mean, n)}. A slug in `pools` is scored over the listed slugs' draws together (the same config
    submitted under several slugs). A 0.00 is a failed run (server down, timeout), not a measurement of the config, so it
    is dropped by default and the count of dropped zeros is returned. The count is of distinct failed DRAWS: a draw
    that belongs to a pool and to its own slug is counted once (2026-10-28: it was counted per slug, so the report said
    5 zero-score draws where Kaggle had 3)."""
    out, used = {}, set()
    for slug in set(scores) | set(pools):
        members = pools.get(slug, [slug])
        used.update(members)
        xs = [x for s in members for x in scores.get(s, [])]
        if drop_zero:
            xs = [x for x in xs if x != 0]
        if xs:
            out[slug] = (sum(xs) / len(xs), len(xs))
    dropped = sum(1 for s in used for x in scores.get(s, []) if x == 0) if drop_zero else 0
    return out, dropped


def screen_values(traces, kernels, baselines, row_fn=None):
    """{slug: official % mean over its mapped dirs that exist and score}; slugs with no scorable dir are left out."""
    if row_fn is None:
        import paper_public_eval as P

        def row_fn(d):
            rows = P.run_rows(d, baselines)
            return P.table_row(rows)["official"] if rows else None
    out = {}
    for slug, dirs in kernels.items():
        vals = []
        for name in dirs:
            p = os.path.join(traces, name)
            if os.path.isdir(p):
                v = row_fn(p)
                if v is not None:
                    vals.append(100.0 * v if v <= 1.5 else float(v))   # paper_public_eval reports a fraction
        if vals:
            out[slug] = sum(vals) / len(vals)
    return out


def report(hidden, screen, base, noise=1.48):
    """Pure. hidden {slug: (mean, n)}, screen {slug: official %} -> output lines (last line = the verdict)."""
    lines = []
    if base not in screen:
        return ["SCREEN_CALIB_INSUFFICIENT base kernel %s has no scorable screen" % base]
    joined = sorted(k for k in screen if k in hidden)
    pending = sorted(k for k in screen if k not in hidden)
    if base not in hidden:
        return ["SCREEN_CALIB_INSUFFICIENT base kernel %s has no hidden score yet" % base]
    hb, sb = hidden[base][0], screen[base]
    lines.append("  %-38s hidden %6.2f (n=%d)  screen %5.1f   [base]" % (base, hb, hidden[base][1], sb))
    agree = disagree = ties = 0
    for k in sorted(joined, key=lambda k: -hidden[k][0]):
        if k == base:
            continue
        dh, ds = hidden[k][0] - hb, screen[k] - sb
        kind = "TIE" if abs(dh) < noise else ("AGREE" if dh * ds > 0 else "DISAGREE")
        agree, disagree, ties = agree + (kind == "AGREE"), disagree + (kind == "DISAGREE"), ties + (kind == "TIE")
        lines.append("  %-38s hidden %6.2f (n=%d)  screen %5.1f   vs base: hidden %+.2f, screen %+.1f  %s"
                     % (k, hidden[k][0], hidden[k][1], screen[k], dh, ds, kind))
    for k in pending:
        lines.append("  %-38s hidden  (none usable: pending or a failed run)  screen %5.1f   prediction on record: screen %+.1f vs base"
                     % (k, screen[k], screen[k] - sb))
    rows = [{"kernel": k, "hidden": hidden[k][0], "offline": screen[k]} for k in joined]
    if len(rows) < 2:
        lines.append("SCREEN_CALIB_INSUFFICIENT %d kernel(s) have both a screen and a hidden score" % len(rows))
        return lines
    pa, pd, pt, _ = R.pairs(rows, noise)
    rho = R.spearman([r["offline"] for r in rows], [r["hidden"] for r in rows]) if len(rows) >= 4 else None
    # cmd-safe on purpose (no | & < > ^ %): the klog task passes this line to the Slack broker on a cmd command line.
    lines.append("SCREEN_CALIB vs-base agree=%d disagree=%d ties=%d; pairs agree=%d disagree=%d ties=%d, noise %.2f; "
                 "rho=%s n=%d" % (agree, disagree, ties, pa, pd, pt, noise, ("%.2f" % rho) if rho is not None else "n/a under 4",
                                  len(rows)))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-file")
    ap.add_argument("--map", default=MAP_FILE)
    ap.add_argument("--traces", default=DEFAULT_TRACES)
    ap.add_argument("--env-dir")
    ap.add_argument("--noise", type=float, default=1.48)
    ap.add_argument("--keep-zero", action="store_true", help="count 0.00 draws (failed runs) as scores")
    a = ap.parse_args(argv)
    base, kernels, pools = load_map(a.map, with_pools=True)
    import duck_score_estimate as DS
    env = a.env_dir or DS.ENV_DIR
    if not os.path.isdir(env) and env.startswith("/mnt/c/"):
        env = "C:/" + env[len("/mnt/c/"):]
    screen = screen_values(a.traces, kernels, DS.load_baselines(env))
    try:
        import experiment_ledger as L
        if a.from_file:
            with open(a.from_file, encoding="utf-8") as f:
                records = L.extract_records(f.read())
        else:
            records = L.fetch_live()
    except Exception as ex:  # live list or file read; report, do not crash a scheduled caller
        print("SCREEN_CALIB_ERROR %s" % str(ex)[:200])
        return 3
    hidden, dropped = pool_hidden(hidden_scores(records), pools, drop_zero=not a.keep_zero)
    out = report(hidden, screen, base, a.noise)
    if dropped:
        out.insert(-1, "  (%d zero-score draw(s) dropped as failed runs; --keep-zero counts them)" % dropped)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
