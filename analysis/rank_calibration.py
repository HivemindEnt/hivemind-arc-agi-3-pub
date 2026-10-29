#!/usr/bin/env python3
"""FI-939: do offline commit-run results rank our kernels the way the hidden leaderboard does?

Why: offline scores (~46) and hidden scores (~25.5) differ by construction (different games, different load; DUCK
38w/38x), so the offset itself is not a target. What matters is whether an offline win predicts a hidden win. Earlier
proxies did not (DUCK 35, 36y), and one loadsim pair only detects ~8.5 points. Each daily submission is the scarce
resource (about 27 left before 11-02). If offline rank tracks hidden rank, offline runs can screen arms before they
use a slot; if not, they stay a crash/regression check only. This gives that answer from data, updated as each
hidden score lands (proposed to the operator 10-05).

Joins, per kernel slug:
  hidden: mean of the COMPLETE scores of all its scored submissions (experiment_ledger grouping, by URL)
  offline: the FINAL summary mean (and levels) of its latest commit-run log in the m2_commit_watch cache
Reports every joined kernel, then pairwise order agreement: for each pair, does offline order = hidden order? Pairs
whose hidden gap is below --noise (default 1.48, the provisional threshold) are counted as TIES, not agreements or
disagreements. Spearman rho over all joined kernels is printed when n >= 4.
Final line: CALIB_RESULT agree=A disagree=D ties=T of P pairs, rho=R | CALIB_INSUFFICIENT <why>.
Usage: rank_calibration.py [--from-file submissions.json] [--cache ~/.m2_commit_watch] [--metric score|levels]
                           [--noise 1.48] [--only-prefix duck-franzen-m2] [--log SLUG=PATH ...]
"""
import argparse, glob, json, os, sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_summary as B  # noqa: E402
import experiment_ledger as L  # noqa: E402


def hidden_by_kernel(records):
    """{slug: (mean, n)} over every scored submission of the slug (all script versions pooled)."""
    out = {}
    for g in L.group([L.normalize(r) for r in records]):
        if not g["kernel"] or not g["n"]:
            continue
        s, n = out.get(g["kernel"], (0.0, 0))
        out[g["kernel"]] = (s + sum(g["scores"]), n + g["n"])
    return {k: (s / n, n) for k, (s, n) in out.items()}


def offline_value(path, metric="score"):
    """Mean per-game score (or levels) of the FINAL summary in a kernel log, or None."""
    s = B.final_summary(B.log_text(open(path, encoding="utf-8", errors="replace").read()))
    if not s or not s.get("complete"):
        return None
    games = s.get("games") or []
    key = "levels" if metric == "levels" else "score"
    vals = [float(g[key]) for g in games if g.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def offline_by_kernel(cache, metric="score", extra=None):
    """{slug: value} from FINAL summaries of <cache>/<slug>/<slug>.log, plus extra {slug: path} logs (older eras
    kept their logs elsewhere, e.g. ~/klog_fn1/duck-qwen38-flashnext.log); levels = mean levels per game."""
    out = {}
    for slug, path in (extra or {}).items():
        if os.path.exists(os.path.expanduser(path)):
            v = offline_value(os.path.expanduser(path), metric)
            if v is not None:
                out[slug] = v
    for p in glob.glob(os.path.join(os.path.expanduser(cache), "*", "*.log")):
        slug = os.path.basename(os.path.dirname(p))
        if os.path.basename(p) != slug + ".log":
            continue
        s = B.final_summary(B.log_text(open(p, encoding="utf-8", errors="replace").read()))
        if not s or not s.get("complete"):
            continue
        games = s.get("games") or []
        if metric == "levels":
            vals = [float(g["levels"]) for g in games if g.get("levels") is not None]
        else:
            vals = [float(g["score"]) for g in games if g.get("score") is not None]
        if vals:
            out[slug] = sum(vals) / len(vals)
    return out


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def pairs(rows, noise):
    agree = disagree = ties = 0
    detail = []
    for a, b in combinations(rows, 2):
        dh = a["hidden"] - b["hidden"]
        do = a["offline"] - b["offline"]
        if abs(dh) < noise:
            ties += 1
            kind = "TIE"
        elif dh * do > 0:
            agree += 1
            kind = "AGREE"
        else:
            disagree += 1
            kind = "DISAGREE"
        detail.append((a["kernel"], b["kernel"], dh, do, kind))
    return agree, disagree, ties, detail


def run(records, cache, metric="score", noise=1.48, only_prefix=None, extra=None):
    hid = hidden_by_kernel(records)
    off = offline_by_kernel(cache, metric, extra)
    rows = [{"kernel": k, "hidden": hid[k][0], "n": hid[k][1], "offline": off[k]}
            for k in sorted(set(hid) & set(off)) if not only_prefix or k.startswith(only_prefix)]
    lines = ["  %-36s hidden %6.2f (n=%d)  offline %s %6.2f" % (r["kernel"], r["hidden"], r["n"], metric, r["offline"])
             for r in sorted(rows, key=lambda r: -r["hidden"])]
    missing = sorted(k for k in hid if k not in off and (not only_prefix or k.startswith(only_prefix)))
    if missing:
        lines.append("  hidden but no offline FINAL in cache: " + ", ".join(missing))
    if len(rows) < 2:
        lines.append("CALIB_INSUFFICIENT %d kernel(s) have both a hidden score and an offline FINAL" % len(rows))
        return lines
    a, d, t, detail = pairs(rows, noise)
    for x, y, dh, do, kind in detail:
        lines.append("  %-8s %s vs %s: hidden %+.2f, offline %+.2f" % (kind, x, y, dh, do))
    rho = spearman([r["offline"] for r in rows], [r["hidden"] for r in rows]) if len(rows) >= 4 else None
    lines.append("CALIB_RESULT agree=%d disagree=%d ties=%d of %d pairs (noise %.2f), rho=%s, n=%d kernels"
                 % (a, d, t, a + d + t, noise, ("%.2f" % rho) if rho is not None else "n/a (<4)", len(rows)))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-file")
    ap.add_argument("--cache", default="~/.m2_commit_watch")
    ap.add_argument("--metric", choices=("score", "levels"), default="score")
    ap.add_argument("--noise", type=float, default=1.48)
    ap.add_argument("--only-prefix")
    ap.add_argument("--log", action="append", default=[], help="SLUG=PATH, an extra offline log (repeatable)")
    a = ap.parse_args(argv)
    try:
        records = (L.extract_records(open(a.from_file, encoding="utf-8").read()) if a.from_file else L.fetch_live())
    except Exception as ex:  # live fetch or file read; report, do not crash the scheduled caller
        print("CALIB_ERROR %s" % str(ex)[:200])
        return 3
    extra = dict(x.split("=", 1) for x in a.log if "=" in x)
    print("\n".join(run(records, a.cache, a.metric, a.noise, a.only_prefix, extra)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
