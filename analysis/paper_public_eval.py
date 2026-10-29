#!/usr/bin/env python3
"""FI-939 paper: public-evaluation table for every offline screen, in the official metric.

Why (2026-10-27): the paper rubric asks for public eval-set results beside the leaderboard numbers (open item in the
paper draft). Every screen already ran the 25 public games x 4 passes; this turns those traces into one table in the
competition's own metric (duck_score_estimate / loss_decompose scoring: per completed level min((human/AI)^2, 1.15),
level-number weighted per game, mean over games), with the pass-to-pass spread so a reader can judge the noise.

Per run: official score as the mean of the 4 per-pass scores (each pass = mean over games) and their sd; the same on
the 14 held-out games the screen gate uses; total levels per pass; and the share of the lost score that is unfinished
levels. Works on full traces or on bundle-only results (it reads only *_events.jsonl).

  paper_public_eval.py LABEL=DIR [LABEL=DIR ...] [--env-dir DIR] [--markdown]
Exit: 0 ok | 2 usage | 3 nothing scorable
"""
import argparse, glob, os, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duck_score_estimate as D  # noqa: E402
import loss_decompose as L  # noqa: E402

HELDOUT = ("ar25", "bp35", "cd82", "dc22", "g50t", "ka59", "lp85", "r11l", "re86", "sb26", "sk48", "su15", "tu93", "vc33")


def run_rows(d, baselines):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "**", "*_events.jsonl"), recursive=True)):
        r = L.score_pass(f, baselines)
        if r:
            rows.append(r)
    return rows


def table_row(rows, heldout=HELDOUT):
    """Pure. rows from loss_decompose.score_pass -> summary dict."""
    by_pass = defaultdict(list)
    for r in rows:
        by_pass[r["pass"]].append(r)

    def pass_scores(sel):
        out = []
        for p, rs in sorted(by_pass.items()):
            rs = [r for r in rs if sel(r)]
            if rs:
                out.append(sum(r["official"] for r in rs) / len(rs))
        return out

    allp = pass_scores(lambda r: True)
    hop = pass_scores(lambda r: r["game"] in heldout)
    lv = [sum(r["levels"] for r in rs) for _, rs in sorted(by_pass.items())]
    s = L.summarize(rows)
    lost = 1 - s["official"]
    sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"games": s["games"], "passes": len(by_pass), "official": statistics.mean(allp) if allp else 0.0,
            "official_sd": sd(allp), "heldout": statistics.mean(hop) if hop else 0.0, "heldout_sd": sd(hop),
            "levels_per_pass": statistics.mean(lv) if lv else 0.0, "unfinished_share": s["unfinished"] / lost if lost else 0.0}


def render(results, markdown=False):
    head = ("run", "games", "passes", "official % (sd)", "held-out 14 % (sd)", "levels/pass", "loss from unfinished")
    lines = []
    if markdown:
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, t in results:
        cells = (label, str(t["games"]), str(t["passes"]), "%.1f (%.1f)" % (100 * t["official"], 100 * t["official_sd"]),
                 "%.1f (%.1f)" % (100 * t["heldout"], 100 * t["heldout_sd"]), "%.1f" % t["levels_per_pass"],
                 "%.0f%%" % (100 * t["unfinished_share"]))
        lines.append(("| " + " | ".join(cells) + " |") if markdown else "  ".join("%-22s" % c if i == 0 else "%-18s" % c
                                                                                 for i, c in enumerate(cells)))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="LABEL=DIR")
    ap.add_argument("--env-dir", default=D.ENV_DIR)
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    baselines = D.load_baselines(a.env_dir)
    results = []
    for spec in a.runs:
        if "=" not in spec:
            print("PUBLIC_EVAL usage: LABEL=DIR, got %r" % spec); return 2
        label, d = spec.split("=", 1)
        rows = run_rows(os.path.expanduser(d), baselines)
        if rows:
            results.append((label, table_row(rows)))
        else:
            print("PUBLIC_EVAL skip %s (no scorable passes)" % label)
    if not results:
        print("PUBLIC_EVAL nothing scorable (check --env-dir)"); return 3
    print(render(results, a.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
