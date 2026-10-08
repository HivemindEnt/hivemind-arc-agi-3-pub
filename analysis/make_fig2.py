#!/usr/bin/env python3
"""FI-939 paper Figure 2: how large an effect one loadsim can detect (DUCK 38bg/38bi).

Panel A: the 80%-power minimum detectable effect (levels won, % of the base total) against passes per arm, for the
25-game design, computed from pooled pass-to-pass variance (loadsim_power.py). Measured arm effects are drawn as
reference lines so the reader sees at once which ones a single 4-pass run could have confirmed.
Panel B: where that noise lives - per-game pooled pass variance, largest first.

Usage: make_fig2.py base=DIR arm=DIR [...] [--out paper/fig2_power] [--max-passes 32]
  The first spec is the base; every other spec is drawn as a measured effect (diff vs base).
Exit: 0 ok | 2 usage | 4 no traces
"""
import argparse, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loadsim_power as P  # noqa: E402

INK, MUTED, GRID, HUE = "#1f1f1e", "#6b6a63", "#d9d8d2", "#2a78d6"


def compute(specs, max_passes=32):
    """Pure: the figure's numbers. Returns dict with curve [(n, mde_pct)], effects [(label, pct)], variances."""
    r = P.analyze(specs, "levels", tuple(range(2, max_passes + 1, 2)))
    if r.get("error"):
        return r
    base_tot = r["totals"][r["base"]] or 1.0
    curve = [(m["passes"], m["mde80_pct"]) for m in r["mde"]]
    effects = [(c["arm"], c["diff_pct"]) for c in r["compare"]]
    arms = [P.load_arm(d, "levels") for _, d in specs]
    games = sorted(set.intersection(*[set(a) for a in arms]))
    var = sorted(P.pooled_var(arms, games).items(), key=lambda kv: -kv[1])
    return {"curve": curve, "effects": effects, "variances": var, "base_total": base_tot, "games": len(games),
            "passes": r["passes"]}


def passes_for(curve, pct):
    """Smallest passes on the curve whose MDE is <= pct (None if none)."""
    for n, m in curve:
        if m <= pct:
            return n
    return None


def draw(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    xs = [n for n, _ in d["curve"]]
    ys = [m for _, m in d["curve"]]
    a1.plot(xs, ys, color=HUE, linewidth=2)
    a1.scatter([4], [dict(d["curve"]).get(4, ys[0])], s=40, color=HUE, zorder=3, edgecolor="white", linewidth=2)
    a1.annotate("one 4-pass run: %.0f%%" % dict(d["curve"]).get(4, ys[0]), (4, dict(d["curve"]).get(4, ys[0])),
                xytext=(10, 4), textcoords="offset points", color=INK, fontsize=9)
    seen = []
    for lab, pct in d["effects"]:
        y = abs(pct)
        a1.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        dy = 3 if any(abs(y - s) < 0.6 for s in seen) else -11
        seen.append(y)
        n = passes_for(d["curve"], y)
        a1.annotate("%s %+.1f%%: %s" % (lab, pct, ("about %d passes per arm" % n) if n else
                                           "needs > %d passes per arm" % xs[-1]),
                    (xs[-1], y), xytext=(-4, dy), textcoords="offset points", ha="right", color=MUTED, fontsize=8.5,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
    a1.set_xlim(0, xs[-1] + 1)
    a1.set_ylim(0, max(ys) * 1.08)
    a1.set_xlabel("passes per arm (25 games)", color=INK, fontsize=10)
    a1.set_ylabel("minimum detectable effect, % of levels won", color=INK, fontsize=10)
    a1.set_title("A. What one loadsim can confirm (80% power)", color=INK, fontsize=11, loc="left")
    top = d["variances"][:12]
    a2.bar(range(len(top)), [v for _, v in top], color=HUE, width=0.7)
    a2.set_xticks(range(len(top)))
    a2.set_xticklabels([g for g, _ in top], rotation=45, ha="right", color=INK, fontsize=9)
    a2.set_ylabel("pass-to-pass variance of levels won", color=INK, fontsize=10)
    a2.set_title("B. Where the noise lives (12 of %d games)" % d["games"], color=INK, fontsize=11, loc="left")
    fig.text(0.01, 0.01, "Pooled within-arm variance over %s; normal approximation; passes in one run may not be "
             "independent." % ", ".join("%s (%d)" % kv for kv in d["passes"].items()), color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=200, facecolor="white")
    fig.savefig(out + ".svg", facecolor="white")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "paper", "fig2_power"))
    ap.add_argument("--max-passes", type=int, default=32)
    a = ap.parse_args(argv)
    specs = []
    for s in a.specs:
        if "=" not in s:
            print(__doc__); return 2
        specs.append(tuple(s.split("=", 1)))
    if len(specs) < 2 or a.max_passes < 4:
        print(__doc__); return 2
    d = compute(specs, a.max_passes)
    if d.get("error"):
        print("NO_TRACES_OR_PASSES: %s" % d.get("arm")); return 4
    draw(d, a.out)
    print("FIG2_OK %s.png  MDE at 4 passes %.1f%%; effects %s" % (
        a.out, dict(d["curve"]).get(4, float("nan")), ", ".join("%s %+.1f%%" % e for e in d["effects"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
