#!/usr/bin/env python3
"""FI-939: build the paper-track reproduction notebook from the repo's own analysis code (DUCK 38bl).

Why: the ARC Prize 2026 paper track requires a PUBLIC Kaggle notebook next to the Writeup (deadline 2026-11-09).
Publishing the notebook is a separate, human decision; this only BUILDS the notebook
locally so it is ready, reproducible and tested. Hand-copying analysis code into a notebook would drift from the
code that produced the paper's numbers, so the builder embeds the current source of each module verbatim and the
test executes the generated notebook end to end.

The notebook:
  1. writes the embedded modules into the working directory (plain Python, no notebook magics, so the same cells
     run under the test harness);
  2. finds the arm trace directories under TRACE_ROOT (env FI939_TRACE_ROOT, default the Kaggle input dataset);
  3. reproduces: the loadsim power table (levels, L2+, official estimate), Figure 2, the ctrlprobe probe audit, the
     within-game control-family table (T4), the per-action effect-perception results (claim_null 38bu,
     replay_tests 38bw, replay_revise 38bx) on base + D', and the pre-registered targeted gate on whatever targeted
     runs exist.
  4. (2026-10-28, DUCK 38fb) reproduces the paper's CURRENT result tables from the HF screen traces: the offline
     screen gate for every arm (fi939_gate.py, pooled base, 14 held-out games, +5% bar), the public-set official
     metric table (paper_public_eval.py) and the screen-vs-hidden calibration (screen_calibration.py). The hidden
     scores are Kaggle leaderboard draws, so they are embedded as data (HIDDEN_SCORES, with its as-of date); the
     kernel-to-screen map is read from screen_kernel_map.json at build time.
Each step prints what it read; a missing arm is reported and skipped, never invented.

Usage: build_paper_notebook.py [--out paper/fi939_paper_repro.ipynb]
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ["duck_score_estimate.py", "loadsim_power.py", "control_mix.py", "probe_audit.py", "targeted_gate.py",
           "make_fig2.py", "replay_claims.py", "claim_null.py", "replay_tests.py", "replay_revise.py",
           # 2026-10-28: the HF screen tables (gate, public-set official metric, screen-vs-hidden calibration)
           "ab_metrics.py", "fi939_gate.py", "loss_decompose.py", "paper_public_eval.py", "bench_summary.py",
           "experiment_ledger.py", "rank_calibration.py", "screen_calibration.py"]
SCREEN_BASE = ["hf-base", "hf-base-r2"]
SCREEN_ARMS = [("freshlook", "hf-harn-freshlook"), ("pastnotes", "hf-harn-pastnotes"),
               ("triedhere", "hf-harn-triedhere"), ("worldmodel", "hf-worldmodel"), ("worldmodel2", "hf-harn-wm2"),
               ("automacro", "hf-harn-amacro"), ("levelpin", "hf-harn-levelpin-r3"),
               ("compact32", "hf-harn-compact32-r2"), ("deep140", "hf-harn-deep140"), ("aggr1", "hf-harn-aggr1"),
               ("effort-medium", "hf-effort-medium"), ("effort-low", "hf-effort-low"), ("KTO-G5", "hf-lora-g5"),
               ("KTO-G12", "hf-lora-g12"), ("KTO-G12r", "hf-lora-g12r")]
HELDOUT = ["ar25", "bp35", "cd82", "dc22", "g50t", "ka59", "lp85", "r11l", "re86", "sb26", "sk48", "su15", "tu93", "vc33"]
# Hidden-set draws per kernel from the Kaggle submissions API (hivemindadmin), as of 2026-10-28 14:00Z. 0.00 = a failed
# scored run (platform slow-load failure); screen_calibration drops those. Leaderboard data cannot be recomputed from
# traces, so it is embedded here and dated.
HIDDEN_AS_OF = "2026-10-28"
HIDDEN_SCORES = {
    "duck-franzen-m2-serverguard-lb": [25.54, 25.51, 26.59, 26.43, 26.93, 27.16, 29.10, 26.65, 27.01, 24.47, 0.0, 0.0],
    "duck-franzen-m2-loadguard-lb": [26.04],
    "duck-franzen-m2-levelpin-lb": [28.46],
    "duck-franzen-m2-lora-g5-lb": [24.12],
    "duck-franzen-m2-aggr1-lb": [18.03],
    "duck-franzen-m2-deep140-lb": [19.99],
    "duck-franzen-m2-freshlook-lb": [0.0],
}


def calib_map():
    with open(os.path.join(HERE, "screen_kernel_map.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {"base": d["base"], "kernels": d["kernels"], "hidden_pool": d.get("hidden_pool") or {}}
ARMS = [("base", "duck-franzen-m2-loadsim"), ("D'", "duck-franzen-m2-dprime-loadsim"),
        ("unanchor", "duck-franzen-m2-unanchor-loadsim"), ("ctrlprobe", "duck-franzen-m2-ctrlprobe-loadsim")]
TARGETED = [("tgt-base", "duck-franzen-m2-tgt-base"), ("ctrlprobe2", "duck-franzen-m2-tgt-ctrlprobe2"),
            ("inventory", "duck-franzen-m2-tgt-inventory"), ("trackrecord", "duck-franzen-m2-tgt-trackrecord")]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def module_cell(name):
    src = open(os.path.join(HERE, name), encoding="utf-8").read()
    return code("# embedded verbatim from fi939-agent-rebuild/%s\n_SRC = %r\nopen(%r, 'w', encoding='utf-8').write(_SRC)\n"
                "print('wrote %s', len(_SRC), 'chars')\n" % (name, src, name, name))


SETUP = '''import os, sys, glob
TRACE_ROOT = os.environ.get("FI939_TRACE_ROOT", "/kaggle/input/fi939-m2-traces")
ENV_DIR = os.environ.get("FI939_ENV_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files")
OUT = os.environ.get("FI939_OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
sys.path.insert(0, os.getcwd())
ARMS = %r
TARGETED = %r
def present(pairs):
    got = []
    for label, d in pairs:
        p = os.path.join(TRACE_ROOT, d)
        n = len(glob.glob(os.path.join(p, "**", "*_events.jsonl"), recursive=True))
        print("%%-10s %%-40s %%s" %% (label, d, ("%%d events files" %% n) if n else "MISSING (skipped)"))
        if n:
            got.append((label, p))
    return got
arms = present(ARMS)
targeted = present(TARGETED)
'''

POWER = '''import loadsim_power as P
if len(arms) >= 2:
    for metric in ("levels", "l2plus"):
        print(P.render(P.analyze(arms, metric, (4, 8, 16, 32))))
        print()
    if os.path.isdir(ENV_DIR):
        import duck_score_estimate as S
        r = P.analyze(arms, "score", (4, 16), S.load_baselines(ENV_DIR))
        print(P.render(r) if not r.get("error") else r)
    else:
        print("official-estimate metric skipped: no environment_files at", ENV_DIR)
else:
    print("power table needs at least two arms")
'''

FIG2 = '''import make_fig2 as F
if len(arms) >= 2:
    d = F.compute(arms, 32)
    if d.get("error"):
        print("figure 2 skipped:", d)
    else:
        try:   # drawing needs matplotlib; without it only this figure is skipped (2026-10-28: the push gate's
            F.draw(d, os.path.join(OUT, "fig2_power"))   # python has none, and the error killed every later section)
            print("Figure 2 written to", os.path.join(OUT, "fig2_power.png"), "- MDE at 4 passes %.1f%%" % dict(d["curve"])[4])
        except ImportError as ex:
            print("figure 2 skipped:", ex)
'''

AUDIT = '''import probe_audit as A
cp = dict(arms).get("ctrlprobe")
if cp:
    per, n = A.audit(cp)
    print(A.render(per)[0])
else:
    print("probe audit skipped: no ctrlprobe arm")
'''

T4 = '''import control_mix as C
base = dict(arms).get("base")
if base:
    rows = C.analyse(C.load([base]))
    wg = C.within_game(rows)
    both = {g: v for g, v in wg.items() if v["won"] and v["failed"]}
    mean = lambda xs: sum(xs) / len(xs)
    for g in sorted(wg):
        v = wg[g]
        print("%-5s won %2d %s   failed %2d %s" % (g, len(v["won"]), ("%.2f" % mean(v["won"])) if v["won"] else "   -",
              len(v["failed"]), ("%.2f" % mean(v["failed"])) if v["failed"] else "   -"))
    if both:
        later = sum(1 for v in both.values() if mean(v["failed"]) > mean(v["won"]))
        print("WITHIN-GAME over %d games with both: mean won=%.2f failed=%.2f; failed later in %d/%d" % (
            len(both), mean([mean(v["won"]) for v in both.values()]), mean([mean(v["failed"]) for v in both.values()]),
            later, len(both)))
'''

GATE = '''import targeted_gate as G
tb = dict(targeted).get("tgt-base")
for label in ("ctrlprobe2", "inventory", "trackrecord"):
    arm = dict(targeted).get(label)
    if tb and arm:
        r = G.evaluate(P.load_arm(tb, "levels"), P.load_arm(arm, "levels"), ["lf52", "bp35", "sk48"], ["r11l", "sc25", "ar25"])
        print("== %s vs tgt-base" % label)
        print(G.render(r) if not r.get("error") else r)
    else:
        print("targeted gate for %s skipped: runs not present" % label)
'''


PERCEPTION = '''import random, claim_null as CN, replay_tests as RT, replay_revise as RV
pair = [p for l, p in arms if l in ("base", "D'")]
if pair:
    print("## negative-control statements vs matched null (38bu)")
    print(CN.summarize(CN.run(pair, 20), random.Random(939), 1000))
    print("## announced tests: expected information and surprise (38bw)")
    r, ann, _ = RT.run(pair, 5)
    print(RT.summarize(r, ann))
    print(RT.strata(r, ann))
    print("## revision after a contradicted no-op statement (38bx)")
    print(RV.summarize(RV.run(pair, ("DURABLE", "REPORT"))))
else:
    print("effect-perception results skipped: neither base nor D' present")
'''


SCREEN_SETUP = '''SCREEN_BASE = %r
SCREEN_ARMS = %r
HELDOUT = %r
def screen_dir(name):
    return os.path.join(TRACE_ROOT, name)
def screen_present(pairs):
    got = []
    for label, d in pairs:
        n = len(glob.glob(os.path.join(screen_dir(d), "**", "*_events.jsonl"), recursive=True))
        print("%%-14s %%-26s %%s" %% (label, d, ("%%d pass logs" %% n) if n else "MISSING screen (skipped)"))
        if n:
            got.append((label, d))
    return got
base_screens = screen_present([("base", d) for d in SCREEN_BASE])
screens = screen_present(SCREEN_ARMS)
'''

SCREEN_GATE = '''import fi939_gate as FG
if len(base_screens) == len(SCREEN_BASE):
    base_pg = FG.pooled_per_game(TRACE_ROOT, ",".join(SCREEN_BASE))
    for label, d in screens:
        r = FG.evaluate(base_pg, FG.per_game(screen_dir(d)), HELDOUT, min_heldout_gain_pct=5.0)
        if r["verdict"] == "CANNOT_EVALUATE":
            print("%-14s cannot evaluate (%d shared games, missing held-out %s)" % (label, r["games"], r["missing_heldout"]))
        else:
            print("%-14s held-out %+6.1f%%  all levels %+6.1f%%  p_worse %.3f  %s" % (
                label, r["heldout_pct"], r["total_pct"], r["p_worse"], r["verdict"]))
else:
    print("screen gate skipped: base screens missing")
'''

PUBLIC_EVAL = '''import paper_public_eval as PE, duck_score_estimate as S
if os.path.isdir(ENV_DIR):
    bl = S.load_baselines(ENV_DIR)
    res = []
    for label, d in [("base", SCREEN_BASE[0]), ("base, repeat", SCREEN_BASE[1])] + screens:
        rows = PE.run_rows(screen_dir(d), bl) if os.path.isdir(screen_dir(d)) else []
        if rows:
            res.append((label, PE.table_row(rows)))
        else:
            print("%s: no scorable passes" % label)
    print(PE.render(res))
else:
    print("public-set official metric skipped: no environment_files at", ENV_DIR)
'''

CALIBRATION = '''import screen_calibration as SC
HIDDEN_AS_OF = %r
HIDDEN_SCORES = %r
CAL_MAP = %r
if os.path.isdir(ENV_DIR):
    hidden, dropped = SC.pool_hidden(HIDDEN_SCORES, CAL_MAP["hidden_pool"])
    screen = SC.screen_values(TRACE_ROOT, CAL_MAP["kernels"], S.load_baselines(ENV_DIR))
    print("hidden draws as of %%s; failed 0.00 draws dropped: %%d" %% (HIDDEN_AS_OF, dropped))
    print("\\n".join(SC.report(hidden, screen, CAL_MAP["base"])))
else:
    print("screen-vs-hidden calibration skipped: no environment_files at", ENV_DIR)
'''


def build():
    cells = [md("# FI-939: where a strong ARC-AGI-3 agent loses its points - reproduction notebook\n\n"
                "Reproduces the paper's power analysis (Figure 2), the probe audit, the within-game control-family "
                "table (T4) and the pre-registered targeted gate from the commit-run traces. Every analysis module "
                "is embedded verbatim from the source repository at build time.")]
    cells += [module_cell(m) for m in MODULES]
    cells += [md("## Trace directories"), code(SETUP % (ARMS, TARGETED)),
              md("## Power of one loadsim (paper section 5)"), code(POWER),
              md("## Figure 2"), code(FIG2),
              md("## What probe_controls executed (probe audit)"), code(AUDIT),
              md("## T4: last control family, within game (won vs failed L2+ levels)"), code(T4),
              md("## Per-action effect perception: statements, tests, revision (38bu-38bx)"), code(PERCEPTION),
              md("## Pre-registered targeted gate (6 games x 16 passes)"), code(GATE),
              md("## HF screens (25 games x 4 passes): trace directories"),
              code(SCREEN_SETUP % (SCREEN_BASE, SCREEN_ARMS, HELDOUT)),
              md("## Offline screen gate for every arm (paper Results table)"), code(SCREEN_GATE),
              md("## Public evaluation set, official metric"), code(PUBLIC_EVAL),
              md("## Does the offline screen predict the hidden score? (screen-vs-hidden calibration)"),
              code(CALIBRATION % (HIDDEN_AS_OF, HIDDEN_SCORES, calib_map()))]
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "paper", "fi939_paper_repro.ipynb"))
    a = ap.parse_args(argv)
    nb = build()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:   # .ipynb is eol=lf (line-endings gate)
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("NOTEBOOK_OK %s (%d cells, %d embedded modules)" % (a.out, len(nb["cells"]), len(MODULES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
