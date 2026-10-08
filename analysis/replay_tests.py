#!/usr/bin/env python3
"""FI-939 DUCK 38bw: does the model's hypothesis TESTING target what it does not know? (replay, no GPU)

Hypothesis generate/test/revise is one of this project's research lines (set 2026-10-07). 38bu showed the model's negative
control statements carry no information. This asks the matching question about its experiments: when the
reasoning announces a test of a control ("let me test whether UP moves the block", "I'll check if clicking ..."),
(1) is that control actually executed soon after, and (2) was the test aimed at an uncertain control?

Uncertainty is measured from the environment log only. Before each executed action, the control's effect rate so
far on this level is p = (effects + 1) / (tries + 2) (Laplace), and the action's expected information is the binary
entropy H(p) in bits: 1.0 for a control never tried or 50/50, near 0 for one that always (or never) worked.
'Effect' is replay_claims.events_timeline's (interior change outside the level's learned HUD/animation mask).

Reported per arm:
  announcements  test sentences naming a control (deduped per level, step, control)
  executed@K     share whose control is executed within K actions on the same level after the announcement
  H(test)        mean H(p) at the first such execution
  H(all)         mean H(p) over every executed action (the null: what a typical action is expected to reveal)
  surprise       mean -log2 P(observed outcome) at the test execution vs over all actions
A test that targets uncertainty has H(test) > H(all). A test that only re-confirms has H(test) near 0.

Usage: replay_tests.py DIR [DIR ...] [--k 5] [--json OUT] [--per-game]
"""
import argparse, glob, json, math, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_claims import CTRL, DIRS, STATE, chronological, control_of, events_file_for, events_timeline  # noqa: E402

TEST = re.compile(r"[^.\n]*\b(let me|let's|lets|i'll|i will|i need to|i should|need to|going to|now)\s+"
                  r"(test|check|verify|probe|see|try)\b[^.\n]*", re.I)
CTRL_RE = re.compile(r"\b" + CTRL + r"\b", re.I)


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def annotate(timeline):
    """[(num, level, control, effect, H, surprise, tries_before)]; p from the control's record on the level BEFORE it."""
    rec = defaultdict(lambda: [0, 0])
    out = []
    for num, lv, c, eff in timeline:
        t = rec[(lv, c)]
        p = (t[1] + 1.0) / (t[0] + 2.0)
        out.append((num, lv, c, eff, h2(p), -math.log2(p if eff else 1 - p), t[0]))
        t[0] += 1
        t[1] += 1 if eff else 0
    return out


def announcements(msgs):
    level, step, seen, out = 1, 0, set(), []
    for role, tx, rc in msgs:
        for m in STATE.finditer(tx):
            step, level = int(m.group(1)), int(m.group(2))
        if role != "assistant" or not rc:
            continue
        for m in TEST.finditer(rc):
            sent = " ".join(m.group(0).split())
            for cm in CTRL_RE.finditer(sent):
                c = control_of(cm.group(1))
                if (level, step, c) in seen:
                    continue
                seen.add((level, step, c))
                out.append({"level": level, "step": step, "control": c, "sentence": sent[:200]})
    return out


def analyse(msgs, timeline, k=5):
    ann = annotate(timeline)
    rows = []
    for a in announcements(msgs):
        ctrls = DIRS if a["control"] == "DIRS" else (a["control"],)
        lvl = [x for x in ann if x[1] == a["level"] and x[0] >= a["step"]][:k]
        hit = next((x for x in lvl if x[2] in ctrls), None)
        a.update(executed=hit is not None, H=(hit[4] if hit else None), surprise=(hit[5] if hit else None),
                 effect=(hit[3] if hit else None), tries_before=(hit[6] if hit else None))
        rows.append(a)
    return rows, ann


STRATA = ((0, 0), (1, 2), (3, 5), (6, 10 ** 9))


def strata(rows, ann):
    """H and surprise of test executions vs all actions, within bands of prior tries of that control on the level.
    Without this, tests look well aimed just because they come early (a first try is always H=1)."""
    out = []
    for lo, hi in STRATA:
        ex = [r for r in rows if r["executed"] and lo <= r["tries_before"] <= hi]
        al = [x for x in ann if lo <= x[6] <= hi]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        out.append("  prior tries %s: tests n=%d H=%.2f surprise=%.2f | all n=%d H=%.2f surprise=%.2f" % (
            ("%d" % lo) if lo == hi else ("%d+" % lo if hi > 10 ** 6 else "%d-%d" % (lo, hi)),
            len(ex), mean([r["H"] for r in ex]), mean([r["surprise"] for r in ex]),
            len(al), mean([x[4] for x in al]), mean([x[5] for x in al])))
    return "\n".join(out)


def summarize(rows, ann, label="ALL"):
    n = len(rows)
    ex = [r for r in rows if r["executed"]]
    ha = sum(x[4] for x in ann) / len(ann) if ann else float("nan")
    sa = sum(x[5] for x in ann) / len(ann) if ann else float("nan")
    ht = sum(r["H"] for r in ex) / len(ex) if ex else float("nan")
    st = sum(r["surprise"] for r in ex) / len(ex) if ex else float("nan")
    low = sum(1 for r in ex if r["H"] < 0.5)
    return ("%-6s announcements=%d executed=%d (%s) | H(test)=%.2f H(all)=%.2f bits | surprise test=%.2f all=%.2f | "
            "tests on near-certain controls (H<0.5): %d (%s) | actions=%d"
            % (label, n, len(ex), ("%.0f%%" % (100.0 * len(ex) / n)) if n else "-", ht, ha, st, sa, low,
               ("%.0f%%" % (100.0 * low / len(ex))) if ex else "-", len(ann)))


def run(paths, k=5):
    rows, ann_all, per_game = [], [], defaultdict(lambda: ([], []))
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, "*_requests.jsonl"))) if os.path.isdir(p) else [p]
        for f in files:
            ev = events_file_for(f)
            if not ev:
                continue
            r, a = analyse(chronological(f), events_timeline(ev), k)
            g = os.path.basename(f).split("-")[0]
            for x in r:
                x.update(game=g, file=os.path.basename(f))
            rows += r
            ann_all += a
            per_game[g][0].extend(r)
            per_game[g][1].extend(a)
    return rows, ann_all, per_game


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--json")
    ap.add_argument("--per-game", action="store_true")
    a = ap.parse_args(argv)
    rows, ann, per_game = run(a.paths, a.k)
    print("REPLAY_TESTS k=%d from %s" % (a.k, ", ".join(a.paths)))
    print(summarize(rows, ann))
    print(strata(rows, ann))
    if a.per_game:
        for g in sorted(per_game):
            print(summarize(per_game[g][0], per_game[g][1], g))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0 if ann else 3


if __name__ == "__main__":
    sys.exit(main())
