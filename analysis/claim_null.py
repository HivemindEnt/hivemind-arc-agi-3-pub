#!/usr/bin/env python3
"""FI-939 DUCK 38bu: are the model's "control X does nothing" claims informative? (per-action effect perception)

38ay withdrew "83% of inert-control claims are contradicted later" because 73-100% of all actions change the board,
so "the control changed something later" is near-certain whatever the claim said. It named two things needed to
revive the question; this tool supplies both, using only recorded traces (no GPU):

1. A claim classifier. Each negative-control sentence (replay_claims.CLAIM) is put in one class, first match wins:
   OTHER_LEVEL   appeals to an earlier level ("as in level 1")
   HYPOTHETICAL  conditional or speculative ("would", "might", "if", "maybe")
   SITUATIONAL   about one position or one try ("blocked by the wall", "the second RIGHT", "here", "now")
   REPORT        past-tense report of a try ("RIGHT was a no-op", "click -> no-op", "UP = no-op")
   DURABLE       present-tense inertness ("UP is a no-op", "MOUSE does nothing"): the belief that matters
   OTHER         none of the above (tallies, plans, fragments)
   GENERAL = REPORT + DURABLE + OTHER (the first, coarser cut; kept so earlier readouts stay comparable).
2. A matched null instead of "contradicted later". For each claim, the same control's per-try effect rate on the
   same level, measured outside the claim's window, gives an expected number of effects for the tries the model
   actually made in the W actions after the claim. Observed / expected (O/E) near 1 means the claim predicted
   nothing about the control; O/E well below 1 means the claim was informative.
   Also reported: whether the control's most recent try before the claim had an effect (a direct misperception
   when the class is DURABLE), and the usage ratio U: tries of the control in the window over the tries expected
   from its share of the level's actions outside the window. U < 1 means the claim changed what the model did.
   Caveat: O/E is conditioned on the model choosing to try; a model that avoids the control except where it
   expects it to work can show O/E near 1 with an informative claim. Read O/E together with U.

Effect = replay_claims.events_timeline: an interior board change outside the level's learned HUD/animation mask.
It is still a pixel-level effect, so O/E is an upper bound on "the control really worked"; the --sample listing
exists so the classes can be read by hand.

Usage: claim_null.py DIR [DIR ...] [--window 20] [--sample N] [--seed 939] [--json OUT]
"""
import argparse, glob, json, os, random, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_claims import (CLAIM, DIRS, EARLIER, STATE, chronological, control_of,  # noqa: E402
                           events_file_for, events_timeline)

CLASSES = ("GENERAL", "SITUATIONAL", "HYPOTHETICAL", "OTHER_LEVEL")
SUBCLASSES = ("REPORT", "DURABLE", "OTHER")          # GENERAL split (38bu, after hand-reading the first sample)
HYPO = re.compile(r"\b(would|could|might|may|if|whether|unless|maybe|perhaps|possibly|suppose|assume|"
                  r"let me (try|test|check)|test(ing)? (if|whether))\b", re.I)
SITU = re.compile(r"\b(block(ed|s|ing)?|wall|walls|edge|boundary|border|obstacle|here|now|currently|this time|"
                  r"again|anymore|any more|at (this|that|the current) (position|point|spot|location|cell)|"
                  r"from (here|this position)|(first|second|third|fourth|fifth|last|\d+(st|nd|rd|th)) "
                  r"(up|down|left|right|click|press|move|try|attempt|one|time)|in this state|in that state|"
                  r"already at|at the (top|bottom|left|right))\b", re.I)


REPORT = re.compile(r"\b(was|were|did|got|gave|returned|produced|resulted|happened|tried|tested|pressed|clicked)\b"
                    r"|->|\u2192|=", re.I)
DURABLE = re.compile(r"\b(is|are|does|do|has|have|seems?|appears?)\b[^.]{0,40}?\b(inert|no-?ops?|nothing|"
                     r"no effect|useless|non-functional)\b", re.I)


def subclass(sentence):
    """Split of GENERAL; other classes map to themselves."""
    c = classify(sentence)
    if c != "GENERAL":
        return c
    if REPORT.search(sentence):
        return "REPORT"
    if DURABLE.search(sentence):
        return "DURABLE"
    return "OTHER"


def classify(sentence):
    if EARLIER.search(sentence):
        return "OTHER_LEVEL"
    if HYPO.search(sentence):
        return "HYPOTHETICAL"
    if SITU.search(sentence):
        return "SITUATIONAL"
    return "GENERAL"


def claims_with_null(msgs, timeline, window=20):
    """One row per (level, control, class): the first such sentence. Positions use 'Current state: step S'."""
    by_lc = defaultdict(list)                      # (level, control) -> [(num, effect)]
    lv_nums = defaultdict(list)                    # level -> [action num]
    for num, lv, c, eff in timeline:
        by_lc[(lv, c)].append((num, eff))
        lv_nums[lv].append(num)
    level, step = 1, 0
    rows, seen = [], set()
    for role, tx, rc in msgs:
        for m in STATE.finditer(tx):
            step, level = int(m.group(1)), int(m.group(2))
        if role != "assistant" or not rc:
            continue
        for m in CLAIM.finditer(rc):
            c = control_of(m.group(1))
            sent = " ".join(m.group(0).split())
            cls = classify(sent)
            if (level, c, cls) in seen:
                continue
            seen.add((level, c, cls))
            ctrls = DIRS if c == "DIRS" else (c,)
            tries = sorted(t for x in ctrls for t in by_lc.get((level, x), []))
            before = [t for t in tries if t[0] < step]
            post = [t for t in tries if step <= t[0] < step + window]
            outside = [t for t in tries if not (step <= t[0] < step + window)]
            rate = (sum(1 for t in outside if t[1]) / len(outside)) if outside else None
            win_n = sum(1 for n in lv_nums.get(level, []) if step <= n < step + window)
            out_n = len(lv_nums.get(level, [])) - win_n
            share = (len(outside) / out_n) if out_n else None
            rows.append({"level": level, "control": c, "cls": cls, "sub": subclass(sent), "step": step,
                         "sentence": sent[:220],
                         "win_actions": win_n, "use_share": share,
                         "expected_tries": (share * win_n) if share is not None else None,
                         "tries_before": len(before),
                         "last_try_effect": (before[-1][1] if before else None),
                         "post_tries": len(post), "post_effects": sum(1 for t in post if t[1]),
                         "base_rate": rate,
                         "expected": (rate * len(post)) if rate is not None else None})
    return rows


def oe(rows, rng=None, boots=0):
    """Observed/expected effects over rows that have post tries and a base rate; optional bootstrap 95% CI."""
    use = [r for r in rows if r["post_tries"] and r["expected"] is not None]
    o = sum(r["post_effects"] for r in use)
    e = sum(r["expected"] for r in use)
    out = {"n": len(use), "obs": o, "exp": round(e, 2), "oe": (o / e if e else None), "ci": None}
    if boots and use and rng is not None:
        vals = []
        for _ in range(boots):
            s = [use[rng.randrange(len(use))] for _ in use]
            se = sum(r["expected"] for r in s)
            if se:
                vals.append(sum(r["post_effects"] for r in s) / se)
        vals.sort()
        if vals:
            out["ci"] = (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])
    return out


def usage(rows):
    """U = observed tries in the window / tries expected from the control's share outside it."""
    use = [r for r in rows if r.get("expected_tries") is not None and r.get("win_actions")]
    o = sum(r["post_tries"] for r in use)
    e = sum(r["expected_tries"] for r in use)
    return {"n": len(use), "obs": o, "exp": round(e, 2), "u": (o / e if e else None)}


def summarize(rows, rng=None, boots=0):
    lines = []
    for cls in CLASSES + SUBCLASSES + ("ALL",):
        if cls == "ALL":
            rs = rows
        elif cls in SUBCLASSES:
            rs = [r for r in rows if r.get("sub") == cls]
        else:
            rs = [r for r in rows if r["cls"] == cls]
        if not rs:
            lines.append("%-12s n=0" % cls)
            continue
        k = oe(rs, rng, boots)
        lt = [r for r in rs if r["last_try_effect"] is not None]
        mis = sum(1 for r in lt if r["last_try_effect"])
        ci = (" CI95 %.2f-%.2f" % k["ci"]) if k["ci"] else ""
        u = usage(rs)
        ci += (" | usage U=%.2f (tries %d vs %.1f expected)" % (u["u"], u["obs"], u["exp"])) if u["u"] is not None else ""
        lines.append(("  " if cls in SUBCLASSES else "") + "%-12s n=%d | last try before claim HAD an effect: %d/%d (%s) | post-claim O/E=%s (obs %d, "
                     "exp %.1f, %d claims with tries)%s"
                     % (cls, len(rs), mis, len(lt), ("%.0f%%" % (100.0 * mis / len(lt))) if lt else "-",
                        ("%.2f" % k["oe"]) if k["oe"] is not None else "-", k["obs"], k["exp"], k["n"], ci))
    return "\n".join(lines)


def run(dirs, window=20):
    rows = []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*_requests.jsonl"))) if os.path.isdir(d) else [d]
        for f in files:
            ev = events_file_for(f)
            if not ev:
                continue
            for r in claims_with_null(chronological(f), events_timeline(ev), window):
                r.update(game=os.path.basename(f).split("-")[0], file=os.path.basename(f), arm=os.path.basename(
                    os.path.dirname(os.path.abspath(f))))
                rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--sample", type=int, default=0, help="print N random claims per class for hand reading")
    ap.add_argument("--seed", type=int, default=939)
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = run(a.paths, a.window)
    rng = random.Random(a.seed)
    print("CLAIM_NULL window=%d actions, %d claim rows from %s" % (a.window, len(rows), ", ".join(a.paths)))
    print(summarize(rows, rng, a.boots))
    if a.sample:
        for cls in SUBCLASSES + CLASSES[1:]:
            rs = [r for r in rows if r.get("sub") == cls]
            for r in rng.sample(rs, min(a.sample, len(rs))):
                print("SAMPLE %-12s %s L%d %-5s last=%s post=%d/%d base=%s | %s" % (
                    cls, r["game"], r["level"], r["control"], r["last_try_effect"], r["post_effects"],
                    r["post_tries"], ("%.2f" % r["base_rate"]) if r["base_rate"] is not None else "-",
                    r["sentence"][:160]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0 if rows else 3


if __name__ == "__main__":
    sys.exit(main())
