#!/usr/bin/env python3
"""FI-939 DUCK 38bx: the REVISE step - after the environment contradicts "X does nothing", does the model update?

38bu: inert-control statements carry no information. 38bw: the experiments behind them are well aimed. So the loss
should sit in revision. This measures it directly from recorded traces (no GPU).

For each pass and each (level, control) with a negative statement (claim_null classes; default: DURABLE and REPORT,
i.e. GENERAL beliefs), find the first CONTRADICTION: an action of that control on the same level, at or after the
statement, whose board effect is real (replay_claims.events_timeline: interior change outside the learned HUD mask).
Then read the model's later reasoning on that level, in order, for the first sentence that names the control and is
either
  POSITIVE  the control moved / changed / worked / rotated / selected / did something, or
  NEGATIVE  another inert statement (replay_claims.CLAIM).
Outcome per contradiction: REVISED (positive first), REASSERTED (negative first) or SILENT (neither before the level
ends). Also: assistant turns from contradiction to the first positive mention.

Pixel-level effect is an upper bound on "the control did what the model cared about", so REASSERTED can include
fair calls (an effect the model rightly treats as irrelevant). The --sample listing exists for hand reading.

Usage: replay_revise.py DIR [DIR ...] [--classes DURABLE,REPORT] [--sample N] [--seed 939] [--json OUT]
"""
import argparse, glob, json, os, random, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_claims import CLAIM, CTRL, DIRS, STATE, chronological, control_of, events_file_for, events_timeline  # noqa: E402
from claim_null import subclass  # noqa: E402

POS = re.compile(r"[^.\n]*\b" + CTRL + r"\b[^.\n]{0,60}?\b(moves?|moved|works?|worked|changes?|changed|rotates?|"
                 r"rotated|shifts?|shifted|selects?|selected|toggles?|toggled|did something|does something|"
                 r"has an effect|had an effect|is effective|responds?|responded|pushes|pushed|activates?|activated)\b"
                 r"[^.\n]*", re.I)


def mentions(sentence_text):
    """[(kind, control, sentence)] for positive and negative statements in a reasoning block, in text order."""
    out = []
    for kind, rx in (("POS", POS), ("NEG", CLAIM)):
        for m in rx.finditer(sentence_text):
            out.append((m.start(), kind, control_of(m.group(1)), " ".join(m.group(0).split())))
    out.sort()
    res = []
    for _, kind, c, s in out:
        if kind == "POS" and CLAIM.search(s):      # "UP moved nothing" style: negative wins
            kind = "NEG"
        res.append((kind, c, s))
    return res


def _match(c, ctrls):
    return c in ctrls or (c == "DIRS" and set(ctrls) <= set(DIRS)) or (c in DIRS and "DIRS" in ctrls)


def analyse(msgs, timeline, classes=("DURABLE", "REPORT")):
    # message index -> (level, step) as of that message; reasoning blocks of assistant turns
    level, step = 1, 0
    turns = []                                       # (idx, level, step, reasoning)
    for i, (role, tx, rc) in enumerate(msgs):
        for m in STATE.finditer(tx):
            step, level = int(m.group(1)), int(m.group(2))
        if role == "assistant" and rc:
            turns.append((i, level, step, rc))
    rows, seen = [], set()
    for ti, (i, lv, st, rc) in enumerate(turns):
        for kind, c, sent in mentions(rc):
            if kind != "NEG" or subclass(sent) not in classes or (lv, c) in seen:
                continue
            seen.add((lv, c))
            ctrls = DIRS if c == "DIRS" else (c,)
            contra = next((t for t in timeline if t[1] == lv and t[2] in ctrls and t[0] >= st and t[3]), None)
            row = {"level": lv, "control": c, "step": st, "sentence": sent[:200], "contradicted": contra is not None,
                   "contra_step": contra[0] if contra else None, "outcome": None, "turns_to": None, "reply": None}
            if contra:
                later = [(tj, x) for tj, x in enumerate(turns) if tj > ti and x[2] > contra[0]]
                n = 0
                for tj, (_, lv2, st2, rc2) in later:
                    if lv2 != lv:
                        break
                    n += 1
                    hit = next(((k, s) for k, c2, s in mentions(rc2) if _match(c2, ctrls)), None)
                    if hit:
                        row.update(outcome="REVISED" if hit[0] == "POS" else "REASSERTED", turns_to=n,
                                   reply=hit[1][:200])
                        break
                if row["outcome"] is None:
                    row["outcome"] = "SILENT"
            rows.append(row)
    return rows


def summarize(rows):
    c = [r for r in rows if r["contradicted"]]
    k = Counter(r["outcome"] for r in c)
    rev = sorted(r["turns_to"] for r in c if r["outcome"] == "REVISED")
    med = rev[len(rev) // 2] if rev else None
    n = len(c)
    pct = lambda x: ("%.0f%%" % (100.0 * x / n)) if n else "-"
    return ("REVISE statements=%d contradicted on the same level=%d | REVISED %d (%s), REASSERTED %d (%s), SILENT %d (%s) "
            "| median turns to revision %s" % (len(rows), n, k["REVISED"], pct(k["REVISED"]), k["REASSERTED"],
                                                pct(k["REASSERTED"]), k["SILENT"], pct(k["SILENT"]), med))


def run(paths, classes):
    rows = []
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, "*_requests.jsonl"))) if os.path.isdir(p) else [p]
        for f in files:
            ev = events_file_for(f)
            if not ev:
                continue
            for r in analyse(chronological(f), events_timeline(ev), classes):
                r.update(game=os.path.basename(f).split("-")[0], file=os.path.basename(f))
                rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--classes", default="DURABLE,REPORT")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=939)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    classes = tuple(x.strip() for x in a.classes.split(",") if x.strip())
    rows = run(a.paths, classes)
    print("REPLAY_REVISE classes=%s from %s" % (",".join(classes), ", ".join(a.paths)))
    print(summarize(rows))
    for cls in classes:
        print("  %-8s %s" % (cls, summarize([r for r in rows if subclass(r["sentence"]) == cls])))
    if a.sample:
        rng = random.Random(a.seed)
        for oc in ("REVISED", "REASSERTED", "SILENT"):
            rs = [r for r in rows if r["outcome"] == oc]
            for r in rng.sample(rs, min(a.sample, len(rs))):
                print("SAMPLE %-10s %s L%d %-5s | %s || %s" % (oc, r["game"], r["level"], r["control"],
                                                             r["sentence"][:110], (r["reply"] or "")[:110]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0 if rows else 3


if __name__ == "__main__":
    sys.exit(main())
