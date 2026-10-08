#!/usr/bin/env python3
"""FI-939: audit what probe_controls() actually did in a ctrlprobe run (DUCK 38bh).

Why: the ctrlprobe arm (38av) gives the model a controls diagnostic that tries keys, clicks one object per kind,
and re-tries keys after an effective click. An arm readout counts levels; it cannot say whether the probe ran as
designed. Reading the first probe table in the 2026-10-08 run showed `probe_controls: 1 actions` followed by every
later row `REFUSED StaleStateActionError`: the M2 sandbox refuses any action in a snippet after a no-op, so one
inert key ended the whole probe. This tool measures how often that happened, over every game and pass.

Method: each `analysis` event's transcript carries the turn history, so the same tool result repeats across
events. Per (game, pass) we keep each distinct probe table once (header line plus its rows). A table is
  BLOCKED      if it executed <= 1 action and has >= 2 rows refused by the stale-state guard;
  PARTIAL      if any row was refused but it is not BLOCKED;
  CLEAN        otherwise.
We also count distinct python tool-call code blocks containing `probe_controls(` (calls whose table never printed,
e.g. because a BaseException guard ended the snippet, show up as calls without a table).
Caveat: two byte-identical tables in one pass are counted once.

Usage: probe_audit.py TRACE_DIR [--games g1,g2] [--json out.json]
Exit: 0 ok | 2 usage | 4 no events files
"""
import glob, json, os, re, sys
from collections import Counter, defaultdict

NAME_RE = re.compile(r"^(?P<g>[a-z0-9]+)-[0-9a-f]+_p(?P<p>\d+)_events\.jsonl$")
HEAD_RE = re.compile(r"^probe_controls: (\d+) actions(?: \(stopped: ([^)]*)\))?\s*$")
ROW_RE = re.compile(r"^  (.+?)\s+-> (.*)$")
CALL_RE = re.compile(r"\[TOOL CALL: python\]\s*<tool_call>(.*?)</tool_call>", re.S)


def parse_tables(text):
    """Distinct probe tables in one transcript: [(header_line, [(label, effect), ...])]."""
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = HEAD_RE.match(lines[i])
        if not m:
            i += 1
            continue
        rows = []
        j = i + 1
        while j < len(lines):
            r = ROW_RE.match(lines[j])
            if not r:
                break
            rows.append((r.group(1).strip(), r.group(2).strip()))
            j += 1
        out.append((lines[i].strip(), rows))
        i = j
    return out


def classify(header, rows):
    m = HEAD_RE.match(header)
    used = int(m.group(1)) if m else 0
    refused = Counter()
    for _, eff in rows:
        if eff.startswith("REFUSED"):
            refused[eff.split(" ", 1)[1] if " " in eff else "?"] += 1
    stale = refused.get("StaleStateActionError", 0)
    if used <= 1 and stale >= 2:
        kind = "BLOCKED"
    elif refused:
        kind = "PARTIAL"
    else:
        kind = "CLEAN"
    changed = sum(1 for _, eff in rows if not eff.startswith("REFUSED") and not eff.startswith("nothing"))
    return {"used": used, "kind": kind, "refused": dict(refused), "rows": len(rows), "changed": changed}


def audit_file(path):
    tables, calls = {}, set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "analysis":
                continue
            t = str(e.get("transcript") or "")
            if "probe_controls" not in t:
                continue
            for code in CALL_RE.findall(t):
                if "probe_controls(" in code and "def probe_controls" not in code:
                    calls.add(code.strip())
            for h, rows in parse_tables(t):
                tables.setdefault((h, tuple(rows)), classify(h, rows))
    return {"calls": len(calls), "tables": list(tables.values())}


def audit(trace_dir, games=None):
    paths = [p for p in glob.glob(os.path.join(trace_dir, "**", "*_events.jsonl"), recursive=True)
             if NAME_RE.match(os.path.basename(p))]
    best = {}
    for p in sorted(paths, key=lambda x: (x.count(os.sep), x)):
        best.setdefault(os.path.basename(p), p)
    per = defaultdict(lambda: {"calls": 0, "tables": 0, "BLOCKED": 0, "PARTIAL": 0, "CLEAN": 0,
                               "actions": 0, "changed_rows": 0, "refused": Counter()})
    for name, p in sorted(best.items()):
        g = NAME_RE.match(name).group("g")
        if games and g not in games:
            continue
        r = audit_file(p)
        d = per[g]
        d["calls"] += r["calls"]
        for t in r["tables"]:
            d["tables"] += 1
            d[t["kind"]] += 1
            d["actions"] += t["used"]
            d["changed_rows"] += t["changed"]
            d["refused"].update(t["refused"])
    return per, len(best)


def render(per):
    lines = ["%-6s %5s %6s %7s %7s %5s %7s %7s  refused" % ("game", "calls", "tables", "BLOCKED", "PARTIAL", "CLEAN",
                                                             "actions", "changed")]
    tot = defaultdict(int)
    ref = Counter()
    for g in sorted(per):
        d = per[g]
        lines.append("%-6s %5d %6d %7d %7d %5d %7d %7d  %s" % (g, d["calls"], d["tables"], d["BLOCKED"], d["PARTIAL"],
                     d["CLEAN"], d["actions"], d["changed_rows"], dict(d["refused"]) or "-"))
        for k in ("calls", "tables", "BLOCKED", "PARTIAL", "CLEAN", "actions", "changed_rows"):
            tot[k] += d[k]
        ref.update(d["refused"])
    lines.append("%-6s %5d %6d %7d %7d %5d %7d %7d  %s" % ("TOTAL", tot["calls"], tot["tables"], tot["BLOCKED"],
                 tot["PARTIAL"], tot["CLEAN"], tot["actions"], tot["changed_rows"], dict(ref) or "-"))
    if tot["tables"]:
        lines.append("BLOCKED share %.0f%% of tables; mean executed actions per table %.1f" % (
            100.0 * tot["BLOCKED"] / tot["tables"], tot["actions"] / tot["tables"]))
    return "\n".join(lines), dict(tot), ref


def main(argv):
    games, out_json, args = None, None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--games" and i + 1 < len(argv):
            games = set(x for x in argv[i + 1].split(",") if x); i += 2
        elif argv[i] == "--json" and i + 1 < len(argv):
            out_json = argv[i + 1]; i += 2
        elif argv[i].startswith("--"):
            print(__doc__); return 2
        else:
            args.append(argv[i]); i += 1
    if len(args) != 1:
        print(__doc__); return 2
    per, n = audit(args[0], games)
    if not n:
        print("NO_EVENTS: %s" % args[0]); return 4
    text, tot, ref = render(per)
    print(text)
    if out_json:
        with open(out_json, "w") as f:
            json.dump({"per_game": {g: dict(d, refused=dict(d["refused"])) for g, d in per.items()},
                       "total": tot, "refused": dict(ref)}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
