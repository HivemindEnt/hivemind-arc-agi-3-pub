#!/usr/bin/env python3
"""FI-939 DUCK 38bz: on a new level, does the agent click the object types that just appeared? (premise check)

38bc found late control discovery only in games that add a mechanic at level 2 (bp35, lf52, sk48). The queued
tgt-inventory arm turns on M2's ARC3_LEVEL_INVENTORY, which flags object types that appeared since the previous
level's start as "likely new mechanics". Before spending GPU on it, check on recorded traces whether the agent
already goes to those objects. If it already clicks them early and often, the arm has little room.

From *_events.jsonl only:
- level start board: the board of the first row whose level == L (the completing action reports the new level and
  the new board; level 1 uses the initial row);
- objects: 4-connected same-colour components, excluding the background (most common colour) and components of
  one cell; type = (colour, shape normalised to its bounding box), position-free;
- appeared types on level L: types at L's start that were absent at L-1's start;
- every MOUSE(row=r, col=c) action on level L >= 2 is located on the board BEFORE it; it is an "appeared" click if
  the component there has an appeared type.
Null: the share of non-background cells covered by appeared-type objects on that same board, so a click landing at
random on some object would hit an appeared one at that rate. O/E = appeared clicks / sum of those shares.

Reported per game (levels >= 2 that have at least one appeared type and one click): clicks, appeared clicks, O/E,
and the median position of the first appeared click as a fraction of the level's actions (1.0 = never).

Usage: new_object_clicks.py DIR [DIR ...] [--games bp35,lf52,sk48] [--json OUT]
"""
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict

CLICK = re.compile(r"MOUSE\(row=(\d+),\s*col=(\d+)\)")


def components(board):
    """{(r, c): comp_id}, [(colour, shape, cells)] for non-background components of size >= 2."""
    h, w = len(board), len(board[0]) if board else 0
    cnt = Counter(v for row in board for v in row)
    bg = cnt.most_common(1)[0][0] if cnt else None
    seen, where, comps = set(), {}, []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or board[r][c] == bg:
                continue
            col, stack, cells = board[r][c], [(r, c)], []
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and board[ny][nx] == col:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            if len(cells) < 2:
                continue
            y0, x0 = min(y for y, _ in cells), min(x for _, x in cells)
            shape = frozenset((y - y0, x - x0) for y, x in cells)
            for cell in cells:
                where[cell] = len(comps)
            comps.append((col, shape, cells))
    return where, comps


def types_of(board):
    return {(col, shape) for col, shape, _ in components(board)[1]}


def load(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") not in ("initial", "action"):
                continue
            if not (isinstance(r.get("board"), list) and r["board"]):
                # older traces carry only board_ascii (one character per colour); same components either way
                asc = [ln for ln in str(r.get("board_ascii") or "").split("\n") if ln]
                if not asc:
                    continue
                r["board"] = [list(ln) for ln in asc]
            rows.append(r)
    return rows


def analyse(rows):
    """Per level >= 2 with appeared types: {'level', 'n_actions', 'clicks', 'hits', 'exp', 'first'}."""
    starts, order = {}, []
    for i, r in enumerate(rows):
        lv = 1 if r.get("type") == "initial" else r.get("level")
        if lv is not None and lv not in starts:
            starts[lv] = i
            order.append(lv)
    out = []
    for lv in order:
        if lv < 2 or (lv - 1) not in starts:
            continue
        appeared = types_of(rows[starts[lv]]["board"]) - types_of(rows[starts[lv - 1]]["board"])
        if not appeared:
            continue
        idx = [i for i in range(starts[lv] + 1, len(rows)) if rows[i].get("level") == lv
               and rows[i].get("type") == "action"]
        clicks, hits, exp, first = 0, 0, 0.0, None
        for k, i in enumerate(idx):
            m = CLICK.search(str(rows[i].get("action_display") or ""))
            if not m:
                continue
            before = rows[i - 1]["board"]
            where, comps = components(before)
            area = sum(len(c[2]) for c in comps)
            app_area = sum(len(c[2]) for c in comps if (c[0], c[1]) in appeared)
            if not area:
                continue
            clicks += 1
            exp += app_area / area
            ci = where.get((int(m.group(1)), int(m.group(2))))
            if ci is not None and (comps[ci][0], comps[ci][1]) in appeared:
                hits += 1
                if first is None:
                    first = k / max(1, len(idx))
        out.append({"level": lv, "n_actions": len(idx), "appeared_types": len(appeared), "clicks": clicks,
                    "hits": hits, "exp": exp, "first": first})
    return out


def summarize(rows, label):
    use = [r for r in rows if r["clicks"]]
    c = sum(r["clicks"] for r in use)
    h = sum(r["hits"] for r in use)
    e = sum(r["exp"] for r in use)
    firsts = sorted(r["first"] if r["first"] is not None else 1.0 for r in use)
    med = firsts[len(firsts) // 2] if firsts else None
    never = sum(1 for r in use if r["first"] is None)
    return ("%-6s levels=%d clicks=%d appeared-object clicks=%d expected=%.1f O/E=%s | first appeared click at "
            "median %s of the level's actions; never in %d/%d levels"
            % (label, len(use), c, h, e, ("%.2f" % (h / e)) if e else "-",
               ("%.2f" % med) if med is not None else "-", never, len(use)))


def run(paths, games=None):
    per = defaultdict(list)
    for p in paths:
        for f in sorted(glob.glob(os.path.join(p, "*_events.jsonl"))):
            g = os.path.basename(f).split("-")[0]
            if games and g not in games:
                continue
            for r in analyse(load(f)):
                r.update(game=g, file=os.path.basename(f))
                per[g].append(r)
    return per


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--games", default="")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    games = set(x for x in a.games.split(",") if x) or None
    per = run(a.paths, games)
    print("NEW_OBJECT_CLICKS from %s" % ", ".join(a.paths))
    allrows = [r for g in sorted(per) for r in per[g]]
    for g in sorted(per):
        print(summarize(per[g], g))
    print(summarize(allrows, "ALL"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(allrows, f, indent=1)
    return 0 if allrows else 3


if __name__ == "__main__":
    sys.exit(main())
