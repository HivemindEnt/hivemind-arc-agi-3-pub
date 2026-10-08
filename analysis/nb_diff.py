#!/usr/bin/env python3
"""FI-939 DUCK 38cf: what does a public fork change relative to the M2 notebook? (static, never runs code)

The top public ARC-AGI-3 notebooks are mostly forks of dfranzen's Milestone-2 notebook. To turn their differences
into one-variable arms we first need the differences themselves, separated into:
  knobs    ARC3_* / environment settings (os.environ[...] = ..., os.environ.setdefault, export X=...) that differ
  cells    code cells added, removed or changed (matched by content similarity), with line counts
  diff     a unified diff of the changed code (optional, --diff)
Only notebook JSON is read. Nothing from the notebooks is imported or executed.

Usage: nb_diff.py BASE.ipynb OTHER.ipynb [--diff] [--max-lines 400]
"""
import argparse, difflib, json, re, sys

KNOB = re.compile(r"""(?:os\.environ\[\s*['"](?P<k1>[A-Z][A-Z0-9_]+)['"]\s*\]\s*=\s*(?P<v1>[^\n#]+)"""
                  r"""|os\.environ\.setdefault\(\s*['"](?P<k2>[A-Z][A-Z0-9_]+)['"]\s*,\s*(?P<v2>[^\n#)]+)\)"""
                  r"""|['"](?P<k3>ARC3_[A-Z0-9_]+)['"]\s*:\s*(?P<v3>[^,\n}]+)"""
                  r"""|^\s*(?:export\s+|%env\s+)(?P<k4>[A-Z][A-Z0-9_]+)\s*=\s*(?P<v4>\S+))""", re.M)


def code_cells(path):
    nb = json.load(open(path, encoding="utf-8"))
    out = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            src = c.get("source", "")
            out.append("".join(src) if isinstance(src, list) else str(src))
    return out


def knobs(cells):
    out = {}
    for src in cells:
        for m in KNOB.finditer(src):
            k = m.group("k1") or m.group("k2") or m.group("k3") or m.group("k4")
            v = (m.group("v1") or m.group("v2") or m.group("v3") or m.group("v4") or "").strip().strip(",")
            out[k] = v
    return out


def match_cells(a, b, threshold=0.6):
    """Greedy pairing by similarity. Returns (pairs [(i, j, ratio)], removed [i], added [j])."""
    pairs, used = [], set()
    for i, x in enumerate(a):
        best, bj = 0.0, None
        for j, y in enumerate(b):
            if j in used:
                continue
            r = 1.0 if x == y else difflib.SequenceMatcher(None, x[:20000], y[:20000], autojunk=False).quick_ratio()
            if r > best:
                best, bj = r, j
        if bj is not None and best >= threshold:
            pairs.append((i, bj, best))
            used.add(bj)
    removed = [i for i in range(len(a)) if i not in {p[0] for p in pairs}]
    added = [j for j in range(len(b)) if j not in used]
    return pairs, removed, added


def compare(a_cells, b_cells):
    pairs, removed, added = match_cells(a_cells, b_cells)
    changed = []
    for i, j, _ in pairs:
        if a_cells[i] != b_cells[j]:
            d = list(difflib.unified_diff(a_cells[i].splitlines(), b_cells[j].splitlines(), lineterm="", n=0))
            plus = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
            minus = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
            changed.append((i, j, plus, minus, d))
    ka, kb = knobs(a_cells), knobs(b_cells)
    knob_diff = {k: (ka.get(k), kb.get(k)) for k in sorted(set(ka) | set(kb)) if ka.get(k) != kb.get(k)}
    return {"pairs": pairs, "removed": removed, "added": added, "changed": changed, "knobs": knob_diff}


def render(r, a_cells, b_cells, show_diff=False, max_lines=400):
    lines = ["CELLS base=%d other=%d | identical=%d changed=%d removed=%d added=%d" % (
        len(a_cells), len(b_cells), sum(1 for i, j, _ in r["pairs"] if a_cells[i] == b_cells[j]), len(r["changed"]),
        len(r["removed"]), len(r["added"]))]
    for k, (va, vb) in r["knobs"].items():
        lines.append("KNOB %s: %s -> %s" % (k, va, vb))
    for i, j, plus, minus, _ in r["changed"]:
        first = b_cells[j].strip().splitlines()[0][:90] if b_cells[j].strip() else ""
        lines.append("CHANGED base#%d -> other#%d +%d -%d | %s" % (i, j, plus, minus, first))
    for j in r["added"]:
        first = b_cells[j].strip().splitlines()[0][:90] if b_cells[j].strip() else ""
        lines.append("ADDED other#%d (%d lines) | %s" % (j, len(b_cells[j].splitlines()), first))
    for i in r["removed"]:
        first = a_cells[i].strip().splitlines()[0][:90] if a_cells[i].strip() else ""
        lines.append("REMOVED base#%d (%d lines) | %s" % (i, len(a_cells[i].splitlines()), first))
    if show_diff:
        budget = max_lines
        for i, j, _, _, d in r["changed"]:
            for l in d:
                if budget <= 0:
                    lines.append("... (diff truncated)")
                    return "\n".join(lines)
                lines.append(l)
                budget -= 1
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--max-lines", type=int, default=400)
    a = ap.parse_args(argv)
    ac, bc = code_cells(a.base), code_cells(a.other)
    print(render(compare(ac, bc), ac, bc, a.diff, a.max_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
