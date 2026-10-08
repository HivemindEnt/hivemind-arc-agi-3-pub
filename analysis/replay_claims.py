#!/usr/bin/env python3
"""FI-939: test DUCK 38ap H1/H2 on recorded M2 traces - how much evidence stands behind the model's "this control
does nothing" claims, and how often such a claim is later contradicted on the same level.

Input: *_requests.jsonl files (one per game pass). Each request carries the (trimmed) conversation history, so the
union of all requests, in first-seen order, reconstructs the pass chronologically.
- level: from user/tool text "Current state: step N, level L".
- per-action verdicts: tool stdout lines "[action] NAME(...) -> effect" / "-> NO-OP".
- claims: assistant reasoning sentences that pair a control (UP/DOWN/LEFT/RIGHT, arrows, MOUSE/click, UNDO, SPACE)
  with a negative word (inert, no-op, does nothing, no effect, useless).

For the first claim about each control on each level it records: tries of that control on that level before the
claim and how many were no-ops, whether that control had already produced an effect (claim contradicts evidence),
whether it produces an effect LATER on the same level (claim falsified) and how many actions later, and whether the
sentence appeals to an earlier level ("as in level 1"), which is H2.

Usage: replay_claims.py FILE_OR_DIR [...] [--json OUT]
"""
import argparse, glob, hashlib, json, os, re, sys
from collections import defaultdict

DIRS = ("UP", "DOWN", "LEFT", "RIGHT")
STATE = re.compile(r"Current state: step (\d+), level (\d+)")
VERDICT = re.compile(r"\[action\] ([A-Z0-9_]+)(?:\([^)]*\))? -> (effect|NO-OP|GAME OVER|LEVEL COMPLETED|RUN COMPLETE)")
NEG = r"(inert|no-?ops?\b|does nothing|do nothing|did nothing|no effect|has no effect|useless|non-functional)"
CTRL = r"(UP|DOWN|LEFT|RIGHT|arrows?|arrow keys|directional(?: keys)?|MOUSE|clicks?|clicking|UNDO|SPACE(?:BAR)?|ACTION[1-7])"
CLAIM = re.compile(r"[^.\n]*\b" + CTRL + r"\b[^.\n]{0,60}?\b" + NEG + r"[^.\n]*", re.I)
EARLIER = re.compile(r"(as in|like in|same as in|carried from|from) (the )?(level \d|previous level|earlier level|last level|level one)", re.I)


def control_of(word):
    w = word.upper()
    if w.startswith("ARROW") or w.startswith("DIRECTIONAL"):
        return "DIRS"
    if w.startswith("CLICK") or w == "MOUSE" or w == "ACTION6":
        return "MOUSE"
    if w.startswith("SPACE") or w == "ACTION5":
        return "SPACE"
    if w == "ACTION7":
        return "UNDO"
    return {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT"}.get(w, w)


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")


def chronological(path):
    """Unique history messages of a pass in first-seen order: [(role, text, reasoning)]."""
    seen, out = set(), []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("event") not in (None, "request"):
                continue
            for m in r.get("messages", [])[1:]:
                rc = str(m.get("reasoning_content") or m.get("reasoning") or "")
                tx = text_of(m.get("content"))
                if m.get("role") == "assistant":
                    for tc in m.get("tool_calls") or []:
                        tx += "\n" + str((tc.get("function") or {}).get("arguments", ""))
                k = hashlib.sha1((m.get("role", "") + "\x00" + tx + "\x00" + rc).encode()).hexdigest()
                if k in seen:
                    continue
                seen.add(k)
                out.append((m.get("role"), tx, rc))
    return out


def analyse(msgs):
    level, n_action = 1, 0
    tries = defaultdict(lambda: defaultdict(lambda: [0, 0]))      # level -> control -> [tries, noops]
    effects = defaultdict(lambda: defaultdict(list))               # level -> control -> [action index of effect]
    claims = []
    seen_claim = set()
    for role, tx, rc in msgs:
        for m in STATE.finditer(tx):
            level = int(m.group(2))
        if role == "tool":
            for m in VERDICT.finditer(tx):
                n_action += 1
                c = control_of(m.group(1))
                t = tries[level][c]
                t[0] += 1
                if m.group(2) == "NO-OP":
                    t[1] += 1
                elif m.group(2) == "effect":
                    effects[level][c].append(n_action)
        elif role == "assistant" and rc:
            for m in CLAIM.finditer(rc):
                c = control_of(m.group(1))
                if (level, c) in seen_claim:
                    continue
                seen_claim.add((level, c))
                ctrls = DIRS if c == "DIRS" else (c,)
                before = sum(tries[level][x][0] for x in ctrls)
                noops = sum(tries[level][x][1] for x in ctrls)
                untried = [x for x in ctrls if tries[level][x][0] == 0]
                claims.append({"level": level, "control": c, "at_action": n_action, "tries_before": before,
                               "noops_before": noops, "untried": untried,
                               "effect_before": any(effects[level][x] for x in ctrls),
                               "earlier_level": bool(EARLIER.search(m.group(0))),
                               "sentence": " ".join(m.group(0).split())[:220], "_ctrls": ctrls})
    for cl in claims:
        later = sorted(a for x in cl.pop("_ctrls") for a in effects[cl["level"]][x] if a > cl["at_action"])
        cl["falsified_later"] = bool(later)
        cl["actions_until_effect"] = (later[0] - cl["at_action"]) if later else None
    return claims


EDGE = 4


def _interior(board_ascii):
    rows = str(board_ascii or "").split("\n")
    return tuple(r[EDGE:-EDGE] for r in rows[EDGE:-EDGE]) if len(rows) > 2 * EDGE else tuple(rows)


HUD_FREQ = 0.5


def events_timeline(events_file, hud_freq=HUD_FREQ):
    """Executed actions from *_events.jsonl: [(action_num, level, control, effect)].

    effect = some board cell changed that is NOT part of the level's always-changing background. DUCK 38ay:
    excluding only the 4 edge cells (still done first) left 73-100% of ALL actions (any control) looking effective, because timers,
    step bars and animations sit deeper than the edge. So the mask is learned per level: a cell that changes on
    more than `hud_freq` of that level's actions is treated as HUD/animation and ignored. DUCK 38au: the verdict
    lines analyse() counts miss up to 94% of executed actions; this is the ground-truth alternative."""
    acts, prev = [], None
    with open(events_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            board = _interior(r.get("board_ascii"))  # edge HUD strips first; the learned mask catches deeper timers
            if r.get("type") == "action" and r.get("action_num") is not None and r.get("level") is not None:
                name = str(r.get("action_name") or r.get("action_display") or "").split("(")[0]
                changed = set()
                if prev is not None:
                    for i, (x, y) in enumerate(zip(prev, board)):
                        if x != y:
                            changed.update((i, j) for j, (p_, q_) in enumerate(zip(x, y)) if p_ != q_)
                acts.append((int(r["action_num"]), int(r["level"]), control_of(name), changed))
            if r.get("type") in ("action", "initial"):
                prev = board
    freq = defaultdict(lambda: defaultdict(int))
    n_lv = defaultdict(int)
    for num, lv, c, ch in acts:
        n_lv[lv] += 1
        for cell in ch:
            freq[lv][cell] += 1
    noisy = {lv: {cell for cell, k in freq[lv].items() if n_lv[lv] >= 4 and k > hud_freq * n_lv[lv]}
             for lv in n_lv}  # once per level (38bu: was rebuilt per action, quadratic on long passes)
    return [(num, lv, c, bool(ch - noisy[lv])) for num, lv, c, ch in acts]


def analyse_events(msgs, timeline):
    """Same claims as analyse(), but tries/no-ops/effects come from the events timeline. The claim's position is the
    latest 'Current state: step S' seen before it; actions with action_num < S count as before the claim."""
    level, step = 1, 0
    claims, seen_claim = [], set()
    for role, tx, rc in msgs:
        for m in STATE.finditer(tx):
            step, level = int(m.group(1)), int(m.group(2))
        if role == "assistant" and rc:
            for m in CLAIM.finditer(rc):
                c = control_of(m.group(1))
                if (level, c) in seen_claim:
                    continue
                seen_claim.add((level, c))
                ctrls = DIRS if c == "DIRS" else (c,)
                mine = [t for t in timeline if t[1] == level and t[2] in ctrls]
                before = [t for t in mine if t[0] < step]
                after = sorted(t[0] for t in mine if t[0] >= step and t[3])
                claims.append({"level": level, "control": c, "at_action": step, "tries_before": len(before),
                               "noops_before": sum(1 for t in before if not t[3]),
                               "untried": [x for x in ctrls if not any(t[2] == x for t in before)],
                               "effect_before": any(t[3] for t in before),
                               "earlier_level": bool(EARLIER.search(m.group(0))),
                               "sentence": " ".join(m.group(0).split())[:220],
                               "falsified_later": bool(after),
                               "actions_until_effect": (after[0] - step + 1) if after else None,
                               "counted_from": "events"})
    return claims


def events_file_for(req_path):
    d, b = os.path.split(req_path)
    b = b.replace("_requests.jsonl", "_events.jsonl")
    for c in (os.path.join(d, b), os.path.join(d, "artifacts", b)):
        if os.path.exists(c):
            return c
    return None


def summarize(rows):
    n = len(rows)
    if not n:
        return "CLAIMS_NONE"
    thin = sum(1 for r in rows if r["tries_before"] <= 2)
    fals = [r for r in rows if r["falsified_later"]]
    contra = sum(1 for r in rows if r["effect_before"])
    earlier = [r for r in rows if r["earlier_level"]]
    l2 = [r for r in rows if r["level"] >= 2]
    gaps = sorted(r["actions_until_effect"] for r in fals)
    med = gaps[len(gaps) // 2] if gaps else None
    return ("CLAIMS n=%d | <=2 tries behind claim: %d (%.0f%%) | already contradicted at claim time: %d | falsified later "
            "on same level: %d (%.0f%%), median %s actions later | appeal to an earlier level: %d (of %d claims on L2+)"
            % (n, thin, 100.0 * thin / n, contra, len(fals), 100.0 * len(fals) / n, med, len(earlier), len(l2)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "*_requests.jsonl"))) if os.path.isdir(p) else [p]
    rows = []
    for f in files:
        game = os.path.basename(f).split("-")[0]
        ev = events_file_for(f)
        msgs = chronological(f)
        for c in (analyse_events(msgs, events_timeline(ev)) if ev else analyse(msgs)):
            c.update(game=game, file=os.path.basename(f))
            rows.append(c)
    for r in rows:
        print("%-5s L%d %-5s tries=%d noop=%d untried=%s contra=%s later=%s(%s) earlier=%s | %s" % (
            r["game"], r["level"], r["control"], r["tries_before"], r["noops_before"], ",".join(r["untried"]) or "-",
            int(r["effect_before"]), int(r["falsified_later"]), r["actions_until_effect"], int(r["earlier_level"]),
            r["sentence"][:120]))
    print(summarize(rows))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
