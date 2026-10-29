#!/usr/bin/env python3
"""FI-939 tier-2 A/B metrics extractor.

Parses a tier-2 run directory (produced by tier2_ab_run.sh / inference-taaf-run inline) and
prints per-game behavioral signals for baseline-vs-patched comparison:
  analyses, actions, board-changes, game_over/resets, levels_completed, plus note-quality
  from the transcript (Progress-ledger present, unique Confirmed-mechanics lines).

Usage: ~/duck-venv/bin/python ab_metrics.py <run_dir> [run_dir2 ...]
A run_dir is ~/duck_ab/<name> (the parent; the timestamped subdir is found automatically).
"""
import sys, os, json, glob, re

def _bool(v): return bool(v) and v not in (0, "0", "false", "False", "")

def game_from_events(path):
    n_analysis = n_action = n_board = n_go = n_level = n_run = n_levels = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get("type") or e.get("event")
        if t == "analysis":
            n_analysis += 1
        elif t == "action":
            n_action += 1
            # result fields may be nested under 'result'/'last_action_result' or top-level
            r = e.get("result") or e.get("last_action_result") or e
            if _bool(r.get("board_changed")): n_board += 1
            if _bool(r.get("game_over")):      n_go += 1
            if _bool(r.get("level_completed")):n_level += 1
            if _bool(r.get("run_complete")):   n_run += 1
            # 2026-10-29: the action that clears a game's LAST level carries run_complete, not level_completed
            # (measured: 32/32 won games in hf-base). 'levels' counts it, matching the official scorer, which
            # counts score increments. levels_completed keeps its old meaning for existing readouts.
            if _bool(r.get("level_completed")) or _bool(r.get("run_complete")): n_levels += 1
    return dict(analyses=n_analysis, actions=n_action, board_changes=n_board,
                resets_game_over=n_go, levels_completed=n_level, run_complete=n_run, levels=n_levels)

def note_quality(transcript):
    if not transcript or not os.path.exists(transcript):
        return dict(progress_ledger=False, confirmed_unique=0)
    txt = open(transcript, encoding="utf-8", errors="replace").read()
    ledger = "Progress ledger" in txt
    # unique confirmed-mechanics bullet lines the model emitted (rough flip-flop/accumulation proxy)
    conf = set()
    for m in re.finditer(r"[Cc]onfirmed mechanics:\s*(.+)", txt):
        frag = m.group(1).strip()
        if frag and "PROVEN" not in frag and "each with" not in frag:  # skip the prompt template line
            conf.add(frag[:80].lower())
    return dict(progress_ledger=ledger, confirmed_unique=len(conf))

def resolve(run_dir):
    run_dir = os.path.expanduser(run_dir)
    subs = sorted(glob.glob(os.path.join(run_dir, "*/")))
    return subs[-1].rstrip("/") if subs else run_dir

def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    for rd in argv[1:]:
        d = resolve(rd)
        print(f"\n=== RUN: {d} ===")
        evs = sorted(glob.glob(os.path.join(d, "artifacts", "*events.jsonl")))
        if not evs:
            print("  (no events artifacts)"); continue
        for ev in evs:
            base = os.path.basename(ev).replace("_events.jsonl", "")
            g = game_from_events(ev)
            tr = glob.glob(os.path.join(d, "transcripts", base + "*.txt"))
            nq = note_quality(tr[0] if tr else None)
            print(f"  {base}: analyses={g['analyses']} actions={g['actions']} "
                  f"board_changes={g['board_changes']} resets(game_over)={g['resets_game_over']} "
                  f"levels={g['levels_completed']} | ledger={nq['progress_ledger']} "
                  f"confirmed_unique={nq['confirmed_unique']}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
