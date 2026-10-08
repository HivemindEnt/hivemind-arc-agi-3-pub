#!/usr/bin/env python3
"""Tests for replay_claims.py (FI-939, DUCK 38aq). Hermetic synthetic request traces. Exit 0 = pass."""
import json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay_claims as R  # noqa: E402

F, OK = [], 0


def check(n, c):
    global OK
    if c:
        OK += 1
    else:
        F.append(n); print("FAIL:", n)


def user(step, level):
    return {"role": "user", "content": [{"type": "text", "text": "Current state: step %d, level %d." % (step, level)}, {"type": "image_url"}]}


def tool(*lines):
    return {"role": "tool", "content": json.dumps({"stdout": "\n".join(lines)})}


def asst(reasoning):
    return {"role": "assistant", "content": "", "reasoning_content": reasoning, "tool_calls": []}


SYS = {"role": "system", "content": "sys"}
# Pass: L1 arrows no-op; L2: UP once no-op, then claim "arrows inert, as in level 1", later LEFT has an effect.
h = [SYS, user(1, 1), asst("Probe."), tool("[action] UP -> NO-OP (executed, no change inside the board area)"),
     asst("UP does nothing here."), user(5, 2), asst("New level."),
     tool("[action] MOUSE(row=1, col=2) -> effect", "[action] UP -> NO-OP (executed, no change inside the board area)"),
     asst("The arrows are inert, as in level 1. Clicks matter."),
     tool("[action] MOUSE(row=3, col=4) -> effect", "[action] MOUSE(row=3, col=5) -> effect",
          "[action] LEFT -> effect"),
     asst("UNDO is useless after all?")]
td = tempfile.mkdtemp()
p = os.path.join(td, "lf52-abc_p0_requests.jsonl")
with open(p, "w") as f:
    # two growing requests, as in real traces (history repeats; must be deduplicated)
    f.write(json.dumps({"event": "request", "messages": h[:6]}) + "\n")
    f.write(json.dumps({"event": "response", "messages": h[:2]}) + "\n")
    f.write(json.dumps({"event": "request", "messages": h}) + "\n")

msgs = R.chronological(p)
check("history deduplicated across growing requests", len(msgs) == len(h) - 1)
claims = R.analyse(msgs)
by = {(c["level"], c["control"]): c for c in claims}
check("L1 claim about UP found", (1, "UP") in by and by[(1, "UP")]["tries_before"] == 1 and by[(1, "UP")]["noops_before"] == 1)
d2 = by.get((2, "DIRS"))
check("L2 'arrows inert' claim normalised to DIRS", d2 is not None)
check("evidence behind L2 claim: 1 try, untried directions listed",
      d2["tries_before"] == 1 and d2["untried"] == ["DOWN", "LEFT", "RIGHT"])
check("L2 claim falsified later by LEFT effect, 3 actions after", d2["falsified_later"] and d2["actions_until_effect"] == 3)
check("appeal to an earlier level detected (H2)", d2["earlier_level"])
check("claim about a control never tried counts 0 tries", by[(2, "UNDO")]["tries_before"] == 0 and not by[(2, "UNDO")]["falsified_later"])
check("control_of maps ACTIONn and words", R.control_of("ACTION3") == "LEFT" and R.control_of("clicks") == "MOUSE"
      and R.control_of("arrow keys") == "DIRS" and R.control_of("SPACEBAR") == "SPACE")
s = R.summarize(claims)
check("summary line", s.startswith("CLAIMS n=3 | <=2 tries behind claim: 3 (100%)") and "falsified later on same level: 1" in s)
check("empty -> CLAIMS_NONE", R.summarize([]) == "CLAIMS_NONE")
check("cli runs on a dir", R.main([td, "--json", os.path.join(td, "o.json")]) == 0 and len(json.load(open(os.path.join(td, "o.json")))) == 3)

# --- events-based counting (DUCK 38au) ---
def board(mark_r, mark_c, hud):
    rows = [["." for _ in range(64)] for _ in range(64)]
    rows[mark_r][mark_c] = "X"
    for c in range(hud):
        rows[0][c] = "H"  # HUD strip at the top edge: must NOT count as an effect
    return "\n".join("".join(r) for r in rows)

ev_rows = [
    {"type": "initial", "level": 1, "action_num": 0, "board_ascii": board(30, 30, 40)},
    {"type": "action", "level": 1, "action_num": 1, "action_name": "ACTION1", "board_ascii": board(30, 30, 39)},
    {"type": "analysis", "level": 1, "action_num": 1},
    {"type": "action", "level": 1, "action_num": 2, "action_name": "ACTION1", "board_ascii": board(30, 30, 38)},
    {"type": "action", "level": 1, "action_num": 3, "action_name": "ACTION6", "action_display": "MOUSE(row=3, col=4)", "board_ascii": board(31, 30, 37)},
    {"type": "action", "level": 1, "action_num": 4, "action_name": "ACTION1", "board_ascii": board(30, 30, 36)},
]
ev_file = os.path.join(td, "evt-x_p0_events.jsonl")
with open(ev_file, "w") as f:
    for r in ev_rows:
        f.write(json.dumps(r) + "\n")
    f.write("garbage\n")
tl = R.events_timeline(ev_file)
check("events timeline: 4 actions, HUD-only change is not an effect",
      [(n, c, e) for n, lv, c, e in tl] == [(1, "UP", False), (2, "UP", False), (3, "MOUSE", True), (4, "UP", True)])
emsgs = [("user", "Current state: step 3, level 1.", ""), ("assistant", "", "So UP does nothing here."),
         ("user", "Current state: step 5, level 1.", ""), ("assistant", "", "UP does nothing, again.")]
ec = R.analyse_events(emsgs, tl)
check("events claim: tries/noops before the claim come from events", len(ec) == 1 and ec[0]["tries_before"] == 2
      and ec[0]["noops_before"] == 2 and not ec[0]["effect_before"] and ec[0]["counted_from"] == "events")
check("events claim: falsified later by UP effect at action 4", ec[0]["falsified_later"] and ec[0]["actions_until_effect"] == 2)
os.makedirs(os.path.join(td, "artifacts"), exist_ok=True)
open(os.path.join(td, "artifacts", "evt-y_p1_events.jsonl"), "w").close()
check("events_file_for finds beside and artifacts/, None when absent",
      R.events_file_for(os.path.join(td, "evt-x_p0_requests.jsonl")) == ev_file
      and R.events_file_for(os.path.join(td, "evt-y_p1_requests.jsonl")).endswith(os.path.join("artifacts", "evt-y_p1_events.jsonl"))
      and R.events_file_for(os.path.join(td, "zz-q_p9_requests.jsonl")) is None)

# --- learned per-level mask (DUCK 38ay): an interior timer that changes on every action is not an effect ---
def board2(mark_r, tick):
    rows = [["." for _ in range(64)] for _ in range(64)]
    rows[mark_r][30] = "X"
    rows[20][20] = "0123456789"[tick % 10]  # deep, always-changing timer cell
    return "\n".join("".join(r) for r in rows)

ev2 = os.path.join(td, "evt-z_p0_events.jsonl")
with open(ev2, "w") as f:
    f.write(json.dumps({"type": "initial", "level": 1, "action_num": 0, "board_ascii": board2(30, 0)}) + "\n")
    for i, (name, r) in enumerate([("ACTION1", 30), ("ACTION2", 30), ("ACTION3", 31), ("ACTION4", 31), ("ACTION1", 31)], 1):
        f.write(json.dumps({"type": "action", "level": 1, "action_num": i, "action_name": name, "board_ascii": board2(r, i)}) + "\n")
tl2 = R.events_timeline(ev2)
check("learned mask: deep timer ignored, only the real move (LEFT, action 3) is an effect",
      [(n, e) for n, lv, c, e in tl2] == [(1, False), (2, False), (3, True), (4, False), (5, False)])
check("mask needs >=4 actions on the level (no masking on tiny levels)",
      R.events_timeline(ev2, hud_freq=1.0)[0][3] is True)

print("replay_claims tests: %d passed, %d failed" % (OK, len(F)))
sys.exit(1 if F else 0)
