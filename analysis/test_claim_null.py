#!/usr/bin/env python3
"""Tests for claim_null.py (DUCK 38bu). Run: python3 test_claim_null.py"""
import json, os, random, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claim_null as cn  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


# --- classifier: the 38ay hand-read cases ---
check("hypothetical", cn.classify("LEFT would be blocked here so it does nothing") == "HYPOTHETICAL")
check("situational wall", cn.classify("UP is blocked by the wall, no effect") == "SITUATIONAL")
check("situational nth try", cn.classify("the 5th RIGHT was a no-op") == "SITUATIONAL")
check("other level", cn.classify("in level 1, SPACE was a no-op too, as in level 1") == "OTHER_LEVEL")
check("general", cn.classify("clicking yellow with no selection = pure no-op") == "GENERAL")
check("general inert", cn.classify("SPACE does nothing in this game") == "GENERAL")
check("word boundary: 'nowhere' is not 'now'", cn.classify("UP does nothing, nowhere to go") == "GENERAL")


# --- null model on a synthetic pass ---
def msgs_at(step, level, sentence):
    return [("user", "Current state: step %d, level %d" % (step, level), ""), ("assistant", "", sentence)]


# Level 1: SPACE tried 10 times outside the window (5 effects -> base 0.5), then claim at step 20,
# then 4 tries in the window with 0 effects -> expected 2, observed 0, O/E 0.
tl = [(i, 1, "SPACE", i % 2 == 0) for i in range(10)]
tl += [(20 + i, 1, "SPACE", False) for i in range(4)]
rows = cn.claims_with_null(msgs_at(20, 1, "SPACE does nothing"), tl, window=10)
check("one row", len(rows) == 1)
r = rows[0]
check("base rate from outside window", abs(r["base_rate"] - 0.5) < 1e-9)
check("expected = rate x post tries", abs(r["expected"] - 2.0) < 1e-9 and r["post_tries"] == 4)
check("observed 0", r["post_effects"] == 0)
check("last try before claim (i=9 odd) had no effect", r["last_try_effect"] is False)
k = cn.oe(rows)
check("O/E 0", k["oe"] == 0.0 and k["n"] == 1)

# Uninformative claim: post effects at the base rate -> O/E 1
tl2 = [(i, 1, "UP", i % 2 == 0) for i in range(10)] + [(20, 1, "UP", True), (21, 1, "UP", False)]
r2 = cn.claims_with_null(msgs_at(20, 1, "UP does nothing"), tl2, window=10)
check("O/E 1 when claim predicts nothing", abs(cn.oe(r2)["oe"] - 1.0) < 1e-9)

# DIRS claim pools the four directions; other levels are ignored
tl3 = [(0, 1, "UP", True), (1, 1, "LEFT", True), (2, 2, "UP", False), (5, 1, "DOWN", False)]
r3 = cn.claims_with_null(msgs_at(5, 1, "the arrow keys do nothing"), tl3, window=3)
check("DIRS pooled, level filter", r3[0]["control"] == "DIRS" and r3[0]["tries_before"] == 2
      and r3[0]["post_tries"] == 1 and r3[0]["last_try_effect"] is True)

# Dedup per (level, control, class), different classes kept
m = msgs_at(5, 1, "UP does nothing. UP does nothing at all. UP would do nothing if blocked.")
check("dedup per class", sorted(x["cls"] for x in cn.claims_with_null(m, [], 5)) == ["GENERAL", "HYPOTHETICAL"])

# No tries in window -> excluded from O/E, still counted in n
r4 = cn.claims_with_null(msgs_at(50, 1, "SPACE does nothing"), tl, window=5)
check("no post tries -> not in O/E", cn.oe(r4)["n"] == 0 and cn.oe(r4)["oe"] is None)

# Bootstrap CI brackets the point estimate
many = []
for j in range(30):
    many += cn.claims_with_null(msgs_at(20, 1, "SPACE does nothing"),
                                [(i, 1, "SPACE", i % 2 == 0) for i in range(10)] +
                                [(20 + i, 1, "SPACE", (i + j) % 3 == 0) for i in range(4)], window=10)
k = cn.oe(many, random.Random(1), 300)
check("bootstrap CI brackets estimate", k["ci"][0] <= k["oe"] <= k["ci"][1])

# summarize prints every class
s = cn.summarize(rows)
check("summarize lists all classes", all(c in s for c in cn.CLASSES + ("ALL",)))

# --- GENERAL split (hand-read sample, 38bu) ---
check("report past tense", cn.subclass("Also the RIGHT press was a NO-OP (no change in the board area)") == "REPORT")
check("report arrow", cn.subclass("click (20,19) \u2192 no-op") == "REPORT")
check("durable present tense", cn.subclass("MOUSE is a no-op") == "DURABLE")
check("durable does nothing", cn.subclass("SPACE does nothing in this game") == "DURABLE")
check("other tally", cn.subclass("I had 4 no-ops (3 mouse no-ops + 1 RIGHT no-op)") == "OTHER")
check("non-GENERAL maps to itself", cn.subclass("LEFT would be blocked here so it does nothing") == "HYPOTHETICAL")

# --- usage ratio: control used 5 of 10 actions outside the window, 0 of 4 inside -> U 0 ---
tlu = [(i, 1, "SPACE" if i % 2 else "UP", True) for i in range(10)] + [(20 + i, 1, "UP", True) for i in range(4)]
ru = cn.claims_with_null(msgs_at(20, 1, "SPACE does nothing"), tlu, window=10)
check("use share 0.5", abs(ru[0]["use_share"] - 0.5) < 1e-9 and ru[0]["win_actions"] == 4)
check("expected tries 2", abs(ru[0]["expected_tries"] - 2.0) < 1e-9)
check("U 0 when avoided", cn.usage(ru)["u"] == 0.0)
check("rows carry sub", ru[0]["sub"] == "DURABLE")
check("summarize lists subclasses", all(c in cn.summarize(ru) for c in cn.SUBCLASSES))

# end-to-end on files: requests + events pair, main() rc 0 and JSON written
d = tempfile.mkdtemp()
req = os.path.join(d, "g1-abc_p0_requests.jsonl")
ev = os.path.join(d, "g1-abc_p0_events.jsonl")
blank = "\n".join("." * 12 for _ in range(12))
moved = blank[:13 * 6 + 6] + "X" + blank[13 * 6 + 7:]
with open(ev, "w") as f:
    f.write(json.dumps({"type": "initial", "board_ascii": blank}) + "\n")
    for i in range(6):
        f.write(json.dumps({"type": "action", "action_num": i, "level": 1, "action_name": "ACTION5",
                            "board_ascii": moved if i % 2 == 0 else blank}) + "\n")
with open(req, "w") as f:
    f.write(json.dumps({"messages": [{"role": "system", "content": "s"},
                                     {"role": "user", "content": "Current state: step 3, level 1"},
                                     {"role": "assistant", "content": "", "reasoning_content": "SPACE does nothing."}]}) + "\n")
out = os.path.join(d, "o.json")
rc = cn.main([d, "--json", out, "--boots", "0"])
check("main rc 0 + json", rc == 0 and json.load(open(out))[0]["control"] == "SPACE")
check("main rc 3 on empty dir", cn.main([tempfile.mkdtemp(), "--boots", "0"]) == 3)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
