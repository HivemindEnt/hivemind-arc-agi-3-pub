#!/usr/bin/env python3
"""Tests for probe_audit.py (stdlib only): python3 test_probe_audit.py"""
import io, json, os, shutil, sys, tempfile, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_audit as A  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


BLOCKED = ("[TOOL CALL: python]\n<tool_call>\n<function=python>\n<parameter=code>\nout = probe_controls(max_actions=30)\n"
           "</parameter>\n</function>\n</tool_call>\n\n[TOOL RESULT: python]\nprobe_controls: 1 actions\n"
           "  UP                           -> nothing\n"
           "  DOWN                         -> REFUSED StaleStateActionError\n"
           "  LEFT                         -> REFUSED StaleStateActionError\n\n")
CLEAN = ("[TOOL CALL: python]\n<tool_call>\n<function=python>\n<parameter=code>\nprobe_controls()\n"
         "</parameter>\n</function>\n</tool_call>\n\n[TOOL RESULT: python]\nprobe_controls: 3 actions (stopped: LEVEL COMPLETED)\n"
         "  UP                           -> b moved (-1,+0)\n"
         "  CLICK b#2(3,4)               -> nothing\n"
         "  then LEFT                    -> b moved (+0,-1) [LEVEL COMPLETED]\n")
PARTIAL = ("[TOOL RESULT: python]\nprobe_controls: 4 actions\n"
           "  UP                           -> b moved (-1,+0)\n"
           "  DOWN                         -> REFUSED KnownNoOpActionError\n")
SYSTEM = "System: `probe_controls(max_actions=16, ...)` is available inside python.\n"

t = A.parse_tables(BLOCKED + CLEAN)
check("parse two tables", len(t) == 2 and len(t[0][1]) == 3 and len(t[1][1]) == 3)
check("header with stop parsed", t[1][0].startswith("probe_controls: 3 actions"))
c = A.classify(*t[0])
check("blocked classified", c["kind"] == "BLOCKED" and c["used"] == 1 and c["refused"] == {"StaleStateActionError": 2})
c2 = A.classify(*t[1])
check("clean classified, changed rows counted", c2["kind"] == "CLEAN" and c2["changed"] == 2)
c3 = A.classify(*A.parse_tables(PARTIAL)[0])
check("partial classified", c3["kind"] == "PARTIAL" and c3["used"] == 4)
check("no table in system prompt text", A.parse_tables(SYSTEM) == [])

tmp = tempfile.mkdtemp()
try:
    def write(name, transcripts):
        with open(os.path.join(tmp, name), "w") as f:
            f.write(json.dumps({"type": "initial"}) + "\n")
            for tr in transcripts:
                f.write(json.dumps({"type": "analysis", "transcript": tr}) + "\n")
                f.write(json.dumps({"type": "action", "action_name": "ACTION1"}) + "\n")
    # history repeats: the blocked table appears in two consecutive transcripts, then the clean one is added
    write("lf52-0a1b2c3d_p0_events.jsonl", [SYSTEM + BLOCKED, SYSTEM + BLOCKED + CLEAN])
    write("bp35-0a1b2c3d_p0_events.jsonl", [SYSTEM])
    os.makedirs(os.path.join(tmp, "artifacts"))
    shutil.copy(os.path.join(tmp, "lf52-0a1b2c3d_p0_events.jsonl"), os.path.join(tmp, "artifacts"))
    per, n = A.audit(tmp)
    check("dedupe across history and artifacts copy", per["lf52"]["tables"] == 2 and per["lf52"]["calls"] == 2)
    check("blocked counted once", per["lf52"]["BLOCKED"] == 1 and per["lf52"]["CLEAN"] == 1)
    check("actions summed", per["lf52"]["actions"] == 4)
    check("game without probe present with zero", per["bp35"]["tables"] == 0)
    check("files counted once", n == 2)
    per2, _ = A.audit(tmp, {"bp35"})
    check("--games filter", "lf52" not in per2)
    buf = io.StringIO()
    out = os.path.join(tmp, "o.json")
    with contextlib.redirect_stdout(buf):
        rc = A.main([tmp, "--json", out])
    check("cli rc 0 and BLOCKED share line", rc == 0 and "BLOCKED share 50%" in buf.getvalue())
    check("json written", json.load(open(out))["total"]["tables"] == 2)
    with contextlib.redirect_stdout(io.StringIO()):
        check("usage rc 2", A.main([]) == 2)
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        check("no events rc 4", A.main([empty]) == 4)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
