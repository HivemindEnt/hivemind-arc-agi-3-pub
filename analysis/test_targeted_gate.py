#!/usr/bin/env python3
"""Tests for targeted_gate.py (stdlib only): python3 test_targeted_gate.py"""
import contextlib, io, json, os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import targeted_gate as G  # noqa: E402

FAILS = []


def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c:
        FAILS.append(n)


def arm(spec):
    """spec: {game: [levels per pass]} -> loadsim_power arm shape."""
    return {g: {p: float(v) for p, v in enumerate(vals)} for g, vals in spec.items()}


base = arm({"lf52": [1] * 15 + [2], "bp35": [1] * 16, "sk48": [0] * 16,
            "r11l": [6, 5, 6, 6] * 4, "sc25": [6, 5, 0, 6] * 4, "ar25": [5, 8, 8, 8] * 4})
good = arm({"lf52": [2] * 8 + [1] * 8, "bp35": [1] * 16, "sk48": [0] * 16,
            "r11l": [6, 5, 6, 6] * 4, "sc25": [6, 5, 0, 6] * 4, "ar25": [5, 8, 8, 8] * 4})
r = G.evaluate(base, good, ["lf52", "bp35", "sk48"], ["r11l", "sc25", "ar25"])
check("lf52 1->8 of 16 passes the mechanism", r["mechanism"][0]["pass"] and r["mechanism"][0]["arm_l2"] == 8)
check("unchanged games do not pass", not r["mechanism"][1]["pass"] and not r["mechanism"][2]["pass"])
check("no cost change -> cost pass", r["cost"]["pass"] and abs(r["cost"]["diff"]) < 1e-9)
check("verdict GATE_PASS", r["verdict"] == "GATE_PASS")

small = arm({"lf52": [2] * 4 + [1] * 12, "bp35": [1] * 16, "sk48": [0] * 16,
             "r11l": [6, 5, 6, 6] * 4, "sc25": [6, 5, 0, 6] * 4, "ar25": [5, 8, 8, 8] * 4})
check("gain of 3 (1->4) is not enough", not G.evaluate(base, small, ["lf52"], ["r11l"])["mechanism_pass"])
base_high = arm({"lf52": [2] * 2 + [1] * 14, "r11l": [6] * 16})
arm_high = arm({"lf52": [2] * 8 + [1] * 8, "r11l": [6] * 16})
check("base already clearing 2/16 blocks the pass (base-max 1)",
      not G.evaluate(base_high, arm_high, ["lf52"], ["r11l"])["mechanism_pass"])
check("base-max 2 lets it pass", G.evaluate(base_high, arm_high, ["lf52"], ["r11l"], base_max=2)["mechanism_pass"])

costly = arm({"lf52": [2] * 8 + [1] * 8, "bp35": [1] * 16, "sk48": [0] * 16,
              "r11l": [2, 1, 2, 2] * 4, "sc25": [0, 1, 0, 0] * 4, "ar25": [5, 8, 8, 8] * 4})
rc = G.evaluate(base, costly, ["lf52", "bp35", "sk48"], ["r11l", "sc25", "ar25"])
check("large drop on cost games fails cost", not rc["cost"]["pass"] and rc["cost"]["diff"] < -rc["cost"]["mde80"])
check("verdict GATE_FAIL when cost fails", rc["verdict"] == "GATE_FAIL")
check("missing game reported", G.evaluate(base, arm({"lf52": [1, 1]}), ["lf52"], ["r11l"]).get("error") == "missing")
check("render has verdict line", "GATE_PASS (mechanism PASS, cost PASS)" in G.render(r))

tmp = tempfile.mkdtemp()
try:
    def write(d, spec):
        os.makedirs(d, exist_ok=True)
        for g, vals in spec.items():
            for p, v in enumerate(vals):
                with open(os.path.join(d, "%s-0a1b2c3d_p%d_events.jsonl" % (g, p)), "w") as f:
                    for L in range(1, v + 1):
                        f.write(json.dumps({"type": "action", "action_name": "ACTION1", "score": L}) + "\n")
    B, A = os.path.join(tmp, "b"), os.path.join(tmp, "a")
    write(B, {"lf52": [1] * 16, "r11l": [3, 4] * 8})
    write(A, {"lf52": [2] * 6 + [1] * 10, "r11l": [3, 4] * 8})
    buf = io.StringIO()
    js = os.path.join(tmp, "g.json")
    with contextlib.redirect_stdout(buf):
        rc_ = G.main([B, A, "--targets", "lf52", "--cost", "r11l", "--json", js])
    check("cli rc 0 and GATE_PASS", rc_ == 0 and "GATE_PASS" in buf.getvalue())
    check("json verdict", json.load(open(js))["verdict"] == "GATE_PASS")
    with contextlib.redirect_stdout(io.StringIO()):
        check("cli missing rc 4", G.main([B, A, "--targets", "sk48", "--cost", "r11l"]) == 4)
        try:
            rcu = G.main([B, A, "--targets", "", "--cost", "r11l"])
        except SystemExit as e:
            rcu = e.code
        check("cli usage rc 2", rcu == 2)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
