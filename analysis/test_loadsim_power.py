#!/usr/bin/env python3
"""Tests for loadsim_power.py (stdlib only): python3 test_loadsim_power.py"""
import io, json, math, os, shutil, sys, tempfile, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loadsim_power as P  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def write_pass(d, game, p, levels, sub=""):
    """A trace whose score rises to `levels`, 5 actions per level, plus a game_over and a RESET."""
    dd = os.path.join(d, sub) if sub else d
    os.makedirs(dd, exist_ok=True)
    ev = [{"type": "action", "action_name": "RESET", "score": 0}]
    for L in range(1, levels + 1):
        for k in range(4):
            ev.append({"type": "action", "action_name": "ACTION1", "score": L - 1})
        ev.append({"type": "action", "action_name": "ACTION1", "score": L})
    ev.append({"type": "action", "action_name": "ACTION2", "score": levels, "game_over": True})
    with open(os.path.join(dd, "%s-0a1b2c3d_p%d_events.jsonl" % (game, p)), "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = P.main(argv)
    return rc, buf.getvalue()


tmp = tempfile.mkdtemp()
try:
    A, B, C = (os.path.join(tmp, x) for x in ("a", "b", "c"))
    # arm A: g1 passes 2,4,2,4 (var 4/3*... ), g2 constant 3
    for p, v in enumerate([2, 4, 2, 4]):
        write_pass(A, "g1", p, v)
    for p in range(4):
        write_pass(A, "g2", p, 3)
    # arm B: g1 3,5,3,5 ; g2 3,3,3,3 ; duplicate copy under artifacts/ must not double count
    for p, v in enumerate([3, 5, 3, 5]):
        write_pass(B, "g1", p, v)
        write_pass(B, "g1", p, v, sub="artifacts")
    for p in range(4):
        write_pass(B, "g2", p, 3)

    arm = P.load_arm(A, "levels")
    check("load_arm reads 2 games x 4 passes", sorted(arm) == ["g1", "g2"] and len(arm["g1"]) == 4)
    check("levels per pass", [arm["g1"][p] for p in range(4)] == [2, 4, 2, 4])
    check("l2plus metric", P.load_arm(A, "l2plus")["g1"][1] == 3.0)
    armb = P.load_arm(B, "levels")
    check("dedupe: artifacts copy not double counted", len(armb["g1"]) == 4)

    s2 = P.pooled_var([arm, armb], ["g1", "g2"])
    # each arm: deviations +-1 -> ss=4, df=3 per arm -> pooled 8/6
    check("pooled variance g1", abs(s2["g1"] - 8.0 / 6.0) < 1e-9)
    check("zero-variance game", s2["g2"] == 0.0)
    check("sd_diff formula", abs(P.sd_diff(s2, 4, 4) - math.sqrt(8.0 / 6.0 * 0.5)) < 1e-9)
    check("p two-sided z=1.96 ~ .05", abs(P.p_two_sided(1.959964) - 0.05) < 1e-4)
    check("passes_needed none for zero effect", P.passes_needed(s2, 0) is None)
    n1 = P.passes_needed(s2, 1.0)
    check("passes_needed formula", n1 == math.ceil(2 * P.K ** 2 * (8.0 / 6.0)))
    check("passes_needed shrinks with bigger effect", P.passes_needed(s2, 2.0) < n1)

    r = P.analyze([("base", A), ("arm", B)], "levels", (4, 16))
    check("totals", abs(r["totals"]["base"] - 6.0) < 1e-9 and abs(r["totals"]["arm"] - 7.0) < 1e-9)
    c = r["compare"][0]
    check("compare diff +1", abs(c["diff"] - 1.0) < 1e-9)
    check("z = diff/sd", abs(c["z"] - 1.0 / P.sd_diff(s2, 4, 4)) < 1e-9)
    check("MDE halves from 4 to 16 passes", abs(r["mde"][0]["mde80"] / r["mde"][1]["mde80"] - 2.0) < 1e-9)
    check("noisiest lists g1 first", r["noisiest"][0][0] == "g1")

    rc, out = run(["base=" + A, "arm=" + B, "--passes", "4,8"])
    check("cli rc 0", rc == 0)
    check("cli prints MDE and COMPARE", "MDE80 at  4" in out and "COMPARE arm - base" in out)
    js = os.path.join(tmp, "o.json")
    rc, _ = run(["base=" + A, "arm=" + B, "--json", js])
    check("cli json written", rc == 0 and json.load(open(js))["games"] == 2)

    rc, _ = run(["base=" + A])
    check("usage: one arm -> 2", rc == 2)
    rc, _ = run(["base=" + A, "arm=" + B, "--metric", "bogus"])
    check("usage: bad metric -> 2", rc == 2)
    rc, _ = run(["base=" + A, "arm=" + B, "--passes", "x"])
    check("usage: bad passes -> 2", rc == 2)
    os.makedirs(C)
    rc, out = run(["base=" + A, "arm=" + C])
    check("empty arm -> 4", rc == 4 and "NO_TRACES" in out)
    D = os.path.join(tmp, "d")
    write_pass(D, "g1", 0, 1)
    write_pass(D, "g2", 0, 1)
    rc, out = run(["base=" + A, "arm=" + D])
    check("single pass -> 3", rc == 3 and "TOO_FEW_PASSES" in out)

    # score metric with a fake baseline dir
    env = os.path.join(tmp, "env", "g1", "v1")
    os.makedirs(env)
    json.dump({"game_id": "g1-0a1b2c3d", "baseline_actions": [5, 5, 5, 5, 5]}, open(os.path.join(env, "metadata.json"), "w"))
    rc, out = run(["base=" + A, "arm=" + B, "--metric", "score", "--env-dir", os.path.join(tmp, "env")])
    check("score metric runs on games with baselines", rc == 0 and "games=1" in out)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
