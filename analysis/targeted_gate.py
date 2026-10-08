#!/usr/bin/env python3
"""FI-939: evaluate the pre-registered 38bi gate for a targeted multi-pass arm against its same-shape base.

Why (DUCK 38bi): targeted arms (ctrlprobe2, inventory) run 6 games x 16 passes next to a 6 x 16 base. The reading
was written down before any data; this applies it mechanically, so the verdict cannot drift toward whatever the
numbers turn out to be.

  1. MECHANISM, per target game: count passes that cleared level 2 (levels >= 2) in base and arm. A target game
     PASSES when the arm clears L2 in at least `--min-gain` (default 4) more passes than the base AND the base
     cleared it in at most `--base-max` (default 1) passes. The arm passes MECHANISM if any target game passes.
  2. COST, over the cost games together: total levels (sum of per-game pass means) must not fall by more than the
     80%-power MDE of that difference (loadsim_power pooled variance over the two runs, their own pass counts).
     A drop larger than the MDE is FAIL; any smaller change is PASS (not "no effect", just not a detected cost).
  3. VERDICT: GATE_PASS (1 and 2) -> earns a 25 x 4 confirmation run before any hidden slot; otherwise GATE_FAIL.

Usage: targeted_gate.py BASE_DIR ARM_DIR [--targets lf52,bp35,sk48] [--cost r11l,sc25,ar25]
                        [--min-gain 4] [--base-max 1] [--json out.json]
Exit: 0 evaluated (verdict in output) | 2 usage | 4 a target or cost game missing from either run
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loadsim_power as P  # noqa: E402


def l2_passes(arm, game):
    return sum(1 for v in arm[game].values() if v >= 2)


def evaluate(base, arm, targets, cost, min_gain=4, base_max=1):
    missing = [g for g in list(targets) + list(cost) if g not in base or g not in arm]
    if missing:
        return {"error": "missing", "games": missing}
    mech = []
    for g in targets:
        b, a = l2_passes(base, g), l2_passes(arm, g)
        mech.append({"game": g, "base_l2": b, "arm_l2": a, "base_n": len(base[g]), "arm_n": len(arm[g]),
                     "pass": (a - b) >= min_gain and b <= base_max})
    s2 = P.pooled_var([base, arm], list(cost))
    nb = min(len(base[g]) for g in cost)
    na = min(len(arm[g]) for g in cost)
    tb = sum(P.mean(list(base[g].values())) for g in cost)
    ta = sum(P.mean(list(arm[g].values())) for g in cost)
    sd = math.sqrt(sum(s2.values()) * (1.0 / nb + 1.0 / na)) if nb and na else 0.0
    mde = P.K * sd
    diff = ta - tb
    cost_pass = not (diff < 0 and -diff > mde)
    mech_pass = any(m["pass"] for m in mech)
    return {"mechanism": mech, "mechanism_pass": mech_pass,
            "cost": {"games": list(cost), "base_total": tb, "arm_total": ta, "diff": diff, "mde80": mde,
                     "pass": cost_pass},
            "verdict": "GATE_PASS" if (mech_pass and cost_pass) else "GATE_FAIL"}


def render(r):
    lines = ["MECHANISM (L2 cleared in how many passes; pass = arm - base >= gain and base <= base-max):"]
    for m in r["mechanism"]:
        lines.append("  %-5s base %2d/%-2d  arm %2d/%-2d  %s" % (m["game"], m["base_l2"], m["base_n"], m["arm_l2"],
                                                            m["arm_n"], "PASS" if m["pass"] else "-"))
    c = r["cost"]
    lines.append("COST (%s total levels per pass-set): base %.2f arm %.2f diff %+.2f, MDE80 %.2f -> %s" % (
        ",".join(c["games"]), c["base_total"], c["arm_total"], c["diff"], c["mde80"], "PASS" if c["pass"] else "FAIL"))
    lines.append("%s (mechanism %s, cost %s)" % (r["verdict"], "PASS" if r["mechanism_pass"] else "FAIL",
                                                 "PASS" if c["pass"] else "FAIL"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("arm")
    ap.add_argument("--targets", default="lf52,bp35,sk48")
    ap.add_argument("--cost", default="r11l,sc25,ar25")
    ap.add_argument("--min-gain", type=int, default=4)
    ap.add_argument("--base-max", type=int, default=1)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    targets = [g for g in a.targets.split(",") if g]
    cost = [g for g in a.cost.split(",") if g]
    if not targets or not cost or a.min_gain < 1 or a.base_max < 0:
        print(__doc__); return 2
    r = evaluate(P.load_arm(a.base, "levels"), P.load_arm(a.arm, "levels"), targets, cost, a.min_gain, a.base_max)
    if r.get("error"):
        print("GATE_MISSING: %s" % ",".join(r["games"])); return 4
    print(render(r))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
