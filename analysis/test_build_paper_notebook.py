#!/usr/bin/env python3
"""Tests for build_paper_notebook.py: build the notebook, then EXECUTE its code cells (in order, plain Python) against a
synthetic trace root in a temp dir. Run: python3 test_build_paper_notebook.py (figure step needs matplotlib)."""
import json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_paper_notebook as B  # noqa: E402

FAILS = []


def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c:
        FAILS.append(n)


def write_pass(d, game, p, levels, fams=("ACTION1",), probe=False):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "%s-0a1b2c3d_p%d_events.jsonl" % (game, p)), "w") as f:
        if probe:
            t = ("[TOOL RESULT: python]\nprobe_controls: 1 actions\n  UP                           -> nothing\n"
                 "  DOWN                         -> REFUSED StaleStateActionError\n"
                 "  LEFT                         -> REFUSED StaleStateActionError\n")
            f.write(json.dumps({"type": "analysis", "transcript": t}) + "\n")
        for L in range(1, levels + 1):
            for a in fams:
                f.write(json.dumps({"type": "action", "action_name": a, "score": L - 1}) + "\n")
            f.write(json.dumps({"type": "action", "action_name": fams[0], "score": L, "level_completed": True}) + "\n")
        for a in fams * 6:
            f.write(json.dumps({"type": "action", "action_name": a, "score": levels}) + "\n")


nb = B.build()
codes = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
check("one cell per embedded module + 11 analysis cells", len(codes) == len(B.MODULES) + 11)
check("modules embedded verbatim", all(repr(open(os.path.join(HERE, m), encoding="utf-8").read()) in c
                                       for m, c in zip(B.MODULES, codes)))
check("no notebook magics", not any(line.lstrip().startswith(("%", "!")) for c in codes for line in c.split("\n")))
check("no em dash in notebook text", "—" not in json.dumps(nb, ensure_ascii=False))

tmp = tempfile.mkdtemp()
try:
    root = os.path.join(tmp, "traces")
    for label, d in B.ARMS:
        for p in range(4):
            for g, lv in (("lf52", 1 + (p % 2)), ("r11l", 3 + (p % 2)), ("sc25", 2)):
                write_pass(os.path.join(root, d), g, p, lv, fams=("ACTION1", "ACTION6"), probe=(label == "ctrlprobe"))
    for label, d in B.TARGETED:
        for p in range(16):
            for g in ("lf52", "bp35", "sk48", "r11l", "sc25", "ar25"):
                lv = 2 if (label == "ctrlprobe2" and g == "lf52" and p < 8) else 1
                write_pass(os.path.join(root, d), g, p, lv)
    work = os.path.join(tmp, "work")
    os.makedirs(work)
    script = os.path.join(work, "nb.py")
    open(script, "w", encoding="utf-8").write("\n\n".join(codes))
    env = dict(os.environ, FI939_TRACE_ROOT=root, FI939_OUT=work, FI939_ENV_DIR=os.path.join(tmp, "noenv"),
               MPLBACKEND="Agg")
    cp = subprocess.run([sys.executable, script], cwd=work, env=env, capture_output=True, text=True, timeout=600)
    out = cp.stdout
    check("notebook code runs to completion", cp.returncode == 0)
    if cp.returncode:
        print(cp.stderr[-2000:])
    check("all arms found", out.count("events files") == len(B.ARMS) + len(B.TARGETED))
    check("power table printed", "COMPARE ctrlprobe - base" in out and "MDE80 at 32" in out)
    check("score metric skipped without env files", "official-estimate metric skipped" in out)
    check("probe audit printed", "BLOCKED share" in out)
    check("T4 rows printed", "lf52" in out and "won" in out)
    check("gate PASS for synthetic ctrlprobe2", "== ctrlprobe2 vs tgt-base" in out and "GATE_PASS" in out)
    check("gate FAIL for unchanged inventory", "== inventory vs tgt-base" in out and "GATE_FAIL" in out)
    check("trackrecord gated too", "== trackrecord vs tgt-base" in out)
    check("perception section runs on base + D'", "## revision after a contradicted no-op statement (38bx)" in out
          and "REVISE statements=" in out and "announcements=" in out and "DURABLE" in out)
    try:
        import matplotlib  # noqa: F401
        check("figure 2 written", os.path.exists(os.path.join(work, "fig2_power.png")))
    except ImportError:
        print("skip figure check: no matplotlib")
    # missing arms are reported, not invented
    shutil.rmtree(os.path.join(root, B.TARGETED[1][1]))
    cp2 = subprocess.run([sys.executable, script], cwd=work, env=env, capture_output=True, text=True, timeout=600)
    check("missing arm reported and skipped", cp2.returncode == 0 and "MISSING (skipped)" in cp2.stdout
          and "targeted gate for ctrlprobe2 skipped" in cp2.stdout)
    # 2026-10-28 (38fb): HF screen tables. 25 synthetic games incl. the 14 held-out; one arm better on held-out,
    # one worse everywhere; environment_files present, so the official metric and the calibration run too.
    games = B.HELDOUT + ["cn04", "ft09", "lf52", "ls20", "m0r0", "s5i5", "sc25", "sp80", "tn36", "tr87", "wa30"]
    env = os.path.join(tmp, "env")
    for g in games:
        os.makedirs(os.path.join(env, g, "v1"), exist_ok=True)
        json.dump({"game_id": g + "-0a1b2c3d", "baseline_actions": [3, 3, 3, 3, 3]},
                  open(os.path.join(env, g, "v1", "metadata.json"), "w"))
    lv_of = {"hf-base": 2, "hf-base-r2": 2, "hf-harn-levelpin-r3": 3, "hf-harn-deep140": 1, "hf-harn-aggr1": 1,
             "hf-lora-g5": 2, "hf-harn-freshlook": 1}
    for d, lv in lv_of.items():
        for p in range(4):
            for g in games:
                write_pass(os.path.join(root, d, "artifacts"), g, p, lv if (g in B.HELDOUT or d.startswith("hf-base")
                                                                            or lv < 2) else 2)
    env3 = dict(os.environ, FI939_TRACE_ROOT=root, FI939_OUT=work, FI939_ENV_DIR=env, MPLBACKEND="Agg")
    cp3 = subprocess.run([sys.executable, script], cwd=work, env=env3, capture_output=True, text=True, timeout=900)
    o3 = cp3.stdout
    check("notebook runs with HF screens and env files", cp3.returncode == 0)
    if cp3.returncode:
        print(cp3.stderr[-2000:])
    check("screen gate: better arm PASS, worse arm FAIL",
          any(l.startswith("levelpin") and l.rstrip().endswith("PASS") for l in o3.splitlines())
          and any(l.startswith("deep140") and l.rstrip().endswith("FAIL") for l in o3.splitlines()))
    check("screen gate: arms without traces reported missing", "hf-harn-pastnotes" in o3 and "MISSING screen (skipped)" in o3)
    check("public-set table printed", "base, repeat" in o3)
    check("calibration printed with its as-of date and dropped zeros",
          "hidden draws as of 2026-10-28; failed 0.00 draws dropped: 3" in o3 and "SCREEN_CALIB vs-base" in o3)
    check("calibration map is the repo's screen_kernel_map.json", B.calib_map()["kernels"] ==
          json.load(open(os.path.join(HERE, "screen_kernel_map.json"), encoding="utf-8"))["kernels"])
    rc = B.main(["--out", os.path.join(tmp, "nb", "x.ipynb")])
    check("cli writes a valid notebook", rc == 0 and json.load(open(os.path.join(tmp, "nb", "x.ipynb")))["nbformat"] == 4)
    check("cli writes LF line endings on every OS (.ipynb is eol=lf)",
          b"\r\n" not in open(os.path.join(tmp, "nb", "x.ipynb"), "rb").read())
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
