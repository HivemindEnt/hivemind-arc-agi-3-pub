#!/usr/bin/env python3
"""FI-939: build a one-variable arm that inserts ONE fleet-written patch cell into an M2 notebook (DUCK 38ar).

Why: our own-work arms (2026-10-07 direction: centre on fleet-original experimentation) change harness behaviour
by monkeypatching `inference.agent.tool_agent` in a cell placed just before the benchmark cell, the same
mechanism the D' cell uses. This builder keeps that mechanical: the cell source lives in CELLS below, it is
inserted right before the cell that starts with BENCH_MARKER, and the build refuses if the marker is missing
or not unique, or if the cell is already present.

Arms:
  unanchor  - replace the level-start user prompt. M2's prompt (ARC3_LEVEL_TRANSFER_GUIDANCE=1) says
              "Start from the mechanics you established on the previous level; do not rediscover them
              without reason." 38ar measured 2.6x more appeals to the previous level per turn on failed L2+
              levels, and human play-test notes say level 2 often breaks a level-1 rule. The replacement
              asks for win attribution with two explanations, a once-per-control probe on the new board
              (including controls that were inert last level), and a goal hypothesis plus an alternative.

  ctrlprobe - (DUCK 38av, 2026-10-07) inject probe_controls() into the python sandbox: a bounded
              controls diagnostic (each key once, one click per object class, keys again after any click that
              changed something), plus system-prompt guidance on when to call it (L1 start, new levels whose
              controls may have changed, and when blocked). Source of the probe: ctrlprobe.py.

  ctrlprobe2 - (DUCK 38bi) the same probe (v2: stale-guard safe, 38bh) with ON-DEMAND guidance only: call it when
              blocked or before concluding a control is inert, never routinely at level start. 38bh: v1 helped
              lf52 L2 (3/4 vs 0/4) but cost 7.5% of levels overall, more than its ~2% action share explains; the
              level-start instruction is the likely channel. Run on a targeted multi-pass base (m2_loadsim
              --only-games) next to that base.

  inventory - (DUCK 38be) env-only: ARC3_LEVEL_INVENTORY=1, an existing M2 knob left off upstream. On the first turn of
              a new level it shows a START-vs-START object inventory against the previous level, with 'appeared'
              object types flagged as "likely new mechanics, investigate these first". 38bc found late control
              discovery only in the games that add a mechanic at level 2 (bp35/lf52/sk48); this is the cheapest
              targeted lever for exactly that. The knob is read at call time (tool_agent._level_inventory_enabled),
              so setting it in a cell before the benchmark is enough; the cell asserts that.

  trackrecord - (DUCK 38bv) every NO-OP verdict gets the control's own per-level record, e.g.
              "[UP on this level: 3 of 7 tries changed the board]". 38bu: inert-control statements do not predict
              later effects (O/E 1.09) and durable ones cut use of the control ~20%. Information only, no advice.
              Source: trackrecord.py (host-side solver patch; counts persist across snippets, one tally per pass).

  knobs6    - (DUCK 38cg) the switch set of the top public M2 fork (sujanmajhisuzan, 10-08), as one arm: six existing
              M2 switches that are off upstream turned on together - NOOP / DEATH / REPEAT_STATE guards, AUTO_FRAME_DIFF
              (budget 300), LEVEL_INVENTORY, GUARDS_FROM_LEVEL 2 -> 1. Some are read at call time and two are module
              constants read at import (tool_agent._AUTO_FRAME_DIFF*, solver.GUARDS_FROM_LEVEL), so the cell sets both the
              environment and those module attributes, then asserts every switch is live.

  --bypass   apply commit_bypass.py (DUCK 38cg): the scored rerun is unchanged; the commit run skips the model. Only for
              arms judged on the hidden score (no offline traces).

Usage: build_arm_cell.py ARM [--src ~/m2/kaggle-franzen-m2-loadsim] [--name unanchor-loadsim] [--root ~/m2]
"""
import argparse, json, os, sys

BENCH_MARKER = "print('Starting benchmark...')"

UNANCHOR_PROMPT = (
    "You have completed the previous level. `current_frame` now contains the starting "
    "board of the next level; any accompanying current-grid image shows this new board. "
    "Build a new plan for this layout rather than continuing the previous level's action "
    "sequence.\n\n"
    "Treat what you learned on the previous level as hypotheses, not facts. In these games "
    "the next level often breaks a rule of the previous one: the object you were chasing may "
    "now be a tool or a decoy, a control that did nothing may now move something, and a new "
    "element usually carries the mechanic this level needs.\n\n"
    "Before you plan:\n"
    "1. Win attribution: say what changed on the action that completed the previous level, and "
    "give at least two different explanations for why that completed it. Do not assume the "
    "object you were pursuing was the cause.\n"
    "2. Probe: on this board, try each control family once (each direction, the action key, a "
    "click on each new or changed object) and record what each did, including nothing. Re-test "
    "controls that were inert on the previous level.\n"
    "3. Goal: write your current goal hypothesis and one alternative, and name the observation "
    "that would tell them apart. Revise it when the evidence disagrees."
)

CTRLPROBE_GUIDANCE = (
    "\n\nControls diagnostic (FI-939 ctrlprobe):\n"
    "- `probe_controls(max_actions=16, clicks=True, combos=True, use_space=True, skip_hashes=())` is available "
    "inside `python`. It tries each key once, clicks one object of each kind, and after any click that changed "
    "something tries the keys again right after it (select-then-move), then prints an action -> effect table "
    "at object level (edge HUD and background ignored). Every probe action is a real, counted game action.\n"
    "- Call it (1) at the start of level 1, before settling on a goal; (2) at the start of a later level when "
    "new kinds of objects appear or the controls may have changed - pass `skip_hashes` for object kinds you "
    "already probed; (3) whenever you are blocked: several turns without progress, or about to conclude that a "
    "control does nothing.\n"
    "- Pass `use_space=False` once SPACE is known to commit, release or reset something in this game.\n"
    "- Treat the table as evidence, not a verdict: a control that did nothing once may work after a selection, "
    "on another object, or later in the level."
)


CTRLPROBE2_GUIDANCE = (
    "\n\nControls diagnostic (FI-939 ctrlprobe, on demand):\n"
    "- `probe_controls(max_actions=16, clicks=True, combos=True, use_space=True, skip_hashes=())` is available "
    "inside `python`. It tries each key once, clicks one object of each kind, and after any click that changed "
    "something tries the keys again right after it (select-then-move), then prints an action -> effect table "
    "at object level (edge HUD and background ignored). Every probe action is a real, counted game action.\n"
    "- Use it only when you are stuck: several turns on this level without progress, or before you conclude that "
    "a control does nothing. Do not call it routinely at the start of a level.\n"
    "- Pass `use_space=False` once SPACE is known to commit, release or reset something in this game.\n"
    "- Treat the table as evidence, not a verdict: a control that did nothing once may work after a selection, "
    "on another object, or later in the level."
)


def ctrlprobe_source():
    """`make_probe` from ctrlprobe.py, verbatim (the cell embeds it into the sandbox bootstrap)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctrlprobe.py"), encoding="utf-8").read()
    i = src.index("def make_probe(")
    return src[i:].rstrip() + "\n"


def _ctrlprobe_cell(guide=None, label="ctrlprobe"):
    guide = CTRLPROBE_GUIDANCE if guide is None else guide
    return (
        "# --- FI-939 arm " + label + " (fleet-original, DUCK 38av/38bi; 2026-10-07): controls diagnostic ---\n"
        "# Injects probe_controls() into the python sandbox and tells the model when to use it. One variable.\n"
        "import re as _cp_re, textwrap as _cp_tw\n"
        "import inference.agent.python_tool_sandbox as _cp_sb\n"
        "import inference.agent.tool_agent as _cp_ta\n"
        "_CP_SRC = " + repr(ctrlprobe_source()) + "\n"
        "_CP_GUIDE = " + repr(guide) + "\n"
        "_cp_boot = _cp_sb._SANDBOX_BOOTSTRAP\n"
        "_cp_hits = list(_cp_re.finditer(r'^( *)runtime_globals\\[\"action\"\\] = action\\n', _cp_boot, _cp_re.M))\n"
        "if len(_cp_hits) != 1:\n"
        "    raise RuntimeError('[CTRLPROBE] sandbox anchor found %d times: harness version changed' % len(_cp_hits))\n"
        "_cp_ind = _cp_hits[0].group(1)\n"
        "_cp_call = 'make_probe(runtime_globals, action, _stale_state)' if '_stale_state: list' in _cp_boot else 'make_probe(runtime_globals, action)'\n"
        "_cp_ins = _cp_tw.indent(_CP_SRC, _cp_ind) + _cp_ind + 'runtime_globals[\"probe_controls\"] = ' + _cp_call + '\\n'\n"
        "_cp_sb._SANDBOX_BOOTSTRAP = _cp_boot[:_cp_hits[0].end()] + _cp_ins + _cp_boot[_cp_hits[0].end():]\n"
        "compile(_cp_sb._SANDBOX_BOOTSTRAP, '<ctrlprobe-bootstrap>', 'exec')\n"
        "_cp_orig_build = _cp_ta._build_system_prompt\n"
        "def _cp_build(*a, **k):\n"
        "    return _cp_orig_build(*a, **k) + _CP_GUIDE\n"
        "_cp_ta._build_system_prompt = _cp_build\n"
        "print('[CTRLPROBE] sandbox probe_controls injected + guidance appended (DUCK 38av); v2 stale-clear=%s (38bh)' % ('_stale_state' in _cp_call), flush=True)\n"
    )


CELLS = {
    "unanchor": (
        "# --- FI-939 arm unanchor (fleet-original, DUCK 38ar): replace the level-start prompt ---\n"
        "# M2's level-start prompt tells the model to start from the previous level's mechanics and not\n"
        "# rediscover them. Failed L2+ levels lean on the previous level 2.6x more per turn (38ar), and\n"
        "# level 2 often breaks a level-1 rule (human play-tests 2026-10-07). One variable: this prompt.\n"
        "import os as _ua_os\n"
        "import inference.agent.tool_agent as _ua_ta\n"
        "import inference.agent.prompts as _ua_pr\n"
        "_UA_PROMPT = " + repr(UNANCHOR_PROMPT) + "\n"
        "if _ua_os.environ.get('ARC3_LEVEL_TRANSFER_GUIDANCE', '').strip().lower() not in ('1', 'true', 'yes', 'on'):\n"
        "    raise RuntimeError('[UNANCHOR] ARC3_LEVEL_TRANSFER_GUIDANCE is off: the level-start prompt would never be shown')\n"
        "if not isinstance(getattr(_ua_ta, 'LEVEL_START_USER_PROMPT', None), str):\n"
        "    raise RuntimeError('[UNANCHOR] tool_agent.LEVEL_START_USER_PROMPT missing: harness version changed')\n"
        "_ua_ta.LEVEL_START_USER_PROMPT = _UA_PROMPT\n"
        "_ua_pr.LEVEL_START_USER_PROMPT = _UA_PROMPT\n"
        "print('[UNANCHOR] level-start prompt replaced (DUCK 38ar), chars=%d' % len(_UA_PROMPT), flush=True)\n"
    ),
}


def src_of(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def insert_cell(nb, cell_src):
    code = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    hits = [i for i in code if src_of(nb["cells"][i]).startswith(BENCH_MARKER)]
    if len(hits) != 1:
        raise ValueError("benchmark marker found %d times (need exactly 1)" % len(hits))
    if any(src_of(nb["cells"][i]) == cell_src for i in code):
        raise ValueError("arm cell already present")
    new = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
           "source": cell_src.splitlines(keepends=True)}
    nb["cells"].insert(hits[0], new)
    return hits[0]


CELLS["inventory"] = (
    "# --- FI-939 arm inventory (DUCK 38be): ARC3_LEVEL_INVENTORY=1, nothing else ---\n"
    "# Existing M2 knob, off upstream: START-vs-START object inventory on a new level, 'appeared' types flagged as\n"
    "# likely new mechanics. 38bc: late control discovery is confined to games adding a mechanic at L2.\n"
    "import os as _inv_os\n"
    "import inference.agent.tool_agent as _inv_ta\n"
    "if not callable(getattr(_inv_ta, '_level_inventory_enabled', None)):\n"
    "    raise RuntimeError('[INVENTORY] tool_agent._level_inventory_enabled missing: harness version changed')\n"
    "_inv_os.environ['ARC3_LEVEL_INVENTORY'] = '1'\n"
    "if _inv_ta._level_inventory_enabled() is not True:\n"
    "    raise RuntimeError('[INVENTORY] knob did not take effect at call time')\n"
    "print('[INVENTORY] ARC3_LEVEL_INVENTORY=1 active (DUCK 38be)', flush=True)\n"
)

def trackrecord_source():
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trackrecord.py"), encoding="utf-8").read()


def _trackrecord_cell():
    return (
        "# --- FI-939 arm trackrecord (fleet-original, DUCK 38bv): per-level track record on NO-OP verdicts ---\n"
        "# One variable: a NO-OP verdict also states how often that control changed the board on this level.\n"
        "import inference.framework.solver as _tr_sv\n"
        "_TR_SRC = " + repr(trackrecord_source()) + "\n"
        "_tr_ns = {}\n"
        "exec(compile(_TR_SRC, '<trackrecord>', 'exec'), _tr_ns)\n"
        "print(_tr_ns['install'](_tr_sv), flush=True)\n"
    )


CELLS["knobs6"] = (
    "# --- FI-939 arm knobs6 (DUCK 38cg): the top public fork's six M2 switches, on together ---\n"
    "import os as _k6_os\n"
    "import inference.agent.tool_agent as _k6_ta\n"
    "import inference.framework.solver as _k6_sv\n"
    "_K6 = {'ARC3_NOOP_REPEAT_GUARD': '1', 'ARC3_DEATH_REPEAT_GUARD': '1', 'ARC3_REPEAT_STATE_GUARD': '1',\n"
    "       'ARC3_AUTO_FRAME_DIFF': '1', 'ARC3_AUTO_FRAME_DIFF_BUDGET': '300', 'ARC3_LEVEL_INVENTORY': '1',\n"
    "       'ARC3_GUARDS_FROM_LEVEL': '1'}\n"
    "for _k, _v in _K6.items():\n"
    "    _k6_os.environ[_k] = _v\n"
    "for _m, _a in ((_k6_ta, '_AUTO_FRAME_DIFF'), (_k6_ta, '_AUTO_FRAME_DIFF_BUDGET'), (_k6_sv, 'GUARDS_FROM_LEVEL')):\n"
    "    if not hasattr(_m, _a):\n"
    "        raise RuntimeError('[KNOBS6] %s.%s missing: harness version changed' % (_m.__name__, _a))\n"
    "_k6_ta._AUTO_FRAME_DIFF = True\n"
    "_k6_ta._AUTO_FRAME_DIFF_BUDGET = 300\n"
    "_k6_sv.GUARDS_FROM_LEVEL = 1\n"
    "_k6_live = {'repeat_state': _k6_ta._repeat_state_guard_enabled(), 'inventory': _k6_ta._level_inventory_enabled(),\n"
    "            'guards_level1': _k6_ta._guards_active(1) and _k6_sv.GUARDS_FROM_LEVEL == 1,\n"
    "            'noop_env': _k6_ta._get_env_bool('ARC3_NOOP_REPEAT_GUARD', False),\n"
    "            'death_env': _k6_ta._get_env_bool('ARC3_DEATH_REPEAT_GUARD', False),\n"
    "            'frame_diff': _k6_ta._AUTO_FRAME_DIFF is True}\n"
    "if not all(_k6_live.values()):\n"
    "    raise RuntimeError('[KNOBS6] not all switches live: %r' % _k6_live)\n"
    "print('[KNOBS6] six M2 switches live (DUCK 38cg): %s' % ', '.join(sorted(_k6_live)), flush=True)\n"
)

CELLS["trackrecord"] = None
CELLS["ctrlprobe"] = None  # built lazily from ctrlprobe.py so the cell always embeds the current source
CELLS["ctrlprobe2"] = None


def cell_for(arm):
    if arm not in CELLS:
        raise ValueError("unknown arm %s (have %s)" % (arm, ", ".join(sorted(CELLS))))
    if arm == "ctrlprobe":
        return _ctrlprobe_cell()
    if arm == "ctrlprobe2":
        return _ctrlprobe_cell(CTRLPROBE2_GUIDANCE, "ctrlprobe2")
    if arm == "trackrecord":
        return _trackrecord_cell()
    return CELLS[arm]


def build(arm, src, root, name, bypass=False):
    if arm not in CELLS:
        raise ValueError("unknown arm %s (have %s)" % (arm, ", ".join(sorted(CELLS))))
    meta = json.load(open(os.path.join(src, "kernel-metadata.json")))
    nb = json.load(open(os.path.join(src, meta["code_file"]), encoding="utf-8"))
    at = insert_cell(nb, cell_for(arm))
    if bypass:
        import commit_bypass
        commit_bypass.apply(nb)
    slug = "duck-franzen-m2-" + name
    out = os.path.join(root, "kaggle-franzen-" + name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, slug + ".ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    meta = dict(meta, id="hivemindadmin/" + slug, title=slug, code_file=slug + ".ipynb")
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
    return out, at


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arm")
    ap.add_argument("--src", default=os.path.expanduser("~/m2/kaggle-franzen-m2-loadsim"))
    ap.add_argument("--root", default=os.path.expanduser("~/m2"))
    ap.add_argument("--name")
    ap.add_argument("--bypass", action="store_true", help="commit-run bypass (leaderboard-only arms)")
    a = ap.parse_args(argv)
    out, at = build(a.arm, a.src, a.root, a.name or a.arm, a.bypass)
    print("built %s (cell inserted at index %d)" % (out, at))
    return 0


if __name__ == "__main__":
    sys.exit(main())
