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

  worldmodel2 - (plan, plan step 2) the worldmodel sandbox helpers plus a REQUIRED checkpoint: after 35
              actions on a level without completion (every 20 after), the user prompt opens with a
              step/check_model/plan instruction. worldmodel alone: check_model in 2 of 100 passes. Source:
              wm2nudge.py.

  worldmodel3 - (FI-939 plan step 3) worldmodel2 whose checkpoint executes the plan with run_plan(step, actions):
              move by move inside one turn, stopping at the first board the model got wrong. Base hf-base:
              36% of turns execute one action (7% of actions), mean 4.9 actions per turn.

  automacro - (FI-939 plan step 3', DUCK 38ec) code-side: when a turn asks for ONE keyboard move and it changed the
              board, the HARNESS repeats it up to 3 more times while each repeat keeps changing the board (stops on
              no change, level/game end, a guard refusal). Clicks/RESET/batches are never repeated; the echo states
              exactly what ran. 38eb: prompt-level steering does not move this model. Source: automacro.py.

  trackrecord - (DUCK 38bv) every NO-OP verdict gets the control's own per-level record, e.g.
              "[UP on this level: 3 of 7 tries changed the board]". 38bu: inert-control statements do not predict
              later effects (O/E 1.09) and durable ones cut use of the control ~20%. Information only, no advice.
              Source: trackrecord.py (host-side solver patch; counts persist across snippets, one tally per pass).

  knobs6    - (DUCK 38cg) the switch set of the top public M2 fork (sujanmajhisuzan, 10-08), as one arm: six existing
              M2 switches that are off upstream turned on together - NOOP / DEATH / REPEAT_STATE guards, AUTO_FRAME_DIFF
              (budget 300), LEVEL_INVENTORY, GUARDS_FROM_LEVEL 2 -> 1. Some are read at call time and two are module
              constants read at import (tool_agent._AUTO_FRAME_DIFF*, solver.GUARDS_FROM_LEVEL), so the cell sets both the
              environment and those module attributes, then asserts every switch is live.

  effort-medium / effort-low - (DUCK 38cu) throughput arms: reasoning_effort=medium|low on every request (the Qwen3.8
              chat template's switch; M2 leaves it at the template default xhigh). Base loadsim: ~2k completion tokens per
              request, decode-bound, unfinished levels time-bound (38ci), so shorter thinking buys more turns per game.

  worldmodel - (DUCK 38cv) agent-written simulator + planning: grid / check_model / plan in the python sandbox (source:
              worldmodel.py) and system-prompt guidance: write step(board, action), check it against this level's real
              transitions, then breadth-first search the model for the shortest route to the goal and run it in one batch.

  A+B        (DUCK 38cw) combined arm: the named arms' cells joined into one cell in the given order, e.g.
              effort-low+worldmodel. Only for arms that each read positive on their own loadsim.

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


def automacro_source():
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "automacro.py"), encoding="utf-8").read()


def _automacro_cell():
    return (
        "# --- FI-939 arm automacro (fleet-original, plan step 3'): harness repeats an effective single move ---\n"
        "import inference.framework.solver as _am_sv\n"
        "_AM_SRC = " + repr(automacro_source()) + "\n"
        "_am_ns = {}\n"
        "exec(compile(_AM_SRC, '<automacro>', 'exec'), _am_ns)\n"
        "print(_am_ns['install'](_am_sv), flush=True)\n"
    )


def serverguard_source():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "serverguard.py"), encoding="utf-8") as f:
        return f.read()


def _serverguard_cell():
    # DUCK 38ed: robustness only. Relaunches a model server that exited (at startup or mid-run); no policy change.
    return (
        "# --- FI-939 serverguard (DUCK 38ed): relaunch a dead model server; robustness only, no policy change ---\n"
        "_SG_SRC = " + repr(serverguard_source()) + "\n"
        "_sg_ns = {}\n"
        "exec(compile(_SG_SRC, '<serverguard>', 'exec'), _sg_ns)\n"
        "# Commit-bypass runs (commit_bypass.py) never start the server: nothing to guard there.\n"
        "if globals().get('TRUE_SUBMISSION', True) is False:\n"
        "    print('[SERVERGUARD] skipped: commit run without a model server (TRUE_SUBMISSION=False)', flush=True)\n"
        "else:\n"
        "    # A guard must never be the thing that ends a run: any install error is printed and the run goes on.\n"
        "    try:\n"
        "        print(_sg_ns['install'](globals()), flush=True)\n"
        "    except Exception as _sg_e:\n"
        "        print('[SERVERGUARD] NOT installed: %r' % (_sg_e,), flush=True)\n"
    )


def loadguard_source():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "loadguard.py"), encoding="utf-8") as f:
        return f.read()


def _loadguard_cell():
    # DUCK 38ep: robustness only. Holds games until the model server is healthy (with keepalive RESETs) instead
    # of letting a slow load end the scored rerun at ~30 min with 0.00; no policy change.
    return (
        "# --- FI-939 loadguard (DUCK 38ep): survive a slow model load in the scored rerun; robustness only ---\n"
        "_LG_SRC = " + repr(loadguard_source()) + "\n"
        "_lg_ns = {}\n"
        "exec(compile(_LG_SRC, '<loadguard>', 'exec'), _lg_ns)\n"
        "# Commit-bypass runs (commit_bypass.py) never start the server: holding there would wait for nothing.\n"
        "if globals().get('TRUE_SUBMISSION', True) is False:\n"
        "    print('[LOADGUARD] skipped: not the scored rerun (TRUE_SUBMISSION=False)', flush=True)\n"
        "else:\n"
        "    # A guard must never be the thing that ends a run: any install error is printed and the run goes on.\n"
        "    try:\n"
        "        print(_lg_ns['install'](globals()), flush=True)\n"
        "    except Exception as _lg_e:\n"
        "        print('[LOADGUARD] NOT installed: %r' % (_lg_e,), flush=True)\n"
    )


def levelpin_source():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "levelpin.py"), encoding="utf-8") as f:
        return f.read()


def _levelpin_cell():
    # DUCK 38es: pin the agent's own reasoning/code from the turns that cleared each level in the system prompt
    # (Siriki M2 write-up: the only text addition that helped, +2.8). Harness-side: nothing for the model to call.
    return (
        "# --- FI-939 arm levelpin (DUCK 38es): solved-level evidence pinned in the system prompt ---\n"
        "import inference.agent.tool_agent as _lp_ta\n"
        "_lp_ns = {}\n"
        "exec(compile(" + repr(levelpin_source()) + ", '<levelpin>', 'exec'), _lp_ns)\n"
        "print(_lp_ns['install'](_lp_ta.ToolAgent), flush=True)\n"
    )


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

CELLS["compact"] = (
    "# --- FI-939 arm compact (DUCK 38cl): rolling history summaries (compaction), M2's own switches ---\n"
    "# 38ce field scan: OpenAI tripled its ARC-AGI-3 score with compaction in place of rolling truncation. M2 ships the\n"
    "# mechanism (harness patch: _maybe_append_rolling_summary) but leaves it off, so history is only truncated.\n"
    "# Rolling mode per the harness docstrings: a summary every 12k new tokens (well under the 58k drain, several\n"
    "# candidates per drop), the drain may stop early at a summary, and the request is told what just happened.\n"
    "# Worst case equals today's drain. Not the aggressive SUMMARY_REPLACES_HISTORY mode; thinking stays off.\n"
    "import os as _cp_os\n"
    "import inference.agent.tool_agent as _cp_ta\n"
    "for _a in ('_summary_interval_tokens', '_drain_stop_at_summaries', '_summary_turn_context', '_summary_replaces_history'):\n"
    "    if not callable(getattr(_cp_ta, _a, None)):\n"
    "        raise RuntimeError('[COMPACT] tool_agent.%s missing: harness version changed' % _a)\n"
    "_CP = {'ARC3_SUMMARY_INTERVAL_TOKENS': '12288', 'ARC3_DRAIN_STOP_AT_SUMMARIES': '1', 'ARC3_SUMMARY_TURN_CONTEXT': '1'}\n"
    "_cp_os.environ.update(_CP)\n"
    "_cp_live = {'interval': _cp_ta._summary_interval_tokens() == 12288, 'drain_stop': _cp_ta._drain_stop_at_summaries() is True,\n"
    "            'turn_context': _cp_ta._summary_turn_context() is True, 'replaces_off': _cp_ta._summary_replaces_history() is False}\n"
    "if not all(_cp_live.values()):\n"
    "    raise RuntimeError('[COMPACT] switches not live: %r' % _cp_live)\n"
    "print('[COMPACT] rolling summaries every 12288 tokens, drain stops at summaries (DUCK 38cl)', flush=True)\n"
)

# DUCK 38eu (2026-10-26): compaction at a gentler interval. 38cp measured the 12k interval at -37% levels overall
# but with large single-game gains (cd82 0->95, s5i5 8->42), and read the loss as summary calls costing turns
# faster than the shorter context saved them; it named ~32k as the follow-up. Identical cell, interval 32768.
CELLS["compact32"] = (CELLS["compact"]
    .replace("arm compact (DUCK 38cl)", "arm compact32 (DUCK 38eu; 38cl cell at a 32k interval)")
    .replace("a summary every 12k new tokens", "a summary every 32k new tokens")
    .replace("12288", "32768"))
assert CELLS["compact32"].count("32768") == 3 and "12288" not in CELLS["compact32"]

CELLS["pace"] = (
    "# --- FI-939 arm pace (DUCK 38cl): pace-aware GPU-slot priority, one M2 switch ---\n"
    "# 38ci: unfinished levels look time-bound. M2's priority scheduler is on (refresh, tail fade) but ARC3_PRIORITY_PACE\n"
    "# is '0'. With it on, a game whose levels have cost more tokens than the reference gets a lower slot priority\n"
    "# (multiplier clamped to 0.5..2, confidence grows with completed levels), so slots move to games clearing cheaply.\n"
    "import os as _pc_os\n"
    "import inference.agent.tool_agent as _pc_ta\n"
    "for _a in ('_pace_reference_tokens', '_get_env_bool'):\n"
    "    if not callable(getattr(_pc_ta, _a, None)):\n"
    "        raise RuntimeError('[PACE] tool_agent.%s missing: harness version changed' % _a)\n"
    "if not _pc_ta._get_env_bool('ARC3_PRIORITY_REFRESH_QUEUE', False):\n"
    "    raise RuntimeError('[PACE] ARC3_PRIORITY_REFRESH_QUEUE is off: pace would never be used')\n"
    "_pc_os.environ['ARC3_PRIORITY_PACE'] = '1'\n"
    "_pc_ref = float(_pc_ta._pace_reference_tokens(1))\n"
    "if not (_pc_ta._get_env_bool('ARC3_PRIORITY_PACE', False) and _pc_ref > 0):\n"
    "    raise RuntimeError('[PACE] switch not live or reference missing (ref=%r)' % _pc_ref)\n"
    "print('[PACE] ARC3_PRIORITY_PACE=1 live, L1 reference %.0f tokens (DUCK 38cl)' % _pc_ref, flush=True)\n"
)

def _effort_cell(level):
    """DUCK 38cu: a fixed reasoning_effort on every request (the Qwen3.8 chat template's own switch).

    The template defaults to xhigh ("think carefully ..., validate key assumptions, consider plausible alternatives");
    medium adds no instruction, low asks for brief thinking. M2 never sets it, so every turn runs at xhigh. Base
    loadsim: mean 1,986 completion tokens per request (median 1,250, p90 4,503), decode-bound at ~77 tok/s per stream,
    and unfinished levels are time-bound (38ci). Fewer thinking tokens per turn = more turns per game in the same wall
    clock. One variable; M2's own effort ladder (ARC3_REASONING_EFFORT_LADDER, off) still wins when it is set."""
    return (
        "# --- FI-939 arm effort-" + level + " (fleet-original, DUCK 38cu): reasoning_effort=" + level + " on every request ---\n"
        "# Throughput arm: the chat template defaults to xhigh; this sends " + level + " instead. One variable.\n"
        "import inference.agent.tool_agent as _ef_ta\n"
        "_EF_LEVEL = " + repr(level) + "\n"
        "_ef_cls = _ef_ta.ToolAgent\n"
        "if not callable(getattr(_ef_cls, '_harness_template_kwargs', None)):\n"
        "    raise RuntimeError('[EFFORT] ToolAgent._harness_template_kwargs missing: harness version changed')\n"
        "_ef_orig = _ef_cls._harness_template_kwargs\n"
        "def _ef_kwargs(self):\n"
        "    kw = _ef_orig(self)\n"
        "    kw.setdefault('reasoning_effort', _EF_LEVEL)\n"
        "    return kw\n"
        "# the served template must know the switch, or the run is a silent no-op: check the model dir's template\n"
        "import os as _ef_os\n"
        "_ef_t = _ef_os.path.join(str(globals().get('MODEL_DIR', '')), 'chat_template.jinja')\n"
        "if _ef_os.path.isfile(_ef_t):\n"
        "    _ef_src = open(_ef_t, encoding='utf-8').read()\n"
        "    if 'reasoning_effort' not in _ef_src or repr(_EF_LEVEL) not in _ef_src:\n"
        "        raise RuntimeError('[EFFORT] served chat template does not support reasoning_effort=%s' % _EF_LEVEL)\n"
        "    print('[EFFORT] served chat template supports reasoning_effort=%s' % _EF_LEVEL, flush=True)\n"
        "else:\n"
        "    print('[EFFORT] WARNING: chat template not found at %s; relying on the request log' % _ef_t, flush=True)\n"
        "_ef_cls._harness_template_kwargs = _ef_kwargs\n"
        "_ef_probe = _ef_cls.__new__(_ef_cls)\n"
        "if _ef_cls._harness_template_kwargs(_ef_probe).get('reasoning_effort') != _EF_LEVEL:\n"
        "    raise RuntimeError('[EFFORT] override not live')\n"
        "print('[EFFORT] reasoning_effort=%s on every request (DUCK 38cu)' % _EF_LEVEL, flush=True)\n"
    )


CELLS["effort-medium"] = None
CELLS["effort-low"] = None
WORLDMODEL_GUIDANCE = (
    "\n\nWorld model and planning (FI-939 worldmodel):\n"
    "- Inside `python` you also have `grid(frame=None)`, `check_model(step, max_report=3, edge=0)` and "
    "`plan(step, is_goal, actions=None, start=None, max_depth=60, max_nodes=20000)`. They read and compute only; "
    "none of them spends an action.\n"
    "- Once you have seen what the controls do on this level, write `step(board, action)` returning the next board "
    "(`board` is a list of row strings as from `grid()`, `action` an action string such as 'LEFT') as a top-level "
    "function so it is retained, then call `check_model(step)`. It replays this level's real transitions through your "
    "function and shows the first boards it gets wrong. Fix those before trusting the model; pass `edge=N` when only "
    "a HUD or timer strip along the border differs.\n"
    "- When the model reproduces the recent transitions, write `is_goal(board)` for your current goal hypothesis and "
    "call `plan(step, is_goal)`. It returns the shortest action list under your model; run it with "
    "`action(result['actions'])` in one batch and compare the real outcome with the prediction. A surprise means the "
    "model or the goal is wrong: correct it, re-check, re-plan. For click games pass the candidate clicks as "
    "`actions=[{'action': 'MOUSE', 'row': r, 'col': c}, ...]`.\n"
    "- Each level is scored by how few actions you use, so once the mechanics are known a planned route is worth more "
    "than exploring move by move. Do not build a model before you have observed the controls; update it when a level "
    "changes the rules."
)


def worldmodel_source():
    """`make_worldmodel` from worldmodel.py, verbatim (embedded into the sandbox bootstrap)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "worldmodel.py"), encoding="utf-8").read()
    i = src.index("def make_worldmodel(")
    return src[i:].rstrip() + "\n"


def _worldmodel_cell():
    return (
        "# --- FI-939 arm worldmodel (fleet-original, DUCK 38cv): model-written simulator + planning in the REPL ---\n"
        "# Adds grid/check_model/plan to the python sandbox and tells the model how to use them. One variable.\n"
        "import re as _wm_re, textwrap as _wm_tw\n"
        "import inference.agent.python_tool_sandbox as _wm_sb\n"
        "import inference.agent.tool_agent as _wm_ta\n"
        "_WM_SRC = " + repr(worldmodel_source()) + "\n"
        "_WM_GUIDE = " + repr(WORLDMODEL_GUIDANCE) + "\n"
        "_wm_boot = _wm_sb._SANDBOX_BOOTSTRAP\n"
        "_wm_hits = list(_wm_re.finditer(r'^( *)runtime_globals\\[\"action\"\\] = action\\n', _wm_boot, _wm_re.M))\n"
        "if len(_wm_hits) != 1:\n"
        "    raise RuntimeError('[WORLDMODEL] sandbox anchor found %d times: harness version changed' % len(_wm_hits))\n"
        "_wm_ind = _wm_hits[0].group(1)\n"
        "_wm_ins = _wm_tw.indent(_WM_SRC, _wm_ind) + _wm_ind + 'runtime_globals.update(make_worldmodel(runtime_globals))\\n'\n"
        "_wm_sb._SANDBOX_BOOTSTRAP = _wm_boot[:_wm_hits[0].end()] + _wm_ins + _wm_boot[_wm_hits[0].end():]\n"
        "compile(_wm_sb._SANDBOX_BOOTSTRAP, '<worldmodel-bootstrap>', 'exec')\n"
        "_wm_orig_build = _wm_ta._build_system_prompt\n"
        "def _wm_build(*a, **k):\n"
        "    return _wm_orig_build(*a, **k) + _WM_GUIDE\n"
        "_wm_ta._build_system_prompt = _wm_build\n"
        "print('[WORLDMODEL] sandbox grid/check_model/plan injected + guidance appended (DUCK 38cv)', flush=True)\n"
    )


def wm2nudge_source():
    """wm2nudge.py body from its first definition, verbatim (embedded into the worldmodel2 cell)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wm2nudge.py"), encoding="utf-8").read()
    i = src.index("WM2_STALL = ")
    return src[i:].rstrip() + "\n"


WM3_GUIDANCE = (
    "\n- `run_plan(step, actions, edge=0)` (worldmodel3) executes a planned action list move by move in ONE turn "
    "and stops at the first board your model got wrong, a level change or a refused action. Prefer it to "
    "`action(list)` whenever the moves come from your model: a wrong model then costs one action, not the batch."
)


def _wm2nudge_cell(use_run_plan=False):
    tag = "WM3" if use_run_plan else "WM2"
    extra = (
        "_wm3_orig_build = _wm2_ta._build_system_prompt\n"
        "_WM3_GUIDE = " + repr(WM3_GUIDANCE) + "\n"
        "def _wm3_build(*a, **k):\n"
        "    return _wm3_orig_build(*a, **k) + _WM3_GUIDE\n"
        "_wm2_ta._build_system_prompt = _wm3_build\n"
    ) if use_run_plan else ""
    return (
        "# --- FI-939 arm worldmodel2 (plan step 2, plan): enforced world-model checkpoint on stalls ---\n"
        "# On top of the worldmodel cell: after WM2_STALL actions on a level without completion (and every WM2_GAP\n"
        "# after), the user prompt opens with a required step/check_model/plan instruction. Source: wm2nudge.py.\n"
        "import inference.agent.tool_agent as _wm2_ta\n"
        "exec(compile(" + repr(wm2nudge_source()) + ", '<wm2nudge>', 'exec'), globals())\n"
        "_wm2_cls = getattr(_wm2_ta, 'ToolAgent', None)\n"
        "if _wm2_cls is None or not callable(getattr(_wm2_cls, '_build_user_prompt', None)):\n"
        "    raise RuntimeError('[WM2] ToolAgent._build_user_prompt not found: harness version changed')\n"
        "if getattr(_wm2_cls._build_user_prompt, '__wm2__', False):\n"
        "    raise RuntimeError('[WM2] checkpoint already installed')\n"
        "_wm2_cls._build_user_prompt = wm2_wrap(_wm2_cls._build_user_prompt, use_run_plan=" + repr(bool(use_run_plan)) + ")\n"
        + extra +
        "print('[" + tag + "] stall checkpoint installed (stall=%d gap=%d)' % (WM2_STALL, WM2_GAP), flush=True)\n"
    )


def triedhere_source():
    """triedhere.py body from its first definition, verbatim (embedded into the triedhere cell)."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "triedhere.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.index("TH_MARK = ")
    return src[i:].rstrip() + "\n"


def _triedhere_cell():
    # DUCK 38ef: information only - prepends a per-board "already tried from here" ledger when the board was seen
    # before on this level. Never acts, never blocks; novel boards get the unchanged prompt.
    return (
        "# --- FI-939 arm triedhere (DUCK 38ef): board memory - what each action already did from this exact board ---\n"
        "import inference.agent.tool_agent as _th_ta\n"
        "exec(compile(" + repr(triedhere_source()) + ", '<triedhere>', 'exec'), globals())\n"
        "_th_cls = getattr(_th_ta, 'ToolAgent', None)\n"
        "if _th_cls is None or not callable(getattr(_th_cls, '_build_user_prompt', None)):\n"
        "    raise RuntimeError('[TRIEDHERE] ToolAgent._build_user_prompt not found: harness version changed')\n"
        "if getattr(_th_cls._build_user_prompt, '__triedhere__', False):\n"
        "    raise RuntimeError('[TRIEDHERE] already installed')\n"
        "_th_cls._build_user_prompt = th_wrap(_th_cls._build_user_prompt)\n"
        "print('[TRIEDHERE] board-memory ledger installed (max %d action lines)' % TH_MAX_ACTIONS, flush=True)\n"
    )


def freshlook_source():
    """freshlook.py body from its first definition, verbatim (embedded into the freshlook cell)."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "freshlook.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.index("FL_MARK = ")
    return src[i:].rstrip() + "\n"


def _freshlook_cell():
    # DUCK 38ei: when a level stalls, empty the agent's conversation history before the turn is built, so the
    # model re-reads the board without the reasoning it has been defending. Non-reset turns are unchanged.
    return (
        "# --- FI-939 arm freshlook (DUCK 38ei): clear the conversation history when a level stalls ---\n"
        "import inference.agent.tool_agent as _fl_ta\n"
        "exec(compile(" + repr(freshlook_source()) + ", '<freshlook>', 'exec'), globals())\n"
        "_fl_cls = getattr(_fl_ta, 'ToolAgent', None)\n"
        "if _fl_cls is None or not callable(getattr(_fl_cls, '_build_user_prompt', None)):\n"
        "    raise RuntimeError('[FRESHLOOK] ToolAgent._build_user_prompt not found: harness version changed')\n"
        "if not callable(getattr(_fl_cls, '_note_history_evicted', None)):\n"
        "    raise RuntimeError('[FRESHLOOK] ToolAgent._note_history_evicted not found: harness version changed')\n"
        "if getattr(_fl_cls._build_user_prompt, '__freshlook__', False):\n"
        "    raise RuntimeError('[FRESHLOOK] already installed')\n"
        "_fl_cls._build_user_prompt = fl_wrap(_fl_cls._build_user_prompt)\n"
        "print('[FRESHLOOK] installed: first reset after %d actions on a level, then every %d, at most %d per level'\n"
        "      % (FL_FIRST, FL_EVERY, FL_MAX_PER_LEVEL), flush=True)\n"
    )


def pastnotes_source():
    """pastnotes.py body from its first import, verbatim (embedded into the pastnotes cell)."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pastnotes.py"), encoding="utf-8") as f:
        src = f.read()
    i = src.index("import hashlib as _pn_hashlib")
    return src[i:].rstrip() + "\n"


def _pastnotes_cell():
    # DUCK 38ek: keep the agent's own reasoning/code/output from every turn of a game, never evicted, in the REPL
    # variable `past_notes` (the PRO-LONG gap: M2 keeps the action history but drops old turns' own findings).
    return (
        "# --- FI-939 arm pastnotes (DUCK 38ek): a never-evicted, searchable log of the agent's own turns ---\n"
        "import inference.agent.tool_agent as _pn_ta\n"
        "import inference.agent.python_tool_sandbox as _pn_sb\n"
        "exec(compile(" + repr(pastnotes_source()) + ", '<pastnotes>', 'exec'), globals())\n"
        "_pn_cls = getattr(_pn_ta, 'ToolAgent', None)\n"
        "for _pn_m in ('_build_user_prompt', '_persistent_history_messages', '_retained_sources'):\n"
        "    if _pn_cls is None or not callable(getattr(_pn_cls, _pn_m, None)):\n"
        "        raise RuntimeError('[PASTNOTES] ToolAgent.%s not found: harness version changed' % _pn_m)\n"
        "    if getattr(getattr(_pn_cls, _pn_m), '__pastnotes__', False):\n"
        "        raise RuntimeError('[PASTNOTES] already installed')\n"
        "if not callable(getattr(_pn_ta, 'run_sandboxed_python', None)) or not isinstance(getattr(_pn_sb, '_SANDBOX_BOOTSTRAP', None), str):\n"
        "    raise RuntimeError('[PASTNOTES] sandbox entry points not found: harness version changed')\n"
        "_pn_sb._SANDBOX_BOOTSTRAP = pn_patch_bootstrap(_pn_sb._SANDBOX_BOOTSTRAP)\n"
        "_pn_ta.run_sandboxed_python = pn_wrap_sandbox(_pn_ta.run_sandboxed_python)\n"
        "_pn_cls._persistent_history_messages = pn_wrap_history(_pn_cls._persistent_history_messages)\n"
        "_pn_cls._build_user_prompt = pn_wrap_prompt(_pn_cls._build_user_prompt)\n"
        "_pn_cls._retained_sources = pn_wrap_sources(_pn_cls._retained_sources)\n"
        "print('[PASTNOTES] installed: up to %d notes per game, reminder every %d turns after the first eviction'\n"
        "      % (PN_MAX, PN_REMIND), flush=True)\n"
    )


CELLS["triedhere"] = None
CELLS["freshlook"] = None
CELLS["pastnotes"] = None
CELLS["worldmodel"] = None
CELLS["worldmodel2"] = None
CELLS["worldmodel3"] = None
CELLS["trackrecord"] = None
CELLS["automacro"] = None
CELLS["serverguard"] = None
CELLS["loadguard"] = None
CELLS["levelpin"] = None
CELLS["ctrlprobe"] = None  # built lazily from ctrlprobe.py so the cell always embeds the current source
CELLS["ctrlprobe2"] = None


def cell_for(arm):
    if "+" in arm:  # DUCK 38cw: combined arm, e.g. "effort-low+worldmodel" -> both cells, in order, as one cell
        parts = arm.split("+")
        if len(set(parts)) != len(parts) or any(not p for p in parts):
            raise ValueError("bad combined arm %r" % arm)
        return "\n".join(cell_for(p) for p in parts)  # each part's own markers and refusals are kept
    if arm not in CELLS:
        raise ValueError("unknown arm %s (have %s)" % (arm, ", ".join(sorted(CELLS))))
    if arm == "ctrlprobe":
        return _ctrlprobe_cell()
    if arm == "ctrlprobe2":
        return _ctrlprobe_cell(CTRLPROBE2_GUIDANCE, "ctrlprobe2")
    if arm == "trackrecord":
        return _trackrecord_cell()
    if arm == "automacro":
        return _automacro_cell()
    if arm == "serverguard":
        return _serverguard_cell()
    if arm == "loadguard":
        return _loadguard_cell()
    if arm == "levelpin":
        return _levelpin_cell()
    if arm == "triedhere":
        return _triedhere_cell()
    if arm == "freshlook":
        return _freshlook_cell()
    if arm == "pastnotes":
        return _pastnotes_cell()
    if arm == "worldmodel":
        return _worldmodel_cell()
    if arm == "worldmodel2":
        return _worldmodel_cell() + "\n" + _wm2nudge_cell()
    if arm == "worldmodel3":
        return _worldmodel_cell() + "\n" + _wm2nudge_cell(use_run_plan=True)
    if arm.startswith("effort-"):
        return _effort_cell(arm.split("-", 1)[1])
    return CELLS[arm]


def build(arm, src, root, name, bypass=False):
    for part in arm.split("+"):
        if part not in CELLS:
            raise ValueError("unknown arm %s (have %s)" % (part, ", ".join(sorted(CELLS))))
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
