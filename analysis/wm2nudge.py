"""wm2nudge: the stall checkpoint for the worldmodel2 arm (FI-939 plan step 2).

Why: the worldmodel arm (DUCK 38cv) gave the model grid/check_model/plan in the python sandbox, but only as an
option, and the model called check_model in 2 of 100 passes. The step-1 failure analysis of hf-base (100 passes)
found 85 passes active without a solve, only 0-7% of actions on the unfinished level without effect, a median of
35 actions per completed level and 78 on the unfinished one: the model explores productively but does not
converge. worldmodel2 keeps the same sandbox helpers and adds this checkpoint: once a level has taken
WM2_STALL actions without completion, and again every WM2_GAP actions after that, the user prompt opens with an
instruction to write step/is_goal, run check_model and execute plan() before any other action.

Pure helpers, no imports, so build_arm_cell.py can embed this file's body verbatim into a notebook cell.
The wrapper never raises: any error returns the original prompt unchanged.
"""

WM2_STALL = 35   # median actions per COMPLETED level in hf-base (step-1 analysis)
WM2_GAP = 20     # actions between repeat checkpoints on the same level
WM2_MARK = "World-model checkpoint (FI-939 wm2)"


def wm2_level_run(history_entries, current_level=None):
    """(level, start_step, n_actions) for the trailing run of history entries on the current level.

    Entries are objects with .action and .frame (.level, .step). The trailing run ends at the first entry
    (walking back) whose frame is on another level. Entries without an action (the initial frame) are not counted.
    """
    ents = list(history_entries or [])
    if not ents:
        return (current_level, None, 0)
    lvl = current_level
    if lvl is None:
        fr = getattr(ents[-1], "frame", None)
        lvl = getattr(fr, "level", None)
    n = 0
    start = None
    for e in reversed(ents):
        fr = getattr(e, "frame", None)
        if fr is None or getattr(fr, "level", None) != lvl:
            break
        start = getattr(fr, "step", None)
        if str(getattr(e, "action", "") or "").strip():
            n += 1
    return (lvl, start, n)


def wm2_should_fire(state, key, n, stall=WM2_STALL, gap=WM2_GAP):
    """Decide and record. state: dict mutated in place {'key', 'last', 'fires'}; key is the level.

    A new key, or a count below the last checkpoint (new game, reset history), starts the level over.
    """
    if not isinstance(state, dict):
        return False
    last = state.get("last")
    if state.get("key") != key or (last is not None and n < int(last)):
        state["key"] = key
        state["last"] = None
    if n < stall:
        return False
    last = state.get("last")
    if last is not None and n - int(last) < gap:
        return False
    state["last"] = n
    state["fires"] = int(state.get("fires", 0)) + 1
    return True


WM2_HEAD = (
    WM2_MARK + ": you have used %d actions on this level without completing it. Exploring one move at a time "
    "is not converging, so this turn starts with the world model, before any other action:\n"
    "1) In `python`, write or update a top-level `step(board, action)` from what the controls have done on this "
    "level and run `check_model(step)`. Fix the first mismatches it reports (pass `edge=N` if only a HUD or "
    "timer strip differs).\n"
    "2) Write `is_goal(board)` for your best current goal hypothesis and run `result = plan(step, is_goal)` "
    "(for click games pass `actions=[{'action': 'MOUSE', 'row': r, 'col': c}, ...]`).\n"
)
WM2_STEP3 = (
    "3) If a plan is found, execute it in one batch with `action(result['actions'])` and compare the real "
    "outcome with the prediction. If no plan is found or the outcome surprises you, use the mismatches to pick "
    "one short test (1-5 actions) that separates your goal hypotheses.\n"
)
# worldmodel3 (plan step 3): the plan runs move by move inside one turn and stops at the first surprise.
WM3_STEP3 = (
    "3) If a plan is found, execute it with `run_plan(step, result['actions'])`: it plays the plan move by move in "
    "this one turn and stops at the first board your model got wrong. If no plan is found or run_plan stops on a "
    "surprise, fix the model from the reported cells, or pick one short test (1-5 actions) that separates your goal "
    "hypotheses.\n"
)


def wm2_block(n, use_run_plan=False):
    return (WM2_HEAD % int(n)) + (WM3_STEP3 if use_run_plan else WM2_STEP3)


def wm2_wrap(orig, state_attr="_fi939_wm2", use_run_plan=False):
    """Wrap ToolAgent._build_user_prompt. Prepends wm2_block on a checkpoint; never raises."""

    def _wrapped(self, action_num, *args, **kwargs):
        prompt = orig(self, action_num, *args, **kwargs)
        try:
            hist = kwargs.get("history_entries")
            cur = kwargs.get("current_frame")
            lvl = getattr(cur, "level", None) if cur is not None else None
            level, start, n = wm2_level_run(hist, lvl)
            st = getattr(self, state_attr, None)
            if not isinstance(st, dict):
                st = {}
                setattr(self, state_attr, st)
            if wm2_should_fire(st, level, n):
                return wm2_block(n, use_run_plan) + "\n" + prompt
        except Exception:  # noqa: BLE001 - a checkpoint failure never breaks a turn
            pass
        return prompt

    _wrapped.__wm2__ = True
    return _wrapped
