"""triedhere: the "already tried from this board" ledger for the triedhere arm (FI-939, DUCK 38ef).

Why: 38eb closed prompt-level steering and 38ee closed "the harness acts for the model". Both took a choice away
from the model. This arm only ADDS information. hf-base explores productively but does not converge (38dz:
median 35 actions per completed level, 78 on the unfinished one), and nothing in the M2 prompt tells the model
that the board in front of it is one it has already stood on, or what each action did from there. Each turn,
when the current board has been seen before on this level, the user prompt opens with a short ledger:
  - how many times this exact board has appeared on this level;
  - for every action already taken FROM this board: how often, and what happened (no change / changed the board /
    went back to a board seen before / reset to the level start / completed the level).
Boards seen for the first time get no block, so a novel state's prompt is byte-identical to the base.

Pure helpers, no imports, so build_arm_cell.py can embed this file verbatim into a notebook cell.
The wrapper never raises: any error returns the original prompt unchanged.
History contract (M2 solver.py): entries are (action display string, RESULTING frame); the first entry of a run
has action "" and the initial frame. Frames carry .grid (tuple of tuples), .level, .step.
"""

TH_MARK = "Board memory (FI-939 triedhere)"
TH_MAX_ACTIONS = 10     # at most this many action lines; the rest are summarised in one line


def th_level_frames(history_entries, level):
    """[(action, grid)] for the trailing run of entries on `level` (oldest first). Entry 0 of the run is the
    board the level started from (its action is whatever entered the level)."""
    run = []
    for e in reversed(list(history_entries or [])):
        fr = getattr(e, "frame", None)
        if fr is None or getattr(fr, "level", None) != level:
            break
        run.append((str(getattr(e, "action", "") or "").strip(), getattr(fr, "grid", None)))
    run.reverse()
    return run


def th_outcomes(run, current_grid):
    """(times_seen, {action: {outcome: count}}) for the current board, from the level's transitions."""
    seen = sum(1 for _a, g in run if g == current_grid)
    if not run:
        return (0, {})
    start = run[0][1]
    boards_before = set()
    out = {}
    for i in range(1, len(run)):
        prev = run[i - 1][1]
        boards_before.add(_key(prev))
        act, nxt = run[i]
        if prev != current_grid or not act:
            continue
        if nxt == prev:
            o = "no change"
        elif nxt == start and i > 1:
            o = "reset to level start"
        elif _key(nxt) in boards_before:
            o = "back to a board seen before"
        else:
            o = "changed the board"
        out.setdefault(act, {})
        out[act][o] = out[act].get(o, 0) + 1
    return (seen, out)


def _key(grid):
    try:
        return hash(grid)
    except TypeError:
        return id(grid)


def th_block(seen, outcomes, level_completed_actions=()):
    if seen < 2 and not outcomes:
        return ""
    lines = [TH_MARK + ": this exact board has appeared %d times on this level." % seen]
    if outcomes:
        lines.append("Actions already taken from this board, and what they did:")
        items = sorted(outcomes.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
        for act, res in items[:TH_MAX_ACTIONS]:
            n = sum(res.values())
            parts = ", ".join("%s x%d" % (k, v) for k, v in sorted(res.items(), key=lambda kv: (-kv[1], kv[0])))
            lines.append("- %s (%d): %s" % (act[:60], n, parts))
        if len(items) > TH_MAX_ACTIONS:
            lines.append("- ... %d more actions" % (len(items) - TH_MAX_ACTIONS))
    lines.append("This is information, not an instruction: choose your next action as you normally would.")
    return "\n".join(lines) + "\n"


def th_for_prompt(history_entries, current_frame):
    lvl = getattr(current_frame, "level", None) if current_frame is not None else None
    grid = getattr(current_frame, "grid", None) if current_frame is not None else None
    if grid is None:
        return ""
    run = th_level_frames(history_entries, lvl)
    seen, outcomes = th_outcomes(run, grid)
    return th_block(seen, outcomes)


def th_wrap(orig, stats_attr="_fi939_th"):
    """Wrap ToolAgent._build_user_prompt. Prepends the ledger when the board was seen before; never raises."""

    def _wrapped(self, action_num, *args, **kwargs):
        prompt = orig(self, action_num, *args, **kwargs)
        try:
            block = th_for_prompt(kwargs.get("history_entries"), kwargs.get("current_frame"))
            st = getattr(self, stats_attr, None)
            if not isinstance(st, dict):
                st = {"turns": 0, "fires": 0}
                setattr(self, stats_attr, st)
            st["turns"] += 1
            if block:
                st["fires"] += 1
                return block + "\n" + prompt
        except Exception:  # noqa: BLE001 - the ledger never breaks a turn
            pass
        return prompt

    _wrapped.__triedhere__ = True
    return _wrapped
