"""freshlook: clear the conversation history when a level stalls (FI-939 arm, DUCK 38ei).

Why: the failure that separates stuck runs from winning runs is an early goal judgement that the model then defends
for the rest of the level (Makarov's controlled study, Kaggle topic 743723: stuck runs had usually already tried the
decisive action; our own 38dz: hf-base explores but does not converge, median 35 actions per completed level vs 78
on the unfinished one; v11 0.34 -> 0.04: carried wrong beliefs locked the agent in). Every arm so far either added
text the model could ignore (worldmodel2, triedhere) or took choices away from it (automacro), and all failed. This
arm removes the defended context instead: after FL_FIRST actions on a level without completing it, and again every
FL_EVERY actions, the agent's conversation history is emptied before the turn is built, so the model re-reads the
current board without its own earlier reasoning. It is not summarisation (38cp: worse): nothing is carried over.

What the model sees on a reset turn: the normal user prompt for the current board, prefixed by one factual line
saying earlier turns on this level were removed. Nothing else changes; turns that are not reset turns are
byte-identical to the base.

Pure helpers plus one wrapper, no imports, so build_arm_cell.py can embed this file verbatim into a notebook cell.
The wrapper never raises: on any error it returns the base prompt and leaves the history untouched.
History contract (M2 solver.py): entries carry .frame with .level; the trailing run of entries on the current
level counts every action taken on it, across GAME_OVER resets of the same level.
"""

FL_MARK = "Context note (FI-939 freshlook)"
FL_FIRST = 50      # first reset after this many actions on one level without completing it
FL_EVERY = 50      # then again every this many actions
FL_MAX_PER_LEVEL = 4


def fl_actions_on_level(history_entries, level):
    """Number of actions taken on `level` so far: the trailing run of entries on that level, minus the entry the
    level started from."""
    n = 0
    for e in reversed(list(history_entries or [])):
        fr = getattr(e, "frame", None)
        if fr is None or getattr(fr, "level", None) != level:
            break
        n += 1
    return max(0, n - 1)


def fl_due(actions, done_on_level, first=FL_FIRST, every=FL_EVERY, cap=FL_MAX_PER_LEVEL):
    """True when a reset is due: the number of resets the action count calls for exceeds the resets already done."""
    if actions < first or done_on_level >= cap:
        return False
    wanted = 1 + (actions - first) // every
    return min(wanted, cap) > done_on_level


def fl_note(actions):
    return ("%s: your earlier turns on this level (%d actions, level not yet completed) were removed from this "
            "conversation. The board below is the current state.\n\n" % (FL_MARK, actions))


def fl_wrap(orig, stats_attr="_fi939_fl"):
    """Wrap ToolAgent._build_user_prompt: on a due turn, empty self._history_messages first, then prefix the note."""
    def _wrapped(self, action_num, *args, **kwargs):
        prompt = orig(self, action_num, *args, **kwargs)
        try:
            frame = kwargs.get("current_frame")
            level = getattr(frame, "level", None)
            if level is None:
                return prompt
            st = getattr(self, stats_attr, None)
            if st is None:
                st = {"level": level, "done": 0, "resets": 0, "turns": 0}
                setattr(self, stats_attr, st)
            st["turns"] += 1
            if st["level"] != level:
                st["level"], st["done"] = level, 0
            actions = fl_actions_on_level(kwargs.get("history_entries"), level)
            if not fl_due(actions, st["done"]):
                return prompt
            if not getattr(self, "_history_messages", None):
                st["done"] += 1          # nothing to clear (already empty): count it, add no note
                return prompt
            self._history_messages = []
            note = getattr(self, "_note_history_evicted", None)
            if callable(note):
                note()
            st["done"] += 1
            st["resets"] += 1
            print("[FRESHLOOK] level %s: history cleared after %d actions (reset %d on this level)"
                  % (level, actions, st["done"]), flush=True)
            return fl_note(actions) + prompt
        except Exception:
            return prompt
    _wrapped.__freshlook__ = True
    return _wrapped
