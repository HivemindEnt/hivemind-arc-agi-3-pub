"""levelpin: pin the agent's own evidence from each solved level in the system prompt (FI-939 arm, DUCK 38es).

Why: Lohit Siriki's Milestone-2 write-up (3rd place, Apache-2.0) reports that keeping "the rule the agent wrote for a
cleared level ... pinned in the prompt for the rest of the game" was the only text addition that helped (+2.8),
"because it returns evidence the agent would otherwise lose rather than adding advice". Our own screens agree on the
direction: removing history hurts most on later levels (freshlook, DUCK 38en: L2+ -17.7%), while tools the model must
choose to open go unused (pastnotes 4.0%, worldmodel 2/100 passes). So this arm offers nothing to call: the harness
itself keeps, for every level the agent clears, the reasoning and code of the turn(s) that cleared it, and shows them
in the system prompt for the rest of the game, where M2's history drain cannot remove them.

Mechanics (wrappers never raise; on any error they fall back to the base behaviour):
- ToolAgent._persistent_history_messages: new assistant turns (fingerprinted) are logged, stamped with the current
  level (only the last LP_KEEP_TURNS per level are kept).
- ToolAgent._build_user_prompt: reads the level from current_frame; when it rises from k to a higher value, level k
  is recorded as solved with its last LP_TURNS turns, and self._system_prompt is rebuilt as base + pinned block.
  The block changes only when a level is solved, so the server's prefix cache is reused between solves.
Pure helpers plus wrappers; imports only hashlib/json, so build_arm_cell.py can embed this file.
"""
import hashlib as _lp_hashlib
import json as _lp_json

LP_MARK = "SOLVED LEVELS IN THIS GAME (kept by the harness, FI-939 levelpin)"
LP_TURNS = 2            # final turns of a solved level that are pinned
LP_KEEP_TURNS = 4       # turns remembered per level while it is being played
LP_REASONING = 700      # tail chars of reasoning kept per pinned turn
LP_CODE = 500           # tail chars of code kept per pinned turn
LP_MAX_LEVELS = 4       # most recent solved levels shown
LP_MAX_CHARS = 5000     # hard cap on the pinned block


def lp_tail(s, n):
    s = "" if s is None else (s if isinstance(s, str) else str(s))
    return s if len(s) <= n else "...[%d chars cut] " % (len(s) - n) + s[-n:]


def lp_code_of(tool_calls):
    out = []
    for tc in tool_calls or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = _lp_json.loads(args)
            except ValueError:
                args = {"raw": args}
        if isinstance(args, dict) and "code" in args:
            out.append(str(args["code"]))
        else:
            out.append("%s(%s)" % (fn.get("name", "?"), _lp_json.dumps(args, default=str)[:300]))
    return "\n".join(out)


def lp_fp(m):
    key = "|".join(str(m.get(k, ""))[:400] for k in ("role", "content", "reasoning_content"))
    key += "|" + lp_code_of(m.get("tool_calls"))[:400]
    return _lp_hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()


def lp_state(agent):
    st = getattr(agent, "_fi939_lp", None)
    if st is None:
        st = {"seen": set(), "turns": {}, "level": None, "solved": [], "base_sys": None, "pins": 0, "turn": 0}
        setattr(agent, "_fi939_lp", st)
    return st


def lp_absorb(st, messages):
    """Log new assistant turns under the current level. Returns the number added."""
    added = 0
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        fp = lp_fp(m)
        if fp in st["seen"]:
            continue
        st["seen"].add(fp)
        st["turn"] += 1
        lvl = st["level"]
        rec = {"turn": st["turn"],
               "reasoning": lp_tail(m.get("reasoning_content") or m.get("reasoning") or "", LP_REASONING),
               "code": lp_tail(lp_code_of(m.get("tool_calls")) or str(m.get("content") or ""), LP_CODE)}
        bucket = st["turns"].setdefault(lvl, [])
        bucket.append(rec)
        del bucket[:-LP_KEEP_TURNS]
        added += 1
    return added


def lp_on_level(st, new_level):
    """Record solved levels when the level number rises. Returns the list of levels newly solved."""
    old = st["level"]
    st["level"] = new_level
    if old is None or new_level is None:
        return []
    try:
        if int(new_level) <= int(old):
            return []
    except (TypeError, ValueError):
        return []
    turns = list(st["turns"].get(old, []))[-LP_TURNS:]
    st["solved"].append({"level": old, "turn": st["turn"], "turns": turns})
    st["turns"].pop(old, None)
    return [old]


def lp_block(solved):
    if not solved:
        return ""
    parts = ["\n\n## %s\nYou already solved these levels. Below is your own reasoning and code from the turns that "
             "cleared each one, so their mechanics are not lost when old turns are trimmed:" % LP_MARK]
    for s in solved[-LP_MAX_LEVELS:]:
        parts.append("\n### Level %s (solved at turn %s)" % (s["level"], s["turn"]))
        for t in s["turns"]:
            if t["reasoning"]:
                parts.append("Reasoning: " + t["reasoning"])
            if t["code"]:
                parts.append("Code:\n" + t["code"])
    text = "\n".join(parts)
    return text if len(text) <= LP_MAX_CHARS else text[:LP_MAX_CHARS] + "\n...[pinned block capped]"


def lp_wrap_history(orig):
    def _wrapped(self, messages, *args, **kwargs):
        try:
            lp_absorb(lp_state(self), list(messages or [])[1:])
        except Exception:
            pass
        return orig(self, messages, *args, **kwargs)
    _wrapped.__levelpin__ = True
    return _wrapped


def lp_wrap_prompt(orig):
    def _wrapped(self, action_num, *args, **kwargs):
        prompt = orig(self, action_num, *args, **kwargs)
        try:
            st = lp_state(self)
            frame = kwargs.get("current_frame")
            lvl = getattr(frame, "level", None)
            if lvl is None:
                return prompt
            if st["base_sys"] is None:
                st["base_sys"] = getattr(self, "_system_prompt", None)
            new = lp_on_level(st, lvl)
            if new and isinstance(st["base_sys"], str):
                self._system_prompt = st["base_sys"] + lp_block(st["solved"])
                st["pins"] += 1
                print("[LEVELPIN] pinned level %s (%d solved so far, block %d chars)"
                      % (new[0], len(st["solved"]), len(self._system_prompt) - len(st["base_sys"])), flush=True)
        except Exception:
            pass
        return prompt
    _wrapped.__levelpin__ = True
    return _wrapped


def install(tool_agent_cls):
    """Patch ToolAgent; refuses to double-patch. Returns a status line."""
    for name, wrap in (("_persistent_history_messages", lp_wrap_history), ("_build_user_prompt", lp_wrap_prompt)):
        fn = getattr(tool_agent_cls, name, None)
        if fn is None:
            raise RuntimeError("[LEVELPIN] ToolAgent.%s missing: harness changed" % name)
        if getattr(fn, "__levelpin__", False):
            return "[LEVELPIN] already installed"
    if not hasattr(tool_agent_cls, "_system_prompt") and "_system_prompt" not in getattr(tool_agent_cls, "__init__").__code__.co_names:
        raise RuntimeError("[LEVELPIN] ToolAgent no longer sets _system_prompt: harness changed")
    tool_agent_cls._persistent_history_messages = lp_wrap_history(tool_agent_cls._persistent_history_messages)
    tool_agent_cls._build_user_prompt = lp_wrap_prompt(tool_agent_cls._build_user_prompt)
    return ("[LEVELPIN] installed: last %d turns of each solved level pinned in the system prompt "
            "(max %d levels, %d chars)" % (LP_TURNS, LP_MAX_LEVELS, LP_MAX_CHARS))
