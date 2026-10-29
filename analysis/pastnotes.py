"""pastnotes: a never-evicted log of the agent's own turns, searchable from the Python tool (FI-939 arm, DUCK 38ek).

Why: PRO-LONG (arXiv 2607.20064, 2026-07-23) reports +15.7 to +21.0 points pass@1 on ARC-AGI-3 from a full
action/observation log the model searches with code instead of reading it in context, and the full log beat a
last-25-actions window. M2 already exposes the full action/frame history in the REPL (`history`, `transitions`), so
the gap is the model's OWN work: once M2 trims old turns from the conversation (150 assistant turns, token drain),
the reasoning, code and printed findings of those turns are gone. This arm keeps them, per game, in `past_notes`.

What changes for the model:
- In the Python tool, `past_notes` is a list of dicts, oldest first: {"turn", "level", "reasoning", "code",
  "output"}. It is always defined (empty until the first turn completes).
- After M2 first evicts history, the user prompt gets one short line saying the removed turns are in `past_notes`,
  repeated every PN_REMIND turns. Turns before the first eviction are byte-identical to the base apart from the
  extra REPL variable.

Mechanics (all wrappers never raise; on error they fall back to the base behaviour):
- ToolAgent._persistent_history_messages: every assistant message and its tool results in the request, the first
  time they are seen (fingerprint), are appended to self._fi939_pn["notes"], truncated, capped at PN_MAX notes.
- ToolAgent._build_user_prompt: records the current level for stamping, appends the reminder line when due.
- ToolAgent._retained_sources is evaluated as an argument right before run_sandboxed_python, in the same thread, so
  wrapping it records the calling agent in a thread-local; the run_sandboxed_python wrapper then adds that agent's
  notes to initial_state["past_notes"].
- The sandbox bootstrap is patched once so _refresh_state sets `past_notes` when the payload carries it and keeps a
  default empty list otherwise.

Pure helpers plus wrappers, imports only threading/hashlib/json, so build_arm_cell.py can embed this file.
"""
import hashlib as _pn_hashlib
import json as _pn_json
import threading as _pn_threading

PN_MARK = "Context note (FI-939 pastnotes)"
PN_MAX = 250          # notes kept per game (oldest dropped)
PN_REASONING = 1200   # chars kept per field
PN_CODE = 1500
PN_OUTPUT = 1200
PN_REMIND = 30        # turns between reminder lines once history has been evicted
PN_ANCHOR = '        runtime_globals["last_action_call_result"] = action_result\n'
PN_INJECT = ('        if "past_notes" in state_payload:\n'
             '            runtime_globals["past_notes"] = list(state_payload.get("past_notes") or [])\n'
             '        runtime_globals.setdefault("past_notes", [])\n')

_pn_local = _pn_threading.local()


def pn_clip(s, n):
    s = "" if s is None else (s if isinstance(s, str) else str(s))
    return s if len(s) <= n else s[:n] + " ...[%d chars cut]" % (len(s) - n)


def pn_code_of(tool_calls):
    """The `code` argument(s) of an assistant message's tool calls, joined; other tools render as name(args)."""
    out = []
    for tc in tool_calls or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = _pn_json.loads(args)
            except ValueError:
                args = {"raw": args}
        if isinstance(args, dict) and "code" in args:
            out.append(str(args["code"]))
        else:
            out.append("%s(%s)" % (fn.get("name", "?"), _pn_json.dumps(args, default=str)[:400]))
    return "\n".join(out)


def pn_fingerprint(msg):
    m = msg or {}
    key = "|".join(str(m.get(k, ""))[:400] for k in ("role", "tool_call_id", "content", "reasoning_content"))
    key += "|" + pn_code_of(m.get("tool_calls"))[:400]
    return _pn_hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()


def pn_text(content):
    if isinstance(content, list):   # multimodal parts: keep the text parts only
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


def pn_absorb(state, messages, level=None):
    """Append notes for assistant messages (and their following tool results) not seen before. Returns count added."""
    seen = state.setdefault("seen", set())
    notes = state.setdefault("notes", [])
    added, cur = 0, None
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            fp = pn_fingerprint(m)
            if fp in seen:
                cur = None
                continue
            seen.add(fp)
            state["turn"] = state.get("turn", 0) + 1
            cur = {"turn": state["turn"], "level": level,
                   "reasoning": pn_clip(m.get("reasoning_content") or m.get("reasoning") or "", PN_REASONING),
                   "code": pn_clip(pn_code_of(m.get("tool_calls")) or pn_text(m.get("content")), PN_CODE),
                   "output": ""}
            notes.append(cur)
            added += 1
        elif role == "tool" and cur is not None:
            fp = pn_fingerprint(m)
            if fp in seen:
                continue
            seen.add(fp)
            joined = (cur["output"] + "\n" if cur["output"] else "") + pn_text(m.get("content"))
            cur["output"] = pn_clip(joined, PN_OUTPUT)
        elif role == "user":
            cur = None
    if len(notes) > PN_MAX:
        del notes[:len(notes) - PN_MAX]
    return added


def pn_note():
    return ("%s: earlier turns were removed from this conversation, but your reasoning, code and printed output "
            "from every turn of this game are kept in the Python variable `past_notes` (list of dicts with keys "
            "turn, level, reasoning, code, output; oldest first). Search it with Python when an earlier finding "
            "would help, e.g. [n['turn'] for n in past_notes if 'goal' in n['reasoning']].\n\n" % PN_MARK)


def pn_state(agent):
    st = getattr(agent, "_fi939_pn", None)
    if st is None:
        st = {"notes": [], "seen": set(), "turn": 0, "level": None, "reminded_at": None, "reminders": 0,
              "absorbed": 0}
        setattr(agent, "_fi939_pn", st)
    return st


def pn_wrap_history(orig):
    def _wrapped(self, messages, *args, **kwargs):
        try:
            st = pn_state(self)
            st["absorbed"] += pn_absorb(st, list(messages or [])[1:], st.get("level"))
        except Exception:
            pass
        return orig(self, messages, *args, **kwargs)
    _wrapped.__pastnotes__ = True
    return _wrapped


def pn_wrap_prompt(orig):
    def _wrapped(self, action_num, *args, **kwargs):
        prompt = orig(self, action_num, *args, **kwargs)
        try:
            st = pn_state(self)
            frame = kwargs.get("current_frame")
            lvl = getattr(frame, "level", None)
            if lvl is not None:
                st["level"] = lvl
            if not getattr(self, "_has_evicted", False) or not st["notes"]:
                return prompt
            last = st["reminded_at"]
            if last is not None and st["turn"] - last < PN_REMIND:
                return prompt
            st["reminded_at"] = st["turn"]
            st["reminders"] += 1
            print("[PASTNOTES] reminder %d at turn %d (%d notes)" % (st["reminders"], st["turn"], len(st["notes"])),
                  flush=True)
            return pn_note() + prompt
        except Exception:
            return prompt
    _wrapped.__pastnotes__ = True
    return _wrapped


def pn_wrap_sources(orig):
    def _wrapped(self, *args, **kwargs):
        _pn_local.agent = self
        return orig(self, *args, **kwargs)
    _wrapped.__pastnotes__ = True
    return _wrapped


def pn_wrap_sandbox(orig):
    def _wrapped(*args, **kwargs):
        try:
            agent = getattr(_pn_local, "agent", None)
            _pn_local.agent = None
            init = kwargs.get("initial_state")
            if agent is not None and isinstance(init, dict):
                st = pn_state(agent)
                kwargs["initial_state"] = dict(init, past_notes=[dict(n) for n in st["notes"]])
        except Exception:
            pass
        return orig(*args, **kwargs)
    _wrapped.__pastnotes__ = True
    return _wrapped


def pn_patch_bootstrap(src):
    """Insert the past_notes refresh after the one anchor line; refuses (ValueError) unless it hits exactly once."""
    if PN_INJECT in src:
        raise ValueError("pastnotes bootstrap patch already applied")
    if src.count(PN_ANCHOR) != 1:
        raise ValueError("pastnotes anchor hit %d times" % src.count(PN_ANCHOR))
    return src.replace(PN_ANCHOR, PN_ANCHOR + PN_INJECT, 1)
