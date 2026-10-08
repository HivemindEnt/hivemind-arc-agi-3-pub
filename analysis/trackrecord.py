"""FI-939 DUCK 38bv: 'trackrecord' arm - give every NO-OP verdict the control's own track record on this level.

Why (38bu, claim_null.py): the model's "control X does nothing" statements do not predict X's later behaviour
(observed/expected 1.09) and present-tense ones cut later use of X by about 20%. Most rest on one or two tries
(38ax). M2 already tells the model when an action changed nothing; it does not say whether that control has
worked elsewhere on the same level. This arm adds exactly that fact to the NO-OP verdict, and nothing else:

    [action] UP -> NO-OP (executed, no change inside the board area) [UP on this level: 3 of 7 tries changed the board]

Mechanism (host side, so the count survives across snippets and turns):
- `_HarnessGameSession._execute_action` is wrapped. After each executed, non-automatic, non-terminal action it
  bumps a per-session tally keyed by (level, engine action name) and attaches `fi939_track = (tries, changed)` to
  the payload. One session = one game pass, so counts never mix games or passes.
- `solver._action_verdict` is wrapped. It is the single verdict function behind both the per-call echo and the
  batch trace (looked up as a module global at call time), so wrapping the module attribute covers both. A NO-OP
  verdict whose item carries `fi939_track` gets the bracketed suffix. Every other verdict is returned unchanged.

The suffix states a count; it gives no advice. That keeps the arm one variable (information), not a prompt change.
Any failure in the tally is swallowed so the arm can never stop a game; install() itself refuses loudly if the
harness shape has changed.
"""

TAG = "fi939_track"
CLICK = "ACTION6"


def _label(item):
    disp = str(item.get("action_display") or item.get("action_name") or "?")
    if item.get("action_name") == CLICK:
        return "clicks"
    return disp.split("(")[0].strip() or "?"


def suffix(item):
    tr = item.get(TAG) if isinstance(item, dict) else None
    if not tr:
        return ""
    n, c = tr
    return " [%s on this level: %d of %d tries changed the board]" % (_label(item), c, n)


def install(sv):
    """Patch an imported inference.framework.solver module in place. Returns a short description."""
    cls = getattr(sv, "_HarnessGameSession", None)
    if cls is None or not callable(getattr(cls, "_execute_action", None)):
        raise RuntimeError("[TRACKRECORD] solver._HarnessGameSession._execute_action missing: harness version changed")
    if not callable(getattr(sv, "_action_verdict", None)):
        raise RuntimeError("[TRACKRECORD] solver._action_verdict missing: harness version changed")
    if getattr(cls._execute_action, "_fi939_track", False) or getattr(sv._action_verdict, "_fi939_track", False):
        raise RuntimeError("[TRACKRECORD] already installed")
    orig_exec = cls._execute_action
    orig_verdict = sv._action_verdict

    def _execute_action(self, *a, **k):
        payload = orig_exec(self, *a, **k)
        try:
            if (isinstance(payload, dict) and payload.get("executed") and not payload.get("automatic")
                    and payload.get("action_name") != "RESET"
                    and payload.get("gameplay_changed") in (True, False)
                    and not (payload.get("level_completed") or payload.get("game_over") or payload.get("done"))):
                tally = self.__dict__.setdefault("_fi939_tally", {})
                t = tally.setdefault((payload.get("level"), payload.get("action_name")), [0, 0])
                t[0] += 1
                t[1] += 1 if payload["gameplay_changed"] is True else 0
                payload[TAG] = (t[0], t[1])
        except Exception:  # the arm must never end a game
            pass
        return payload

    def _action_verdict(item):
        v = orig_verdict(item)
        try:
            if isinstance(v, str) and v.startswith("NO-OP"):
                return v + suffix(item)
        except Exception:
            pass
        return v

    _execute_action._fi939_track = True
    _action_verdict._fi939_track = True
    cls._execute_action = _execute_action
    sv._action_verdict = _action_verdict
    return "[TRACKRECORD] NO-OP verdicts carry the control's per-level track record (DUCK 38bv)"
