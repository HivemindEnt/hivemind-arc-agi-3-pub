"""FI-939 'automacro' arm: the harness, not the model, repeats an effective single move (plan step 3').

Why (DUCK 38eb): instructions in the prompt have not changed what this model does in play (optional tools, required
checkpoints, effort hints), and 68% of base passes end on the clock, not on a solve. A code-side lever that does not
depend on the model choosing to call anything: when a turn asks for ONE directional/keyboard action and that action
changed the board without ending the level or the game, the harness plays the same action again, up to
AUTOMACRO_MAX extra times, while each repeat keeps changing the board. That is the commonest human pattern on these
games (walk until something stops you) and each repeat saves a model turn. Base hf-base: 36% of turns execute one
action, so the arm acts on about a third of turns.

Stops at the first repeat that: changed nothing (that repeat is the one wasted action, and it ends the macro), ended
the level or the game, completed the run, was refused by a harness guard (death/repeat/stale guards run inside the
normal step_env, so every repeat is judged against the board it actually faces), or raised. Clicks (ACTION6),
RESET, multi-action batches and non-dict payloads are never repeated.

What the model sees: the normal payload of the LAST executed action, with executed_count / executed_actions covering
every move, and an action_echo line that says plainly what the harness did, e.g.
    [action] RIGHT -> effect | [auto-macro] harness repeated RIGHT 3 more times (stopped: no change)
so the model is never told it made one move when it made four.

Mechanism: `_HarnessGameSession.step_env` is wrapped. The wrapper calls the original step_env for the model's request
and, if the conditions hold, again with the same single action. Any failure in the wrapper returns the payload
already obtained, so the arm can never stop a game. install() refuses loudly if the harness shape has changed.
"""

AUTOMACRO_MAX = 3   # extra repeats per turn; small so a bad repeat costs at most a few actions
TAG = "fi939_automacro"
NEVER_REPEAT = ("ACTION6", "RESET")


def _single_action(arguments):
    """Return the one requested action spec, or None for a batch / query / unparseable request."""
    if not isinstance(arguments, dict) or str(arguments.get("query") or "").strip():
        return None
    acts = arguments.get("actions")
    if isinstance(acts, list):
        return acts[0] if len(acts) == 1 else None
    return arguments.get("action") if arguments.get("action") is not None else None


def _effective(p):
    return (isinstance(p, dict) and p.get("executed", True) and p.get("gameplay_changed") is True
            and int(p.get("executed_count", 1) or 1) == 1
            and not (p.get("level_completed") or p.get("game_over") or p.get("run_complete") or p.get("done")
                     or p.get("stop_reason") or p.get("error")))


def repeat_arguments(arguments):
    """The arguments for one repeat: same request, without per-turn markers (stale_after)."""
    rep = {k: v for k, v in arguments.items() if k != "stale_after"}
    return rep


def stop_label(p):
    if not isinstance(p, dict):
        return "error"
    if p.get("error") and not p.get("executed_count"):
        return "refused" if p.get("stop_reason") else "error"
    for k, lab in (("run_complete", "run complete"), ("game_over", "game over"),
                   ("level_completed", "level completed")):
        if p.get(k):
            return lab
    if p.get("stop_reason"):
        return str(p.get("stop_reason"))
    if p.get("gameplay_changed") is False:
        return "no change"
    return "limit"


def merge(first, payloads, label):
    """Payload the model reads: the last executed one, with counts/echo covering the whole macro."""
    executed = [p for p in payloads if isinstance(p, dict) and p.get("executed", True) and not (
        p.get("error") and not p.get("executed_count"))]
    last = dict(executed[-1]) if executed else dict(first)
    disp = str(first.get("action_display") or (first.get("executed_actions") or ["?"])[0])
    extra = len(executed) - 1
    acts = []
    for p in executed:
        acts.extend(p.get("executed_actions") or [str(p.get("action_display") or "")])
    last["executed_count"] = len(acts)
    last["executed_actions"] = acts
    last["requested_count"] = 1
    last["reward"] = sum(float(p.get("reward", 0.0) or 0.0) for p in executed)
    last["board_changed"] = any(bool(p.get("board_changed")) for p in executed)
    last[TAG] = {"extra": extra, "stopped": label}
    if extra > 0:
        note = " | [auto-macro] harness repeated %s %d more time%s (stopped: %s)" % (
            disp, extra, "" if extra == 1 else "s", label)
        base_echo = first.get("action_echo") or ("[action] %s -> effect" % disp)
        last["action_echo"] = base_echo + note
    elif first.get("action_echo"):
        last["action_echo"] = first.get("action_echo")
    return last


def install(sv, max_extra=AUTOMACRO_MAX):
    cls = getattr(sv, "_HarnessGameSession", None)
    if cls is None or not callable(getattr(cls, "step_env", None)):
        raise RuntimeError("[AUTOMACRO] solver._HarnessGameSession.step_env missing: harness version changed")
    if getattr(cls.step_env, "_fi939_automacro", False):
        raise RuntimeError("[AUTOMACRO] already installed")
    orig = cls.step_env

    def step_env(self, arguments, *a, **k):
        first = orig(self, arguments, *a, **k)
        try:
            spec = _single_action(arguments)
            if spec is None or not _effective(first):
                return first
            if str(first.get("action_name") or "") in NEVER_REPEAT:
                return first
            payloads = [first]
            label = "limit"
            rep = repeat_arguments(arguments)
            for _ in range(int(max_extra)):
                try:
                    if getattr(self, "should_stop", lambda: False)():
                        label = "stopped"
                        break
                    p = orig(self, rep, *a, **k)
                except Exception:  # noqa: BLE001 - a repeat failure ends the macro, never the game
                    label = "error"
                    break
                payloads.append(p)
                if not _effective(p):
                    label = stop_label(p)
                    break
            if len(payloads) == 1:
                return first
            st = self.__dict__.setdefault("_fi939_automacro_stats", {"macros": 0, "extra": 0})
            st["macros"] += 1
            st["extra"] += sum(1 for p in payloads[1:] if isinstance(p, dict) and not (
                p.get("error") and not p.get("executed_count")))
            return merge(first, payloads, label)
        except Exception:  # noqa: BLE001
            return first

    step_env._fi939_automacro = True
    cls.step_env = step_env
    return "[AUTOMACRO] effective single moves are repeated by the harness (max %d extra per turn)" % int(max_extra)
