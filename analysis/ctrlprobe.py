"""ctrlprobe: a bounded controls diagnostic the model can call inside the M2 python sandbox (FI-939 DUCK 38av).

Design brief (project lead, 2026-10-07): begin each game - and any level where the controls may have changed, or whenever the
model is blocked - with a routine pass over what is interactive and how, including COMBINATIONS (click
objects first, THEN try movement keys and SPACE), the way a human explores controls before reasoning about
the goal. Evidence it targets: lf52 L2 arrows concluded inert after one no-op (38ap), g50t SPACE
record/replay never found (38ar), 59-69% of "control X does nothing" claims made after <=2 tries (38au).

This file is the single source: build_arm_cell.py embeds `make_probe` verbatim into the sandbox bootstrap
(it runs in the sandbox child with restricted builtins and NO imports), and the tests exercise it with a
fake environment. Keep it import-free and use only basic builtins.

Sandbox surface it relies on (documented in the M2 system prompt): action(list) -> result dict,
runtime globals current_frame (.level, .shape, .segmentation{'nodes'}) and valid_actions.
"""


def make_probe(rg, action, stale=None):
    """Return probe_controls bound to the sandbox's runtime globals `rg` and `action` function.

    stale (v2, DUCK 38bh): the sandbox's per-snippet `_stale_state` list. The M2 sandbox refuses every action
    in a snippet after one that changed nothing (StaleStateActionError) until something changes. That guard is
    meant for plans built on a stale belief; a controls probe EXPECTS inert actions, so with v1 one inert key
    refused the rest of the probe (10% of probe tables in the 2026-10-08 run executed <= 1 action). When given,
    the probe clears the marker before each of its own actions. Guard refusals that derive from BaseException
    (known no-op / known death / repeated action) are recorded as REFUSED rows instead of ending the snippet;
    a terminal-state refusal stops the probe; anything else (e.g. the sandbox time limit) propagates.
    """

    GUARDS = ("StaleStateActionError", "KnownNoOpActionError", "KnownDeathActionError",
              "RepeatedActionInStateError")

    EDGE = 4

    def _bbox(node):
        pts = node.get("boundary") or []
        if not pts:
            return None
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        return (min(rows), min(cols), max(rows), max(cols))

    def _objects(frame):
        try:
            nodes = frame.segmentation.get("nodes") or []
        except Exception:
            return []
        try:
            h, w = frame.shape
        except Exception:
            h, w = 64, 64
        out = []
        for n in nodes:
            bb = _bbox(n)
            if bb is None:
                continue
            r0, c0, r1, c1 = bb
            # HUD strips hug an edge: skip objects lying entirely within EDGE cells of one edge
            if r1 < EDGE or c1 < EDGE or r0 >= h - EDGE or c0 >= w - EDGE:
                continue
            # skip background-sized regions (more than a quarter of the board)
            if (n.get("pixels") or 0) > (h * w) // 4:
                continue
            out.append({"id": n.get("id"), "hash": n.get("hash"), "color": n.get("color"),
                        "bbox": bb, "pixels": n.get("pixels") or 0})
        return out

    def _diff(before, after):
        """Object-level change summary (HUD and background excluded). Empty string = nothing changed."""
        b = {}
        a = {}
        for o in _objects(before):
            b.setdefault(o["hash"], []).append(o["bbox"])
        for o in _objects(after):
            a.setdefault(o["hash"], []).append(o["bbox"])
        parts = []
        for hsh in sorted(set(b) | set(a), key=lambda x: str(x)):
            pb = sorted(b.get(hsh, []))
            pa = sorted(a.get(hsh, []))
            if pb == pa:
                continue
            col = ""
            for o in _objects(after) + _objects(before):
                if o["hash"] == hsh:
                    col = str(o["color"])
                    break
            if len(pa) > len(pb):
                parts.append("%s appeared x%d" % (col, len(pa) - len(pb)))
            elif len(pa) < len(pb):
                parts.append("%s vanished x%d" % (col, len(pb) - len(pa)))
            else:
                moved = [(x, y) for x, y in zip(pb, pa) if x != y]
                if moved:
                    (r0, c0, _, _), (r1, c1, _, _) = moved[0]
                    parts.append("%s moved (%+d,%+d)%s" % (col, r1 - r0, c1 - c0,
                                                           " +%d more" % (len(moved) - 1) if len(moved) > 1 else ""))
        return "; ".join(parts[:4]) + (" ..." if len(parts) > 4 else "")

    def probe_controls(max_actions=16, clicks=True, combos=True, use_space=True, skip_hashes=(), verbose=True):
        """Bounded controls diagnostic. Tries each key once, clicks one instance of each object class, and
        after any click that changed something tries the directions (and SPACE) right after it.
        Stops on level change, game over, refusal storms or the action budget. Every probe action is a real,
        counted game action. Returns the list of rows and prints a compact table."""
        rows = []
        state = {"used": 0, "stop": ""}
        start_level = getattr(rg.get("current_frame"), "level", None)
        valid = [str(v) for v in (rg.get("valid_actions") or [])]

        def _do(label, act):
            if state["stop"] or state["used"] >= max_actions:
                if not state["stop"]:
                    state["stop"] = "budget %d reached" % max_actions
                return None
            before = rg.get("current_frame")
            if stale is not None:
                del stale[:]
            try:
                res = action([act]) or {}
            except BaseException as e:  # harness guards: known no-op / death / repeated / stale / terminal
                name = type(e).__name__
                if name == "TerminalStateActionError":
                    state["stop"] = "terminal state"
                    rows.append({"action": label, "effect": "REFUSED " + name})
                    return None
                if not isinstance(e, Exception) and name not in GUARDS:
                    raise
                rows.append({"action": label, "effect": "REFUSED " + name})
                return None
            state["used"] += 1
            after = rg.get("current_frame")
            eff = _diff(before, after)
            flags = []
            if res.get("level_completed"):
                flags.append("LEVEL COMPLETED")
            if res.get("game_over") or str(res.get("state", "")).upper() == "GAME_OVER":
                flags.append("GAME OVER")
            if not eff and res.get("gameplay_changed"):
                eff = "interior changed (no object-level move)"
            row = {"action": label, "effect": eff or "nothing", "flags": flags}
            rows.append(row)
            if flags or getattr(after, "level", start_level) != start_level:
                state["stop"] = ", ".join(flags) or "level changed"
            return row

        keys = [k for k in ("UP", "DOWN", "LEFT", "RIGHT") if k in valid]
        if use_space and "SPACE" in valid:
            keys.append("SPACE")
        for k in keys:
            _do(k, k)

        if clicks and "MOUSE" in valid and not state["stop"]:
            classes = {}
            for o in _objects(rg.get("current_frame")):
                if o["hash"] in skip_hashes:
                    continue
                cur = classes.get(o["hash"])
                if cur is None or o["pixels"] > cur["pixels"]:
                    classes[o["hash"]] = o
            for o in sorted(classes.values(), key=lambda x: -x["pixels"]):
                if state["stop"]:
                    break
                r0, c0, r1, c1 = o["bbox"]
                rr, cc = (r0 + r1) // 2, (c0 + c1) // 2
                label = "CLICK %s#%s(%d,%d)" % (o["color"], o["id"], rr, cc)
                row = _do(label, {"action": "MOUSE", "row": rr, "col": cc})
                if combos and row is not None and row["effect"] != "nothing" and not state["stop"]:
                    for k in keys:
                        if state["stop"]:
                            break
                        _do("  then " + k, k)

        if verbose:
            print("probe_controls: %d actions%s" % (state["used"], (" (stopped: %s)" % state["stop"]) if state["stop"] else ""))
            for r in rows:
                print("  %-28s -> %s%s" % (r["action"], r["effect"], (" [" + ", ".join(r.get("flags") or []) + "]") if r.get("flags") else ""))
        return rows

    return probe_controls
