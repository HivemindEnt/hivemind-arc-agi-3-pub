"""worldmodel: helpers for a model-written simulator inside the M2 python sandbox (FI-939 DUCK 38cv).

Idea (big-swing arm 3, owner direction 2026-10-12): instead of reasoning one action per turn, the model writes
`step(grid, action) -> grid` for the current level, checks it against the level's REAL transitions, and searches
its own model for a short action sequence to the goal. Scoring squares the human/agent action ratio per level, so a
plan that is short in real actions pays twice: fewer wasted actions and fewer turns.

M2 already exposes `transitions` (each with .action, .before_frame, .after_frame, .result) and retains model-written
functions across snippets, but gives no way to score a hypothesised step function against what happened, and no
search. This file adds three functions to the sandbox:

  grid(frame=None)                       -> list of row strings of a frame's ASCII board (default current_frame)
  check_model(step, max_report=3, edge=0)-> how often step(before, action) reproduces the real after-board on this level
  plan(step, is_goal, actions=None, ...) -> shortest action list under the model, by breadth-first search
  run_plan(step, actions, edge=0)        -> executes a plan move by move, stopping at the first surprise (step 3)

Single source: build_arm_cell.py embeds `make_worldmodel` verbatim into the sandbox bootstrap (it runs in the
sandbox child, NO imports, basic builtins only). Tests exercise it with a fake environment.
"""


def make_worldmodel(rg):
    """Return {'grid', 'check_model', 'plan'} bound to the sandbox runtime globals `rg` (read at call time)."""

    def _rows(frame):
        txt = getattr(frame, "ascii", None)
        if not isinstance(txt, str):
            return None
        return [r for r in txt.split("\n") if r != ""]

    def _norm(board):
        """Accept a list/tuple of row strings, a list of lists of chars, or one newline-joined string."""
        if board is None:
            return None
        if isinstance(board, str):
            return tuple(r for r in board.split("\n") if r != "")
        out = []
        for row in board:
            out.append(row if isinstance(row, str) else "".join(str(c) for c in row))
        return tuple(out)

    def grid(frame=None):
        """ASCII board of `frame` (default current_frame) as a list of row strings; grid()[r][c] is one cell."""
        f = rg.get("current_frame") if frame is None else frame
        rows = _rows(f)
        return list(rows) if rows is not None else None

    def _level_transitions():
        cur = rg.get("current_frame")
        lvl = getattr(cur, "level", None)
        out = []
        for t in rg.get("transitions") or []:
            res = getattr(t, "result", None) or {}
            if isinstance(res, dict) and res.get("automatic"):
                continue  # automatic RESET after a death is not the model's action
            b, a = getattr(t, "before_frame", None), getattr(t, "after_frame", None)
            if b is None or a is None:
                continue
            if lvl is not None and (getattr(b, "level", lvl) != lvl or getattr(a, "level", lvl) != lvl):
                continue  # level change or other level: not this level's dynamics
            out.append(t)
        return out

    def _cell_diff(pred, real, edge):
        diffs = []
        h = max(len(pred), len(real))
        for r in range(h):
            pr = pred[r] if r < len(pred) else ""
            rr = real[r] if r < len(real) else ""
            w = max(len(pr), len(rr))
            for c in range(w):
                pc = pr[c] if c < len(pr) else "?"
                rc = rr[c] if c < len(rr) else "?"
                if pc != rc:
                    on_edge = edge > 0 and (r < edge or c < edge or r >= h - edge or c >= w - edge)
                    diffs.append((r, c, pc, rc, on_edge))
        return diffs

    def check_model(step, max_report=3, edge=0):
        """Replay this level's real transitions through `step(board, action)`.

        `board` is a list of row strings (as from grid()); `action` is the transition's action string.
        `step` may return rows, a list of lists of chars, or one newline-joined string.
        edge > 0 reports mismatches within `edge` cells of the border (HUD, timers) separately.
        Returns a dict: checked, exact, exact_ignoring_edge, accuracy, errors, mismatches (first max_report).
        """
        ts = _level_transitions()
        exact = 0
        exact_core = 0
        errors = 0
        mismatches = []
        for i, t in enumerate(ts):
            before = _rows(t.before_frame)
            real = tuple(_rows(t.after_frame) or ())
            try:
                pred = _norm(step(list(before), t.action))
            except Exception as e:  # the model's step raised: report it, keep going
                errors += 1
                if len(mismatches) < max_report:
                    mismatches.append({"i": i, "action": t.action, "error": "%s: %s" % (type(e).__name__, e)})
                continue
            if pred is None:
                errors += 1
                continue
            d = _cell_diff(pred, real, edge)
            if not d:
                exact += 1
                exact_core += 1
                continue
            if all(x[4] for x in d):
                exact_core += 1
            if len(mismatches) < max_report:
                mismatches.append({"i": i, "action": t.action, "cells_wrong": len(d),
                                   "first": [(r, c, "pred=" + p, "real=" + a) for r, c, p, a, _ in d[:8]]})
        n = len(ts)
        return {"checked": n, "exact": exact, "exact_ignoring_edge": exact_core, "errors": errors,
                "accuracy": (exact / n) if n else None, "mismatches": mismatches}

    def plan(step, is_goal, actions=None, start=None, max_depth=60, max_nodes=20000):
        """Breadth-first search over YOUR model: the shortest action list from `start` (default the current board)
        to a board where is_goal(board) is true. Nothing is executed; pass the result to action(...) yourself.

        actions: the moves to try (default: valid_actions minus RESET and MOUSE; pass click dicts explicitly,
        e.g. [{'action': 'MOUSE', 'row': 4, 'col': 7}]). Boards that the model maps to themselves are pruned.
        Returns {'found': bool, 'actions': [...], 'nodes': n, 'depth': d, 'reason': ...}.
        """
        if actions is None:
            actions = [a for a in (rg.get("valid_actions") or []) if str(a).upper() not in ("RESET", "MOUSE")]
        acts = list(actions)
        if not acts:
            return {"found": False, "actions": [], "nodes": 0, "depth": 0, "reason": "no actions to try"}
        s0 = _norm(start) if start is not None else tuple(_rows(rg.get("current_frame")) or ())
        if is_goal(list(s0)):
            return {"found": True, "actions": [], "nodes": 1, "depth": 0, "reason": "start already satisfies goal"}
        parent = {s0: None}
        frontier = [s0]
        depth = 0
        nodes = 1
        while frontier and depth < max_depth:
            depth += 1
            nxt = []
            for s in frontier:
                for a in acts:
                    try:
                        s2 = _norm(step(list(s), a))
                    except Exception as e:
                        return {"found": False, "actions": [], "nodes": nodes, "depth": depth,
                                "reason": "step raised on %r: %s: %s" % (a, type(e).__name__, e)}
                    if s2 is None or s2 == s or s2 in parent:
                        continue
                    parent[s2] = (s, a)
                    nodes += 1
                    if is_goal(list(s2)):
                        path = []
                        cur = s2
                        while parent[cur] is not None:
                            prev, act = parent[cur]
                            path.append(act)
                            cur = prev
                        path.reverse()
                        return {"found": True, "actions": path, "nodes": nodes, "depth": depth, "reason": "goal"}
                    if nodes >= max_nodes:
                        return {"found": False, "actions": [], "nodes": nodes, "depth": depth,
                                "reason": "node limit %d reached (tighten the model or the goal)" % max_nodes}
                    nxt.append(s2)
            frontier = nxt
        return {"found": False, "actions": [], "nodes": nodes, "depth": depth,
                "reason": "no goal state within depth %d under this model" % max_depth if frontier else
                "model state space exhausted without reaching the goal"}

    def run_plan(step, actions, edge=0):
        """Execute `actions` ONE AT A TIME with the real action([a]), checking each real board against step's
        prediction, and stop at the first surprise: a board your model got wrong (outside `edge` border cells),
        a level change, an action that raised (terminal state, harness guard), or a step that raised.
        Use it for a plan you are confident in: the whole plan costs one turn instead of one turn per move, and a
        wrong model costs only one wasted action. Returns {'executed', 'planned', 'stopped', 'at', 'action',
        'cells_wrong', 'first'}; stopped is None when every action matched.
        """
        act = rg.get("action")
        acts = list(actions or [])
        out = {"executed": 0, "planned": len(acts), "stopped": None, "at": None, "action": None,
               "cells_wrong": 0, "first": []}
        if not callable(act):
            out["stopped"] = "no action() in this sandbox"
            return out
        for i, a in enumerate(acts):
            cur = rg.get("current_frame")
            before = tuple(_rows(cur) or ())
            lvl = getattr(cur, "level", None)
            try:
                pred = _norm(step(list(before), a))
            except Exception as e:  # noqa: BLE001
                out.update(stopped="step raised: %s: %s" % (type(e).__name__, e), at=i, action=a)
                return out
            try:
                act([a])
            except Exception as e:  # noqa: BLE001 - terminal state or a harness guard: report, do not hide
                out.update(stopped="action raised: %s: %s" % (type(e).__name__, e), at=i, action=a)
                return out
            out["executed"] += 1
            after = rg.get("current_frame")
            if getattr(after, "level", lvl) != lvl:
                out.update(stopped="level changed", at=i, action=a)
                return out
            real = tuple(_rows(after) or ())
            d = [x for x in _cell_diff(pred or (), real, edge) if not x[4]]
            if d:
                out.update(stopped="surprise", at=i, action=a, cells_wrong=len(d),
                           first=[(r, c, "pred=" + p, "real=" + q) for r, c, p, q, _ in d[:8]])
                return out
        return out

    return {"grid": grid, "check_model": check_model, "plan": plan, "run_plan": run_plan}
