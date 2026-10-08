"""Tests for ctrlprobe.make_probe (FI-939 DUCK 38av) against a fake sandbox environment."""
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctrlprobe  # noqa: E402


def box(r0, c0, r1, c1):
    return [[r0, c0], [r0, c1], [r1, c1], [r1, c0]]


class Frame:
    def __init__(self, objs, level=1):
        self.level = level
        self.shape = (64, 64)
        self.segmentation = {"nodes": [dict(o) for o in objs]}


class World:
    """A platform (blue) that moves with arrows only after it is clicked; a HUD bar at the top edge that
    shrinks on every action; a background region; optionally SPACE = game over."""

    def __init__(self, space_kills=False, valid=("UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE")):
        self.selected = False
        self.pos = [30, 20]
        self.hud = 40
        self.space_kills = space_kills
        self.rg = {"valid_actions": list(valid)}
        self.calls = []
        self.refresh()

    def objs(self):
        r, c = self.pos
        return [
            {"id": 0, "hash": "hud", "color": "g", "pixels": self.hud, "boundary": box(0, 0, 1, self.hud)},
            {"id": 1, "hash": "bg", "color": "B", "pixels": 3000, "boundary": box(5, 0, 63, 63)},
            {"id": 2, "hash": "plat", "color": "b", "pixels": 24, "boundary": box(r, c, r + 2, c + 7)},
            {"id": 3, "hash": "wall", "color": "w", "pixels": 10, "boundary": box(50, 50, 51, 54)},
        ]

    def refresh(self):
        self.rg["current_frame"] = Frame(self.objs())

    def action(self, acts):
        a = acts[0]
        self.calls.append(a)
        self.hud -= 1
        res = {"executed": True, "gameplay_changed": False}
        if isinstance(a, dict):
            r, c = a["row"], a["col"]
            pr, pc = self.pos
            self.selected = pr <= r <= pr + 2 and pc <= c <= pc + 7
            if self.selected:
                self.pos = [pr + 1, pc]  # selection nudges it visibly (outline stand-in)
        elif a == "SPACE" and self.space_kills:
            res["game_over"] = True
        elif self.selected and a in ("LEFT", "RIGHT"):
            self.pos[1] += -1 if a == "LEFT" else 1
        self.refresh()
        return res


class Probe(unittest.TestCase):
    def run_probe(self, world, **kw):
        p = ctrlprobe.make_probe(world.rg, world.action)
        return p(verbose=False, **kw)

    def test_keys_alone_do_nothing_but_click_then_move_works(self):
        w = World()
        rows = self.run_probe(w, use_space=False)
        eff = {r["action"].strip(): r["effect"] for r in rows}
        self.assertEqual(eff["RIGHT"], "nothing")  # HUD shrink ignored
        click = [r for r in rows if r["action"].startswith("CLICK b#2")]
        self.assertEqual(len(click), 1)
        self.assertIn("moved", click[0]["effect"])
        self.assertIn("moved (+0,+1)", eff["then RIGHT"])
        self.assertIn("moved (+0,-1)", eff["then LEFT"])

    def test_background_and_hud_are_never_clicked(self):
        w = World()
        self.run_probe(w, use_space=False)
        clicks = [a for a in w.calls if isinstance(a, dict)]
        self.assertTrue(all(not (a["row"] <= 1) for a in clicks), clicks)  # no HUD click
        self.assertEqual(len(clicks), 2)  # platform + wall, not the background

    def test_budget_is_respected(self):
        w = World()
        rows = self.run_probe(w, max_actions=3)
        self.assertEqual(len(w.calls), 3)
        self.assertEqual(len(rows), 3)

    def test_stops_on_game_over(self):
        w = World(space_kills=True)
        rows = self.run_probe(w)
        self.assertEqual(w.calls[-1], "SPACE")
        self.assertIn("GAME OVER", rows[-1]["flags"])

    def test_use_space_false_never_presses_space(self):
        w = World(space_kills=True)
        self.run_probe(w, use_space=False)
        self.assertNotIn("SPACE", w.calls)

    def test_skip_hashes_and_no_mouse(self):
        w = World()
        self.run_probe(w, use_space=False, skip_hashes=("plat",))
        self.assertFalse(any(isinstance(a, dict) and a["row"] < 40 for a in w.calls))
        w2 = World(valid=("UP", "DOWN"))
        self.run_probe(w2)
        self.assertEqual(w2.calls, ["UP", "DOWN"])

    def test_refused_actions_are_recorded_not_raised(self):
        w = World()

        def refusing(acts):
            raise RuntimeError("KnownNoOpActionError: blocked")
        p = ctrlprobe.make_probe(w.rg, refusing)
        rows = p(verbose=False, clicks=False)
        self.assertTrue(rows and all(r["effect"].startswith("REFUSED") for r in rows))

    def test_v2_stale_guard_does_not_end_the_probe(self):
        """38bh: M2 refuses every action in a snippet after a no-op until something changes. v1 lost the
        rest of the probe after one inert key; v2 clears the sandbox's stale marker before each probe action."""

        class StaleStateActionError(Exception):
            pass

        def guarded(world, stale):
            def act(acts):
                if stale:
                    raise StaleStateActionError("stale")
                before = world.rg["current_frame"].segmentation["nodes"][2]["boundary"]
                res = world.action(acts)
                if world.rg["current_frame"].segmentation["nodes"][2]["boundary"] == before:
                    stale.append(str(acts[0]))
                return res
            return act

        w1, s1 = World(), []
        v1 = ctrlprobe.make_probe(w1.rg, guarded(w1, s1))(verbose=False, use_space=False)
        self.assertEqual(len(w1.calls), 1)  # UP was inert, everything after it refused
        self.assertTrue(all(r["effect"] == "REFUSED StaleStateActionError" for r in v1[1:]))
        w2, s2 = World(), []
        v2 = ctrlprobe.make_probe(w2.rg, guarded(w2, s2), s2)(verbose=False, use_space=False)
        self.assertFalse(any(r["effect"].startswith("REFUSED") for r in v2))
        self.assertIn("moved (+0,+1)", {r["action"].strip(): r["effect"] for r in v2}["then RIGHT"])

    def test_v2_base_exception_guards_recorded_terminal_stops_others_propagate(self):
        class KnownNoOpActionError(BaseException):
            pass

        class TerminalStateActionError(BaseException):
            pass

        w = World()
        rows = ctrlprobe.make_probe(w.rg, lambda a: (_ for _ in ()).throw(KnownNoOpActionError("x")), [])(
            verbose=False, clicks=False, use_space=False)
        self.assertEqual([r["effect"] for r in rows], ["REFUSED KnownNoOpActionError"] * 4)
        rows = ctrlprobe.make_probe(w.rg, lambda a: (_ for _ in ()).throw(TerminalStateActionError("x")), [])(
            verbose=False, clicks=False, use_space=False)
        self.assertEqual(len(rows), 1)
        with self.assertRaises(KeyboardInterrupt):
            ctrlprobe.make_probe(w.rg, lambda a: (_ for _ in ()).throw(KeyboardInterrupt()), [])(
                verbose=False, clicks=False)

    def test_source_is_import_free(self):
        src = open(ctrlprobe.__file__).read()
        body = src[src.index("def make_probe"):]
        self.assertNotIn("import ", body)


if __name__ == "__main__":
    unittest.main()
