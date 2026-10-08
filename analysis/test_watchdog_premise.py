"""Tests for watchdog_premise.py (DUCK 38ci). Run: python3 test_watchdog_premise.py"""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import watchdog_premise as W


def act(n, name, level, step, lc=False, go=False):
    return {"type": "action", "action_num": n, "action_name": name, "level": level, "analysis_step": step,
            "level_completed": lc, "game_over": go}


def game():
    """Level 1: 3 actions over 2 turns, cleared (the clearing event is logged at level 2).
    Level 2: game over, forced RESET, voluntary RESET, then 2 more actions, unfinished."""
    ev = [{"type": "initial", "level": 1},
          act(1, "ACTION1", 1, 0), act(2, "ACTION2", 1, 0), act(3, "ACTION6(3,4)", 2, 1, lc=True),
          act(4, "ACTION1", 2, 2, go=True), act(5, "RESET", 2, 2), act(6, "ACTION3", 2, 3),
          act(7, "RESET", 2, 3), act(8, "ACTION4", 2, 4), act(9, "ACTION4", 2, 4)]
    return ev


class T(unittest.TestCase):
    def test_attempts(self):
        a = W.attempts(game())
        self.assertEqual([x["level"] for x in a], [1, 2])
        l1, l2 = a
        self.assertEqual((l1["actions"], l1["turns"], l1["cleared"], l1["clear_at"]), (3, 2, True, 3))
        self.assertEqual((l2["actions"], l2["turns"], l2["cleared"]), (6, 3, False))
        self.assertEqual((l2["resets_forced"], l2["resets_vol"], l2["restarts"], l2["first_restart_at"]),
                         (1, 1, 1, 1))

    def test_skips_non_actions_and_bad_rows(self):
        ev = [{"type": "analysis"}, {"type": "action"}, act(1, "ACTION1", 1, 0)]
        a = W.attempts(ev)
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]["actions"], 1)

    def test_no_initial_defaults_level1(self):
        a = W.attempts([act(1, "ACTION1", 1, 0)])
        self.assertEqual(a[0]["level"], 1)

    def _rows(self):
        rows = []
        for i, (n, cleared) in enumerate([(10, True), (25, True), (50, True), (70, False), (20, False)]):
            rows.append({"game": "g", "level": 1, "actions": n, "turns": n // 5, "cleared": cleared,
                         "resets_vol": 0, "resets_forced": 0, "restarts": 0, "clear_at": n if cleared else None,
                         "first_restart_at": None})
        return rows

    def test_summarize_thresholds(self):
        s = W.summarize(self._rows(), [18, 45])
        b18, b45 = s["by"][("actions", 18)], s["by"][("actions", 45)]
        self.assertEqual((b18["reached"], b18["cleared_after"]), (4, 2))   # 25,50,70,20 open after 18; 25,50 clear
        self.assertAlmostEqual(b18["share_of_clears"], 2 / 3.0)
        self.assertEqual((b45["reached"], b45["cleared_after"]), (2, 1))   # 50 and 70
        self.assertIsNone(b18["late_score"])                                # no baselines

    def test_late_score_uses_baselines(self):
        s = W.summarize(self._rows(), [18], {"g": [25]})
        sc, n = s["by"][("actions", 18)]["late_score"]
        self.assertEqual(n, 2)
        self.assertAlmostEqual(sc, (1.0 + 0.25) / 2)                       # (25/25)^2 and (25/50)^2

    def test_restart_matched(self):
        rows = self._rows()
        rows[3].update(restarts=1, first_restart_at=30)                    # unfinished level restarted at 30
        rows[2].update(restarts=1, first_restart_at=20)                    # cleared at 50 after a restart at 20
        s = W.summarize(rows, [18])
        self.assertEqual((s["restart_levels"], s["restart_cleared_after"]), (2, 1))
        # pools without restarts: >30 -> none; >20 -> [25 cleared] -> 1.0; mean over non-empty pools = 1.0
        self.assertAlmostEqual(s["restart_matched_clear"], 1.0)

    def test_run_and_main(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "ab12-0123abcd_p0_events.jsonl"), "w") as f:
            for r in game():
                f.write(json.dumps(r) + "\n")
            f.write("not json\n")
        with open(os.path.join(d, "ignored.jsonl"), "w") as f:
            f.write("{}\n")
        per, n = W.run([d])
        self.assertEqual(n, 1)
        self.assertEqual(sorted(per), ["ab12"])
        self.assertEqual(W.main([d, "--thresholds", "1,2", "--env-dir", d, "--per-game"]), 0)
        self.assertEqual(W.main([tempfile.mkdtemp()]), 3)

    def test_render(self):
        s = W.summarize(self._rows(), [18])
        txt = W.render(s, "ALL", [18])
        self.assertIn("RESET voluntary=0", txt)
        self.assertIn("T=18 open 4, cleared later 2 (50%)", txt)


if __name__ == "__main__":
    unittest.main()
