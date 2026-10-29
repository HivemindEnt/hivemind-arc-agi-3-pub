"""Tests for ab_metrics.game_from_events (the level/action counter the screens and readouts share)."""
import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_metrics as M


class GameFromEvents(unittest.TestCase):
    def test_counts(self):
        rows = [{"type": "analysis"}, {"type": "analysis"},
                {"type": "action", "board_changed": True},
                {"type": "action", "result": {"game_over": True}},
                {"event": "action", "last_action_result": {"level_completed": "true", "board_changed": "true"}},
                {"type": "action", "run_complete": 1},
                {"type": "initial"}]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write("not json\n\n")
            p = f.name
        try:
            g = M.game_from_events(p)
        finally:
            os.unlink(p)
        self.assertEqual(g, {"analyses": 2, "actions": 4, "board_changes": 2, "resets_game_over": 1,
                             "levels_completed": 1, "run_complete": 1, "levels": 2})

    def test_levels_counts_the_winning_last_level_once(self):
        rows = [{"type": "action", "level_completed": True}, {"type": "action", "level_completed": True},
                {"type": "action", "run_complete": True},                       # last level: won game
                {"type": "action", "level_completed": True, "run_complete": True}]  # both flags: one level
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(json.dumps(r) for r in rows))
            p = f.name
        try:
            g = M.game_from_events(p)
        finally:
            os.unlink(p)
        self.assertEqual((g["levels_completed"], g["run_complete"], g["levels"]), (3, 2, 4))


if __name__ == "__main__":
    unittest.main()
