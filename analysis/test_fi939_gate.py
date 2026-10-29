"""Tests for fi939_gate.py (synthetic per-game data; no traces needed)."""
import os, tempfile, unittest
import fi939_gate as G

GAMES = ["g%03d" % i for i in range(25)]
HELD = GAMES[:5]


def mk(f):
    return {k: [f(i)] * 4 for i, k in enumerate(GAMES)}


class Gate(unittest.TestCase):
    def test_identical_passes(self):
        b = mk(lambda i: i % 4)
        r = G.evaluate(b, b, HELD)
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["p_worse"], 1.0)

    def test_uniformly_worse_fails_significance(self):
        b = mk(lambda i: 3)
        c = mk(lambda i: 2)
        r = G.evaluate(b, c, HELD)
        self.assertFalse(r["rule_total_not_sig_worse"])
        self.assertLess(r["p_worse"], 0.001)
        self.assertEqual(r["verdict"], "FAIL")

    def test_heldout_worse_fails_even_if_total_better(self):
        b = mk(lambda i: 2)
        c = mk(lambda i: 1 if i < 5 else 4)
        r = G.evaluate(b, c, HELD)
        self.assertTrue(r["rule_total_not_sig_worse"])
        self.assertFalse(r["rule_heldout_not_worse"])
        self.assertEqual(r["verdict"], "FAIL")

    def test_small_noisy_loss_not_significant(self):
        b = mk(lambda i: 2)
        c = mk(lambda i: 2 - (1 if i == 7 else 0) + (1 if i == 0 else 0))  # held-out +1, one other -1
        r = G.evaluate(b, c, HELD)
        self.assertEqual(r["verdict"], "PASS")

    def test_per_game_counts_the_winning_last_level(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "artifacts"))
        with open(os.path.join(d, "artifacts", "ab12-0123abcd_p0_events.jsonl"), "w") as f:
            f.write('{"type": "action", "level_completed": true}\n{"type": "action", "run_complete": true}\n')
        self.assertEqual(G.per_game(d), {"ab12": [2]})

    def test_missing_heldout_cannot_evaluate(self):
        b = mk(lambda i: 1)
        c = {k: v for k, v in b.items() if k != HELD[0]}
        self.assertEqual(G.evaluate(b, c, HELD)["verdict"], "CANNOT_EVALUATE")

    def test_exact_matches_known_value(self):
        # 3 nonzero diffs all negative: only the all-negative assignment is <= obs -> 1/8
        self.assertAlmostEqual(G.sign_flip_p_worse([-1, -1, -1, 0]), 1 / 8)

    def test_per_game_parses_event_names(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "artifacts"))
        for n in ("ab12-0123abcd_p1_events.jsonl", "ab12-0123abcd_p2_events.jsonl", "junk.jsonl"):
            open(os.path.join(d, "artifacts", n), "w").close()
        g = G.per_game(d, levels_fn=lambda p: 3)
        self.assertEqual(g, {"ab12": [3, 3]})

    def test_min_heldout_gain(self):
        b = mk(lambda i: 2)
        c = mk(lambda i: 2.1 if i < 5 else 2)      # held-out +5%
        self.assertEqual(G.evaluate(b, c, HELD)["verdict"], "PASS")
        self.assertEqual(G.evaluate(b, c, HELD, min_heldout_gain_pct=5.0)["verdict"], "PASS")
        self.assertEqual(G.evaluate(b, c, HELD, min_heldout_gain_pct=10.0)["verdict"], "FAIL")
        self.assertEqual(G.evaluate(b, b, HELD, min_heldout_gain_pct=10.0)["verdict"], "FAIL")



class Pooling(unittest.TestCase):
    def test_pooled_per_game_concatenates_passes_across_run_dirs(self):
        fake = {"/t/a": {"ls20": [1, 2], "bp35": [0]}, "/t/b": {"ls20": [3], "cd82": [4]}}
        got = G.pooled_per_game("/t", "a, b", per_game_fn=lambda d: fake[d.replace(os.sep, "/")])
        self.assertEqual(got, {"ls20": [1, 2, 3], "bp35": [0], "cd82": [4]})

    def test_single_name_is_unchanged(self):
        fake = {"/t/a": {"ls20": [1, 2]}}
        self.assertEqual(G.pooled_per_game("/t", "a", per_game_fn=lambda d: fake[d.replace(os.sep, "/")]),
                         {"ls20": [1, 2]})

if __name__ == "__main__":
    unittest.main()
