"""Tests for paper_public_eval.table_row/render (pure, no files)."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_public_eval as P


def r(game, p, official, compl, levels):
    return {"game": game, "pass": p, "official": official, "compl": compl, "levels": levels, "stuck": None, "n_levels": 5}


class Table(unittest.TestCase):
    def test_row(self):
        rows = [r("ar25", 0, 0.8, 1.0, 5), r("zz99", 0, 0.2, 0.4, 2),
                r("ar25", 1, 0.6, 0.8, 4), r("zz99", 1, 0.0, 0.0, 0)]
        t = P.table_row(rows)
        self.assertEqual((t["games"], t["passes"]), (2, 2))
        self.assertAlmostEqual(t["official"], (0.5 + 0.3) / 2)          # pass means 0.5 and 0.3
        self.assertAlmostEqual(t["heldout"], (0.8 + 0.6) / 2)           # only ar25 is held-out
        self.assertAlmostEqual(t["levels_per_pass"], (7 + 4) / 2)
        # loss 1-0.4 = 0.6; unfinished = 1 - mean(compl over games) = 1 - (0.9+0.2)/2 = 0.45 -> 75%
        self.assertAlmostEqual(t["unfinished_share"], 0.45 / 0.6)
        self.assertGreater(t["official_sd"], 0)

    def test_render(self):
        t = P.table_row([r("ar25", 0, 0.5, 0.5, 3)])
        md = P.render([("base", t)], markdown=True)
        self.assertTrue(md.startswith("| run |"))
        self.assertIn("| base | 1 | 1 | 50.0 (0.0) |", md)
        self.assertIn("base", P.render([("base", t)]))

    def test_cli_usage(self):
        self.assertEqual(P.main(["nolabel"]), 2)


if __name__ == "__main__":
    unittest.main()
