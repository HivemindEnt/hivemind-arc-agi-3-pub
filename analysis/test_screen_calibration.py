"""Tests for screen_calibration.py (FI-939: does the offline HF screen predict the hidden score?)."""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_calibration as S  # noqa: E402

BASE = "k-base"


class Report(unittest.TestCase):
    def test_vs_base_agree_disagree_tie_and_pending(self):
        hidden = {BASE: (26.6, 9), "k-up": (29.0, 1), "k-down": (23.0, 1), "k-flat": (26.9, 1), "k-wrong": (28.9, 1)}
        screen = {BASE: 44.7, "k-up": 47.0, "k-down": 39.1, "k-flat": 30.0, "k-wrong": 40.0, "k-new": 30.6}
        out = S.report(hidden, screen, BASE, noise=1.48)
        text = "\n".join(out)
        self.assertRegex(text, r"k-up .*hidden \+2\.40, screen \+2\.3\s+AGREE")
        self.assertRegex(text, r"k-down .*hidden -3\.60, screen -5\.6\s+AGREE")
        self.assertRegex(text, r"k-flat .*TIE", "a hidden gap inside the noise is a tie, whatever the screen says")
        self.assertRegex(text, r"k-wrong .*DISAGREE")
        self.assertRegex(text, r"k-new .*none usable.*prediction on record: screen -14\.1 vs base")
        self.assertTrue(out[-1].startswith("SCREEN_CALIB vs-base agree=2 disagree=1 ties=1;"), out[-1])
        self.assertIn("n=5", out[-1])
        self.assertNotIn("n/a", out[-1], "rho is computed with 5 kernels")

    def test_insufficient_cases(self):
        self.assertTrue(S.report({}, {"x": 1.0}, BASE)[-1].startswith("SCREEN_CALIB_INSUFFICIENT base kernel"))
        self.assertIn("no hidden score", S.report({}, {BASE: 44.0}, BASE)[-1])
        out = S.report({BASE: (26.0, 3)}, {BASE: 44.0, "k-new": 40.0}, BASE)
        self.assertTrue(out[-1].startswith("SCREEN_CALIB_INSUFFICIENT 1 kernel"))
        self.assertTrue(any("k-new" in l and "prediction on record" in l for l in out), "a pending arm is still listed")

    def test_rho_needs_four(self):
        out = S.report({BASE: (26.0, 3), "a": (28.0, 1), "b": (22.0, 1)}, {BASE: 44.0, "a": 46.0, "b": 40.0}, BASE)
        self.assertIn("rho=n/a under 4 n=3", out[-1])


class ScreenValues(unittest.TestCase):
    def test_mean_over_existing_replicates_fraction_to_percent(self):
        t = tempfile.mkdtemp()
        for d in ("r1", "r2", "solo"):
            os.makedirs(os.path.join(t, d))
        vals = {"r1": 0.44, "r2": 0.46, "solo": 0.306}
        got = S.screen_values(t, {"k": ["r1", "r2", "missing"], "s": ["solo"], "none": ["missing"]}, None,
                              row_fn=lambda p: vals[os.path.basename(p)])
        self.assertAlmostEqual(got["k"], 45.0)
        self.assertAlmostEqual(got["s"], 30.6)
        self.assertNotIn("none", got, "a kernel with no scorable dir is left out, not scored 0")

    def test_unscorable_dir_is_skipped(self):
        t = tempfile.mkdtemp(); os.makedirs(os.path.join(t, "empty"))
        self.assertEqual(S.screen_values(t, {"k": ["empty"]}, None, row_fn=lambda p: None), {})


class MapFile(unittest.TestCase):
    def test_shipped_map_is_valid(self):
        base, kernels = S.load_map()
        self.assertIn(base, kernels)
        self.assertTrue(all(isinstance(v, list) and v and all(x.startswith("hf-") for x in v) for v in kernels.values()))
        self.assertIn("duck-franzen-m2-deep140-lb", kernels, "the pre-registered 10-28 calibration draw is mapped")

    def test_base_must_be_mapped(self):
        p = os.path.join(tempfile.mkdtemp(), "m.json")
        with open(p, "w") as f:
            json.dump({"base": "x", "kernels": {"y": ["hf-y"]}}, f)
        with self.assertRaises(ValueError):
            S.load_map(p)


class Pool(unittest.TestCase):
    def test_pool_and_drop_zero(self):
        scores = {"m2": [25.5, 26.6, 0.0, 27.0], "guard": [26.0], "arm": [28.46], "dead": [0.0]}
        got, dropped = S.pool_hidden(scores, {"guard": ["m2", "guard"]})
        self.assertEqual(dropped, 2, "distinct failed draws: m2's zero once (pooled and own slug), dead's zero")
        self.assertAlmostEqual(got["guard"][0], (25.5 + 26.6 + 27.0 + 26.0) / 4); self.assertEqual(got["guard"][1], 4)
        self.assertEqual(got["arm"], (28.46, 1))
        self.assertNotIn("dead", got, "a slug whose only draw failed has no score")
        kept, d0 = S.pool_hidden(scores, {}, drop_zero=False)
        self.assertEqual(d0, 0); self.assertEqual(kept["m2"][1], 4)

    def test_shipped_map_pools_the_base_and_skips_partial_screens(self):
        base, kernels, pools = S.load_map(with_pools=True)
        self.assertIn("duck-franzen-m2", pools[base])
        self.assertEqual(kernels["duck-franzen-m2-levelpin-lb"], ["hf-harn-levelpin-r3"])


class Wiring(unittest.TestCase):
    def test_final_line_is_cmd_safe(self):
        hidden = {"b": (26.5, 9), "x": (28.0, 1), "y": (20.0, 1), "z": (25.0, 1)}
        screen = {"b": 44.7, "x": 40.0, "y": 32.0, "z": 43.0}
        line = S.report(hidden, screen, "b")[-1]
        self.assertTrue(line.startswith("SCREEN_CALIB "))
        self.assertFalse(any(c in line for c in '|&<>^%()"'), line)

    @unittest.skipUnless(os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_klog_autopull_scheduled.cmd")),
                         "fleet scheduler wiring: the task file exists only in the private repo")
    def test_klog_task_reruns_calibration_on_a_new_score(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_klog_autopull_scheduled.cmd")
        with open(p, encoding="utf-8", newline="") as f:
            s = f.read()
        self.assertIn("\r\n", s)
        i = s.index("duck_screen_calibration.sh")
        self.assertGreater(i, s.index('findstr /b /c:"SCOREWATCH_NEW"'), "runs only after a new score")
        self.assertIn("call node --use-system-ca bin\\notify-slack.mjs", s[i:i + 600], "late expansion of the verdict line")


if __name__ == "__main__":
    unittest.main()
