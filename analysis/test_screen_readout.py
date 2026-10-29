"""Tests for screen_readout.py (no network)."""
import io, json, os, shutil, sys, tempfile, unittest
from contextlib import redirect_stdout
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_readout as S


def w(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def game(levels_done, tail_actions=2):
    ev = []
    for L in range(1, levels_done + 1):
        ev += [{"type": "analysis"}, {"type": "action", "score": L - 1}, {"type": "action", "score": L}]
    ev += [{"type": "analysis"}] + [{"type": "action", "score": levels_done}] * tail_actions
    return ev


class Readout(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.base = os.path.join(self.td, "base"); self.arm = os.path.join(self.td, "arm")
        for p in range(2):
            w(os.path.join(self.base, "artifacts", "ab12-x_p%d_events.jsonl" % p), game(2))
            w(os.path.join(self.arm, "artifacts", "ab12-x_p%d_events.jsonl" % p), game(1))
        # base: full request logs; arm: bundle summary only
        w(os.path.join(self.base, "ab12-x_p0_requests.jsonl"),
          [{"event": "request"}, {"event": "response", "usage": {"prompt_tokens": 100, "completion_tokens": 10}}] * 4)
        json.dump({"ab12-x_p0_requests.jsonl": {"lines": 4, "events": {"request": 2, "response": 2},
                                                "usage": {"prompt_tokens": 300, "completion_tokens": 30}}},
                  open(os.path.join(self.arm, "requests_summary.json"), "w"))
        open(os.path.join(self.arm, "papermill.log"), "w").write("[DEEP140] x\n[DEEP140] y\n")

    def tearDown(self):
        shutil.rmtree(self.td)

    def test_segments(self):
        p = os.path.join(self.base, "artifacts", "ab12-x_p0_events.jsonl")
        self.assertEqual(S.level_segments(p), [(True, 2, 1), (True, 2, 1), (False, 2, 1)])
        self.assertEqual(S.max_level(p), 2)

    def test_stats_from_logs_and_summary(self):
        b = S.run_stats(self.base); a = S.run_stats(self.arm, ["[DEEP140]"])
        self.assertEqual(b["source"], "logs"); self.assertEqual(a["source"], "summary")
        self.assertEqual(b["requests_per_pass"], 2.0)  # 4 requests / 2 passes
        self.assertEqual(a["requests_per_pass"], 1.0)
        self.assertEqual(a["prompt_per_req"], 150.0); self.assertEqual(b["prompt_per_req"], 100.0)
        self.assertEqual(a["markers"], {"[DEEP140]": 2})
        self.assertEqual(b["games"], {"ab12": 2.0}); self.assertEqual(a["games"], {"ab12": 1.0})
        c = S.compare([b], a)
        self.assertAlmostEqual(c["requests_per_pass"]["pct"], -50.0)
        self.assertAlmostEqual(c["prompt_per_req"]["pct"], 50.0)

    def test_cli(self):
        out = os.path.join(self.td, "r.json"); buf = io.StringIO()
        with redirect_stdout(buf):
            rc = S.main([self.base + "," + self.base, self.arm, "--marker", "[DEEP140]", "--json", out])
        self.assertEqual(rc, 0)
        self.assertIn("DELTA requests_per_pass", buf.getvalue())
        self.assertIn("ab12", json.load(open(out))["compare"]["games"])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(S.main([self.base, os.path.join(self.td, "empty")]), 4)
            self.assertEqual(S.main([]), 2)


if __name__ == "__main__":
    unittest.main()
