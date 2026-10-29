"""Tests for hf_results_bundle.py (log-channel results fallback). No network."""
import io, json, os, shutil, sys, tarfile, tempfile, types, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_results_bundle as B
import ab_metrics
import duck_score_estimate as D


def write_events(path, events):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        f.write("not json\n\n")


EVENTS = [
    {"type": "initial", "frame": [[0] * 64] * 64},
    {"type": "analysis", "text": "x" * 5000, "analysis_step": 1},
    {"type": "action", "action_name": "ACTION1", "score": 0, "board_changed": True, "frame": [[1] * 64] * 64},
    {"type": "action", "action_name": "ACTION2", "score": 0, "game_over": True},
    {"type": "action", "action_name": "RESET", "score": 0},
    {"type": "action", "action_name": "ACTION3", "score": 1, "result": {"level_completed": True, "board_changed": True, "frame": [1]}},
    {"event": "action", "action_name": "ACTION4", "last_action_result": {"run_complete": "true"}},
]


class Bundle(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.dest = tempfile.mkdtemp()
        write_events(os.path.join(self.root, "artifacts", "g1-abc_p0_events.jsonl"), EVENTS)
        write_events(os.path.join(self.root, "pass2", "artifacts", "g1-abc_p0_events.jsonl"), EVENTS[:3])
        write_events(os.path.join(self.root, "g1-abc_p0_requests.jsonl"),
                     [{"event": "request", "messages": ["m"] * 50}, {"event": "response", "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
                      {"event": "response", "response": {"usage": {"prompt_tokens": 5}}}])
        write_events(os.path.join(self.root, "venv", "x_events.jsonl"), EVENTS)
        with open(os.path.join(self.root, "papermill.log"), "w") as f:
            f.write("FINAL line\n")

    def tearDown(self):
        shutil.rmtree(self.root); shutil.rmtree(self.dest)

    def roundtrip(self, mangle=lambda ls: ls):
        buf = io.StringIO()
        self.assertEqual(B.emit(self.root, out=buf), 0)
        return B.parse(mangle(buf.getvalue().splitlines(True)))

    def test_roundtrip_readouts_identical(self):
        raw, err = self.roundtrip()
        self.assertIsNone(err)
        names = B.extract(raw, self.dest)
        self.assertIn("requests_summary.json", names)
        self.assertIn("papermill.log", names)
        self.assertFalse(any(n.startswith("venv/") for n in names))
        for rel in ("artifacts/g1-abc_p0_events.jsonl", "pass2/artifacts/g1-abc_p0_events.jsonl"):
            a, b = os.path.join(self.root, rel), os.path.join(self.dest, rel)
            self.assertEqual(ab_metrics.game_from_events(a), ab_metrics.game_from_events(b))
            load = lambda p: [json.loads(l) for l in open(p) if l.strip().startswith("{")]
            self.assertEqual(D.level_actions(load(a)), D.level_actions(load(b)))
        red = open(os.path.join(self.dest, "artifacts/g1-abc_p0_events.jsonl")).read()
        self.assertNotIn("frame", red)
        self.assertNotIn("xxxx", red)
        summ = json.load(open(os.path.join(self.dest, "requests_summary.json")))
        r = summ["g1-abc_p0_requests.jsonl"]
        self.assertEqual(r["events"], {"request": 1, "response": 2})
        self.assertEqual(r["usage"], {"prompt_tokens": 15, "completion_tokens": 3})

    def test_log_prefixes_and_duplicates_tolerated(self):
        def mangle(ls):
            out = ["2026-10-27T07:00:00Z FI939 " + l for l in ls]
            return out[:3] + out[1:]
        raw, err = self.roundtrip(mangle)
        self.assertIsNone(err)

    def test_missing_chunk_and_bad_sha_and_no_end(self):
        buf = io.StringIO(); B.emit(self.root, out=buf)
        ls = buf.getvalue().splitlines()
        self.assertTrue(len(ls) >= 3)
        self.assertEqual(B.parse(ls[:1] + ls[2:])[1].split()[0], "chunks")
        bad = ls[:]; i = bad[1].rfind(" ") + 1; bad[1] = bad[1][:i] + ("A" if bad[1][i] != "A" else "B") + bad[1][i + 1:]
        self.assertEqual(B.parse(bad)[1], "sha256 mismatch")
        self.assertEqual(B.parse(ls[:-1])[1], "no END marker")
        self.assertEqual(B.parse(["nothing"])[1], "no bundle in log")

    def test_unsafe_tar_path_refused(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            ti = tarfile.TarInfo("../evil.txt"); ti.size = 1; t.addfile(ti, io.BytesIO(b"x"))
        with self.assertRaises(ValueError):
            B.extract(buf.getvalue(), self.dest)

    def test_too_large_drops_log_then_skips(self):
        old = B.MAX_B64
        try:
            B.MAX_B64 = 10
            buf = io.StringIO()
            self.assertEqual(B.emit(self.root, out=buf), 1)
            self.assertIn("SKIP", buf.getvalue())
        finally:
            B.MAX_B64 = old


try:   # the fetch side lives in the private job launcher; the published layout carries only the bundle module
    import hf_jobs_loadsim as _H  # noqa: F401
    _HAVE_LAUNCHER = True
except ImportError:
    _HAVE_LAUNCHER = False


@unittest.skipUnless(_HAVE_LAUNCHER, "hf_jobs_loadsim not present (published layout)")
class FetchFallback(unittest.TestCase):
    def test_fetch_from_log_rebuilds(self):
        import hf_jobs_loadsim as H
        root, dest = tempfile.mkdtemp(), tempfile.mkdtemp()
        try:
            write_events(os.path.join(root, "artifacts", "g2-x_p0_events.jsonl"), EVENTS)
            buf = io.StringIO(); B.emit(root, out=buf)
            log = ["FI939 07:00:00 EXIT rc=1\n"] + buf.getvalue().splitlines(True)
            hf = types.SimpleNamespace(fetch_job_logs=lambda job_id: iter(log))
            self.assertEqual(H.fetch_from_log({"job_id": "j1"}, dest, hf), 1)
            self.assertTrue(os.path.exists(os.path.join(dest, "artifacts", "g2-x_p0_events.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(dest, "job_console.log")))
            hf2 = types.SimpleNamespace(fetch_job_logs=lambda job_id: iter(["no bundle\n"]))
            self.assertEqual(H.fetch_from_log({"job_id": "j1"}, dest, hf2), 0)
        finally:
            shutil.rmtree(root); shutil.rmtree(dest)

    def test_embed_rejects_terminator(self):
        import hf_jobs_loadsim as H
        self.assertNotIn("\nFI939PY\n", "\n" + open(os.path.join(H.HERE, "hf_results_bundle.py")).read() + "\n")


if __name__ == "__main__":
    unittest.main()
