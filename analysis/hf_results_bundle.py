"""Results fallback for HF screens: carry reduced results through the job's console log.

Why: result uploads go to presigned S3 PUT URLs. Since 2026-10-26 pc-2 signs them with STS credentials that the
fleet broker caps at 1 h, so any PUT after ~1 h fails with ExpiredToken (harn-levelpin-r2 lost its data this way).
The HF job console log is always retrievable with fetch_job_logs, and a 2026-10-27 CPU test carried 1.6 MB of
base64 through it intact (400 chunks, sha256 match). So the job also prints a reduced bundle to stdout, and fetch
rebuilds the trace directory from the log when the S3 objects are missing.

What the bundle keeps (enough for fi939_gate.py, loadsim_power.py and ab_metrics.game_from_events, which only read
these fields):
  artifacts/*_events.jsonl  -> one line per event, keys limited to KEEP_KEYS (+ the nested result bools)
  papermill.log             -> as is (gzip in the tar)
  requests_summary.json     -> per *_requests.jsonl file: line counts per event value and summed usage tokens
Full request logs are not carried (100+ MB); readouts that need message bodies need the S3 path.

  emit:   python3 hf_results_bundle.py emit <working_dir>       (stdout: FI939BUNDLE BEGIN / FI939B64 n / END)
  decode: python3 hf_results_bundle.py decode <log_file> <dest> (rebuilds files under dest)
"""
import base64, hashlib, io, json, os, sys, tarfile

KEEP_KEYS = ("type", "event", "action_name", "score", "game_over", "board_changed", "level_completed",
             "run_complete", "levels_completed", "state", "analysis_step", "action")
RESULT_KEYS = ("board_changed", "game_over", "level_completed", "run_complete", "score")
CHUNK = 4000
MAX_B64 = 6_000_000  # ~1.5x the size tested; above this the bundle drops papermill.log, then refuses
TAG_BEGIN, TAG_CHUNK, TAG_END = "FI939BUNDLE BEGIN", "FI939B64", "FI939BUNDLE END"


def reduce_event(e):
    if not isinstance(e, dict):
        return None
    out = {k: e[k] for k in KEEP_KEYS if k in e and not isinstance(e[k], (dict, list))}
    for nk in ("result", "last_action_result"):
        r = e.get(nk)
        if isinstance(r, dict):
            sub = {k: r[k] for k in RESULT_KEYS if k in r and not isinstance(r[k], (dict, list))}
            if sub:
                out[nk] = sub
    return out


def reduce_events_file(path):
    lines = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = reduce_event(json.loads(line))
            except Exception:
                continue
            if r is not None:
                lines.append(json.dumps(r, separators=(",", ":")))
    return ("\n".join(lines) + "\n") if lines else ""


def _usage(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("prompt_tokens", "completion_tokens", "total_tokens") and isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0) + int(v)
            elif isinstance(v, dict):
                _usage(v, acc)


def summarize_requests(path):
    counts, usage, n = {}, {}, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            n += 1
            ev = str(e.get("event") or e.get("type") or "?")
            counts[ev] = counts.get(ev, 0) + 1
            u = e.get("usage")
            if not isinstance(u, dict) and isinstance(e.get("response"), dict):
                u = e["response"].get("usage")
            if isinstance(u, dict):
                _usage(u, usage)
    return {"lines": n, "events": counts, "usage": usage}


def _walk(root, suffix):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("venv", "site-packages", ".venv", "__pycache__")]
        for fn in sorted(files):
            if fn.endswith(suffix):
                yield os.path.join(d, fn)


def build(root, with_log=True):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        def add(name, data):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = 0
            tar.addfile(ti, io.BytesIO(data))
        for p in _walk(root, "_events.jsonl"):
            add(os.path.relpath(p, root).replace(os.sep, "/"), reduce_events_file(p).encode())
        summ = {os.path.relpath(p, root).replace(os.sep, "/"): summarize_requests(p) for p in _walk(root, "_requests.jsonl")}
        add("requests_summary.json", json.dumps(summ, indent=1, sort_keys=True).encode())
        pl = os.path.join(root, "papermill.log")
        if with_log and os.path.exists(pl):
            with open(pl, "rb") as f:
                add("papermill.log", f.read())
    return buf.getvalue()


def encode(root):
    raw = build(root)
    b = base64.b64encode(raw).decode()
    if len(b) > MAX_B64:
        b = base64.b64encode(build(root, with_log=False)).decode()
    if len(b) > MAX_B64:
        return None
    return b


def emit(root, out=sys.stdout):
    b = encode(root)
    if b is None:
        print("FI939BUNDLE SKIP too large", file=out, flush=True)
        return 1
    print("%s %s %d" % (TAG_BEGIN, hashlib.sha256(b.encode()).hexdigest(), len(b)), file=out, flush=True)
    for i in range(0, len(b), CHUNK):
        print("%s %d %s" % (TAG_CHUNK, i // CHUNK, b[i:i + CHUNK]), file=out, flush=True)
    print(TAG_END, file=out, flush=True)
    return 0


def parse(lines):
    """Return (bytes, error). Tolerates log prefixes before the tag and duplicate lines."""
    want = size = None
    chunks, ended = {}, False
    for l in lines:
        l = l.rstrip("\r\n")
        i = l.find("FI939B")
        if i < 0:
            continue
        l = l[i:]
        if l.startswith(TAG_BEGIN + " "):
            parts = l.split()
            want, size, chunks, ended = parts[2], int(parts[3]), {}, False
        elif l.startswith(TAG_CHUNK + " ") and want:
            _, n, data = l.split(" ", 2)
            chunks[int(n)] = data.strip()
        elif l.startswith(TAG_END) and want:
            ended = True
    if not want:
        return None, "no bundle in log"
    expected = (size + CHUNK - 1) // CHUNK
    if len(chunks) != expected or set(chunks) != set(range(expected)):
        return None, "chunks %d/%d" % (len(chunks), expected)
    b = "".join(chunks[k] for k in range(expected))
    if hashlib.sha256(b.encode()).hexdigest() != want:
        return None, "sha256 mismatch"
    if not ended:
        return None, "no END marker"
    return base64.b64decode(b), None


def extract(raw, dest):
    os.makedirs(dest, exist_ok=True)
    real = os.path.realpath(dest)
    names = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            target = os.path.realpath(os.path.join(dest, m.name))
            if not target.startswith(real + os.sep):
                raise ValueError("unsafe path in bundle: %r" % m.name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tar.extractfile(m) as src, open(target, "wb") as f:
                f.write(src.read())
            names.append(m.name)
    return names


def main(argv):
    if len(argv) >= 2 and argv[0] == "emit":
        return emit(argv[1])
    if len(argv) >= 3 and argv[0] == "decode":
        with open(argv[1], encoding="utf-8", errors="replace") as f:
            raw, err = parse(f)
        if err:
            print("BUNDLE_ERR", err); return 1
        print("BUNDLE_OK files=%d" % len(extract(raw, argv[2]))); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
