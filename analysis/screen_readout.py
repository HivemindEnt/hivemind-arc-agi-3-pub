#!/usr/bin/env python3
"""FI-939: one-command mechanism readout for an HF screen, from events + the bundle's request summary.

Why (2026-10-27, DUCK 38ew): since result uploads can fail (38ev), a screen may come back with only the reduced bundle
(events, papermill.log, requests_summary.json). The gate (fi939_gate.py) and level power (loadsim_power.py) already run
on that. The cost and mechanism checks the pre-registrations ask for (requests per pass, tokens per request, actions per
turn on completed vs unfinished levels, per-game levels, marker counts) were being computed by one-off scratch scripts.
This makes them one tested tool that reads either a full trace dir (request logs) or a bundle dir (summary).

  screen_readout.py BASE_DIR[,BASE_DIR2] ARM_DIR [--marker TEXT ...] [--json OUT]
Prints: per run requests/pass, completion and prompt tokens per request, actions per turn (completed / unfinished),
per-game mean levels (base mean vs arm), marker line counts in papermill.log, and arm-vs-base percentage deltas.
Exit: 0 ok | 2 usage | 4 a run dir has no events
"""
import glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _events(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def level_segments(path):
    """[(completed, actions, turns)] per level attempt-run; a level ends when the score rises."""
    out, a, t, score = [], 0, 0, 0
    for e in _events(path):
        ty = e.get("type") or e.get("event")
        if ty == "analysis":
            t += 1
        elif ty == "action":
            a += 1
            sc = e.get("score")
            if isinstance(sc, (int, float)) and int(sc) > score:
                out.append((True, a, t))
                a = t = 0
                score = int(sc)
    if a or t:
        out.append((False, a, t))
    return out


def max_level(path):
    m = 0
    for e in _events(path):
        s = e.get("score")
        if (e.get("type") or e.get("event")) == "action" and isinstance(s, (int, float)):
            m = max(m, int(s))
    return m


def event_files(d):
    return sorted(glob.glob(os.path.join(d, "**", "*_events.jsonl"), recursive=True))


def request_stats(d):
    """Totals from requests_summary.json (bundle) or, failing that, from the full *_requests.jsonl logs."""
    p = os.path.join(d, "requests_summary.json")
    if os.path.exists(p):
        summ = json.load(open(p, encoding="utf-8"))
        src = "summary"
    else:
        import hf_results_bundle as B
        files = glob.glob(os.path.join(d, "**", "*_requests.jsonl"), recursive=True)
        if not files:
            return None
        summ = {os.path.relpath(f, d): B.summarize_requests(f) for f in files}
        src = "logs"
    ev, us = defaultdict(int), defaultdict(int)
    for v in summ.values():
        for k, n in v.get("events", {}).items():
            ev[k] += n
        for k, n in v.get("usage", {}).items():
            us[k] += n
    resp = ev.get("response", 0) or 1
    return {"source": src, "files": len(summ), "requests": ev.get("request", 0), "responses": ev.get("response", 0),
            "completion_per_req": us.get("completion_tokens", 0) / resp, "prompt_per_req": us.get("prompt_tokens", 0) / resp}


def run_stats(d, markers=()):
    fs = event_files(d)
    if not fs:
        return None
    c, u = [0, 0], [0, 0]
    games = defaultdict(list)
    for p in fs:
        for done, a, t in level_segments(p):
            x = c if done else u
            x[0] += a
            x[1] += t
        games[os.path.basename(p).split("-")[0]].append(max_level(p))
    rs = request_stats(d)
    out = {"dir": d, "passes": len(fs), "apt_completed": c[0] / max(c[1], 1), "apt_unfinished": u[0] / max(u[1], 1),
           "games": {g: sum(v) / len(v) for g, v in sorted(games.items())}}
    if rs:
        out.update(rs)
        out["requests_per_pass"] = rs["requests"] / len(fs)
    pl = os.path.join(d, "papermill.log")
    if markers and os.path.exists(pl):
        txt = open(pl, encoding="utf-8", errors="replace").read()
        out["markers"] = {m: txt.count(m) for m in markers}
    return out


def pct(a, b):
    return 100.0 * (a - b) / b if b else float("nan")


def compare(bases, arm):
    keys = ["requests_per_pass", "completion_per_req", "prompt_per_req", "apt_completed", "apt_unfinished"]
    res = {}
    for k in keys:
        bv = [b[k] for b in bases if k in b]
        if bv and k in arm:
            bm = sum(bv) / len(bv)
            res[k] = {"base": bm, "arm": arm[k], "pct": pct(arm[k], bm)}
    gb = defaultdict(list)
    for b in bases:
        for g, v in b["games"].items():
            gb[g].append(v)
    res["games"] = {g: {"base": sum(v) / len(v), "arm": arm["games"].get(g)} for g, v in sorted(gb.items())}
    return res


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    base_dirs = [os.path.expanduser(x) for x in argv[0].split(",")]
    arm_dir = os.path.expanduser(argv[1])
    markers, out_json, i = [], None, 2
    while i < len(argv):
        if argv[i] == "--marker" and i + 1 < len(argv):
            markers.append(argv[i + 1]); i += 2
        elif argv[i] == "--json" and i + 1 < len(argv):
            out_json = argv[i + 1]; i += 2
        else:
            print("unknown arg", argv[i]); return 2
    runs = [run_stats(d, markers) for d in base_dirs + [arm_dir]]
    if any(r is None for r in runs):
        print("SCREEN_READOUT_ERR a run dir has no events:", [d for d, r in zip(base_dirs + [arm_dir], runs) if r is None])
        return 4
    bases, arm = runs[:-1], runs[-1]
    cmp_ = compare(bases, arm)
    for r in runs:
        print("%-40s passes=%3d req/pass=%6.1f compl/req=%6.0f prompt/req=%7.0f apt_done=%.2f apt_unfin=%.2f%s" % (
            os.path.basename(r["dir"].rstrip("/\\")), r["passes"], r.get("requests_per_pass", float("nan")),
            r.get("completion_per_req", float("nan")), r.get("prompt_per_req", float("nan")), r["apt_completed"],
            r["apt_unfinished"], ("  markers=%s" % r["markers"]) if r.get("markers") else ""))
    for k, v in cmp_.items():
        if k != "games":
            print("DELTA %-20s base %9.2f arm %9.2f  %+6.1f%%" % (k, v["base"], v["arm"], v["pct"]))
    print("per game (base mean -> arm):")
    for g, v in cmp_["games"].items():
        a = v["arm"]
        flag = "" if a is None else ("  +" if a - v["base"] >= 1 else ("  -" if v["base"] - a >= 1 else ""))
        print("  %-6s %5.2f -> %s%s" % (g, v["base"], "  n/a" if a is None else "%5.2f" % a, flag))
    if out_json:
        with open(out_json, "w") as f:
            json.dump({"runs": runs, "compare": cmp_}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
