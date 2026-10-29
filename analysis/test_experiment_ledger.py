#!/usr/bin/env python3
"""Tests for experiment_ledger.py (FI-939). Stdlib only. Exit 0 = all pass."""
import json
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import experiment_ledger as L  # noqa: E402

FAILS = []
PASSES = 0


def check(name, cond):
    global PASSES
    if cond:
        PASSES += 1
    else:
        FAILS.append(name)
        print("FAIL:", name)


def rec(date, score, sid, desc="d", status="SubmissionStatus.COMPLETE", slug="duck-harness-kaggle"):
    return {"date": date, "public_score": score, "status": status, "description": desc,
            "url": "/code/hivemindadmin/%s?scriptVersionId=%s" % (slug, sid), "ref": sid + "r"}


# 1. JSON extraction survives kaggle's warning lines before the array
raw = "Warning: outdated kaggle\nWarning: again\n" + json.dumps([rec("2026-09-24 08:00:00", "0.34", "111")])
got = L.extract_records(raw)
check("extract_after_warnings", len(got) == 1 and got[0]["public_score"] == "0.34")
check("extract_empty_on_garbage", L.extract_records("no json here") == [])

# 2. normalization
e = L.normalize(rec("2026-09-24 08:00:00", "0.34", "352293386", desc="Duck v6b: x"))
check("norm_kernel", e["kernel"] == "duck-harness-kaggle")
check("norm_script", e["script_version"] == "352293386")
check("norm_score", e["score"] == 0.34)
check("norm_label_from_desc", e["label"] == "Duck v6b")
pending = L.normalize(rec("2026-09-26 01:00:00", "", "5", status="SubmissionStatus.PENDING"))
check("norm_pending_score_none", pending["score"] is None and pending["scored"] is False)
err = L.normalize(rec("2026-09-26 01:00:00", "", "6", status="SubmissionStatus.ERROR"))
check("norm_error_not_scored", err["scored"] is False)
nourl = L.normalize({"date": "2026-09-01", "public_score": "0.1", "status": "SubmissionStatus.COMPLETE",
                     "description": "x", "url": ""})
check("norm_missing_url", nourl["script_version"] == "" and nourl["kernel"] == "")

# 3. labels override
e2 = L.normalize(rec("2026-09-24", "0.34", "352293386"), labels={"352293386": "v8/v6b"})
check("label_override", e2["label"] == "v8/v6b")

# 4. grouping + stats
recs = [rec("2026-09-20 00:55:00", "0.12", "351183033", desc="v13: a"),
        rec("2026-09-21 06:00:00", "0.11", "351183033", desc="daily b"),
        rec("2026-09-24 08:00:00", "0.34", "352293386", desc="Duck v6b: c"),
        rec("2026-09-26 01:00:00", "", "999", status="SubmissionStatus.PENDING")]
groups = L.group([L.normalize(r) for r in recs])
g = {x["script_version"]: x for x in groups}
check("group_count", len(groups) == 3)
check("repeat_n", g["351183033"]["n"] == 2)
check("repeat_mean", abs(g["351183033"]["mean"] - 0.115) < 1e-9)
check("repeat_range", abs(g["351183033"]["range"] - 0.01) < 1e-9)
check("repeat_label_earliest", g["351183033"]["label"] == "v13")
check("single_range_none", g["352293386"]["range"] is None)
check("pending_counted_separately", g["999"]["n"] == 0 and g["999"]["pending"] == 1 and g["999"]["mean"] is None)
check("sorted_by_first_date", [x["script_version"] for x in groups] == ["351183033", "352293386", "999"])

# 5. noise floor + compare
nf = L.noise_floor(groups)
check("noise_floor_value", abs(nf["max_range"] - 0.01) < 1e-9 and nf["groups_with_repeats"] == 1)
check("noise_floor_none_without_repeats", L.noise_floor([g["352293386"]])["max_range"] is None)
v = L.compare(g["352293386"], g["351183033"], nf)
check("compare_delta", abs(v["delta"] - 0.225) < 1e-9)
check("compare_exceeds", v["verdict"] == "exceeds_observed_spread")
small = {"mean": 0.12, "n": 1, "script_version": "a", "label": "a"}
v2 = L.compare(small, g["351183033"], nf)
check("compare_within", v2["verdict"] == "within_observed_spread")
v3 = L.compare(small, g["351183033"], {"max_range": None, "groups_with_repeats": 0})
check("compare_unknown_without_floor", v3["verdict"] == "no_variance_data")
v4 = L.compare(g["999"], g["351183033"], nf)
check("compare_unscored", v4["verdict"] == "unscored")

# 6. render is text and mentions each group
txt = L.render(groups, nf)
check("render_has_rows", "351183033" in txt and "352293386" in txt and "n=2" in txt)
check("render_caveat_small_n", "suggestive" in txt)

# 7. CLI from a file, JSON mode, exit 0; bad file exit 2
fx = os.path.join(HERE, "_ledger_fixture.json")
with open(fx, "w") as f:
    f.write("Warning: x\n" + json.dumps(recs))
p = subprocess.run([sys.executable, os.path.join(HERE, "experiment_ledger.py"), "--from-file", fx, "--json"],
                   capture_output=True, text=True)
check("cli_rc0", p.returncode == 0)
try:
    out = json.loads(p.stdout)
    check("cli_json_groups", len(out["groups"]) == 3 and "noise_floor" in out)
except Exception:
    check("cli_json_parse", False)
p2 = subprocess.run([sys.executable, os.path.join(HERE, "experiment_ledger.py"), "--from-file", fx + ".missing"],
                    capture_output=True, text=True)
check("cli_missing_file_rc2", p2.returncode == 2)
os.remove(fx)

# relative noise floor + arms vs base (M2 era)
def _g(kernel, scores, date):
    return {"kernel": kernel, "script_version": kernel + "1", "label": kernel, "n": len(scores), "pending": 0,
            "scores": scores, "mean": sum(scores) / len(scores) if scores else None,
            "min": min(scores) if scores else None, "max": max(scores) if scores else None,
            "range": (max(scores) - min(scores)) if len(scores) >= 2 else None, "first_date": date, "last_date": date}
G = [_g("old", [4.23, 2.75], "2026-10-02"), _g("m2", [25.54], "2026-10-04"), _g("arm", [27.0], "2026-10-06")]
nfr = L.noise_floor(G)
check("relative floor kept beside the absolute one", abs(nfr["max_rel_range"] - 1.48 / 3.49) < 1e-9 and abs(nfr["max_range"] - 1.48) < 1e-9)
h, rows = L.vs_base(G, "m2", nfr)
check("vs_base: older kernels excluded, base excluded", [r["kernel"] for r in rows] == ["arm"])
check("vs_base: threshold = max(abs, rel x base mean)", abs(h["threshold"] - (1.48 / 3.49) * 25.54) < 1e-9)
check("vs_base: one sample -> needs_samples", rows[0]["verdict"] == "needs_samples")
G2 = [_g("old", [4.23, 2.75], "2026-10-02"), _g("m2", [25.54, 24.0], "2026-10-04"), _g("arm", [40.0, 41.0], "2026-10-06"),
      _g("arm2", [26.0, 25.0], "2026-10-07")]
h2, rows2 = L.vs_base(G2, "m2", L.noise_floor(G2))
check("vs_base: big gap with n>=2 -> better; small -> within_noise",
      [r["verdict"] for r in rows2] == ["better", "within_noise"])
check("vs_base: a repeated base uses its own spread", abs(h2["threshold"] - 1.54) < 1e-9)
# external identical-config samples widen the spread only
hx, rx = L.vs_base(G, "m2", nfr, external=[27.89, 27.80])
check("external: threshold = spread of own + external", abs(hx["threshold"] - max(nfr["max_range"] or 0, 27.89 - 25.54)) < 1e-9
      and hx["external_n"] == 2)
check("external: base mean and n unchanged", hx["mean"] == h["mean"] and hx["n"] == h["n"])
check("external: arm still needs its own samples", rx[0]["verdict"] == "needs_samples")
import tempfile as _t, json as _j, os as _o
_d = _t.mkdtemp(); _p = _o.path.join(_d, "ext.json")
_j.dump({"m2": {"scores": [27.89, "27.80"]}, "_doc": "x"}, open(_p, "w"))
check("load_external: reads and coerces", L.load_external(_p, "m2") == [27.89, 27.8])
check("load_external: missing kernel/file -> []", L.load_external(_p, "zz") == [] and
      L.load_external(_o.path.join(_d, "no.json"), "m2") == [])
open(_p, "w").write("{bad")
check("load_external: corrupt -> []", L.load_external(_p, "m2") == [])
_repo = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "duck_external_samples.json")
check("repo external file: M2 entry present with evidence", L.load_external(_repo, "duck-franzen-m2") == []
      and "_retired_2026-10-05" in _j.load(open(_repo)))
# power / samples needed
M2S = [25.54, 27.62, 27.80, 27.89, 31.47]
sd5 = L.stdev(M2S)
check("stdev: sample sd of the 5 identical M2 runs", abs(sd5 - 2.137) < 0.005)
check("stdev: < 2 samples -> None", L.stdev([1.0]) is None)
check("samples_needed: ~2*(2.8*sd/d)^2", L.samples_needed(sd5, 2.0) == 18 and L.samples_needed(sd5, 4.0) == 5)
import tempfile as _tf
_rp = _tf.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump({"references": {"k": [{"score": 34.3, "who": "x V1", "seen": "2026-10-08"}, {"score": 31.06, "who": "x V3"}]}}, _rp)
_rp.close()
_refs = L.load_references(_rp.name, "k")
check("load_references reads score/who/seen", len(_refs) == 2 and _refs[0]["score"] == 34.3 and _refs[1]["seen"] == "?")
check("load_references missing kernel/file -> []", L.load_references(_rp.name, "zz") == [] and L.load_references("/nope.json", "k") == [])
_g = L.reference_gap([25.54, 25.51], _refs)
check("reference_gap mean difference and range", abs(_g[0] - (32.68 - 25.525)) < 1e-9 and _g[1] == 31.06 and _g[2] == 34.3)
check("reference_gap none without data", L.reference_gap([], _refs) is None and L.reference_gap([1.0], []) is None)
check("sd_upper n=2 is ~31.9x the estimate", abs(L.sd_upper(1.0, 2) - 31.91) < 0.05)
check("sd_upper n=11 from table (df 10)", abs(L.sd_upper(1.0, 11) - (10 / 3.2470) ** 0.5) < 1e-9)
check("sd_upper n=31 Wilson-Hilferty close to exact 1.336", abs(L.sd_upper(1.0, 31) - 1.336) < 0.01)
check("sd_upper shrinks toward the estimate as n grows", L.sd_upper(1.0, 3) > L.sd_upper(1.0, 6) > L.sd_upper(1.0, 31) > 1.0)
check("sd_upper bad input None", L.sd_upper(None, 5) is None and L.sd_upper(1.0, 1) is None)
check("samples_needed: floor 2, bad input None", L.samples_needed(0.1, 10) == 2 and L.samples_needed(None, 2) is None
      and L.samples_needed(2, 0) is None)
check("vs_base: unknown base -> None", L.vs_base(G2, "nope", L.noise_floor(G2))[0] is None)

# 38cl pooled within-version sd
PG = [{"kernel": "a", "scores": [25.51, 25.54], "mean": 25.525},
      {"kernel": "b", "scores": [28.49, 23.63], "mean": 26.06},
      {"kernel": "c", "scores": [2.75, 4.23], "mean": 3.49},          # other regime: excluded
      {"kernel": "d", "scores": [25.96], "mean": 25.96}]              # single score: excluded
_ps = L.pooled_sd(PG, 25.52)
check("pooled_sd uses same-regime repeats only", _ps is not None and _ps[1] == 2 and [k for k, _ in _ps[2]] == ["a", "b"])
check("pooled_sd value", _ps is not None and abs(_ps[0] - ((0.00045 + 11.8098) / 2) ** 0.5) < 1e-3)
check("pooled_sd none without repeats", L.pooled_sd([PG[3]], 25.0) is None and L.pooled_sd(PG, 0) is None)

# FI-939 38eg: a 0.00 COMPLETE draw is a failed run, not a policy sample
_z = L.normalize(rec("2026-10-22 01:05:00", "0.00", "354930583"))
check("zero draw is failed, not scored", _z["failed"] is True and _z["scored"] is False and _z["score"] is None)
check("low nonzero draw is still scored", L.normalize(rec("2026-09-24 08:00:00", "0.04", "1"))["scored"] is True)
check("error draw is neither scored nor failed", L.normalize(rec("2026-10-20 01:05:00", "", "2", status="SubmissionStatus.ERROR"))["failed"] is False)
_ZG = L.group([L.normalize(rec("2026-10-%02d 01:05:00" % d, s, "354930583")) for d, s in
               ((16, "27.16"), (17, "29.10"), (22, "0.00"), (23, "24.47"))])
check("zero excluded from mean/range, counted as failed", _ZG[0]["n"] == 3 and _ZG[0]["failed"] == 1 and _ZG[0]["pending"] == 0
      and abs(_ZG[0]["range"] - (29.10 - 24.47)) < 1e-9 and _ZG[0]["min"] == 24.47)
check("noise floor not inflated by a failed run", abs(L.noise_floor(_ZG)["max_range"] - 4.63) < 1e-9)
check("render reports failed runs", "Failed scored runs (0.00" in L.render(_ZG, L.noise_floor(_ZG)) and "+1 failed" in L.render(_ZG, L.noise_floor(_ZG)))

# paging (moved from finals_plan, 2026-10-24)
class _S:
    def __init__(self, ref):
        self.ref = ref
_pages = {1: [_S(i) for i in range(3)], 2: [_S(i) for i in range(3, 6)], 3: [_S(6)]}
_calls = []
_got = L.collect_pages(lambda n, k: _calls.append(n) or _pages.get(n, []), page_size=3)
check("collect_pages reads until a short page", [x.ref for x in _got] == list(range(7)) and _calls == [1, 2, 3])
check("collect_pages stops when a page repeats", len(L.collect_pages(lambda n, k: [_S(1), _S(2)], page_size=2)) == 2)
try:
    L.collect_pages(lambda n, k: [_S((n, 0)), _S((n, 1))], page_size=2, max_pages=3)
    check("collect_pages refuses to truncate silently", False)
except RuntimeError:
    check("collect_pages refuses to truncate silently", True)
_n = L.normalize(dict(rec("2026-10-17 01:05:00", "29.10", "354930583"), ref="57289794", total_bytes="3814"))
check("normalize keeps ref and bytes", _n["ref"] == "57289794" and _n["bytes"] == "3814")

print("experiment_ledger tests: %d passed, %d failed" % (PASSES, len(FAILS)))
sys.exit(1 if FAILS else 0)
