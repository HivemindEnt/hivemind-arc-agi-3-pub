#!/usr/bin/env python3
"""FI-939 experiment ledger: group every scored Kaggle submission by the exact kernel
script version it came from, and report n / mean / range per version.

WHY: each version was scored once, so 0.34 vs 0.15 vs 0.04 could partly be rerun noise.
The daily tick now resubmits unchanged kernels (REPEAT-SAMPLE) to measure that noise. This
tool reads the scored history (READ-ONLY; never submits) and says whether a gap between two
versions exceeds the largest spread any single unchanged version has shown so far.

Grouping key: (kernel slug, scriptVersionId) parsed from each submission's URL, which
identifies the exact notebook version that was scored. Descriptions are free text and are
used only for a display label.

Usage:
  experiment_ledger.py                       # live, via the kaggle API (read-only)
  experiment_ledger.py --from-file dump.json # offline, from a saved submissions dump
  options: --labels labels.json ({scriptVersionId: label}), --json, --compare SID_A SID_B,
           --base KERNEL_SLUG (every later kernel vs that base, with a level-aware threshold)
Exit: 0 ok, 2 input unreadable, 3 live fetch failed.
"""
import argparse
import json
import os
import re
import sys

COMPETITION = "arc-prize-2026-arc-agi-3"
_URL_RE = re.compile(r"/code/[^/]+/([^?]+)\?scriptVersionId=(\d+)")


def extract_records(text):
    """Return the submissions list from text that may carry CLI warning lines first."""
    i = text.find("[")
    while i != -1:
        try:
            val = json.loads(text[i:])
            if isinstance(val, list):
                return val
        except ValueError:
            pass
        i = text.find("[", i + 1)
    return []


def _label_from_desc(desc):
    d = (desc or "").strip()
    head = re.split(r"[:;(]", d, 1)[0].strip()
    return head[:40] if head else "?"


def normalize(r, labels=None):
    url = str(r.get("url") or "")
    m = _URL_RE.search(url)
    kernel, sid = (m.group(1), m.group(2)) if m else ("", "")
    status = str(r.get("status") or "")
    raw = str(r.get("public_score") or "").strip()
    try:
        score = float(raw) if raw and raw.lower() != "none" else None
    except ValueError:
        score = None
    scored = score is not None and status.upper().endswith("COMPLETE")
    # FI-939 38eg: a COMPLETE draw at exactly 0.00 is a failed scored rerun (the 10-22 default and 10-24 serverguard
    # zeros, byte-identical 3,411-byte files), not a sample of the policy. Counted as a score it made the default's
    # spread 29.10 and its sd ~8, so every arm read "within_noise" and the power plan asked for far too many repeats.
    # It is reported as failed and kept out of every mean, spread and sd.
    failed = scored and score == 0.0
    scored = scored and not failed
    label = (labels or {}).get(sid) or _label_from_desc(r.get("description"))
    return {"date": str(r.get("date") or ""), "kernel": kernel, "script_version": sid,
            "score": score if scored else None, "scored": scored, "failed": failed, "status": status,
            "ref": str(r.get("ref") or ""), "bytes": str(r.get("total_bytes") or ""),
            "label": label, "description": str(r.get("description") or "")}


def group(entries):
    by = {}
    for e in entries:
        key = (e["kernel"], e["script_version"])
        by.setdefault(key, []).append(e)
    out = []
    for (kernel, sid), es in by.items():
        es.sort(key=lambda x: x["date"])
        scores = [x["score"] for x in es if x["scored"]]
        n = len(scores)
        failed = sum(1 for x in es if x.get("failed"))
        out.append({
            "kernel": kernel, "script_version": sid, "label": es[0]["label"],
            "n": n, "failed": failed, "pending": len(es) - n - failed,
            "scores": scores,
            "mean": (sum(scores) / n) if n else None,
            "min": min(scores) if n else None,
            "max": max(scores) if n else None,
            "range": (max(scores) - min(scores)) if n >= 2 else None,
            "first_date": es[0]["date"], "last_date": es[-1]["date"],
        })
    out.sort(key=lambda g: g["first_date"])
    return out


def noise_floor(groups):
    ranges = [g["range"] for g in groups if g.get("range") is not None]
    # FI-939 2026-10-04: the only repeats so far sit at 3-4 points; the M2 base scores ~25. An absolute spread
    # measured at one level says little about another, so the relative spread (range / mean) is kept too.
    rel = [g["range"] / g["mean"] for g in groups if g.get("range") is not None and g.get("mean")]
    return {"max_range": max(ranges) if ranges else None, "groups_with_repeats": len(ranges),
            "max_rel_range": max(rel) if rel else None}


def compare(a, b, nf):
    if a.get("mean") is None or b.get("mean") is None:
        return {"delta": None, "verdict": "unscored"}
    delta = a["mean"] - b["mean"]
    if nf.get("max_range") is None:
        verdict = "no_variance_data"
    elif abs(delta) > nf["max_range"] + 1e-12:
        verdict = "exceeds_observed_spread"
    else:
        verdict = "within_observed_spread"
    return {"delta": delta, "verdict": verdict, "a": a.get("script_version"), "b": b.get("script_version")}


def load_external(path, kernel):
    """Scores of runs of the SAME configuration as `kernel` made by others (public leaderboard), from a JSON file
    {"<kernel slug>": {"scores": [..], "evidence": "..."}}. Only used for the base's spread (noise threshold).
    Missing file or kernel -> []."""
    try:
        doc = json.load(open(path))
        return [float(x) for x in ((doc.get(kernel) or {}).get("scores") or [])]
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def load_references(path, kernel):
    """FI-939 38bn: hidden scores of the SAME code run from OTHER accounts, kept apart from noise samples. They are
    never pooled into the base's spread (38s retired them as noise because our own repeats were tight); they are shown
    next to the base so the identical-code gap stays visible. File: {"references": {"<kernel>": [{"score", "who",
    "seen"}]}}. Missing -> []."""
    try:
        doc = json.load(open(path))
        out = []
        for r in ((doc.get("references") or {}).get(kernel) or []):
            out.append({"score": float(r["score"]), "who": str(r.get("who", "?")), "seen": str(r.get("seen", "?"))})
        return out
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return []


def reference_gap(base_scores, refs):
    """(ref mean - base mean, ref min, ref max) or None."""
    if not base_scores or not refs:
        return None
    xs = [r["score"] for r in refs]
    return (sum(xs) / len(xs) - sum(base_scores) / len(base_scores), min(xs), max(xs))


Z_ALPHA, Z_BETA = 1.96, 0.8416  # two-sided 5% test, 80% power


def stdev(xs):
    xs = [float(x) for x in xs]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def samples_needed(sd, delta, z_a=Z_ALPHA, z_b=Z_BETA):
    """Hidden scores needed PER SIDE to detect a true mean difference `delta` between an arm and the base, for
    run-to-run standard deviation `sd` (normal approximation, two-sample, equal n). FI-939 2026-10-04: identical M2
    configs scored 25.54 .. 31.47, so the planning question is how many repeats an arm needs, not whether n=2
    suffices. Returns None if sd or delta is unusable."""
    if not sd or not delta or delta <= 0:
        return None
    import math
    return max(2, math.ceil(2 * ((z_a + z_b) * sd / delta) ** 2))


# Lower 2.5% quantiles of chi-square for df = 1..10 (tables); beyond 10, Wilson-Hilferty.
_CHI2_LO_025 = {1: 0.000982, 2: 0.0506, 3: 0.2158, 4: 0.4844, 5: 0.8312, 6: 1.2373, 7: 1.6899, 8: 2.1797,
                9: 2.7004, 10: 3.2470}


def sd_upper(sd, n):
    """FI-939 38bm: one-sided-in-effect 95% upper bound for a normal sd estimated from n samples (df = n - 1), i.e.
    sd * sqrt(df / chi2_{0.025, df}). With n = 2 the bound is about 32x the estimate: two identical scores do NOT
    show the noise is small, only that it might be. Returns None for unusable input."""
    if sd is None or n is None or n < 2:
        return None
    df = n - 1
    q = _CHI2_LO_025.get(df)
    if q is None:
        z = 1.959964
        q = df * (1 - 2.0 / (9 * df) - z * (2.0 / (9 * df)) ** 0.5) ** 3
    return sd * (df / q) ** 0.5


def pooled_sd(groups, near_mean, rel=0.3):
    """FI-939 38cl: pooled within-version run-to-run sd over every version scored at least twice whose mean lies
    within +-rel of `near_mean` (same score regime; a 3-point version says little about a 25-point one).
    Returns (sd, df, [(kernel, scores)]) or None. Why: the base's two scores agreed to 0.03 and the power plan rested
    on them, then one public kernel scored 28.49 and 23.63 on consecutive days; the base alone hid the noise."""
    if not near_mean:
        return None
    ss, df, used = 0.0, 0, []
    for g in groups:
        xs = g.get("scores") or []
        if len(xs) < 2 or g.get("mean") is None or abs(g["mean"] - near_mean) > rel * abs(near_mean):
            continue
        m = sum(xs) / len(xs)
        ss += sum((x - m) ** 2 for x in xs)
        df += len(xs) - 1
        used.append((g["kernel"], xs))
    if df == 0:
        return None
    return (ss / df) ** 0.5, df, used


def vs_base(groups, base_kernel, nf, min_n=2, external=None):
    """Each kernel scored since the base's first submission vs the base (all versions of each kernel pooled).
    Threshold: once the base has a repeat, the larger of the absolute floor and the base's own spread; before that,
    the larger of the absolute floor and the relative floor x the base mean (conservative: the relative floor comes
    from lower-score repeats). Verdict needs_samples until both sides have >= min_n scores."""
    pooled = {}
    for g in groups:
        p = pooled.setdefault(g["kernel"], {"kernel": g["kernel"], "label": g["label"], "scores": [],
                                             "first_date": g["first_date"]})
        p["scores"].extend(g["scores"])
        p["first_date"] = min(p["first_date"], g["first_date"])
    base = pooled.get(base_kernel)
    if base is None or not base["scores"]:
        return None, []
    bm = sum(base["scores"]) / len(base["scores"])
    # FI-939 2026-10-04: identical-config public reruns (verified with notebook_env_diff) measure the same noise; they
    # widen the spread sample but never move the base mean or count toward the base's own n.
    bs = base["scores"] + list(external or [])
    if len(bs) >= 2:
        # the base has its own repeat: its measured spread at its own level (never below the absolute floor)
        thr = [x for x in (nf.get("max_range"), max(bs) - min(bs)) if x is not None]
    else:
        # no repeat at this level yet: scale the relative floor (conservative, from lower-score repeats)
        thr = [x for x in (nf.get("max_range"), (nf.get("max_rel_range") or 0) * bm if nf.get("max_rel_range") else None)
               if x is not None]
    threshold = max(thr) if thr else None
    rows = []
    for k, p in sorted(pooled.items(), key=lambda kv: kv[1]["first_date"]):
        if k == base_kernel or p["first_date"] < base["first_date"] or not p["scores"]:
            continue
        m = sum(p["scores"]) / len(p["scores"])
        d = m - bm
        if len(p["scores"]) < min_n or len(base["scores"]) < min_n:
            v = "needs_samples"
        elif threshold is None:
            v = "no_variance_data"
        elif abs(d) > threshold:
            v = "better" if d > 0 else "worse"
        else:
            v = "within_noise"
        rows.append({"kernel": k, "label": p["label"], "n": len(p["scores"]), "mean": m, "delta": d, "verdict": v})
    head = {"kernel": base_kernel, "n": len(base["scores"]), "mean": bm, "threshold": threshold,
            "external_n": len(external or [])}
    return head, rows


def _f(x):
    return "-" if x is None else "%.2f" % x


def render(groups, nf):
    lines = ["%-10s %-22s %-24s %-5s %-6s %-11s %s" % ("first", "kernel", "label", "n", "mean", "min-max", "scriptVersionId")]
    for g in groups:
        rng = "%s-%s" % (_f(g["min"]), _f(g["max"])) if g["n"] else ("pending" if g["pending"] else "-")
        lines.append("%-10s %-22s %-24s %-5s %-6s %-11s %s" % (
            g["first_date"][:10], g["kernel"][:22], g["label"][:24], "n=%d" % g["n"], _f(g["mean"]), rng,
            g["script_version"] + (" (+%d failed 0.00)" % g["failed"] if g.get("failed") else "")))
    nfail = sum(g.get("failed", 0) for g in groups)
    if nfail:
        lines.append("Failed scored runs (0.00, excluded from every mean/spread/sd): %d" % nfail)
    if nf["max_range"] is None:
        lines.append("Noise floor: no version has been scored twice yet; every gap is a single-sample comparison.")
    else:
        lines.append("Noise floor: largest spread for an unchanged version = %.2f across %d repeated version(s)."
                     % (nf["max_range"], nf["groups_with_repeats"]))
    lines.append("Caveat: with so few repeats, gaps are suggestive, not conclusive; a floor from n=2 understates true variance.")
    return "\n".join(lines)


def collect_pages(fetch_page, page_size=100, max_pages=20):
    """Read every page of a paged list call. KaggleApi.competition_submissions returns 20 rows per call by default
    and gives no next-page token, so one call silently dropped older draws (8 of the default's 10 on 2026-10-24).
    Stops on a short page or a page with nothing new; refuses (RuntimeError) to truncate past max_pages."""
    seen, out = set(), []
    for page in range(1, max_pages + 1):
        batch = [x for x in (fetch_page(page, page_size) or []) if x is not None]
        fresh = [x for x in batch if str(getattr(x, "ref", "")) not in seen]
        for x in fresh:
            seen.add(str(getattr(x, "ref", "")))
        out.extend(fresh)
        if len(batch) < page_size or not fresh:
            return out
    raise RuntimeError("more than %d pages of submissions; raise max_pages" % max_pages)


def fetch_live():
    from kaggle.api.kaggle_api_extended import KaggleApi  # read-only list call
    api = KaggleApi()
    api.authenticate()
    out = []
    for s in collect_pages(lambda n, k: api.competition_submissions(COMPETITION, page_number=n, page_size=k)):
        out.append({k: str(getattr(s, k, "") or "") for k in ("date", "description", "public_score", "status", "url",
                                                              "ref", "total_bytes")})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-file")
    ap.add_argument("--labels")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("SID_A", "SID_B"))
    ap.add_argument("--base", metavar="KERNEL_SLUG", help="judge every later kernel against this base kernel")
    ap.add_argument("--power", nargs="*", type=float, metavar="DELTA",
                    help="with --base: hidden scores needed per side to detect these mean gains (default 1 2 3 4 6)")
    ap.add_argument("--external", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "duck_external_samples.json"),
                    help="identical-config public scores per kernel (base spread only)")
    a = ap.parse_args(argv)
    labels = {}
    try:
        if a.labels:
            with open(a.labels) as f:
                labels = json.load(f)
        if a.from_file:
            with open(a.from_file, encoding="utf-8") as f:
                recs = extract_records(f.read())
        else:
            recs = None
    except (OSError, ValueError) as ex:
        print("LEDGER_INPUT_ERROR: %s" % ex, file=sys.stderr)
        return 2
    if recs is None:
        try:
            recs = fetch_live()
        except Exception as ex:  # network / auth
            print("LEDGER_FETCH_ERROR: %s" % ex, file=sys.stderr)
            return 3
    groups = group([normalize(r, labels) for r in recs])
    nf = noise_floor(groups)
    cmp_out = None
    if a.compare:
        idx = {g["script_version"]: g for g in groups}
        ga, gb = idx.get(a.compare[0]), idx.get(a.compare[1])
        cmp_out = compare(ga or {}, gb or {}, nf)
    if a.json:
        print(json.dumps({"groups": groups, "noise_floor": nf, "compare": cmp_out}, indent=1))
    else:
        print(render(groups, nf))
        if cmp_out:
            print("Compare %s vs %s: delta=%s -> %s" % (a.compare[0], a.compare[1], _f(cmp_out["delta"]), cmp_out["verdict"]))
        if a.base:
            head, rows = vs_base(groups, a.base, nf, external=load_external(a.external, a.base))
            if head is None:
                print("Base %s: no scored submission" % a.base)
            else:
                print("Base %s: n=%d mean=%s; decision threshold %s (base's own spread once repeated,"
                      " else the relative floor x base mean; never below the absolute floor; +%d identical-config public scores)" % (
                          head["kernel"], head["n"], _f(head["mean"]), _f(head["threshold"]), head["external_n"]))
                for r in rows:
                    print("  %-28s n=%d mean=%s delta=%s -> %s" % (r["kernel"][:28], r["n"], _f(r["mean"]),
                                                                 _f(r["delta"]), r["verdict"]))
                refs = load_references(a.external, a.base)
                if refs:
                    own = [x for g in groups if g["kernel"] == a.base for x in g["scores"]]
                    gap = reference_gap(own, refs)
                    print("Reference: identical code run from other accounts (NOT pooled as noise): %s" % ", ".join(
                        "%.2f %s (seen %s)" % (r["score"], r["who"], r["seen"]) for r in refs))
                    if gap:
                        print("  gap: reference mean - our base mean = %+.2f (references %.2f..%.2f, our base n=%d)" % (
                            gap[0], gap[1], gap[2], len(own)))
                if a.power is not None:
                    pool = [x for g in groups if g["kernel"] == a.base for x in g["scores"]] + \
                        load_external(a.external, a.base)
                    sd = stdev(pool)
                    if sd is None:
                        print("Power: need >= 2 base samples (own + identical-config public) for a spread")
                    else:
                        print("Power: base run-to-run sd %.2f from %d samples [%s]" % (
                            sd, len(pool), ", ".join("%.2f" % x for x in sorted(pool))))
                        up = sd_upper(sd, len(pool))
                        print("  95%% upper bound on that sd: %.2f (df=%d) - plan with both columns" % (up, len(pool) - 1))
                        for d in (a.power or [1, 2, 3, 4, 6]):
                            print("  detect +%.1f -> %d hidden scores per side at sd %.2f; %d at the upper bound %.2f "
                                  "[5%% two-sided, 80%% power]" % (d, samples_needed(sd, d), sd,
                                                                   samples_needed(up, d), up))
                    ps = pooled_sd(groups, head["mean"])
                    if ps:
                        psd, pdf, used = ps
                        pup = sd_upper(psd, pdf + 1)
                        print("Power (pooled, 38cl): within-version sd %.2f, df=%d, from %s; 95%% upper bound %.2f" % (
                            psd, pdf, "; ".join("%s [%s]" % (k[:28], ", ".join("%.2f" % x for x in xs)) for k, xs in used),
                            pup))
                        for d in (a.power or [1, 2, 3, 4, 6]):
                            print("  detect +%.1f -> %d hidden scores per side at pooled sd %.2f; %d at the upper bound"
                                  " %.2f" % (d, samples_needed(psd, d), psd, samples_needed(pup, d), pup))
    return 0


if __name__ == "__main__":
    sys.exit(main())
