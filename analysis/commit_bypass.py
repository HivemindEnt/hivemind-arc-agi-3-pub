#!/usr/bin/env python3
"""FI-939 DUCK 38cg: commit-run bypass for M2-family notebooks (leaderboard-only arms).

Why (38cf): a Kaggle commit run of the M2 notebook serves the model and plays the public games for hours, all on our
30 h weekly GPU quota. Scored reruns ran while that quota was exhausted (09-26), so only the commit run is the cost.
For an arm judged only on its hidden score, the commit run's offline play buys nothing. This transform keeps the
scored rerun byte-for-byte the same and makes the commit run skip the model:

  launcher cell  (starts SGLang; marker LAUNCHER)   -> if TRUE_SUBMISSION: exec(<original source>) else: skip
  census cell    (system monitor; marker CENSUS)    -> same
  bench cell     (starts with BENCH; has top-level await) -> original body indented under `if TRUE_SUBMISSION:`;
                                                       else writes the one-row placeholder submission.parquet
  cells after the bench cell                         -> exec-wrapped under TRUE_SUBMISSION

exec(compile(src, name, "exec"), globals()) runs the original text unchanged in the notebook namespace. The bench cell
cannot be exec-wrapped (top-level await), so it is indented instead; the transform refuses if that cell contains a
triple-quoted string, where indentation would change the string's content.

Do NOT use this for loadsim or targeted A/B commit runs: those runs exist to produce the offline traces.

Usage: commit_bypass.py IN.ipynb OUT.ipynb
"""
import json, sys

BENCH = "print('Starting benchmark...')"
LAUNCHER = "# Paste this entire file into the launcher cell."
CENSUS = "# --- system census"
PLACEHOLDER = (
    "    import pandas as _fi939_pd\n"
    "    _fi939_pd.DataFrame([[\"1_0\", \"1\", True, 1]], columns=[\"row_id\", \"game_id\", \"end_of_game\", \"score\"]"
    ").to_parquet(WORKING_DIR / \"submission.parquet\", index=False)\n"
    "    print('[FI939-BYPASS] commit run: model not started; placeholder submission.parquet written', flush=True)\n")
MARK = "# --- FI-939 commit bypass (DUCK 38cg)"


def _src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _set(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


def _has_magic(text):
    """IPython-only syntax (shell escapes, magics) cannot go through exec()."""
    return any(ln.lstrip().startswith(("!", "%")) for ln in text.splitlines())


def exec_wrap(text, name):
    if _has_magic(text):
        raise ValueError("%s cell uses IPython magics or shell escapes; exec() cannot run it" % name)
    return (MARK + ": run only in the scored rerun ---\n"
            "if TRUE_SUBMISSION:\n"
            "    exec(compile(%r, %r, 'exec'), globals())\n"
            "else:\n"
            "    print('[FI939-BYPASS] skipped %s in the commit run', flush=True)\n" % (text, "<%s>" % name, name))


def indent_wrap(text):
    if _has_magic(text):
        raise ValueError("bench cell uses IPython magics or shell escapes")
    if '"""' in text or "'''" in text:
        raise ValueError("bench cell has a triple-quoted string; indenting would change it")
    body = "".join(("    " + ln) if ln.strip() else ln for ln in text.splitlines(keepends=True))
    if not body.endswith("\n"):
        body += "\n"
    return (MARK + ": the scored rerun runs the original benchmark unchanged ---\n"
            "if TRUE_SUBMISSION:\n" + body + "else:\n" + PLACEHOLDER)


def apply(nb):
    """Transform in place. Returns a list of (cell index, action)."""
    code = [i for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"]
    if any(MARK in _src(nb["cells"][i]) for i in code):
        raise ValueError("bypass already applied")
    bench = [i for i in code if _src(nb["cells"][i]).startswith(BENCH)]
    launch = [i for i in code if _src(nb["cells"][i]).lstrip().startswith(LAUNCHER)]
    if len(bench) != 1 or len(launch) != 1:
        raise ValueError("expected exactly one bench and one launcher cell, found %d and %d" % (len(bench), len(launch)))
    if not any("TRUE_SUBMISSION" in _src(nb["cells"][i]) and i < launch[0] for i in code):
        raise ValueError("TRUE_SUBMISSION is not defined before the launcher cell")
    done = []
    for i in code:
        src = _src(nb["cells"][i])
        if i == launch[0]:
            _set(nb["cells"][i], exec_wrap(src, "launcher")); done.append((i, "launcher exec-wrapped"))
        elif src.lstrip().startswith(CENSUS):
            _set(nb["cells"][i], exec_wrap(src, "census")); done.append((i, "census exec-wrapped"))
        elif i == bench[0]:
            _set(nb["cells"][i], indent_wrap(src)); done.append((i, "bench indented under TRUE_SUBMISSION"))
        elif i > bench[0]:
            _set(nb["cells"][i], exec_wrap(src, "post-bench cell %d" % i)); done.append((i, "post-bench exec-wrapped"))
    return done


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if len(a) != 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    nb = json.load(open(a[0], encoding="utf-8"))
    for i, what in apply(nb):
        print("BYPASS cell %d: %s" % (i, what))
    with open(a[1], "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
