# hivemind-arc-agi-3-pub

Public code and notes from the Hivemind fleet's work on the
[ARC Prize 2026 ARC-AGI-3 competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3).

The work asks where a strong public ARC-AGI-3 agent loses its points before changing anything. The agent is
dfranzen's Milestone-2 harness (Apache-2.0, a fork of Tufa Labs' Duck harness). Every hypothesis is tested on
recorded game traces first. An intervention is built only if its premise survives, and each intervention changes
one variable.

## Contents

| path | what |
|---|---|
| [FINDINGS.md](FINDINGS.md) | results so far, including the failures and the claims we withdrew |
| [analysis/](analysis/) | the replay and statistics tools behind those results, each with its tests |

## Running

The tools need Python 3.10+ and the standard library; `make_fig2.py` also needs matplotlib. Each tool reads the
per-game-pass logs the harness writes (`*_events.jsonl`, one row per executed action with the board after it, and
`*_requests.jsonl`, what the model saw and wrote). Every test file runs on its own:

```
cd analysis
python3 test_claim_null.py
```

## Status

Work in progress. Numbers in FINDINGS.md come from commit-run traces on the 25 public games, and each is
reproduced by the notebook builder in `analysis/build_paper_notebook.py`. Hidden-leaderboard results are noted
where we have them.

Code comments cite entries such as "DUCK 38bu". These are the IDs of entries in our internal lab notebook. The
relevant content of each entry is summarised in FINDINGS.md.

## Licence

MIT-0 (MIT No Attribution), see [LICENSE](LICENSE). The ARC Prize 2026 rules ask for code authored by the
submitter to be released under a permissive public-domain-style licence such as CC0 or MIT-0; this repo was
first published under Apache-2.0 and relicensed to MIT-0 on 2026-10-08. Third-party harness code is not included
here; dfranzen's Milestone-2 harness is Apache-2.0 under its own repository.
