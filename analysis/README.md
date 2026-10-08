# analysis/

Replay and statistics tools. All of them read recorded traces and need no GPU or model. Each `X.py` has a
`test_X.py` that runs standalone.

| tool | question it answers |
|---|---|
| `duck_score_estimate.py` | Official-formula score estimate from traces: per level min((human/agent actions)^2, 1.15), level-weighted. |
| `loss_decompose.py` | Splits the points lost into unfinished levels vs inefficiency on finished levels; where runs get stuck. |
| `loadsim_power.py` | How large an effect one multi-pass run can detect (pass-to-pass variance, MDE at 80% power). |
| `make_fig2.py` | Figure 2: minimum detectable effect against passes per arm. |
| `targeted_gate.py` | Pre-registered pass/fail reading for a targeted arm against a same-shape base. |
| `control_mix.py` | When each control family first appears within a level, won vs failed levels. |
| `probe_audit.py` | What the `probe_controls()` diagnostic actually executed (clean, partial, blocked). |
| `replay_claims.py` | "Control X does nothing" statements: tries behind each, from the event log. |
| `claim_null.py` | Do those statements predict the control's later behaviour? Matched-null O/E, usage ratio, claim classes. |
| `replay_tests.py` | Are announced tests aimed at uncertain controls? Expected information and surprise, by prior tries. |
| `replay_revise.py` | After the game contradicts a "does nothing" statement, does the agent revise? |
| `new_object_clicks.py` | On a new level, does the agent click object types that just appeared, compared with chance? |
| `ctrlprobe.py` | The `probe_controls()` sandbox diagnostic used by the ctrlprobe arm. |
| `trackrecord_premise.py` | How often a no-op's control had already worked on the level (whether trackrecord's note would be new). |
| `trackrecord.py` | The trackrecord arm: each NO-OP verdict also states the control's record on that level so far. |
| `build_paper_notebook.py` | Builds a notebook that embeds these modules verbatim and reproduces the reported numbers. |
| `build_arm_cell.py` | Builds an intervention ("arm") notebook: inserts one patch cell into a pinned Milestone-2 notebook. Arms: unanchor, ctrlprobe, trackrecord, inventory, knobs6 and others. |
| `commit_bypass.py` | Rewrites a Milestone-2 notebook so the model runs only in Kaggle's scored rerun; the commit run writes a placeholder submission. Saves weekly GPU quota for leaderboard-only arms. |
| `nb_diff.py` | Static diff of two Kaggle notebooks: environment switches, matched cells, unified diff. Used to compare public entries with the base notebook. |

`build_arm_cell.py` and `commit_bypass.py` take a Milestone-2 notebook as input (not included here; it is
public on Kaggle). `ctrlprobe.py` and `trackrecord.py` patch the dfranzen Milestone-2 harness at import time. Their tests fake the
harness, and also check against the real harness source when it is present on disk (set `FI939_M2_SOLVER`).
