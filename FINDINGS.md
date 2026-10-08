# Findings (work in progress)

The baseline agent is dfranzen's public Milestone-2 harness running Qwen3.8-Flash-Next. The data are commit-run
traces on the 25 public games, 4 passes per run, in five runs: the baseline (M2), a second baseline-family run
(D'), and the unanchor, ctrlprobe and ctrlprobe-v2 arms. "O/E" means observed over expected under the stated
null. Every result below is labelled as holding, mixed, or withdrawn.

## Method lessons (both from our own mistakes)

- **Count actions from the environment's event log, not from what the agent was shown.** The agent-visible verdict
  lines missed 15-39% of executed actions per run (up to 94% on one pass). One early finding rested on them, and we
  retracted it.
- **"The control changed something later" tests nothing.** Between 73% and 100% of all actions change the board
  in every game. We withdrew an "83% of 'does nothing' claims were later contradicted" figure for this reason.
  Its replacement is the matched null in `claim_null.py`, described below.

## Where the points go

- **Unfinished levels, not inefficiency (holds).** `loss_decompose.py` splits the official score exactly. On the
  two 25-game, 4-pass runs, 97% of the lost points are levels never finished, and 3% are extra actions on finished
  levels: 51.9 vs 1.6 points (unanchor) and 58.6 vs 1.9 (ctrlprobe). On finished levels the agent is about as
  efficient as the human baseline. Arms that only save actions can therefore recover at most a few points; an arm
  has to clear more levels to matter. The first uncleared level per game-pass is L1 in about 10%, L2 in 11-17% and
  L3 or later in 50%; 22-30% of game-passes clear every level.

- **Goal identification is mostly not the problem.** In 5 of 9 hand-read failed episodes the model stated the
  right goal and kept it.
- **Level-boundary anchoring (holds, replicated).** Failed level-2+ attempts appeal to the previous level
  2.0-2.6x more per turn than won ones: 0.098 vs 0.038 appeals per turn on M2, 0.043 vs 0.022 on D'.
- **Late control discovery is not general (negative result).** Within each game, failed L2+ levels introduce the
  game's last control family at 0.24 of the level's actions, against 0.39 on won levels. Failed levels are later
  in only 3 of 11 games. The exceptions are the three games never cleared at level 2 (bp35, lf52, sk48), and each
  of them adds a new mechanic on level 2.

## Per-action effect perception (2026-10-08)

| question | tool | result | status |
|---|---|---|---|
| Do "control X does nothing" statements predict X's later behaviour on the level? | `claim_null.py` | No. Later tries of X change the board at X's usual per-try rate for that level: O/E 1.09 (95% CI 1.07-1.12), 1,151 statements with later tries. This holds for 10/20/40-action windows and in every run. | holds |
| Do present-tense beliefs ("MOUSE is a no-op") change behaviour? | `claim_null.py` | Yes: use of that control drops about 20% (usage ratio 0.78), with no drop in how often it works when tried. About half of the sentences in this class are true durable beliefs on a hand read of 15. | holds, small n (85) |
| Are the agent's announced tests aimed at uncertain controls? | `replay_tests.py` | Yes. Comparing tests with ordinary actions at the same number of prior tries, tests hit controls whose outcome is less predictable, and the outcome surprises more: 0.65 vs 0.29 bits after 6+ tries. | holds |
| After the game contradicts a "does nothing" statement, does the agent revise? | `replay_revise.py` | Mostly yes. It revises in 59% of cases (67% for present-tense beliefs), usually within one turn. 17% restate a no-op, and on a hand read these are mostly accurate statements about a different board position. | holds |
| Earlier claim: "the agent designs good experiments but draws bad conclusions" | | Too strong. The revision test above contradicts it. Most "does nothing" statements are accurate local reports. | withdrawn |
| On a new level, does the agent click object types that just appeared? | `new_object_clicks.py` | Early, but below chance overall. The first such click comes at a median 12% of the level, but these objects get 0.60-0.62 of the clicks their share of the board would predict. bp35: 0.3-0.4. lf52: 0.2. | holds; premise check for the inventory arm |
| When an action does nothing, had that control already worked on the level? | `trackrecord_premise.py` | Usually. 73-82% of no-ops, about 6-11 per game pass, and typically 2 of 3 earlier tries had worked. The agent is not told this today. | holds; premise check for the trackrecord arm |

## Interventions (offline A/B on the 25 public games)

- **unanchor** replaces the level-start prompt, which tells the model to carry the previous level's mechanics
  forward, with requests to re-test controls and state alternative goals.
  - It did what it was written to do: on failed L2+ levels, previous-level appeals fell from 0.098 to 0.034 per
    turn.
  - Levels won rose only from 402 to 414 (+3%, p = 0.82), which is within noise.
- **ctrlprobe** adds a sandbox `probe_controls()` diagnostic.
  - lf52 cleared level 2 in 3 of 4 passes (base: 0 of 4).
  - Overall levels fell 402 to 372 (-7.5%, p = 0.2).
  - An integration defect blocked 6 of 62 probes. The fixed v2 is queued with on-demand guidance only.
- **How much can one run show?** A 25-game, 4-pass run can detect only about a 16% change in levels won at 80%
  power. A 3% effect would need about 120 passes per arm. Targeted arms therefore run on 6 games x 16 passes
  against a base of the same shape, with a pre-registered gate (`targeted_gate.py`).
- **Queued:** ctrlprobe v2, level inventory, trackrecord (`trackrecord.py`), and Apache-licensed model swaps
  (Qwen3.8-27B, gpt-oss-120b, Gemma-4-31B).

## Reading the hidden leaderboard

- Two identical submissions agreeing closely says little about the noise. With one degree of freedom, the
  standard deviation is bounded only loosely, so we size repeats from that bound.
- A public notebook's "best score" is the maximum over everyone who resubmitted that version, so it overstates
  what the code typically scores.

## Earlier work on the Duck harness (failures included)

These results are from Qwen3.6-27B with Tufa Labs' Duck harness and our patches. Each is a single scored
submission.

| change | score |
|---|---|
| Duck v5 | 0.14 |
| with scaffolding (archetype classification and "deterministic first" prompt lines) | 0.34 |
| same patches with that scaffolding removed | 0.15 |
| carry-forward notes memory added to the 0.34 build | 0.04 |

The memory change kept the agent's beliefs reliably. It also kept a confidently wrong world model for over 100
turns in one game, so carried beliefs are only as good as the beliefs themselves.
