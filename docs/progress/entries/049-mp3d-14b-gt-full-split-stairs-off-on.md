# 049 - MP3D v1 val with the 14B in both LLM roles, ground truth, all fixes: stairs off, then stairs on

**Branch:** `feat/objnav-habitat-mp3d-nadav` (worktree branch `copilot/worktree-2026-11-11T10-35-31-03da`)
**Status:** done (both runs complete; the comparison below)
**Roadmap item:** the method on the full official MP3D split under ideal perception, with the larger model the
prompts were written for (entry 043 ran the 3B), and the first measurement of the multi-storey search on MP3D.

## Goal
Re-run the complete `objectnav_mp3d_v1` val split (2,195 episodes, 11 scenes, 21 categories) under ground-truth
perception with every fix of entries 044-046 in, `qwen2.5:14b-instruct` in both LLM roles, the current defaults
otherwise (stair traversal OFF, the on-object filter ON, the labeller rules OFF), logging what a failure analysis
needs and recording every LLM answer; then the same run with `allow_stair_traversal` TRUE, and the two compared.

## Configuration (frozen in every shard's `run.json`; `preflight.json` at the root)

| | |
|---|---|
| episodes | the published val split, 2,195 in 11 scenes, shards of 10 (222 shards), publisher's order strided |
| protocol | `mp3d-objectnav-v1-val/2`: 500 actions, 0.25 m / 30 deg, 640x480 at HFOV 79 deg and 0.88 m, STOP within 0.1 m geodesic of a view point, the agent's navmesh |
| perception | ground truth (habitat-sim's semantic sensor; `mp3d_categories.py`), `min_pixels` 64, **on-object filter 0.25 m ON** |
| LLM | **`qwen2.5:14b-instruct` for the room labels AND the node oracle** (Q4, digest `7cdf5a01...`), num_ctx 8192, temperature 0, seed 17, 768 / 1536 max tokens; one native Ollama, 4 request slots, f16 KV (15.1 GB) |
| method | `sparx-rpt-llm-host-sweep`, frontier explorer, segmentation on (the oracle's masks); `allow_stair_traversal` **False** (run 1) / **True** (run 2); `floor_plane_bound_m` 0.5; `stair_source` observed |
| fixes in | elevation overshoot, frontal standoff (raised planes only), doorway-only scrape guard, pursuit bounds, re-sighting nearest the memory, inspection caps, tiny-fragment rule, frontier tri-state, reconsideration spacing, label hysteresis (entries 044-046); labeller rules `dining_min_chairs` 0 and `shape_rule` False (off) |
| records | `--full-records` (the whole per-episode report) and `--record-log-only` (steps.jsonl, trajectory.csv, map journal, metrics.json per episode; **no video**) |
| answers | every LLM answer recorded, keyed by episode, into `runs/<run>_support/llm_answers` (run 2 replays run 1's store and records its own) |
| seed | 17 everywhere |
| source | run 1: `a77c946c` (the method code of `60c9823a`; the commits since touch the harness and docs only); run 2: `2f458e20` + this entry, same method code |

Launch (run 1), from `runs/mp3d_14b_support/supervise.sh`:

```bash
HAB_PY=~/miniconda3/envs/objnav-habitat/bin/python MP3D_SCENES_DIR=~/datasets/mp3d/scene_datasets \
OLLAMA_MODE=native LLM_BASE_URL=http://127.0.0.1:11435 LLM_MODEL=qwen2.5:14b-instruct LLM_REASONING_MODEL=qwen2.5:14b-instruct \
OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=8192 HABITAT_VRAM_MIB=1000 VRAM_CAP=0.90 \
LLM_ANSWERS_RECORD=runs/gt_oracle_mp3d_2195ep_14b_20261111T095333Z_support/llm_answers \
  mp3d/run_benchmark.sh --perception ground-truth --record-log-only --full-records \
    --shard-episodes 10 --jobs 8 --jobs-file <support>/jobs --stall-minutes 30 --output runs/gt_oracle_mp3d_2195ep_14b_20261111T095333Z
```

A supervisor re-invokes the launcher on the same root until `campaign_summary.json` says `complete` (every exit
resumes: finished shards kept, started shards continued, merged scenes kept); a keeper restarts Ollama if it stops
answering; `resource_monitor.py` samples the machine every 15 s. Run 2 adds `--policy-config
{"allow_stair_traversal": true}` and `LLM_ANSWERS_REPLAY=<run 1's store>`.

## Sizing the machine (the pilot, 2026-11-11 09:05-09:50 UTC)

RTX 4090 (24 GB), 24 cores, 64 GB. The 14B at 4 slots and 8k context holds 15.1 GB; one MP3D ground-truth
renderer 270-1,180 MiB by scene (mean ~820). The LLM, not the card, is the limit: the client asks Ollama for
`format: json`, whose grammar sampling is CPU work serialised in Ollama's runner -- an idle server answers 0.93
requests/s without the format and 0.36 with it (150-token answers, 4 concurrent) -- and the campaign's rate was
~29 answers/min (0.5/s) with 7 workers, with 12 workers, with 8 slots and with a q8_0 KV cache alike. Seven to
eight renderers saturate it; 12 only queue (8.4 requests in flight, 13 s median latency) for the same
episodes/h. So: `--jobs 8` at a 0.90 VRAM cap (7-8 shards at once), f16 KV, 4 slots. The card's memory runs at
85-92 %, its compute at 40-60 %; CPU 15-25 %; RAM ~33 %.

## Run 1: stairs off (2026-11-11 09:53 -> 2026-11-12 13:07 UTC, 27.2 h)

`runs/gt_oracle_mp3d_2195ep_14b_20261111T095333Z`: `complete: true`, 2,195 of 2,195 episodes in 11 of 11 scenes,
222 shards, every one exit 0 -- no stall, no crash, no resume, no agent error. Against entry 043 (the same split
under ground truth with the 3B in both roles and none of the fixes of entries 044-046): **SR 52.1 -> 65.6 %, SPL
23.6 -> 29.1 %, DTG 4.42 -> 3.82 m.**

| | episodes | SR | SPL | SoftSPL | DTG (m) | mean steps | STOP / limit / error | wrong STOPs: < 0.2 / 0.2-1 / 1-3 / >= 3 m |
|---|---|---|---|---|---|---|---|---|
| **official, 0.1 m** | 2195 | **65.6 %** (CI 63.6-67.6) | **29.1 %** (CI 28.0-30.2) | 31.8 % | 3.82 | 253 | 1527 / 668 / 0 | 24 / 24 / 15 / 23 |
| same rows at ApexNav's 0.2 m | 2195 | 66.7 % | 29.7 % | | | | | 24 episodes flip |

### Same storey against cross storey

A goal view point within +-0.5 m of the start height (entry 043's rule; `storey_report.py`):

| | episodes | SR | SPL | SoftSPL | DTG (m) | steps | STOP / limit |
|---|---|---|---|---|---|---|---|
| same storey | 1807 (82.3 %) | **77.4 %** (CI 75.4-79.2) | 34.3 % | 37.0 % | 1.91 | 208 | 1480 / 327 |
| cross storey | 388 (17.7 %) | **11.1 %** (CI 8.3-14.6) | 4.6 % | 7.3 % | 12.69 | 463 | 47 / 341 |

By the level survey (`sets/mp3d_val_levels.csv`, `start_level == goal_level`): 1,880 same-level episodes 74.9 % /
33.2 %, 315 other-level 10.2 % / 4.6 %. Entry 043's 3B run had 61.8 % / 7.0 % on the same split of the episodes:
the gain is on the start storey (+15.6 pts) and the cross-storey episodes stay where stair traversal off leaves them
(the 43 cross-storey successes are goals whose storey the agent could reach without a staircase it was forbidden --
split levels, ramps, a view point on a landing).

### Per scene

| scene | n | SR | SPL | DTG (m) | steps | cross-storey n | SR cross | SR same |
|---|---|---|---|---|---|---|---|---|
| 2azQ1b91cZZ | 200 | 74.0 % | 34.4 % | 3.29 | 250 | 10 | 70.0 % | 74.2 % |
| 8194nk5LbLH | 201 | 27.9 % | 11.9 % | 10.72 | 390 | 138 | 0.7 % | 87.3 % |
| EU6Fwq7SyZv | 198 | 65.7 % | 33.9 % | 3.35 | 228 | 45 | 26.7 % | 77.1 % |
| QUCTc6BB5sX | 197 | 66.0 % | 27.3 % | 3.69 | 314 | 1 | 100.0 % | 65.8 % |
| TbHJrupSAjP | 200 | 67.0 % | 31.5 % | 3.27 | 273 | 33 | 6.1 % | 79.0 % |
| X7HyMhZNoso | 200 | 29.0 % | 13.3 % | 10.53 | 402 | 103 | 6.8 % | 52.6 % |
| Z6MFQCViBuw | 200 | 73.5 % | 30.8 % | 2.12 | 212 | 0 | - | 73.5 % |
| oLBMNvg9in8 | 201 | 85.6 % | 40.5 % | 1.11 | 167 | 28 | 46.4 % | 91.9 % |
| pLe4wQe7qrG | 200 | 100.0 % | 29.3 % | 0.04 | 93 | 0 | - | 100.0 % |
| x8F5xyUWy9e | 200 | 49.5 % | 23.6 % | 2.98 | 311 | 30 | 0.0 % | 58.2 % |
| zsNo4HB9uLZ | 198 | 84.3 % | 43.7 % | 0.86 | 142 | 0 | - | 84.3 % |

8194nk5LbLH (138 of 201 cross-storey) and X7HyMhZNoso (103 of 200) are, as in entry 043, the two scenes stair
traversal off decides; on their own storey they score 87.3 % and 52.6 %. x8F5xyUWy9e's 58.2 % same-storey is the
lowest of a flat start (the split-level house).

### Per category (SR / SPL / DTG, and SR on the same- and cross-storey halves)

## Per category

| category | n | SR | SPL | DTG |
|---|---|---|---|---|
| bathtub | 9 | 44.4 % | 25.4 % | 9.77 |
| bed | 48 | 43.8 % | 21.7 % | 10.50 |
| cabinet | 180 | 75.6 % | 35.6 % | 2.20 |
| chair | 434 | 61.8 % | 26.8 % | 3.85 |
| chest_of_drawers | 63 | 52.4 % | 25.0 % | 6.83 |
| clothes | 25 | 44.0 % | 22.1 % | 8.28 |
| counter | 66 | 50.0 % | 23.5 % | 5.86 |
| cushion | 239 | 74.5 % | 40.6 % | 2.48 |
| fireplace | 30 | 83.3 % | 42.4 % | 1.18 |
| gym_equipment | 44 | 9.1 % | 2.3 % | 11.10 |
| picture | 161 | 70.8 % | 34.9 % | 1.61 |
| plant | 87 | 47.1 % | 21.8 % | 4.67 |
| seating | 87 | 77.0 % | 21.8 % | 1.94 |
| shower | 70 | 74.3 % | 32.9 % | 3.86 |
| sink | 82 | 73.2 % | 31.9 % | 3.38 |
| sofa | 53 | 50.9 % | 22.7 % | 6.37 |
| stool | 68 | 52.9 % | 24.4 % | 7.54 |
| table | 340 | 84.4 % | 31.1 % | 1.58 |
| toilet | 28 | 42.9 % | 15.9 % | 8.56 |
| towel | 73 | 42.5 % | 20.0 % | 6.22 |
| tv_monitor | 8 | 12.5 % | 10.4 % | 14.76 |


## Per category by storey (SR; n same / n cross)

| category | n same | same | n cross | cross |
|---|---|---|---|---|
| bathtub | 7 | 57.1 % | 2 | 0.0 % |
| bed | 29 | 72.4 % | 19 | 0.0 % |
| cabinet | 161 | 80.7 % | 19 | 31.6 % |
| chair | 338 | 75.4 % | 96 | 13.5 % |
| chest_of_drawers | 56 | 55.4 % | 7 | 28.6 % |
| clothes | 24 | 45.8 % | 1 | 0.0 % |
| counter | 54 | 61.1 % | 12 | 0.0 % |
| cushion | 202 | 85.6 % | 37 | 13.5 % |
| fireplace | 29 | 82.8 % | 1 | 100.0 % |
| gym_equipment | 7 | 57.1 % | 37 | 0.0 % |
| picture | 150 | 72.0 % | 11 | 54.5 % |
| plant | 54 | 75.9 % | 33 | 0.0 % |
| seating | 81 | 82.7 % | 6 | 0.0 % |
| shower | 59 | 86.4 % | 11 | 9.1 % |
| sink | 73 | 80.8 % | 9 | 11.1 % |
| sofa | 38 | 65.8 % | 15 | 13.3 % |
| stool | 46 | 76.1 % | 22 | 4.5 % |
| table | 315 | 89.8 % | 25 | 16.0 % |
| toilet | 19 | 57.9 % | 9 | 11.1 % |
| towel | 63 | 49.2 % | 10 | 0.0 % |
| tv_monitor | 2 | 50.0 % | 6 | 0.0 % |


The 21 categories all gained on the start storey against entry 043's run; gym_equipment (37 of 44 episodes
cross-storey, every goal upstairs in 8194nk5LbLH) and tv_monitor (6 of 8) stay where stair traversal off puts them.

### Failure mechanisms (`single_storey_failures.py` over every failure's recording; `<support>/failures.txt`)

754 failures: 742 the method's, 12 the ground-truth sensor's (phantom ids the on-object filter let through: 10
whole ids, 2 slivers; outside the analysis).

| mechanism | episodes | of the 742 | reading |
|---|---|---|---|
| `limit_locks_released` | 392 | 52.8 % | the step limit after at least one closing lock was taken and released: the target (or a same-class instance) was seen, the closing did not convert it |
| `limit_never_seen` | 222 | 29.9 % | the step limit with no closing lock ever: never saw a goal -- 341 of the 668 step-limit failures are cross-storey episodes, where no lock is possible on the start storey |
| `limit_wedged` | 54 | 7.3 % | the step limit with 40+ blocked forward steps or a stall release: stuck in geometry |
| `stop_near_miss` | 44 | 5.9 % | a STOP within 1 m of a view point: 23 **beside** the object (off its normal, past its width), 19 in the band in front of it, 2 too far |
| `stop_seen_through_barrier` | 21 | 2.8 % | a STOP with the agent nearer a view point than the scorer's geodesic says: a wall, railing or void between |
| `stop_other` | 9 | 1.2 % | a wrong STOP 1 m or more away, memory on a real goal |

Over the 1,807 same-storey episodes: 408 failures; over the 388 cross-storey: 345 (341 step limits, 2 wrong STOPs
3 m or more away, 2 others). The per-episode tags, every lock's first box and memory, the release reasons and the
near-miss standing spots are in `<support>/failures.json`; the lean counters that explain each outcome in every
row of `<scene>/episodes.jsonl` (full records), the decision log in `<scene>/shard_<k>/recordings/<key>/steps.jsonl`.

### Storey transitions (stair traversal off; `storey_report.py`)

The floor atlas registered a `floor_arrival` (settling on another storey, planned or not) in 71 episodes (67 once,
4 twice): 21 of them a completed stair traversal (`stable_platform_and_stair_exit`, 15 of them in cross-storey
episodes), all from stair-start episodes the spawn switch walks off the staircase, the other 50 the atlas's own
reading of a half-level, a ramp or a landing. 121 traversals were attempted, 100 abandoned on the stairs (the
guard's rule). Against the survey's level heights the recorded trajectories show a settled-level change in 15
episodes (9 same-storey, 6 cross), and 1,704 of the 2,017 episodes with a known goal level ended on it (46 of the
344 cross-storey ones -- split levels and landings the +-0.5 m rule calls a storey).

### Runtime and utilisation

| | |
|---|---|
| wall | 27.2 h for 2,195 episodes: **80.7 episodes/h** (entry 043's 3B run: 179/h with three LLM servers); 209 worker-hours; 343 s mean, 172 s median, 2,438 s max per episode |
| LLM | 50,149 answers recorded (39,228 room labels, 10,921 oracle calls; 22.8 per episode), every one HTTP 200; labels 453 prompt / 43 answer tokens at 7.2 s server time, oracle 3,306 / 162 at 19.0 s (queueing included); decode 18 tokens/s per stream under load; the store is 246 MB in 222 files |
| card | VRAM 85 % mean / 97 % peak of 24 GB (the 14B 15.1 GB + 7-8 renderers); GPU compute **37 % mean**, 34 % median, above 80 % in 3.5 % of samples |
| host | CPU 20 % mean (24 cores), RAM 36 % mean / 62 % peak, 8.7 workers mean (the pool's 8, plus another study's 3 for part of the run) |
| disk | 23 GB under the root: 2,195 step logs at ~20-30 KB a decision, map journals, full records; no video |

The machine was never the limit the brief aimed at: with one 14B answering `format: json` the LLM saturates at
~0.5 answers/s whatever the slot or worker count (the pilot above), and 7-8 renderers already queue on it; the
other 55-60 % of the card's compute is idle because the renderers wait for answers. A second study sharing the same
Ollama from 15:21 to about 21:00 local on the 11th (three workers) cost run 1 about a quarter of its rate while it
ran (95 -> 65-70 episodes/h). Faster runs want the grammar constraint off (the client could validate the JSON it
gets instead) or a second GPU, not more workers.

## Run 2: stairs on (2026-11-12 13:07 -> 2026-11-13 00:25 UTC, 11.3 h)

`runs/gt_oracle_mp3d_2195ep_14b_stairs_20261112T130740Z`: `complete: true`, 2,195 of 2,195, 222 shards all exit 0,
no stall, no resume, no agent error. The one difference from run 1 in the frozen configuration is
`allow_stair_traversal: true` (and the answer store's paths; `preflight.json` diffed field by field). Run 1's
answers were replayed: 3,544 of the 23,160 questions came from its store, 19,616 went to the model -- the arms part
early, because stair traversal changes the HOUSE line and the node list of almost every oracle call and the rooms
the labeller sees -- and 653 episodes asked nothing live. 22,817 answers recorded (115 MB); run 2 asked half as
much as run 1 (2.2 oracle and 9.0 label calls an episode against 4.9 and 24.9) because its episodes end sooner.

| | episodes | SR | SPL | SoftSPL | DTG (m) | mean steps | STOP / limit / error | wrong STOPs >= 3 m |
|---|---|---|---|---|---|---|---|---|
| **stairs on, official 0.1 m** | 2195 | **56.4 %** (CI 54.3-58.4) | **25.7 %** | 29.9 % | 4.33 | 222 | 1856 / 339 / 0 | 475 |
| stairs off (run 1) | 2195 | 65.6 % (CI 63.6-67.6) | 29.1 % | 31.8 % | 3.82 | 253 | 1527 / 668 / 0 | 23 |

**Stair traversal on costs 9.3 SR points (CI -11.2 to -7.4, McNemar p < 0.001) and 3.4 SPL points; it flips 132
episodes to success and 336 to failure.** The whole loss is one mechanism, below.

## Comparison (`storey_report.py stairs-off=... stairs-on=...`; `entries/049/comparison_storey_report.md`)

### Overall, same storey, cross storey

| arm | group | n | SR [95 %] | SPL | SoftSPL | DTG (m) | steps | STOP / limit | wrong STOPs >= 3 m | dSR [95 %] | McNemar p | dSPL [95 %] | dDTG; flips + / - |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stairs off | all | 2195 | 65.6 % [63.6, 67.6] | 29.1 % | 31.8 % | 3.82 | 253 | 1527 / 668 | 23 | | | | |
| stairs on | all | 2195 | 56.4 % [54.3, 58.4] | 25.7 % | 29.9 % | 4.33 | 222 | 1856 / 339 | 475 | **-9.3 [-11.2, -7.4]** | < 0.001 | -3.4 [-4.4, -2.5] | +0.51 m; +132 / -336 |
| stairs off | same storey (1807) | 1807 | 77.4 % [75.4, 79.2] | 34.3 % | 37.0 % | 1.91 | 208 | 1480 / 327 | 21 | | | | |
| stairs on | same storey | 1807 | 65.2 % [63.0, 67.4] | 30.0 % | 33.7 % | 3.03 | 201 | 1572 / 235 | 276 | **-12.1 [-14.2, -10.0]** | < 0.001 | -4.3 [-5.4, -3.3] | +1.12 m; +91 / -310 |
| stairs off | cross storey (388) | 388 | 11.1 % [8.3, 14.6] | 4.6 % | 7.3 % | 12.69 | 463 | 47 / 341 | 2 | | | | |
| stairs on | cross storey | 388 | 14.9 % [11.7, 18.8] | 5.5 % | 12.3 % | 10.37 | 319 | 284 / 104 | 199 | +3.9 [-0.3, 8.0] | 0.086 | +0.9 [-1.1, 2.7] | -2.32 m; +41 / -26 |

By the level survey (`start_level == goal_level`): same level 74.9 -> 63.5 % (-11.4 [-13.5, -9.4]), other level
10.2 -> 13.7 % (+3.5 [-0.6, 7.6], p 0.12).

The cross-storey gain is real but small and not significant at n = 388: 41 cross-storey episodes won, 26 lost, DTG
2.3 m shorter, SoftSPL 7.3 -> 12.3 %. The same-storey loss is large and certain: 310 same-storey episodes that run 1
solved fail in run 2, 196 of them by a wrong STOP (146 of those 3 m or more from every view point), 114 at the step
limit.

### Why: the stair traversal's SAFE_HALT ends the episode

`building.phase` at the last action of run 2's 619 wrong STOPs: **541 are `SAFE_HALT`**
(`stair_traversal.py`: `retreat_budget_exhausted_on_unconfirmed_support`), 318 of them same-storey episodes and
221 cross-storey, 459 of them 3 m or more from the goal, median 9.0 m. The sequence, read from the recordings: the
search commits to a staircase (the loop offers it as a node or the fallback takes it); the observed-stairs
traversal stalls on the flight (`blocked_or_stalled` 345 of 616 retreats, `timeout` 271) having climbed a median
0.31 m (under 0.3 m in 250 of 539); it retreats along its trace for `retreat_actions` (80) without the source
platform being *verified* (level support of `min_floor_area_m2` 3 m2, within 0.45 m of the entry, off the flight);
the budget runs out and the traversal plans `NavigationCommand.stop_here(...)` -- a STOP the scorer takes as the
agent's answer, at a median action 241 with half the budget left, in 91 of 200 sampled cases standing within
0.12 m of the source floor's height (back on the storey it left). In no halted episode had the target closing ever
seen a goal. The mechanism tagger reads these as `stop_seen_through_barrier` (292) and `stop_other` (233) because
the agent stands far from every view point with no closing memory; they are one failure, the traversal's.

So the comparison measures the traversal's safety halt, not the multi-storey search: of 391 episodes that
attempted a traversal, 129 completed one or more (`stable_platform_and_stair_exit`: 75 same-storey, 54 cross-storey; SR
41.1 % among them), and 541 ended in the halt (2 of them successes by coincidence). 26 of the 58 cross-storey
successes followed a floor arrival; the other 32 are the split-level goals run 1 also reached. The halt is the
`RETREAT`-never-exits gap `HANDOFF.md` (2026-10-01) records for HM3D, now measured on MP3D: a retreat that cannot
re-verify its source platform should hand the episode back to the search on the storey it stands on, not STOP.

### Failure mechanisms, both arms (`single_storey_failures.py`; `entries/049/run{1,2}_*_failure_mechanisms.txt`)

| mechanism | stairs off | stairs on | of which SAFE_HALT (run 2) |
|---|---|---|---|
| `limit_locks_released` | 392 | 99 | 0 |
| `limit_never_seen` | 222 | 213 | 0 |
| `limit_wedged` | 54 | 27 | 0 |
| `stop_near_miss` (beside / in band / far) | 44 (23 / 19 / 2) | 55 (37 / 16 / 2) | 7 |
| `stop_seen_through_barrier` | 21 | 306 | 292 |
| `stop_other` | 9 | 237 | 233 |
| `stop_other_level` | 0 | 6 | 6 |
| ground-truth phantoms (outside the analysis) | 12 | 15 | 1 |
| **failures** | **754** | **958** | **541** |

Without the 541 halts run 2 would have 417 failures to run 1's 754 -- fewer step limits (339 against 668) because
the episodes that would have run out of budget on the wrong storey now attempt the stairs; whether they would have
converted is what a fix to the halt will measure.

### Per category


| category | n | SR stairs-off | SR stairs-on | SPL stairs-off | SPL stairs-on | DTG stairs-off | DTG stairs-on | dSR stairs-on vs stairs-off |
|---|---|---|---|---|---|---|---|---|
| bathtub | 9 | 44.4 % | 0.0 % | 25.4 % | 0.0 % | 9.77 | 12.01 | -44.4 pts (+0 / -4) |
| bed | 48 | 43.8 % | 35.4 % | 21.7 % | 15.3 % | 10.50 | 10.16 | -8.3 pts (+6 / -10) |
| cabinet | 180 | 75.6 % | 60.6 % | 35.6 % | 29.3 % | 2.20 | 3.78 | -15.0 pts (+6 / -33) |
| chair | 434 | 61.8 % | 55.5 % | 26.8 % | 25.9 % | 3.85 | 4.00 | -6.2 pts (+33 / -60) |
| chest_of_drawers | 63 | 52.4 % | 36.5 % | 25.0 % | 19.5 % | 6.83 | 7.50 | -15.9 pts (+2 / -12) |
| clothes | 25 | 44.0 % | 36.0 % | 22.1 % | 16.6 % | 8.28 | 8.27 | -8.0 pts (+3 / -5) |
| counter | 66 | 50.0 % | 43.9 % | 23.5 % | 23.3 % | 5.86 | 7.79 | -6.1 pts (+6 / -10) |
| cushion | 239 | 74.5 % | 69.5 % | 40.6 % | 37.8 % | 2.48 | 2.45 | -5.0 pts (+14 / -26) |
| fireplace | 30 | 83.3 % | 70.0 % | 42.4 % | 36.1 % | 1.18 | 3.68 | -13.3 pts (+2 / -6) |
| gym_equipment | 44 | 9.1 % | 25.0 % | 2.3 % | 9.1 % | 11.10 | 8.01 | +15.9 pts (+9 / -2) |
| picture | 161 | 70.8 % | 59.0 % | 34.9 % | 31.2 % | 1.61 | 2.44 | -11.8 pts (+11 / -30) |
| plant | 87 | 47.1 % | 56.3 % | 21.8 % | 24.0 % | 4.67 | 3.96 | +9.2 pts (+15 / -7) |
| seating | 87 | 77.0 % | 67.8 % | 21.8 % | 18.9 % | 1.94 | 2.19 | -9.2 pts (+1 / -9) |
| shower | 70 | 74.3 % | 72.9 % | 32.9 % | 32.7 % | 3.86 | 3.61 | -1.4 pts (+5 / -6) |
| sink | 82 | 73.2 % | 57.3 % | 31.9 % | 26.5 % | 3.38 | 4.31 | -15.9 pts (+5 / -18) |
| sofa | 53 | 50.9 % | 37.7 % | 22.7 % | 20.1 % | 6.37 | 7.50 | -13.2 pts (+3 / -10) |
| stool | 68 | 52.9 % | 42.6 % | 24.4 % | 18.0 % | 7.54 | 8.97 | -10.3 pts (+3 / -10) |
| table | 340 | 84.4 % | 68.8 % | 31.1 % | 25.2 % | 1.58 | 2.70 | -15.6 pts (+2 / -55) |
| toilet | 28 | 42.9 % | 7.1 % | 15.9 % | 2.1 % | 8.56 | 10.41 | -35.7 pts (+1 / -11) |
| towel | 73 | 42.5 % | 32.9 % | 20.0 % | 16.3 % | 6.22 | 5.23 | -9.6 pts (+5 / -12) |
| tv_monitor | 8 | 12.5 % | 12.5 % | 10.4 % | 10.4 % | 14.76 | 14.95 | +0.0 pts (+0 / -0) |

### Per category by storey (SR; n same / n cross)


| category | n same | same stairs-off | same stairs-on | n cross | cross stairs-off | cross stairs-on |
|---|---|---|---|---|---|---|
| bathtub | 7 | 57.1 % | 0.0 % | 2 | 0.0 % | 0.0 % |
| bed | 29 | 72.4 % | 44.8 % | 19 | 0.0 % | 21.1 % |
| cabinet | 161 | 80.7 % | 66.5 % | 19 | 31.6 % | 10.5 % |
| chair | 338 | 75.4 % | 67.5 % | 96 | 13.5 % | 13.5 % |
| chest_of_drawers | 56 | 55.4 % | 39.3 % | 7 | 28.6 % | 14.3 % |
| clothes | 24 | 45.8 % | 37.5 % | 1 | 0.0 % | 0.0 % |
| counter | 54 | 61.1 % | 53.7 % | 12 | 0.0 % | 0.0 % |
| cushion | 202 | 85.6 % | 79.7 % | 37 | 13.5 % | 13.5 % |
| fireplace | 29 | 82.8 % | 69.0 % | 1 | 100.0 % | 100.0 % |
| gym_equipment | 7 | 57.1 % | 28.6 % | 37 | 0.0 % | 24.3 % |
| picture | 150 | 72.0 % | 61.3 % | 11 | 54.5 % | 27.3 % |
| plant | 54 | 75.9 % | 68.5 % | 33 | 0.0 % | 36.4 % |
| seating | 81 | 82.7 % | 72.8 % | 6 | 0.0 % | 0.0 % |
| shower | 59 | 86.4 % | 83.1 % | 11 | 9.1 % | 18.2 % |
| sink | 73 | 80.8 % | 64.4 % | 9 | 11.1 % | 0.0 % |
| sofa | 38 | 65.8 % | 52.6 % | 15 | 13.3 % | 0.0 % |
| stool | 46 | 76.1 % | 60.9 % | 22 | 4.5 % | 4.5 % |
| table | 315 | 89.8 % | 72.7 % | 25 | 16.0 % | 20.0 % |
| toilet | 19 | 57.9 % | 10.5 % | 9 | 11.1 % | 0.0 % |
| towel | 63 | 49.2 % | 38.1 % | 10 | 0.0 % | 0.0 % |
| tv_monitor | 2 | 50.0 % | 50.0 % | 6 | 0.0 % | 0.0 % |

Every category but gym_equipment (+15.9 pts: 9 of its 37 cross-storey episodes reached the upstairs gym),
plant (+9.2) and tv_monitor (0) loses; bathtub (-44, 4 of 9), toilet (-36) and table (-15.6, 55 lost) most, the
small bathroom categories because a bathroom search that passes a staircase commits to it.

### Storey transitions per episode

| arm | group | n | floor arrivals / ep | episodes with an arrival | arrivals 0 / 1 / 2 / 3+ | completed stair traversals (attempts) | level changes / ep (trajectory) | episodes with a change | changes 0 / 1 / 2 / 3+ | ended on the goal's level | SR with a change | SR without |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stairs-off | all | 2195 | 0.03 | 71 | 2124 / 67 / 4 / 0 | 21 (121) | 0.01 | 15 | 2180 / 9 / 6 / 0 | 1704 of 2017 | 9 / 15 | 1432 / 2180 |
| stairs-off | same_storey | 1807 | 0.01 | 21 | 1786 / 20 / 1 / 0 | 6 (36) | 0.01 | 9 | 1798 / 5 / 4 / 0 | 1658 of 1673 | 6 / 9 | 1392 / 1798 |
| stairs-off | cross_storey | 388 | 0.14 | 50 | 338 / 47 / 3 / 0 | 15 (85) | 0.02 | 6 | 382 / 4 / 2 / 0 | 46 of 344 | 3 / 6 | 40 / 382 |
| stairs-on | all | 2195 | 0.07 | 129 | 2066 / 110 / 18 / 1 | 149 (601) | 0.10 | 152 | 2043 / 100 / 44 / 8 | 1785 of 2093 | 50 / 152 | 1187 / 2043 |
| stairs-on | same_storey | 1807 | 0.05 | 75 | 1732 / 60 / 14 / 1 | 91 (357) | 0.07 | 94 | 1713 / 60 / 28 / 6 | 1701 of 1741 | 30 / 94 | 1149 / 1713 |
| stairs-on | cross_storey | 388 | 0.15 | 54 | 334 / 50 / 4 / 0 | 58 (244) | 0.20 | 58 | 330 / 40 / 16 / 2 | 84 of 352 | 20 / 58 | 38 / 330 |

Stairs on: 149 completed traversals in 601 attempts, in 129 episodes (run 1: 21 in 121, all off a stair spawn), floor arrivals in
129 episodes (110 once, 18 twice, 1 three times), a settled-level change of the recorded height in 152 episodes
(mean 0.10 a trajectory; cross-storey 0.20); 84 of 352 cross-storey episodes ended on the goal's level (run 1: 46
of 344), and 20 of the 58 cross-storey episodes whose trajectory changed level succeeded against 38 of 330 that
did not. On the same-storey episodes a level change happened in 94 and 30 of them still succeeded; the other 64 are
the agent on the wrong storey when the halt or the budget ended the episode.

### Runtime and utilisation, run 2

11.3 h, **194 episodes/h** (2.4x run 1: half the LLM calls an episode, 15 % of them replayed, shorter episodes --
134 s mean, 64 s median, 81 worker-hours); the card's memory 86 % mean / 91 % peak, its compute 36 % mean (above
80 % in 2.7 % of samples), CPU 23 %, RAM 32 %, 7.3 workers mean; 17 GB of logs. Both runs together: 38.5 h of wall
time, 73,000 LLM answers recorded, every episode of both arms replayable.

## What to do with this
1. **Fix the halt before measuring stair traversal again.** `StairTraversal.plan` returns `stop_here` when the
   retreat budget is spent without a verified source platform; on 91 of 200 sampled halts the agent stood within
   0.12 m of the source floor's height. A retreat that ends on the storey it left (the atlas's `floor_match_m` test,
   as `observe` already accepts for a settled return) should release the portal with a cooldown and return the
   episode to the room search; one that ends mid-flight should keep retreating along the stair terrain rather than
   STOP. Measured target: run 2's 541 halts, 318 of them on same-storey episodes run 1 solved.
2. **Then re-run the stairs-on arm** on the same answers (`LLM_ANSWERS_REPLAY` on both stores): the cross-storey
   388 are where the method can still gain 30-40 SR points overall (entry 043's and run 1's ceiling without them is
   ~77 %), and the gym_equipment / tv_monitor / bed episodes of 8194nk5LbLH and X7HyMhZNoso are the test.
3. The same-storey failures of run 1 (408) are the closing's and the search's: 392 limits after a released lock
   (the target seen, not converted) are the largest remaining mechanism on the start storey; `failures.json` lists
   every lock's release reason.

## Notes
- 2026-11-11: the harness needed three things before launch (CHANGELOG 2026-11-11): a log-only recording (the
  step logs the failure tools read, without 2,195 videos), the stall watchdog on the sharded pool (it was accepted
  and ignored with `--shard-episodes`), and a resource monitor. One model in both roles was a reading of "a 14B
  model" (the previous full run also used one model in both roles); the default split would have kept the 3B on
  the labels.
- 2026-11-15: run 2 was read three days after it finished (2026-11-13 00:25 UTC); the machine sat idle in between.
  The helper processes (supervisors, Ollama keeper, resource monitors) were stopped afterwards; the Ollama on 11435 is
  left running.
- 2026-11-12: the storey report's first count of completed traversals took every `completed_transitions` entry
  with a `destination` key -- the abandoned ones carry one too (the portal's); a traversal that arrived is the one
  with a completion reason and a destination storey (121 -> 21 in run 1).
