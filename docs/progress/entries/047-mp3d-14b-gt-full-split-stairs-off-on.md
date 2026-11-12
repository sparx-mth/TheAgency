# 047 - MP3D v1 val with the 14B in both LLM roles, ground truth, all fixes: stairs off, then stairs on

**Branch:** `feat/objnav-habitat-mp3d-nadav` (worktree branch `copilot/worktree-2026-11-11T10-35-31-03da`)
**Status:** in-progress (run 1 complete and written up below; run 2, stairs on, running since 2026-11-12 13:07 UTC)
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

## Run 2: stairs on (started 2026-11-12 13:07 UTC)

`runs/gt_oracle_mp3d_2195ep_14b_stairs_20261112T130740Z`: the same launch with `--policy-config
{"allow_stair_traversal": true}` and `LLM_ANSWERS_REPLAY` on run 1's store (a question an episode had already had
answered gets the same answer; only the questions stair traversal adds go to the model). Running at ~185
episodes/h on the replayed answers; the comparison (`storey_report.py stairs-off=... stairs-on=...`) follows.

## Notes
- 2026-11-11: the harness needed three things before launch (CHANGELOG 2026-11-11): a log-only recording (the
  step logs the failure tools read, without 2,195 videos), the stall watchdog on the sharded pool (it was accepted
  and ignored with `--shard-episodes`), and a resource monitor. One model in both roles was a reading of "a 14B
  model" (the previous full run also used one model in both roles); the default split would have kept the 3B on
  the labels.
- 2026-11-12: the storey report's first count of completed traversals took every `completed_transitions` entry
  with a `destination` key -- the abandoned ones carry one too (the portal's); a traversal that arrived is the one
  with a completion reason and a destination storey (121 -> 21 in run 1).
