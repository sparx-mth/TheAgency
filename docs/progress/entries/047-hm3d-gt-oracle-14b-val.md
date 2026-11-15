# 047 - HM3D v1 val, ground-truth perception, qwen2.5:14b in both LLM roles, the full split

**Branch:** `feat/objnav-habitat-hm3d-nadav`
**Status:** done
**Roadmap item:** the perception upper bound of the room-search method on HM3D with the reasoning model the method
was designed for (entry 039 measured the 3B; entry 043 is the MP3D counterpart)

## Goal
Run every episode of the published HM3D ObjectNav **v1** val split (2,000 episodes, 20 scenes, 6 categories) under
ideal perception and the current defaults, with the 14B model answering both the room labeller and the node oracle,
logging enough to find every failure's mechanism afterwards without video, and report SR / SPL / DTG overall, per
category and same- vs cross-storey, the failure mechanisms, the runtime and the machine's utilization.

## Configuration (frozen in every scene's `run.json`)
- Tree `0dd7d39c` plus the harness additions of this entry (heartbeat, sampler, breakdown; no method change).
- `--perception ground-truth`: habitat-sim's semantic sensor in place of YOLO-World and SAM; exact instance masks;
  on-object (depth-consistency) filter **on** at 0.25 m, min 64 px; `instances_with_boxes == annotated_instances`
  on every scene.
- Labeller rules **off** (`dining_min_chairs 0`, `shape_rule False`); `allow_stair_traversal` **false** (the
  default -- merged, never validated on HM3D; see "Why stairs stayed off"); frontier local exploration; frontal
  standoff on; view-point overshoot 0 / 0, elevated 3; seed 17; start validation on.
- LLM: `qwen2.5:14b-instruct` for **both** roles (labels at the 8,192 default context, oracle `num_ctx` 8192,
  1,536 reply tokens, 900 s timeout), temperature 0, seed 17, 49/49 layers on the GPU. The 3B was not loaded.
- `--full-records` (every event log kept, ~75-250 KB a row), **no `--record`**; `LLM_ANSWERS_RECORD` →
  `runs/gt_oracle_2000ep_14b_20261111T114322Z_llm_answers` (49,743 answers, 79 MB: any episode replays exactly).
- 9 scene workers at once, Ollama `OLLAMA_NUM_PARALLEL=4`, scenes started longest-first (the Nov-4 run's durations),
  stall watchdog 30 min on the new per-step heartbeat. Launched by `..._support/launch_full_run.sh` (re-running it
  resumes); `..._support/status.sh` is the one-line health check.

## Result (2026-11-12 11:46 UTC)

Run `runs/gt_oracle_2000ep_14b_20261111T114322Z`: **complete, exit 0, 2,000 / 2,000 episodes, 23 h 59 min**
(2026-11-11 11:47 → 2026-11-12 11:46 UTC), 83 episodes an hour, 372 s mean wall an episode. Zero agent errors, zero
stalls, zero model evictions or OOMs, one oracle refusal in ~50,000 calls (00813/000068, action 98: a malformed
reply the repair did not fix; one decision fell to the nearest frontier, the episode succeeded). Non-publishable by
construction (oracle perception). Full tables: `..._support/breakdown.md` (`hm3d/campaign_breakdown.py`).

| group | n | SR [95 % CI] | SPL [95 % CI] | DTG m | steps |
|---|---|---|---|---|---|
| **overall** | 2000 | **0.710** [0.689, 0.729] | **0.398** [0.384, 0.412] | **3.15** | 248 |
| bed | 433 | 0.647 | 0.338 | 4.12 | 280 |
| chair | 428 | 0.862 | 0.540 | 1.07 | 162 |
| plant | 84 | 0.583 | 0.269 | 7.34 | 289 |
| sofa | 376 | 0.798 | 0.512 | 2.05 | 194 |
| toilet | 398 | 0.641 | 0.289 | 3.63 | 304 |
| tv_monitor | 281 | 0.591 | 0.316 | 4.41 | 311 |
| **same-storey goal** | 1652 | **0.835** [0.817, 0.853] | **0.472** | **1.14** | 202 |
| **cross-storey goal** | 348 | **0.112** [0.083, 0.149] | **0.048** | **12.72** | 468 |

Same/cross is the level survey's `goal_same_level` (`hm3d/sets/hm3d_v1_val_levels.csv`: whether the goal's
nearest view point stands on the start's storey). Per category, same-storey SR runs 0.72 (toilet) to 0.92 (sofa);
cross-storey 0.05 (tv_monitor) to 0.21 (chair). The 39 cross-storey successes are 31 stair-start episodes that
left the stairs onto the goal's storey (the designed spawn-stairs departure, not a climb: `allow_stair_traversal`
read False in every row), 6 other stair starts and 2 others.

Per scene (SR): 00820 0.37, 00890 0.40, 00878 0.44, 00814 0.46, 00832 0.50, 00835 0.54, 00891 0.57, 00800 0.68,
00873 0.70, 00843 0.75, 00839 0.75, 00880 0.82, 00802 0.83, 00853 0.85, 00876 0.86, 00848 0.87, 00877 0.89,
00824 0.96, 00813 0.98, 00829 0.99. The five weakest are the multi-storey houses with most cross-storey goals.

**Failures: 581 of 2,000** (564 step limits, 17 scored STOPs), one mechanism tag each from the rows alone
(`campaign_breakdown.MECHANISMS`; a reading to point a replay at, not a verdict):

| mechanism | n | share | what it is |
|---|---|---|---|
| `limit_cross_storey_goal` | 307 | 52.8 % | the goal is on another storey and the search is confined to the spawn storey: unreachable by construction |
| `limit_never_seen` | 160 | 27.5 % | same storey, 500 actions, the target never in an accepted frame (toilet 58, bed 33, chair 22, tv 20, sofa 19, plant 8); median 8 rooms built, 11 oracle calls, start DTG 7.8 m, final DTG 6.7 m |
| `limit_seen_not_converted` | 61 | 10.5 % | a lock, a pursuit or a bearing, and no STOP (toilet 26, tv_monitor 22) |
| `limit_wedged` | 36 | 6.2 % | a lock released for going nowhere, or forward refused 40+ times (tv_monitor 18) |
| `stop_near_miss` | 11 | 1.9 % | a STOP within a metre of a view point |
| `stop_far` / `stop_cross_storey_goal` | 4 / 2 | 1.0 % | a STOP on the wrong instance or across a level |

Last closing release among the failures: "not moving: three steps toward it went nowhere" 33, "no safe path from
here" 26, "lost: 3 frames without the class in view" 21, "unverified (again)" 18, "near and unmeasurable: the box
fills 30-64 % of the frame without depth" 10. So of the failures the method can act on (274 same-storey), 58 % never
saw the target and 35 % saw it and could not stand on it.

**LLM.** 11,015 oracle calls (5.5 an episode) and 55,120 room-label calls (27.6); 324 episodes asked the oracle
nothing -- 98.5 % of them successes at a median 38 steps (the target was in view before a room decision was due),
so the fallback never ran blind. Mean 6.2 requests in flight against 4 slots: the LLM queue was never empty.

**Runtime and utilization** (`utilization_log.py`, 5,744 samples at 15 s): VRAM mean 21.1 GB, max 22.0 GB of 24.5
(86-90 %); GPU compute **39 % mean** (median 37, p90 58, 4 % of samples at 80 %+); CPU 29 % of 24 cores; RAM 18.3 GB
mean, 19.5 max; worker occupancy 96 % (207 worker-hours over 24 h x 9). Scene durations ran 2.5 h (00829) to 17.7 h
(00820); longest-first kept the tail to the last ~4 h. The ~90 % target held for memory but not for GPU compute: the
card's idle share is inside the 14B's `/api/chat` path -- Ollama's runner sat at 100 % of ONE core through the run
(grammar-constrained JSON sampling and prompt handling are per-request CPU work on one thread), so the median call
took 9.0 s (p90 19.7) against 3.0 s in the pilot where the 3B carried the labels. No knob of this run fixes it; more
slots need VRAM the renderers hold, a second Ollama server needs a second copy of the weights.

## Why stairs stayed off
The multi-storey implementation is merged (observed stairs, traversal, the spawn-stairs departure, 16 fixes since
2026-10-21) but has **never been run with `allow_stair_traversal: true` on HM3D**: every HM3D and MP3D campaign on
record ran under the default (`grep '"allow_stair_traversal": true' runs/*/run.json` finds none), the last
multi-storey validation was the Gibson development starts, and the traversal's own account (`HANDOFF.md`
2026-10-01, entry 038 problem 1) names a retreat that never exits. The user's instruction was: enable it only if
merged *and validated* -- it is the first, not the second. The 348 cross-storey episodes (17.4 % of the split) are
therefore the ceiling this run was measured under; `limit_cross_storey_goal` is their tag.

## Reading against the Nov-4 3B run (entry 039 lineage, `gt_oracle_2000ep_3b_20261104T083003Z`)
Not a controlled comparison -- the 3B run used the tree of 2026-11-04 (before the on-object filter had its boxes on
HM3D, the frontal standoff, the merged closing) and the 3B in both roles -- but the same 2,000 episodes: SR 0.711 →
0.710 (unchanged), SPL 0.381 → **0.398** (+1.7 points, CI-separated at the edge), same-storey SR 0.832 → 0.835,
cross-storey 0.132 → 0.112, DTG 3.16 → 3.15. Per category the 14B gains bed (+1.7), tv_monitor (+1.1) and SPL
everywhere but plant; chair SR 0.888 → 0.862 (-2.6) and plant 0.619 → 0.583 (84 episodes). Per scene the largest
moves are 00880 -6.1 and 00839 +5.0 points. A paired A/B (the 3B labeller against this run's stores) is the clean
measurement, and the answer store makes it cheap.

## What to look at next
1. The 160 `limit_never_seen` same-storey failures, toilet first (58): eight rooms built and eleven oracle calls
   without the target entering a frame is the search, not perception. Replay from the answer store; the full rows
   carry `room_search_loop.events`, `last_reasoning` and `building.events`.
2. The 97 seen-not-converted and wedged failures, tv_monitor first (40 of them): the closing's last releases say
   "went nowhere" / "no safe path" -- the standoff and the local planner, where the MP3D closing work (entry 046)
   applies.
3. Stair traversal on, on the 348 cross-storey episodes only (`--episode-ids-file`), before any full-split run with it.

## Notes
- 2026-11-11: the first launch (14B + 3B, 4 slots, 8 renderers) killed the 14B runner with `CUDA error: out of
  memory` on its first busy request (kernels load lazily; 450 MiB free was not enough) and Ollama then evicted one
  model for the other on every call. Measured: 14B runner 15.1 / 16.6 / 18.2 GB at 4 / 5 / 6 slots; 3B 3.5 GB; a GT
  renderer 740 MiB after its first frame. The second launch at 11 workers reached 24.0 of 24.5 GB and was stopped
  before any episode wrote; 9 workers held 22.2 GB steady for 24 h. CHANGELOG 2026-11-11 (Added).
- 2026-11-11: without `--record` a scene child wrote nothing between episodes, so the 30-min stall watchdog would
  have killed live workers on episodes longer than that (the Nov-4 run had 45-66 min episodes). `heartbeat.json`,
  rewritten every 30 s from `ProgressEnv.step`, is now the first file the pool watches.
- 2026-11-12: the user kept the 14B labeller after being told it halves LLM throughput (median call 9.0 s against
  3.0 s): ~24 h against an estimated ~17-20 h with the 3B labeller.
