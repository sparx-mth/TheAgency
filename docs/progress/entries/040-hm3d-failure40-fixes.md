# 040 - The GT-oracle sweep's single-storey failures: what was wrong, the fixes, and 40 of them re-run

**Branch:** `feat/objnav-habitat-hm3d-nadav`
**Status:** done (fixes in the working tree, uncommitted at the time of writing)
**Roadmap item:** the perception upper bound of the room-search method (entry 039, `HANDOFF.md` 2026-10-29)

## What the 2,000-episode run actually measured
`gt_oracle_2000ep_3b_20261028T152707Z` (HM3D v1 val, ground-truth perception, `qwen2.5:3b-instruct` in both LLM roles,
SR 0.691, SPL 0.362) never asked the node oracle once: `llm_queries` and `oracle_repair_successes` are 0 in all 2,000
rows. One model on two routes with two context windows (labels at the server's 4,096, the oracle at 8,192) starved the
oracle's reload under 20 workers; every call hung to its 900 s timeout until the 512-slot queue filled at 6.45 h, then
was refused with HTTP 503 (`LESSONS.md` 2026-10-30). The sweep was the exploration fallback plus the closing.

Single storey only (the user's scope): 1,614 episodes whose shortest path changes height by at most 1 m, SR 84.8 %.
Their failures by cause: trapped in the fallback's squeeze/escape rungs; steps, sunken and raised areas and spawns on a
tread (SR 10-19 % once the path changes height by 0.25-1.0 m); targets seen and given up; false STOPs; searches that
ran out.

## Fixes
| fix | where | what it repairs |
|---|---|---|
| one context window for one model; truncation refused; a real oracle probe at preflight; the run stops after 3 dead-oracle episodes | `core/mapping/topology/llm_client.py`, `services/llm.py`, `policy_factory.py`, `hm3d/run.py` | the dead oracle |
| the lattice plan checked with the start relaxation it was planned with; `dither_replans` cut-off | `navigation/kinematic_follower.py` | left-right turning 1.2 m from a toilet for 18 actions |
| walkable step surfaces (stair terrain, risers <= `max_step_m`, within `step_surface_band_m`) free, never drops, out of the stair mask; the storey plane re-measured on a plane of floor size | `multifloor/spawn_floor_guard.py`, `world_model/stair_terrain.py`, `localization/floor_atlas.py` | steps painted as walls/drops; spawns on a tread boxed in |
| a standoff may lie on the storey's step band, not only 0.2 m off the agent | `closing/target_path.py` | every ring point of a target 0.4 m down skipped |
| a lock released for want of a path, or a bearing approach stopped by a wall, is kept and walked back to once a route exists | `closing/target_pursuit.py` (new) | seen targets never returned to |
| a STOP on the 3-D memory alone needs map line of sight | `closing/target_closing.py` | a STOP 1.0 m from a toilet behind the bathroom wall |
| geometric path glances off for this re-run (cue glances kept); later restored on capped rays, see the addendum | `exploration/path_glances.py` | 232 measured glances: promised 7.4 m2 median, realized 0.20 m2, r = -0.17; no better per action than walking |
| `--episode-ids-file`; `HABITAT_RENDER=software` really leaves the NVIDIA card | `hm3d/campaign.py`, `run_benchmark.sh`, `benchmark_services.sh` | re-running chosen cases; the laptop's 60 % cap |

## The 40-case re-run
40 single-storey episodes the sweep failed or solved with SPL under 0.15, spread over 17 scenes and eight causes
(`runs/failure40/selected40.csv`); re-run recorded on the laptop (one GPU renderer + three Mesa workers), the final code
for every episode (stage 3, with the 11 episodes the last changes can alter re-run as stage 4; unaffected episodes
replayed identically). Report: `runs/failure40/report_final/index.html` (video, trajectory, step log per episode).

| | SR | SPL | DTG (m) |
|---|---|---|---|
| before (2,000-episode run) | 0.125 | 0.014 | 5.17 |
| after | **0.600** | **0.321** | 2.55 |

19 fixed (failed -> succeeded), 10 improved (the 5 poor-SPL successes from SPL 0.112 to 0.690 mean; 5 failures ended
nearer), 7 unchanged, 4 failed both times further from the goal. No success was lost.

| case | n | fixed or improved |
|---|---|---|
| trapped | 9 | 7 |
| steps / step down / spawn off plane | 10 | 7 |
| seen, not reached | 6 | 4 |
| false STOP | 6 | 3 |
| search ran out | 4 | 3 |
| poor SPL | 5 | 5 |

Resources (5 s samples, 08:36-10:33 UTC): CPU mean 10 % (max 24 %), VRAM max 48 %, GPU utilisation mean 37 % (above
60 % in 0.9 % of samples, LLM bursts), RAM mean 46 % -- above 60 % (max 67 %) for 14 minutes while three Mesa workers
ran large scenes beside the GPU worker.

## Open
- Search misses with the 3B oracle remain the largest failure class (00853/000065, 00890/000008, 00848/000004).
- A 0.66 m doorway the map draws 0.4 m wide still traps 00835/000059.
- 00835/000006: treads seen only from the edge; walking further down than `departure_m` triggers the single-storey
  retreat while the step band reaches 0.8 m -- the two bounds disagree.
- Long episodes diverge between GPU and Mesa rendering (00814/000049); per-episode verdicts carry that noise.

## Addendum (2026-10-31, later): geometric glances restored on capped rays
The user's intent for path glances is the look through an open door beside the route. The old promise counted every
unknown cell a ray crossed, an unmapped wall included; on the same 232 recorded glances it had r = -0.17 with what
the map gained, gating on recorded doors/openings r = 0.31, a geometric "gap between walls" gate r = 0.11, and
**counting the first 1 m of unknown per ray r = 0.39 (0.62 for the geometric glances)**, mean predicted 0.86 m2 vs
0.85 realized (`runs/failure40/report_glances/glance_calibration_232.csv`). The scorer: capped rays (`unknown_run_m`
1.0 m), floors 1.0 m2 and 0.15 m2/action; cue glances unchanged. Since the regression review below the default is
OFF again (`RPTSettings.geometric_glances`, False; `{"geometric_glances": true}` in `--policy-config` turns it on).

| the 40 cases | SR | SPL |
|---|---|---|
| 1. original | 0.125 | 0.014 |
| 2. fixes, geometric glances off | 0.600 | 0.321 |
| 3. fixes + capped geometric glances | **0.650** | 0.283 |

Between 2 and 3: 7 fixed, 5 lost (00814/000006, 00824/000018, 00853/000008, 00890/000047, 00890/000078 -- the last
an original success), SPL up 2, down 5, 21 the same: 19 of 40 changed, so per-episode churn dominates and the net
(+2 successes, -0.04 SPL) needs the full single-storey split to settle. RAM max 52 %, VRAM 48 %, CPU 15 %
(two Mesa workers). Recordings: `runs/failure40/report_glances/index.html`.

## Addendum (2026-10-31, evening): what the regressions were
Every episode of columns 2 and 3 is compared action by action (`glance_audit.py`, `glance_patterns.py`,
`glance_repromise.py` in the session scratchpad); both columns used the same renderer for each episode. The step logs
agree on every input up to the warmup's end (step 10). There, in 37 of 40 episodes, the room LLM's text and the
oracle's probabilities differ: same rooms, same labels, temperature 0, seed 17 (`LESSONS.md` 2026-10-31). The
column-2 runs of the 29 stage-3 episodes also predate the stage-4 changes. By what first parted the two runs:

| first parted by | episodes | fixed | lost | SPL change (over 40) |
|---|---|---|---|---|
| an oracle answer, before any geometric glance | 6 | 0 | 3 | -0.067 |
| a geometric glance | 27 | 7 | 2 | +0.032 |
| never parted | 6 | 0 | 0 | 0 |
| unclear (00820/000024) | 1 | 0 | 0 | -0.003 |

The regressions:
- **Oracle, not glances:** 00890/000078 (step 10: room 2 at p 0.40 in column 2, 0.15 in column 3; column 2 saw the
  plant at step 17), 00890/000047 (step 48: room 2 p 0.30 vs 0.05), 00853/000008 (step 14: room 0 p 0.01 vs 0.10).
  00824/000072 took no geometric glance at all; it parted at step 59 on an oracle answer and found the toilet at step
  208 instead of 87.
- **Glance-started:**
  - 00814/000006: a left glance at step 34, 12.1 m from the chair, set a new course. Its 10 geometric glances cost 37
    actions, less than the 101-action margin column 2 finished with, so the different route lost it, not the time.
  - 00824/000018: a right glance at step 61, 11.2 m out. Its 11 geometric glances cost 48 actions for 2.7 m2 of map
    (0.06 m2 per action, walking there 0.19), and 8 of them looked where an earlier glance had. It ended 2.8 m from
    the toilet, out of actions.

The fixes: in all 7 a geometric glance started the new course, in 6 an early full circle. Only one revealed the
target itself: 00839/000044, right glance at step 482, 2 m out, STOP at 500. In 00800/000009, 00853/000065 and
00814/000098 a cue glance first showed it; in 00802/000012 and 00814/000049 no glance did. 00848/000004's toilet was
visible from step 2 in both runs and column 2 never closed on it, which is a closing question for later.

Over all 545 glance blocks of column 3: walking revealed 0.17 m2 of map per action. Geometric glances: 161 blocks,
764 actions (19 per episode), 0.26 m2 per action, 7 first sightings. Cue glances: 384 blocks, 29 actions an episode,
0.38 m2 per action, 17 first sightings. Glances within 2 m of the goal paid best (3 first sightings in 16). By
position, budget, committed room or a known target, no group of glances paid clearly worse.

The one clear pattern is cell-level. A geometric glance can credit unknown cells that an earlier glance already
looked at and left unknown:
- The 38 glances where more than half of the credited unknown was re-promised revealed 0.04-0.10 m2 per action.
- The 123 others revealed 0.30.

Offline, discounting looked-at cells against today's floors would drop 61 glances (275 actions, about 7 an episode).
They revealed 0.15 m2 per action, at walking parity, but include 4 of the 7 first sightings. Two of those led
straight to a success: 00813/000082 at step 213 and 00843/000021 at step 112. Dropping only glances more than half
re-promised removes 38 (141 actions, 0.065 m2 per action) and 2 sightings (00813/000082 and the dubious
00802/000004). Not adopted: objects stand on mapped floor, so map gain undercounts what a re-look can show.

