# 048 - HM3D v1 val, stair traversal on against off: the same 2,000 episodes, paired

**Branch:** `feat/objnav-habitat-hm3d-nadav`
**Status:** done
**Roadmap item:** the multi-storey search measured on HM3D for the first time (entry 047 is the stairs-off arm; `allow_stair_traversal`
had never been run on HM3D before)

## Goal
Run 2 is Run 1 (entry 047) with **one** difference, `allow_stair_traversal: true`
(`hm3d/sets/policy_stairs_on.json`), on the same 2,000 episodes with the same seed, perception, LLM (qwen2.5:14b in
both roles), workers and records, Run 1's LLM answers replayed so the arms part only where the change acts; then
compare the two episode by episode (`hm3d/compare_campaigns.py`): SR, SPL and DTG overall, per category and by
storey, with the number of storey transitions per episode.

## Runs
| | Run 1: stairs off | Run 2: stairs on |
|---|---|---|
| root | `runs/gt_oracle_2000ep_14b_20261111T114322Z` | `runs/gt_oracle_2000ep_14b_stairs_20261112T115622Z` |
| wall | 23 h 59 min, 83 episodes/h | **12 h 31 min**, 160 episodes/h |
| LLM calls | 11,015 oracle + 55,120 labels | 6,155 oracle + 26,887 labels (8,800 replayed from Run 1, 23,010 asked live) |
| health | 0 errors, 0 stalls, 1 oracle refusal | 0 errors, 0 stalls, 0 refusals, 0 evictions |
| utilization | VRAM 21.1 GB, GPU 39 %, CPU 29 % | VRAM 21.0 GB, GPU 37 %, CPU 32 % |

Run 2 is twice as fast because its episodes make half the LLM calls: a traversal makes none, and a `SAFE_HALT`
ends the episode early. A 2-episode smoke before the launch confirmed the flag reaches the policy
(00820/000008: a toilet one storey down, 500-step failure in Run 1, descent completed and STOP at 0.05 m in 161
steps). Pairing integrity: of the 222 Run-2 episodes in which no traversal started, the stairs were never offered,
every answer was replayed and Run 1's spawn-floor guard painted nothing, 220 end identically to Run 1; the two that
differ (00814/000092, 00824/000098) are successes in both arms, 4-6 steps apart.

## Result (2026-11-13 00:27 UTC)

Full tables: `runs/gt_oracle_2000ep_14b_stairs_20261112T115622Z_support/compare_stairs_off_vs_on.md` (and
`breakdown.md` of the arm). Paired B - A differences, percentile-bootstrap 95 % intervals over the per-episode
differences, exact McNemar on the success flips.

| group | n | SR off → on | ΔSR [95 % CI] (McNemar p) | SPL off → on | ΔSPL [95 % CI] | DTG m off → on | ΔDTG [95 % CI] | transitions / episode off → on |
|---|---|---|---|---|---|---|---|---|
| **overall** | 2000 | **0.710 → 0.630** | **−0.080** [−0.100, −0.059] (p < 1e-4) | 0.398 → 0.368 | −0.030 [−0.040, −0.019] | 3.15 → 3.12 | −0.03 [−0.24, +0.18] | 0.07 → 0.28 |
| bed | 433 | 0.647 → 0.577 | −0.069 [−0.113, −0.025] (0.003) | 0.338 → 0.310 | −0.029 | 4.12 → 3.29 | −0.83 [−1.33, −0.33] | 0.08 → 0.34 |
| chair | 428 | 0.862 → 0.750 | −0.112 [−0.149, −0.077] (< 1e-4) | 0.540 → 0.480 | −0.060 | 1.07 → 1.56 | +0.50 [+0.22, +0.77] | 0.07 → 0.18 |
| plant | 84 | 0.583 → 0.607 | +0.024 [−0.059, +0.107] (0.79) | 0.269 → 0.292 | +0.023 | 7.34 → 5.86 | −1.48 [−3.19, +0.14] | 0.08 → 0.12 |
| sofa | 376 | 0.798 → 0.782 | −0.016 [−0.056, +0.027] (0.52) | 0.512 → 0.510 | −0.002 | 2.05 → 1.63 | −0.42 [−0.79, −0.05] | 0.09 → 0.18 |
| toilet | 398 | 0.641 → 0.500 | −0.141 [−0.191, −0.093] (< 1e-4) | 0.289 → 0.247 | −0.042 | 3.63 → 4.45 | +0.82 [+0.32, +1.33] | 0.07 → 0.40 |
| tv_monitor | 281 | 0.591 → 0.516 | −0.075 [−0.135, −0.014] (0.022) | 0.316 → 0.294 | −0.022 | 4.41 → 4.55 | +0.14 [−0.54, +0.81] | 0.03 → 0.32 |
| **same-storey goal** | 1652 | **0.835 → 0.711** | **−0.125** [−0.146, −0.105] (< 1e-4) | 0.472 → 0.420 | −0.052 [−0.064, −0.041] | 1.14 → 2.04 | +0.90 [+0.72, +1.09] | 0.04 → 0.20 |
| **cross-storey goal** | 348 | **0.112 → 0.247** | **+0.135** [+0.083, +0.184] (< 1e-4) | 0.048 → 0.125 | +0.077 [+0.052, +0.105] | 12.72 → 8.26 | −4.46 [−5.17, −3.77] | 0.25 → 0.63 |

Per category by storey (SR off → on): same-storey bed 0.873 → 0.699, chair 0.907 → 0.787, plant 0.780 → 0.814,
sofa 0.918 → 0.846, toilet 0.720 → 0.555, tv_monitor 0.731 → 0.610; cross-storey bed 0.102 → 0.283, chair 0.214 →
0.214, plant 0.120 → 0.120, sofa 0.138 → 0.431, toilet 0.115 → 0.135, tv_monitor 0.052 → 0.155. Every same-storey
category but plant loses significantly; of the cross-storey categories bed and sofa gain significantly, tv_monitor
at the edge, chair / plant / toilet do not move (28, 25 and 52 episodes).

**Storey transitions per episode** (`building.completed_transitions`):

| arm | group | mean | episodes with ≥ 1 | max | distribution (0 / 1 / 2 / 3 / 4+) | mean in successes | mean in failures |
|---|---|---|---|---|---|---|---|
| off | all | 0.073 | 89 | 5 | 1911 / 53 / 21 / 11 / 4 | 0.058 | 0.108 |
| off | cross | 0.247 | 48 | 5 | 300 / 24 / 15 / 6 / 3 | 1.103 | 0.139 |
| on | all | 0.276 | 406 | 4 | 1594 / 292 / 86 / 25 / 3 | 0.113 | 0.551 |
| on | same | 0.200 | 229 | 4 | 1423 / 150 / 59 / 17 / 3 | 0.063 | 0.538 |
| on | cross | 0.632 | 177 | 3 | 171 / 142 / 27 / 8 / 0 | 0.802 | 0.576 |

With stairs off the 89 transitions are stair-start episodes leaving the stairs (entry 047). With stairs on, 177 of
the 348 cross-storey episodes changed storey at least once (171 never did), and **229 same-storey episodes changed
storey although their goal was on the storey they started on** -- a failure in those episodes carries 0.54
transitions on average, a success 0.06.

**Flips.** On succeeds where off failed: **135** (67 cross-storey goals that were unreachable before; 68
same-storey, where the search simply took another path). On fails where off succeeded: **294** (274 same-storey):
175 `stop_safe_halt`, 69 `limit_in_traversal`, 24 never seen, 15 seen and not converted, 7 wedged, 4 near misses.

**Failure mechanisms** (one tag per failed episode, `campaign_breakdown.MECHANISMS`):

| mechanism | off (581 failures) | on (740 failures) |
|---|---|---|
| `stop_safe_halt` -- the traversal gave up and the building layer's SAFE_HALT ended the episode with a STOP | 0 | **358** |
| `limit_in_traversal` -- the 500 steps ran out in TRAVERSE (127) / RETREAT (59) / CONFIRM (2) | 0 | **188** |
| `limit_cross_storey_goal` -- unreachable by construction with stairs off | 307 | 0 |
| `limit_no_transition` -- a cross-storey goal, no storey change completed | 0 | 40 |
| `limit_never_seen` | 160 | 95 |
| `limit_seen_not_converted` | 61 | 36 |
| `limit_wedged` | 36 | 16 |
| `stop_near_miss` / `stop_far` / `stop_cross_storey_goal` | 11 / 4 / 2 | 7 / 0 / 0 |

**What the SAFE_HALTs are.** 358 episodes, 242 of them with a same-storey goal. In 213 of those 242 the agent
completed no transition at all: the traversal started early (median action 63, p10 21), retreated and halted;
in none of the 242 had the target been seen; Run 1 succeeded on 156 of them. 116 more halts are cross-storey
episodes, where the stairs were the right call and the climb failed. The 188 step-limit-in-traversal failures
(113 same-storey) are the same mechanism without the halt. Together, **546 of the 740 failures (74 %) end inside the
traversal machinery**, and 355 of those are same-storey episodes that had no business on the stairs.

## Reading
1. **Climbing works when it is needed**: cross-storey SR 11 → 25 %, DTG 12.7 → 8.3 m, 142 single and 35 multiple
   transitions completed, bed and sofa cross-storey SR up 18 and 29 points. The 348 cross-storey episodes gain 47
   net successes.
2. **Deciding to climb is the problem**: same-storey SR 84 → 71 %, 206 net losses, and the loss is not the stairs
   themselves but the decision taken before the storey was searched (median action 63) and what follows a failed
   traversal -- a RETREAT that ends in SAFE_HALT, which STOPs the episode wherever the agent stands (a safe
   behaviour for an aircraft, a guaranteed failure for a benchmark episode that could have gone on searching).
3. The net −8.0 SR points mean the flag cannot be turned on as it stands. The two levers it names: the room-search
   loop's valuation of the stairs node against unexplored rooms of the current storey (the oracle's "elsewhere"
   mass and `stairs_as_nodes`), and the end state of a failed traversal (resume the search on the storey the agent
   stands on instead of SAFE_HALT → STOP). Both are measurable on the 242 same-storey SAFE_HALT episodes alone,
   by replay from Run 2's answer store.

## Notes
- 2026-11-12: `campaign_breakdown.mechanism` reads `building.allow_stair_traversal` and tags the traversal-on endings
  (`stop_safe_halt`, `limit_in_traversal`, `limit_no_transition`); `compare_campaigns.py` pairs two roots and
  checks the pairing on the episodes in which the change never acted. The check first read 1 of 9 quiet episodes as
  differing (00802/000001): Run 1's spawn-floor guard had painted 952 step cells that Run 2's inert guard did not, so
  the quiet set now requires an unpainted map in A as well.
- Run 2's answer store (`..._stairs_..._llm_answers`, 30,594 answers) is the replay source for anything that
  changes the traversal decision; Run 1's store stays the source for same-storey work.
