# 036 - A target set is a target reached: the committed transit, the throttled scene graph, the log-side review

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** implemented and flown -- the five-scene campaign `runs/benchmark_25episodes_20261022T081925Z` (25 of 25 scored)
**Roadmap item:** ObjectNav exploration efficiency (010) and the room-search loop (011); follows 035

## Diagnosis
The 25-episode campaign of 2026-10-22 04:43 (`runs/benchmark_25episodes_20261022T044344Z`, stopped by the
user after 19) ran at 130 s per episode and lost the Corozal toilet (`c1af8954a378`: 500 actions, 1168 s,
DTG 8.5 m). Read from `steps.jsonl` and the policy's latency ledger, not from the video:

- **Where the time went:** the scene-graph rebuild -- watershed, registry, frontier inventory, per-room facts --
  ran on every one of the 500 actions at 1.43 s median: 701 s of the 1084 s the decisions took. The room LLM was
  203 s (19 calls at ~10 s), the classifier 40 s (82 calls), A* 4 s. Not "the LLM at every step": the geometry
  at every step.
- **Why the LLM was called 19 times:** eleven of the eighteen selections ended `room_missing_held ->
  release(unreachable)` three actions later -- the watershed had renumbered the room the agent was walking to
  (merged, split, re-drawn), the committed transit was dropped, and the order was re-solved with a fresh LLM
  call. At action 137 the agent re-planned to the toilet's room while committed to its neighbour (the user's
  "step 125 -> 134 -> 137" case).
- **Frontiers:** a frontier wrapping two or three sides of an unknown block was ONE 8-connected cluster with one
  goal on whichever side its centroid fell nearest (action 291: O30 on one side of an unknown square, nothing on
  the side the agent could reach); the route back through the open room crossed five times outranked the wings
  never walked; the floor panel drew no frontier markers though the column counted "blue diamonds".

## Goal
Once a room and a route to its entry are chosen, the walk is committed: a renumbered room keeps its turn, a
new name or a finishing verdict waits for the arrival, and only arrival, a refused or stalled route, a wedge,
the clocks or the target takeover end a transit. The geometry is not rebuilt while the committed route is
walked. A place the agent keeps failing to reach is given up with a back-off. Frontiers are one per side of a
block of unknown and rank trodden ground low. And a recording is reviewed by step from its log, never from a
frame: the map of every action is journaled and a tool reads it.

## Steps
- [x] `object_search_supervisor.py` `rebind`; `room_search_loop.py` `LoopSettings.transit_commitment` (on),
      `rebind_reach_m`, `_rebind_room`, no classifier call and no `finished_in_transit` in transit, the grace walk
      for a place no room holds (`room_missing_grace` 10), the geometry rebuilt on the action a transit ends and
      at every release.
- [x] `policy.py` `graph_period` / `refresh_graph`; `RPTSettings.transit_graph_period_steps` 5; the body-radius
      cost grid cached per map.
- [x] Place-level failure memory: `entry_failure_limit` 2, `entry_failure_radius_m` 0.75, `entry_failure_backoff`
      2; `unreachable:<n>` exclusions; the explore entry passes failed places over.
- [x] `core/mapping/topology/frontier_split.py` shared by `room_costs.frontier_clusters` and
      `room_stats.count_frontier_clusters`: a component that wraps its unknown (its centre on the unknown side of
      its cells) is cut per side at a doorway's width; an arc into the open unknown stays whole.
- [x] `frontier_ranking.py` `visit_weight` 0.05 / `visit_radius_m` 1.5, `trail_visits`; the oracle line's
      `~N m2 unseen behind them` on a room's exits; the floor panel's frontier diamonds.
- [x] The recording: `map_journal.npz` (per-action deltas of the occupancy, the settled unknown and the partition),
      the frontier inventory in full on every step row, `review_recording.py` (`summary`, `timeline`, `step N` with
      the map of that action by sector round the agent, `map N` as text, `events`). `LESSONS.md` 2026-10-22.
- [x] Regressions: 21 new across `test_explore_visit.py`, `test_room_search_loop.py`, `test_frontier_ranking.py`,
      `test_search_node_oracle.py`, `test_floor_panels.py`, `test_review_recording.py`; the tests that encode the
      historical release-in-transit pinned to `transit_commitment=False`.
- [x] Fly it: `gibson/run_benchmark.sh --five-scene 5` (seed 17, stairs forbidden, everything recorded).

## Result -- `runs/benchmark_25episodes_20261022T081925Z` (2026-10-22, 25 of 25 scored in 16 minutes)

| | SR | SPL | SoftSPL | DTG (m) | steps | wall per episode |
|---|---:|---:|---:|---:|---:|---:|
| **all 25** | **0.880** | **0.631** | 0.652 | **0.61** | 71.0 | 37.5 s |
| the 19 the morning run scored, this run | 0.895 | 0.693 | | 0.58 | 60.6 | 32.8 s |
| the 19 the morning run scored, morning run | 0.895 | 0.631 | | 0.76 | 98.7 | 130.0 s |

Per scene (SR / SPL): Collierville 0.8 / 0.674, Corozal 1.0 / 0.738, Darden 1.0 / 0.783, Markleeville 0.8 / 0.478,
Wiconisco 0.8 / 0.480. Decision median 35-370 ms per episode (the morning's toilet episode: 1732 ms); the scene
graph rebuilt on 106 of the Corozal toilet's 202 actions, 108 of Wiconisco toilet's 291.

- **Corozal/000003 toilet: SR 0 -> 1** (202 actions, 155 s, SPL 0.354; the morning: 500 actions, 1168 s). Three
  `room_rebound` events, three LLM calls (the morning: nineteen).
- Darden/000000 bed SPL 0.450 -> 0.750 (212 -> 146 actions), Darden/000001 bed 0.379 -> 1.000 (177 -> 60),
  Darden/000004 toilet 0.901 -> 1.000; Collierville/000000 toilet 82 -> 67 actions. Ten of the nineteen are
  identical pose trajectories (the short ones the search changes never touched).
- **Markleeville/000000 bed: SR 1 -> 0** (DTG 5.09 m, STOP at action 119). From the log (`review_recording step
  92..119`): a YOLO `bed` at 0.50-0.82 on a frame-filling box at (3.98, -7.28), locked at 92, INSPECT alternating
  TURN_LEFT / TURN_RIGHT between two boxes for 24 actions, ADVANCE, STOP 0.69 m from the memory -- a detector false
  positive the closing accepted, the historically flaky Markleeville bed (entry 035 notes); the morning run took a
  longer way (361 actions, SPL 0.155) and met the real bed first. Not the search's doing; the closing's.
- **Wiconisco/000003 couch: SR 0** (DTG 4.18 m, STOP at action 49): a `sofa` at 0.90-0.93 in an 80 x 40 px box,
  BEARING walk, lock at 0.37 with a 0.34 m footprint -- an armchair-sized object read as a sofa; a plausibility floor
  on the footprint of a couch would have refused it. Collierville/000004 tv is the same wrong STOP as every run.

## Notes
- 2026-10-22: the review rule, after the user's objection to frames being read: pointed at a step, run
  `review_recording.py <dir> step N` / `map N`. Both failures above were diagnosed that way in under a minute each.
- 2026-10-22: open items the campaign shows and this entry does not touch -- the closing's INSPECT dead-band
  (24 actions of left/right between two boxes, Markleeville/000000), a footprint plausibility floor per target
  class in the closing (Wiconisco/000003), the open-plan R0 the watershed keeps as one room (Corozal), and the
  glance scorer's 0.78 m2 left gain at the morning run's action 261.
