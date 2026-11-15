# 019 - Scan visits, finished rooms, type priors, class-voted objects

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** implemented; 826 runtime/core regressions pass; Ranchester couch-downstairs episode succeeds
in 150 actions (SR 1, SPL 0.71; attempt 7 of seven, see Notes)
**Roadmap item:** ObjectNav exploration efficiency (010) and room-search loop (011); follows 018

## Diagnosis (runs/zson-couch-downstairs-20261004-release, 500 actions, failure)
The 2026-10-04 recording spent 46 actions in R0, left it at 64, was back at the same
spot at 74, did not abandon it until 120; held R18 from 216 to 402 over five visits
(`budget_spent` x4); the LLM gave R18 and R27 p=0.8 on a `sofa` landmark sitting on
three beds. The descent was held 14 times by the peek gate while re-partitioning kept
producing "pending" rooms; the stairs were offered four times and never first. Four
doorway peeks started, four were cancelled. The former sweep visit had no notion of
"finished": every re-partition renumbered the room and bought it a new visit.

## Goal
One look-around per room from the open floor, then on; a room once scanned (or seen
more than half of) is never a node again; rooms whose type cannot hold the target are
never nodes; stairs are a connector, not a room; no coverage gate before a floor
change; object identity is a per-frame vote so a lone misidentification cannot name a
room or start a pursuit; the 3B as the reasoning model for development turnaround.

## Steps
- [x] `room_vantage.py`: the interior cell of greatest clearance the reachable map offers.
- [x] `room_scans.py`: the scan ledger -- where every completed rotation stood; finished by
      `scan_point_inside` or `seen_from_scan` (>= 50 % of cells in clear view within 5 m);
      sticky per `(floor, pid)`.
- [x] `room_search_loop.py`: `visit="scan"` -- transit to the vantage point, approach bound 20,
      full rotation on measured yaw, `exhausted` release, visit bound 36; finished and
      type-implausible rooms excluded from oracle, solver and transit; reconsider deferred
      mid-visit; a relabel ends the visit only when the new type rules the room out.
- [x] `room_priors.py`: target -> room types not worth entering (exclusions, never `unknown`;
      a confirmed target object in the room outranks the prior).
- [x] `scene_graph.py`: `update(exclude=...)` keeps seen stair footprints out of the watershed;
      `reason(exclude=...)` keeps finished/ruled-out rooms out of the prompt with p = 0.
- [x] `discovery.py` / `rpt_settings.py`: the warm-up is one full rotation (12 x 30 deg), recorded
      as the storey's first scan; repeated on every storey first entered.
- [x] `doorway_candidates.py` / `floor_departure.py`: `gate_floor_departure` off by default;
      `doorway_peek.enabled` off by default (a peek costs a scan and finishes nothing).
- [x] `landmarks.py` / `perception.py` / `perception_cycle.py`: footprint radius per detection,
      positional association (dedupe radius or disc IoU >= 0.15) with a height check, one vote
      per frame per class, plurality class with recorded relabels, confirmation by a clear
      plurality; target evidence judged on the map's class; the border gate shared by the
      takeover and the legacy evidence path.
- [x] `target_closing.py`: a fresh box spanning the image centre column is aligned; terminal range
      is the near edge of the measured surface; an exhausted inspection after an in-range sighting
      STOPs (`stop_on_exhausted_inspection`) instead of raising.
- [x] `target_closing.py` (after the restart): a takeover starts only on a **depth-projected**
      candidate; once active, a re-sighting on the anchor counts from `track_confidence` (0.30) --
      detect high, associate low (`weak_resightings`).
- [x] `exploration_fallback.py`: **exits first** -- object shadows (`object_shadow`: beside a
      confirmed landmark's footprint, no longer than it could cast), frontiers of `type:`-excluded
      rooms and frontiers inside the camera's blind radius (1.2 m) wait in a `frontier_demoted`
      rung behind every opening and the stairs; **goal commitment** -- the goal in force is kept
      until gone or refused unless another is worth `goal_switch_gain` (2x) more; `snapshot()` in
      the per-step record (`method.fallback`); `RoomSearchLoop.excluded` public.
- [x] `floor_context.py`: each storey's loop events/stats/exclusions kept in the record.
- [x] HUD: visit budget, scan phase, `NOT NODES` line, excluded rooms dimmed.
- [x] README (runtime, objects), scan/closing/fallback regressions, fixtures pinned to
      `visit="sweep"` for the historical sweep tests.

## Notes
- 2026-10-04 (attempt 1, 197 actions, agent_error): everything up to the couch worked -- stairs
  chosen by RPT* at 66 (p = 0.6 "unvisited ground floor holds the living room"), descent 77-120,
  lock at 157, 0.9 m from the couch at 172 -- then twelve inspect cycles: the box clipped at the
  left edge said "turn", the next heading showed a wall of upholstery and no box. Fixed in
  `target_closing._inspect` (three changes above).
- 2026-10-04 (attempt 2, 185 actions, SR = 1, SPL = 0.82): success. Two inefficiencies in the
  trail: the legacy target-evidence path pursued a 48 px edge sliver for 9 actions (now gated),
  and a potted-plant relabel cut the first room's rotation at 180 deg (now kept unless the new
  type rules the room out).
- 2026-10-04 (attempt 3, 500 actions, DTG 7.2 m): the couch released at 3.8 m from the stair head
  was never re-verified from 1 m -- the rejection memory blocked it. Added `rejection_min_range_m`
  (a close view is new evidence) and the range-growing association radius.
- 2026-10-04 (attempt 4, 500 actions, DTG 3.7 m): six twelve-step verifications from one spot three
  metres from the couch -- the (0, +30, 0, -30) sweep put the candidate at the frame edge on every
  other frame. Added `release_cooldown_actions` and the face-then-step verify.
- 2026-10-04 (attempt 5, 215 actions, SR = 1, SPL = 0.64): success, but 42 actions upstairs in the
  fallback (61-102: ten goal changes among strips of unknown between the beds, a 150-degree spin,
  back within 6 cm of the start, the passage to the stairs 2 m away the whole time) and 42 downstairs
  in two released verifications: the couch read 0.68 / 0.42 / 0.52 at three successive headings and
  the agent's own centring turn reset the consecutive-frame run every cycle.
- 2026-10-04 (attempt 6, 167 actions, SR = 1, SPL = 0.70): with the tracking threshold and exits
  first. Downstairs: first sighting 143, lock 144, STOP 166 (was 165 -> 214). Upstairs the fallback
  still took 62 actions, 37 of them turns: a 180-degree turn toward a demoted goal followed by a
  210-degree turn back toward a frontier that appeared 0.25 m from the agent -- the floor the
  camera cannot see from where it stands. Hence the goal commitment and the blind-radius rule.
- 2026-10-04 (attempt 7, 150 actions, SR = 1, SPL = 0.71): with commitment and the blind radius.
  Upstairs: warm-up 12, first room's vantage 15, fallback 26 (goal kept on 22 actions, switched 4,
  outranked 0; 23 blind, 12 shadow, 130 type demotions), a second room valued and transited 21,
  stairs seen from its vantage, approach 7, traversal 38, confirm 4. Downstairs: first sighting
  129, lock 130, STOP 149. The 15.2 m geodesic was walked in 21.6 m.
- The fallback's frontier order was briefly changed to demote frontiers inside finished rooms
  behind the stairs rung, then reverted: a frontier at a finished room's doorway is how new
  rooms are found, and stairs-vs-floor is the oracle's judgement, not the last resort's. The
  shadow rule is the version of that idea that survives: it demotes the strips behind furniture,
  not the openings.
- `seen_from_scan` finished the two downstairs rooms from the stair-head warm-up; the couch was
  in one of them, and the detector saw it from that very scan -- the takeover does not depend on
  the room being a node.
- The stairs in Ranchester are visible from the upper corridor only as a sliver through a 1 m
  passage: 7 surface samples at 3.6 m from one spot (the minimum is 6), none from 6 cm away. A
  perfect detector does not help a bad viewpoint; the exit rule is what gets the agent to the
  passage mouth.

## Open questions
- `scan_seen_fraction` = 0.5 is aggressive for an L-shaped room; the target closing is
  independent of it, so a missed object only costs the room's node, not the episode.
- The 38-44-action ground-truth traversal of one flight is the single largest fixed cost of the
  episode and the one thing untouched here.
- The exit rule cannot tell a doorway from a strip behind an object the detector does not know
  (a dresser, a desk without a label): both are frontiers beside a wall with unknown behind. The
  landmark map is the only thing that distinguishes them, so the detector vocabulary matters.

## Result
Attempt 7 (150 actions, SR 1, SPL 0.71) is the top-level recording under
`runs/zson-couch-downstairs-20261004-scan/`; attempts 1-6 are kept in its `attempt-*` folders with
the source diff each ran. See the CHANGELOG entry of 2026-10-04.
