# 020 - Openings as nodes, the peek, unique room numbers, the same-storey question

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** implemented; 1069 runtime/core regressions pass; couch-downstairs 154 actions SPL 0.80, same-storey couch found (see Notes)
**Roadmap item:** ObjectNav exploration efficiency (010) and room-search loop (011); follows 019

## Diagnosis (runs/zson-couch-downstairs-20261004-scan, attempt 7, action 75)
The agent stands in the upstairs hallway facing three doors: a toilet through the right one, a
cabinet through the one ahead, nothing known of the left one. The map shows R0, R2, R4 and R6 --
no room for the hallway, none for the three rooms behind the doors. The watershed partitions
*observed* free space, so a room glimpsed through its door is a few cells hanging off the
hallway's region, and the toilet seen through the door named the hallway `bathroom` (weak). The
type prior excluded the hallway; the fallback demoted every frontier of it (130 `type_demoted`);
nothing was left to value but the stairs the agent happened to turn towards. The descent was the
right call for a couch and the wrong reasoning: in a search for something on *this* storey it
would have been fatal. Also: both storeys numbered their rooms from R0.

## Goal
Every door and gap to unseen space is a node of its own, valued by the room likely behind it and
visited by a quick peek; a weak label on a room with several such exits does not rule it out; room
numbers are unique across the building; and the method is tried on a target that lies on the
starting storey but not in the starting room.

## Steps
- [x] `opening_nodes.py`: exits of the mapped floor (the fallback's own test, without the type rule),
      merged within 1.5 m, >= 0.8 m wide, not at a seen staircase's foot, not yet peeked; sticky
      building-wide ids (`OpeningRegistry`); `doorway`/`gap` by a confirmed door frame; objects
      glimpsed beyond the threshold; `OpeningSettings`.
- [x] `search_node_oracle.py`: the OPENING kind, its prompt line, the rule and the worked example.
- [x] `room_costs.build_instance(service_s=...)`: a peek's price on an opening's entering arcs.
- [x] `room_search_loop.py`: openings collected with the stairs at every SELECT, valued in the same
      oracle call, offered to RPT* at their thresholds; `_transit_opening` (walk, 20-action bound,
      look from within 1.5 m or retire); `_peek_visit` (face the unknown, `2*look_turns+1` headings
      swept from the nearer side, one turn per action, retired `exhausted`); the weak-type rule in
      `_exclusions`; openings count as rooms left for the way-back hold; `peek_state()` for the HUD.
- [x] `stair_nodes.py`: stair ids are a band (100 000-199 999); openings from 200 000.
- [x] `room_registry.py` / `scene_graph.py` / `floor_context.py`: `first_pid` -- a new storey numbers
      its rooms after the highest pid any storey has used.
- [x] `rpt_policy.py`: `OpeningRegistry` on the policy, `openings` in `episode_info`/`configuration`.
- [x] HUD: `OPENINGS (nodes)` section, `O<n>` names in the order, the peek in force, `O<n>` markers
      with the heading into the unknown on the active floor map; `openings` in `live.json`.
- [x] Regressions: `test_opening_nodes.py` (23), the oracle's opening line, `service_s`, `first_pid`.
- [x] README (loop steps 3, 4, 6; the openings paragraph; unique room numbers), CHANGELOG.
- [x] Rerun the couch-downstairs episode and read the trail against action 75 (attempts 8-10 below).
- [x] The same-storey episode: `runs/zson-couch-same-floor-20261004` (ground-floor start, 7.35 m from
      the nearest couch, another room; attempts 1-3 below).
- [x] From the trails: the heading into the unknown from the known floor (not the mean of the unknown
      cells); a 35-degree glimpse cone; a distance-aware approach bound with one re-aim; the
      opening-revalue throttle; **target landmarks as nodes**; **a failed inspection releases the lock**
      (blank views, map outvote) instead of ending the episode.

## Notes
- 2026-10-04: the IDE crashed twice during this work; the half-finished loop edits (callers passing
  `openings` to methods that did not take them yet) were completed from the attempt-7 source
  snapshot the run script keeps (`source-changes.diff`).
- 2026-10-04: the weak-type allowance counts *exits* only. A single-sided door's `rooms` list in the
  door links is empty (the watershed knows no room on its far side), so confirmed doors cannot be
  credited to a room without a second lookup; the exits already carry the door when one stands at
  them (`doorway`).
- 2026-10-04: the unknown at the foot or head of a seen staircase is not an opening (the stairs are a
  node already; the unknown beyond a flight is the other storey).
- 2026-10-04 (couch downstairs, attempt 8, 199 actions, SR 1, SPL 0.47): the first peek (O2, 4 m away)
  spent the 20-action approach bound 1.8 m short; O3 -- the stair passage -- read "glimpsed: bed" for
  sixty actions because `unknown_heading` averaged the unknown on every side of the frontier cell and
  pointed back into the hallway, where a half-plane glimpse credited a hallway bed to it; thirty oracle
  calls in sixty fallback actions (one per door that appeared or re-snapped). Fixes: the heading from the
  known floor toward the unknown, the glimpse cone, `approach_bound`, `revalue_actions`.
- 2026-10-04 (attempt 9, 189 actions, SR 1, SPL 0.57): upstairs 74 actions to the stairs -- the peek at
  O3 (step 36) showed the stairs at 42 (the threshold itself unreachable: the stair footprint), stairs
  chosen 56, taken 74. Downstairs the warm-up saw the couch three times at 0.41-0.45, under the
  takeover's 0.50 and without depth from 3.8 m, put a `sofa` on the map, and the search walked west
  for forty actions chasing frontiers; the couch found was a different one. Fix: target landmarks as
  nodes.
- 2026-10-04 (attempt 10, **154 actions, SR 1, SPL 0.80**, 19.0 m walked for 15.2 m): the
  downstairs warm-up ended at 128, T1 (the sofa landmark, p 0.85) headed the order, the takeover started
  six actions into the walk, STOP at 154. Downstairs 40 actions (attempt 9: 75; attempt 7: 38 to a
  nearer couch).
- 2026-10-04 (same storey, attempt 1, 83 actions, `agent_error`, DTG 2.85 m): a `sofa` at 0.74-0.92 from
  four metres locked at step 39; walking to it the box slid to the left image edge, the agent passed it,
  and from one metre the detector saw nothing in 24 inspection views; `fail("terminal visual confirmation
  unavailable")` ended the episode with the real couch 2.85 m away. Fix: `release_on_failed_inspection`.
- 2026-10-04 (same storey, attempt 2, **121 actions, SR 1, SPL 0.64**, 11.5 m for 7.35 m): the phantom
  lock cost 38-82 (44 actions, 24 of them inspection turns), released, then the legacy target-evidence
  path (`target`, 96-103) and the takeover found the couch; STOP at 120. Also R2 was entered twice
  (15, 28): its label flipped between the two visits as the partition changed under it. Fix for the
  inspection: release after one pass over the views, and at once when the map outvotes the lock.
- 2026-10-04 (same storey, attempt 3, **101 actions, SR 1, SPL 0.67**, 11.0 m for 7.35 m): the same
  phantom locked at 39 and was released at 63 after one pass over the inspection views (attempt 2:
  82); the couch then came by the legacy evidence path and the takeover; STOP at 100. What remains is
  the phantom itself (44 -> 25 actions) and the R2 double visit (6).

## Open questions
- A new opening appearing while nothing is in force buys an oracle call at most every 6 actions;
  on a storey the fallback is carrying this is still one 3B call per six actions.
- The oracle valued every downstairs node at 0-2 % in attempts 9 and 10 ("F0: no living room or
  office here") while standing four metres from the couch; the landmark node, not the oracle, found
  it. The 3B's storey reading is the weak link; the 14B is the benchmark model.
- The exploration fallback's exit order is still greedy by utility (size / distance); in attempt 8 it
  walked seven metres west before the two-metre passage east. With openings as nodes the fallback runs
  only when every node is below `min_prob`; a smarter fallback order (nearest exit first when the
  oracle has given up on the storey) is the next lever.
- R2's label flip between two visits (same storey, attempt 2): the partition changed the room's object
  set and the classifier was re-asked; a room once ruled out by a STRONG label could stay ruled out
  for the episode by position (the scan ledger's trick), not by pid.

## Result
Couch downstairs: attempt 10 (154 actions, SR 1, SPL 0.80) is the top-level recording under
`runs/zson-couch-downstairs-20261004-openings/`; attempts 8-9 in its `attempt-*` folders with the
source diff each ran (attempt 7, SPL 0.71, under `runs/zson-couch-downstairs-20261004-scan/`).
Same storey: attempt 3 (101 actions, SR 1, SPL 0.67) is the top-level recording under
`runs/zson-couch-same-floor-20261004/`; attempts 1 (agent error) and 2 (121 actions, SPL 0.64) in its `attempt-*` folders. See the
CHANGELOG entries of 2026-10-04.

