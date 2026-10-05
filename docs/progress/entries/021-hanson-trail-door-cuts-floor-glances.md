# 021 - The Hanson trail: door cuts, furniture, the uncertainty floor, glances along the route

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** implemented; 1040 runtime/core regressions pass (1013 before); Hanson re-flown (see Result)
**Roadmap item:** ObjectNav exploration efficiency (010) and room-search loop (011); follows 020

## Diagnosis (runs/zson-campaign-3x3-20261004, Hanson/000000, toilet, 500 actions, failed)
Read against the user's eight observations on the recording:

1. **R0 and R6 are one bedroom.** The door landmark YOLO confirmed from three frames sat 0.6 m
   off its doorway, inside the bedroom at the junction with R7; the 0.75 m disk carved there
   severed the room around its bed. No disk radius works (0.45 m cuts nothing and merges R7 in;
   0.5 m and above severs the bedroom): the cut has to move onto the doorway.
2. **The bed, the desks and the wardrobe belong to no room.** `_room_objects` credited a
   landmark to the room whose mask holds its cell; furniture stands on OCCUPIED cells, which the
   watershed never labels. The bedroom was "living room?" from its one chair on free floor.
3./5. **A door ahead and no room beyond it.** The space behind a door is an *opening* node, and
   the openings were there (`O200006` 1.9 m away at action 33) -- valued 0.00 by the 3B oracle
   ("no toilet glimpsed through gap") and dropped by `min_prob`.
4. **No look to the sides.** Nothing in the method ever turned toward a door beside the route.
6. **Down the stairs at action 27.** The two never-entered rooms were valued 0.01 ("small
   unknown room, no toilet fits" -- the words of the prompt's own example), the stairs 0.60; the
   order was `[stairs]`.
7. **The sink through the door at 164.** The sink was a 0.21 detection (below 0.35, never a
   landmark); R15 was never entered and valued 0.01; when the sink was confirmed later the room
   was a weak "living room" (two sofas) and the type prior ruled it out for the toilet.
8. **The peek cancelled at 196.** The approach bound was sized on the inventory's 3.5 m geodesic
   (body radius); A* planned 12.5 m around a squeeze (preferred clearance); the bound ran out
   5.8 m short on a route that was fine.

## Goal
Rooms split at doorways and not at misplaced doors; furniture names the room it stands in; a
place the search knows nothing about is never written off; the type prior cannot override a
sink; distances are the planner's; and the agent looks to the side where it pays.

## Steps
- [x] `core/mapping/topology/room_watershed.py`: `door_snap_reach_m` / `door_choke_min_m` /
      `door_choke_max_m`; `door_carve_mask`, `snap_door_to_choke` (narrowest valid choke within
      reach, nearest on a tie; a dead-end stub is not a choke), `_basin_sides`;
      `room_merge.merge_basins_by_dynamics(basin_sides=)`. On in `scene_graph.DEFAULT_SEGMENTATION`
      (0.9 m). Checked on all 15 recorded floors: Hanson F0 4 -> 3 rooms (bedroom whole), the rest
      within one room of before.
- [x] `scene_graph.object_room` / `room_near` / `room_near_cell`: nearest room floor within the
      footprint radius + 0.6 m; used by the room objects, the landmark openings and the snapshot.
- [x] `search_node_oracle.py`: `entered=no` in the room line; rules 2b/2c; the example fixed;
      `SearchNode.unexplored`; `floor_unexplored`; `SearchNodeOracle(unexplored_floor=)` applied
      to fresh and reused replies, `NodeOracleResult.floored`. `LoopSettings.unexplored_floor`
      (0.25) -> `RepairingNodeOracle` in `floor_context.new`.
- [x] `room_priors.HOME_OBJECTS` / `home_object` / `ruled_out`; `room_search_loop._ruled_out`
      (strong labels only; target seen or home object kept; `weak_type_kept` / `home_object_kept`
      events); `_relabel_ends_turn` on the same test; `weak_type_max_openings` retained, unused.
- [x] `room_search_loop._size_approach_to_route` (`peek_approach_extended` events).
- [x] `frontier_ranking.accessible_frontiers(preferred_cost=)`, `room_costs.prefer_cost_matrix` /
      `build_instance(preferred_cost=)`, `RPTSearchPolicy.preferred_cost`, wired through
      `ObservedSceneGraph.update` / `refresh_accessibility` and the loop's `_instance`.
- [x] `core/planning/exploration/view_gain.py` (`ViewCone`, `UnknownView`, `glance_gains`) and
      `methods/path_glances.py` (`GlanceSettings`, `GlanceScheduler`); wired in
      `RPTSearchPolicy._search` around the loop's and the building's follow commands; `glance`
      in `SUSPENDED_PHASES`; `glances` in `configuration()`, `episode_info()`, the `search`
      snapshot and the HUD header.
- [x] Regressions: `test_room_watershed.py` (+5), `test_rooms.py` (+1), `test_search_node_oracle.py`
      (+3, 1 updated), `test_room_scans.py` / `test_opening_nodes.py` (type prior, home objects,
      approach bound), `test_frontier_ranking.py` / `test_room_costs.py` (two-tier distances),
      `test_view_gain.py` (4), `test_path_glances.py` (10).
- [x] README (what is not a node; the uncertainty floor; where a door cuts; distances; the
      glances section), CHANGELOG, LESSONS (2026-10-05).
- [x] Re-fly the three Hanson episodes (`runs/zson-hanson-3x-20261005`, same models and seed).
- [ ] Re-fly the other six; decide the peek economy (below).

## Open questions
- The choke snap is a heuristic over observed geometry: a furniture passage narrower than the
  doorway within 0.9 m of a door landmark would be taken for the doorway. The door detector's
  position error (along the camera ray) is the real problem; a jamb-fitted door plane would be
  the structural fix.
- `heal_free_mask` (one 3x3 closing) erases walls up to two cells thick: 375 of 1088 occupied
  cells on Hanson F1. A pinhole-only heal changes every partition (measured: mostly more rooms);
  not changed here for want of ground truth.
- Glance economy: with unknown on every side early in an episode the scheduler would look every
  ~14 actions (8 cooldown + 6). The 0.5 m2/action bar is reasoned, not measured on a flight.

## Notes
- 2026-10-05: the first choke picker chose a 0.1 m skeleton spur for the corridor door; the
  half-width band (0.25-0.8 m) and the "parts the skeleton into two substantial pieces" test
  are what make the snap safe. A local-minimum test along the skeleton was tried and dropped:
  fragile at skeleton junctions, redundant with "narrowest wins".
- 2026-10-05: the first barrier for a snapped cut was "the disk's throat cells by clearance";
  near the walls every disk cell has low clearance, so it barred the hall's own basins from
  merging. Sides of the carved mask are the clean statement of "parted by a door".
- 2026-10-05: a full circle (12 actions) can never beat the better side glance (6 actions) on
  gain per action unless the rear sector holds something -- `(L + R) / 12 <= max(L, R) / 6`.
  That is the intended economy: the other side is a second glance's business after the cooldown,
  taken where it shows best.

## Result
Shipped as listed under Steps. CHANGELOG `[Unreleased]` 2026-10-05 entries; LESSONS
`2026-10-05 -- Hanson/000000`.

Hanson re-fly (`runs/zson-hanson-3x-20261005`): the toilet episode STOPs at action 359 at 0.68 m
from a real toilet in the upstairs bathroom R21 (was 500 actions, never found) -- scored SR=0 only
because the single annotated toilet is downstairs (DTG 19.2 m; upper-floor annotations are
incomplete). The bedroom is one room labelled `bedroom`; four bedrooms are identified from their
beds and desks; never-entered rooms and openings sit at the 0.25 floor and were all peeked into
before the stairs; 13 glances cost 53 actions where 3-37 m2 of unknown lay to the side; no peek
was cancelled short (`peek_approach_extended` x20). The chair (14 actions, false STOP) and the
potted-plant (161 actions, "closing action bound reached") episodes are identical to the campaign:
both end inside the target-closing takeover before the search runs -- a separate diagnosis.
Open: eight peeks cost 212 actions (~27 each) against the 4 + walk RPT* charges them, which is why
the stairs (0.60) kept slipping behind 0.25 openings; raising `OpeningSettings.service_steps` to
about 12 and/or a lower floor for a `gap` than for a `doorway` are the candidate fixes.
