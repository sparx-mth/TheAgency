# Changelog

All notable changes to this project are logged here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/) — one line per change, written for
future-you, not for a commit log.

## [Unreleased]
### Added (2026-10-22)
- **Five-scene campaign on the committed-target changes** (`runs/benchmark_25episodes_20261022T081925Z`, 25 of 25 scored in
  16 minutes, everything recorded with the map journal; `docs/progress/entries/036-objnav-committed-target-and-log-review.md`):
  SR 0.880, SPL 0.631, SoftSPL 0.652, DTG 0.61 m, 71 actions and 37.5 s per episode. On the nineteen episodes the
  04:43 run had scored before it was stopped: SR 0.895 -> 0.895, SPL 0.631 -> 0.693, DTG 0.76 -> 0.58 m, 98.7 -> 60.6
  actions, 130 -> 32.8 s per episode; Corozal/000003 toilet SR 0 -> 1 (500 -> 202 actions, three LLM calls where
  there were nineteen). Two wrong STOPs on detector false positives (Markleeville/000000 `bed`, Wiconisco/000003
  `sofa` with a 0.34 m footprint), diagnosed from the logs; the closing's, not the search's.
### Changed (2026-10-22, the Corozal/000003 toilet recording `c1af8954a378`: a target set is a target reached)
- **The committed-target rule** (`rooms/room_search_loop.py` `LoopSettings.transit_commitment` on, `rebind_reach_m`
  0.6; `object_search_supervisor.py` `rebind`). Eleven of the episode's eighteen selections were `unreachable`
  releases of a room the watershed had merely renumbered on the way to it (`room_missing_held` -> `release` three
  actions later), each buying a room-LLM call (10 s) and an RPT* solve, and at action 137 the agent re-planned to the
  toilet's room while committed to its neighbour. A transit now ends only by arrival, a route the planner refuses or
  the follower makes no progress on, the follower's wedge, the supervisor's clocks, or the target takeover: the room
  in force is RE-BOUND to the live room holding the committed entry point (else the lost room's centre) under its new
  number, with its entry, estimate and probability (`room_rebound` events, `rooms_rebound`); a visit in force follows
  the room renumbered under the agent's feet; no classifier call is spent in transit (`relabels_deferred`; the arrival
  asks) and the ledger's `finished_in_transit` no longer ends a walk -- both are judged at arrival. The place walked
  to with no room under it at all is given up after `room_missing_grace` (10) actions as before. Off, the 2026-10-21
  behaviour; the regressions that encode it say so.
- **The scene graph is not rebuilt on every action of a committed transit** (`policy.py` `graph_period`,
  `refresh_graph`; `RPTSettings.transit_graph_period_steps` 5). The watershed, registry, frontier inventory and
  per-room facts were 1.4 s of a 1.7 s median decision, 500 times; the follower needs the map and its route, not a
  fresh partition. Rebuilt every fifth action while the route is walked, on the action the transit ends (before the
  visit's first decision), and at every loop point as before; every action in SEARCH and SELECT as before. The
  body-radius cost grid is built once per map and shared (`navigation_cost` cache; the loop asked for it twice).
- **A place the agent keeps failing to reach is given up** (`LoopSettings.entry_failure_limit` 2,
  `entry_failure_radius_m` 0.75, `entry_failure_backoff` 2; `entry_failure` events, `unreachable:<n>` exclusions).
  `unreachable`, `blocked` and `transit_timeout` releases are remembered by the entry's POSITION, so the memory
  survives the renumbering the supervisor's `max_attempts` does not: a room whose failed places add up to the limit,
  or whose candidate entry lies at one, is withheld for `failure_cooldown_steps * 2 ** (n - 2)` actions, and the
  explore entry passes such places over for the room's next frontier or doorstep. 0 disables.
- **A frontier wrapping a block of unknown is one cluster per side** (`core/mapping/topology/frontier_split.py`;
  `room_costs.frontier_clusters`, `room_stats.count_frontier_clusters` `split_sides`). The floor mapped along three
  sides of an unmapped room was one 8-connected component with one goal, on whichever side its centroid fell nearest
  -- action 291: the gap O30 on one side of an unknown square and nothing on the side the agent could reach. A
  component whose cells face different sides of their unknown and whose centre lies on the unknown side of them (it
  wraps a block; an arc bulging into the open unknown has its centre behind it and stays whole) is cut at the corners
  into one cluster per side of at least a doorway's width (`SIDE_MIN_M` 0.9); the goal maker and the scene graph's
  count make the same cut. The per-cluster extraction reads each component's bounding box, not the whole grid.
- **Trodden ground ranks under ground barely visited** (`frontier_ranking.py` `FrontierRankingParams.visit_weight`
  0.05 / `visit_radius_m` 1.5, `trail_visits`; the loop's `_visits`, the fallback's `_by_view_gain`). A frontier
  goal's utility is divided by `1 + 0.05 * actions spent within 1.5 m of it`, read off the sight ledger's trail: the
  route back through the open room crossed five times (action 291) now ranks behind the frontiers in the wings the
  agent never walked. 0 disables.
- **The unseen floor behind a room's exits is on its oracle line** (`opening_nodes.RoomExits.unseen_m2`,
  `search_node_oracle.SearchNode.unseen_m2`, `format_exits`): `exits=2 (1 doorway, ~30m2 unseen behind them)`, the
  rays' reach from the exits' thresholds summed, and the prompt told that a room whose exits open onto a large
  unseen area is where the rooms not found yet must be -- the model can prefer the way to another room over the
  living room it has seen.
- **The floor panel draws the accessible frontier goals** (`floor_panels.py` `FRONTIER_COLOR`; `search_panel.py`):
  the column counted "blue diamonds" the map did not show, so a recording could not say which way the inventory
  pointed. One small diamond per cluster.
- **A recording is reviewed from its log, by step, never from a frame** (`floor_panels.py` map journal ->
  `map_journal.npz`; `visualization.py` `search_snapshot` `frontier_goals` / `transit` / `entry_failures`, the
  explore visit's `ranked` goals; `review_recording.py`; `recording.py` and `episode.json` say so). The per-step
  row had the frontier goals as bare coordinates and the only saved map was the final one, so "what lay to the
  agent's left at action 261 and why was none of it a goal" was answerable only from a video frame. Now every
  action's occupancy, settled-unknown and room-partition deltas are journaled, each frontier goal rides with its
  size, walk, facing, utility and room, and `review_recording.py <dir> summary | timeline | step N | map N |
  events` answers from the log -- `step N` with the map of that action by sector round the agent (free /
  occupied / unknown / settled area, raw frontier cells, the goals and openings in the sector). Older
  recordings fall back to the final map and say so. `LESSONS.md` has the entry.
### Fixed (2026-10-22, after 19 episodes of the 25-episode campaign: stair false positives, classifier churn, update cost)
- **An observed staircase must have a flight's profile** (`world_model/stair_terrain.py` `stair_profile`;
  `MultiFloorParams.stair_min_risers` 3, `stair_riser_min_m` 0.10, `stair_max_tread_m` 0.55): any climbable chain of
  support surfaces 0.4 m up was a candidate -- Corozal/000003 placed a "staircase" on a coffee table's shelf and top
  and the sofa behind them, Darden/000000 one on a bed, each masking a square metre or two of the room on the planning
  grid and excluding it from the partition (Corozal's toilet episode ran to the 500-step limit, DTG 8.5 m). Now the
  path must climb one way in at least three risers of 0.10 m to `max_step_m`, with no level between them walked for
  more than 0.55 m: a table top, a seat or a mattress is a tread half a metre deep or more, and no furniture is three
  tread-sized risers tall; a landing ends one stretch and the next flight starts at it. `risers` / `profile_rise_m`
  on the candidate. One new test in `test_multifloor.py`.
- **The room classifier is keyed by a room's objects alone** (`room_classifier.py` `_signature`,
  `revisable_room_labels.py`): with the geometry bucket in every cache key, a room re-asked the model whenever it grew
  from small to medium or gained a doorway -- 80-100 classifier calls per long episode. The bucket now keys only a
  room with no objects (the shape-only question); the geometry is still shown in the prompt when a query happens.
- **`_parts_skeleton` judges a choke on the skeleton's bounding box, not the whole grid** (`room_watershed.py`): the
  800x800 grid was labelled twice per candidate, a hundred times per scene-graph update -- 0.4 s of the update's
  0.76 s on Corozal/000003's map, the update being 1.4 s median over the 500 steps. Exact (9,600 random verdicts on
  three recorded maps unchanged, the Corozal partition identical); the update halves to 0.37 s on that map.
### Changed (2026-10-21, afternoon: observed stairs, the unvisited raised, wide glances, the room taxonomy)
- **Stairs are observed, never read from the navmesh** (`core/localization/floor_atlas.py`
  `MultiFloorParams.stair_source` default `observed`; `core/planning/object_search/multifloor/multifloor_policy.py`,
  `policy.py`, `spawn_floor_guard.py` docstrings). The ground-truth connector's footprint was the navmesh centreline,
  which in Darden begins 0.45 m before the first riser on a sloped threshold -- drawn on the grid the flight stood
  lower than the stairs do, and with the capsule's foot it closed the junction below. The method now takes its stairs
  from its own RGB-D: a reachable run of measured treads rising `stair_min_rise_m` (0.40 m) within 5 m
  (`world_model/stair_terrain.py` candidates) is a portal of the storey, placed from the measurement, masked by the
  same capsule (`peek_stairs`), with the observed drops and the map's obstacle band as before -- no simulator geometry
  is consulted, the footing of ApexNav, SG-Nav and OSG-Nav. `configuration()["ground_truth_stairs"]` reads False;
  `stair_source="ground_truth"` is the declared ablation and the fixtures that hand an episode a scene structure ask
  for it (`tests/test_method.py setup_policy`, the baseline's `baseline()`).
- **A room never stood in is raised; a room entered decays with the time spent inside** (`rooms/room_search_loop.py`
  `_scaled_values`, `LoopSettings.unvisited_boost` 1.5 / `unvisited_floor` 0.15 / `dwell_half_life_steps` 20 /
  `dwell_floor` 0.05; the oracle prompt's STEP 4, `search_node_oracle.py`). The models read a never-entered room's
  `frontier=0` or `1` as little to find, when the count measures the map and not the room -- one boundary cleared
  opens the whole of it. A never-entered, unfinished room the model gave anything is read at 1.5x its number and no
  less than 0.15 (a room given 0 by the model or the type prior stays 0; the room the agent stands in keeps its
  number); a room entered and left keeps `entered_discount * 0.5 ** (steps inside / 20)`, floored at 0.05, the
  lingered rule still capping it. The prompt: never below 15 for a never-entered unknown room, never lowered for a
  low frontier count; the longer `searched=` reads, the lower. `test_search_economy.py`: the lingered test follows
  the decay; one new test.
- **Side glances turn to 120 degrees when the far band pays** (`exploration/path_glances.py`
  `GlanceSettings.wide_rad` 2pi/3, `GlancePlan.turn_rad`, `wide` plans and counters). A right angle with the 79-degree
  camera shows the floor to about 130 degrees off the route; a pocket beside and a little behind the agent -- the
  alcove just passed, the room whose door opens back toward it -- lies in the 130-160 band only a wider look
  reaches. Every candidate point is scored for the wide look too, taken only when the extra band alone pays for the
  two extra turns each way at `min_gain_per_action_m2`; the sector memory, the committed plan and the HUD carry
  the angle. Two new tests in `test_path_glances.py`.
- **The room taxonomy is residential, and a label is never locked** (`core/mapping/topology/room_classifier.py`
  `HOME_LABEL_SET`, `LABEL_ALIASES`, geometry facts in the prompt and the cache key; `revisable_room_labels.py`
  `shape_only_rooms`; `rooms/scene_graph.py` `_room_geometry`; `core/mapping/objects/landmarks.py` `max_misses`;
  `perception/cycle.py` `_count_misses`; `RPTSettings.landmark_max_misses` 4 and the miss view bounds). Hallway /
  corridor, foyer / entryway, dining room, office / study, laundry / utility, balcony / patio, closet / pantry /
  storage and garage are the set offered first, synonyms and slashed forms are read as the set's labels, and the
  prompt names the detector's confusions (a table or a sofa for a bed). A room is told its floor area, its shape and
  the doorways off it, and a room with nothing in it that reads as a passage -- long and narrow, or two or more
  doorways, 2 m2 or more -- is asked about on its shape alone, once per geometry bucket, WEAK and provisional (it
  rules nothing out). The lock: confirmation was one-way, so a dining table YOLO called `bed` at 0.52 twice named a
  kitchen a bedroom for the episode (Darden/000000). A confirmed landmark that stands in a CLEAR view -- inside the
  frame by 40 px, 0.8-3.5 m away, the depth at its pixel reaching it -- without a detection of its class is missed,
  once per distinct viewpoint (0.5 m or 30 degrees from the last counted miss); as many misses as
  `max(landmark_max_misses, its observations)` -- four for the twice-seen table, twelve for a sofa seen twelve times
  (a flat four retracted 29 of Markleeville's 51 confirmed objects, most of them real) -- retract it from the
  confirmed set (the room's evidence, the target landmark nodes), the evidence gate
  takes the room's label back at the next scene-graph update and the label history records it; the next detection
  that folds into it reinstates it. `perception.counts` `landmark_misses` / `landmark_views_occluded` /
  `landmark_retractions`, `perception.retractions` in the record. Tests: `test_room_classifier.py` (3),
  `test_rooms.py` (1), `test_perception_association.py` (3).
### Fixed (2026-10-21, the audit after the first room-level run: doorways, glances, closing, stairs, spins)
- **The doorway partition: the space beyond an observed open door is a node of its own** (`core/mapping/topology/
  room_watershed.py`, `core/planning/object_search/rooms/scene_graph.py`, `room_search_loop.py`). The sliver of floor
  glimpsed through a doorway had no clearance peak, so no watershed label, and `_reclaim_carved` handed it to the
  nearest seeded room ACROSS the very cut carved to part them (the toilet seen through the door named the hallway a
  bathroom). Now: `ObservedSceneGraph.note_openings` (called by the loop with every action's openings and the viewing
  pose) reads the doorway off the line of sight from the viewpoint to the opening -- `sightline_thresholds`, the
  floor narrowing between OBSERVED jambs with the unknown passable, so an open room's frontier is no doorway -- snaps
  the dip to the doorway's choke (`SIGHTLINE_SNAP_REACH_M` 0.8) and keeps it as a sticky threshold
  (`threshold_events` `source: sightline`); the carve there is a two-cell LINE across the passage (`_severing_cut`,
  the disk of old swallowed a thin sliver or left two crescents of it), chokes are ranked by clearance to observed
  obstacles (`ChokeFields`), a short far side counts when it leads into the unknown (`_parts_skeleton`, `leads_on`,
  `LEADS_ON_CELLS`), a choke must be walled on both sides across the passage (`_walled_both_sides`) and must have a
  cut that parts the floor; and the seedless floor beyond a cut that borders unknown is seeded as a **stub room**
  (`_stub_rooms`, `WatershedRoomParams.stub_min_cells`, 4 in `DEFAULT_SEGMENTATION`, 0 = the older behaviour), exempt
  from `min_room_cells`, keeping its number as it grows. `segment_rooms_watershed(unknown_mask=...)`,
  `door_carve_mask(unknown_mask=...)`. The scene graph's edges now include every threshold with the room pair it
  joins through the doorway coordinate (`graph.doors`, kind `threshold`, index `t<k>`; doors carry kind `door`), and
  a door's annulus starts at the door cell so the stub beyond it is linked. Tests: `tests/test_doorway_partition.py`
  (2), four in `test_room_watershed.py`.
- **Side glances were scheduled and never taken** (`core/planning/object_search/exploration/path_glances.py`). The
  audit of the fifteen recordings of 2026-10-21 found the scheduler wired and firing (40 glances, 33 of them side or
  cue), but the geometric side looks re-planned away before the agent reached them -- Collierville/000000: 27
  scheduled, 4 taken; Markleeville/000000: 27 and none -- the re-scoring every two actions moving the plan to the next
  doorway as the map grew. The plan in force is now **committed**: re-scored on the fresh map, kept while its gain
  clears the floors, replaced only by a candidate at or nearer than its point, taken when its point is reached or
  passed. Three new tests.
- **Redundant circles** (`path_glances.py`): a full circle must pay for its four extra turns with the circle's extra over
  the better side glance (`_full_worth_it`), and is never taken inside a room the search has finished, that has no
  frontier left, or whose frontier walk is in force (`_full_allowed_at`): the recordings' circles promised 5.5 and
  18.5 m2 and revealed 0.8 and 1.9. The loop already re-selects on the action a room's visit ends; no other rotation
  source exists (the sweep's look-around allowance is 0, the fallback never spins).
- **The takeover starts on an RGB detection without depth** (`core/planning/object_search/closing/target_closing.py`):
  a confident whole box of the target's class with no usable depth -- beyond the sensor's ~5 m -- begins a bounded
  BEARING approach once the class has been in the frame on two consecutive actions (`bearing_start`): face the box by
  its image-space bearing, step toward it, re-face it each frame it is seen, until a frame projects with depth and the
  ordinary VERIFY starts there with its own clock. Given up without a rejection memory after `bearing_max_actions`
  (16), `bearing_lost_frames` (3) frames without the class, three steps that moved the agent nowhere,
  `bearing_max_walk_m` (3.0 m) walked toward the box -- a box first seen within the depth cap plus that distance is in
  range by then, and one still depthless is unmeasurable (a television screen is a hole in the Gibson mesh,
  Corozal/000001: the unbounded walk went 3.75 m, never projected, and left the agent to STOP on a second television
  6.95 m from the goal; with the bound the episode succeeds) or not where the bearing pointed -- or a box
  `bearing_near_box_frac` (0.30) of the frame's height tall without depth, which is within reach and cannot be
  positioned, so no blind STOP is taken on it; no retry within `bearing_retry_m` (1.5 m) of the spot the walk began.
  `phase=target_bearing`; counters `bearing_starts` / `bearing_resolved` / `bearing_abandoned` / `bearing_refusals`.
  `test_invalid_depth_never_starts_a_takeover` became `test_invalid_depth_never_locks_but_begins_a_bearing_approach`;
  four more tests.
- **The stair capsule's half-width is a setting, slightly reduced** (`settings.py` `RPTSettings.stair_margin_m` 0.45 m,
  five cells drawn; `world_model/peek_stairs.py`): before, body radius + arrival tolerance + a cell, 0.6 m drawn, which
  with the planner's body-radius inflation on top narrowed the passages beside Markleeville's stairwell on the 2D grid.
  Stairs only; the adaptive capsule still narrows it to the body radius where a corridor would close. Not scene-specific
  by design: the method names no scene. Tests updated in `test_spawn_floor_guard.py`.
- **The stair capsule's storey end is square** (`world_model/peek_stairs.py` `_foot_of` / `_square_foot`, Darden,
  same day): OpenCV's thick line ends in half-discs, and the one at the storey end lay on the floor in front of the
  first riser -- the margin's whole radius of plain floor, 0.55 m with five cells drawn. Darden's flight has its foot
  in the junction of the hallway and the room beside it, and that half-disc, with the planner's inflation on top,
  drew the junction as one oval (`Darden/000000`, the campaign of 2026-10-21: 204 cells, 50 of them hallway floor
  between y 1.03 and 1.53 across the junction's whole width). The end nearest the storey plane is now cut flat across
  the flight's direction, inside every candidate radius of `adaptive_capsule` as well; the sides keep their margin and
  the far end its cap, more flight being behind it. The connector's anchor itself -- the storey floor in front of the
  flight -- is no longer masked or refused as a goal; the first treads still are. Not scene-specific. One new test in
  `test_spawn_floor_guard.py`; two probes moved from the anchor to the first treads. *What the square foot does not
  open:* the doorways between Darden's west and east halves are 0.2 m humps in the mesh (the navmesh walks them; the
  observed map's obstacle band begins 0.15 m above the plane), see progress entry 035.
### Changed (2026-10-21, room-level GAPs: the decision hierarchy of the room search)
- **Exits are a room's attributes, never nodes; the room's visit walks them** (`core/mapping/topology/
  search_node_oracle.py`, `core/planning/object_search/rooms/{room_search_loop,opening_nodes,scene_graph}.py`,
  `policy.py`, `objnav_benchmark_runtime/search_panel.py`). A standalone LLM number per doorway or gap let the RPT*
  order chase open passes for their own sake -- every gap of a living room, in a search for a toilet -- instead of
  the room-level prior. Under `explore` every exit read off the live map is now folded into the room it opens from
  (`_fold_openings`); the room's node line carries its exits as attributes (`SearchNode.exits` / `doorways` /
  `glimpsed`, rendered `exits=2 (1 doorway)  glimpsed through its exits: toilet`, `opening_nodes.room_exits`;
  `ObservedSceneGraph.reason(exits=...)`); the system prompt's new EXITS rule ties them to the room's type (exits
  raise a room only where its type fits the target's home; a hallway's or an unidentified room's by what the home
  still lacks behind them; a living room's gaps never for a toilet), its worked example values rooms by their
  exits, and the OPENING node rules moved to `OPENINGS_SUPPLEMENT`, appended only when an exit node is offered.
  The global decision is over rooms and stairs; the look through a doorway is a step of the chosen room's local
  exploration (the explore visit's frontier walk), which ends when the room's GAPs are exhausted or
  `explore_visit_steps` runs out and whose release is the loop point that re-selects. A ledger-finished room with
  GAPs left to walk (`_gaps_left`: significant live frontier on a room with an exit) is **reopened** for them
  (`_exclusions` -> `_reopened`, event `room_reopened`, stat `rooms_reopened`, `reopened_for_gaps` in the
  diagnostics, `exits` / `reopened` in the estimate events): shown as `status=scanned(...)` with its `exits=`,
  entered at its best frontier, priced by its frontiers, discounted as an entered room, walked and not re-scanned;
  a mid-visit rotation finishes the room in the ledger but no longer ends the visit while GAPs are left, and a
  room in transit with GAPs left is not released `finished_in_transit`. A new exit on an offered room buys the
  oracle a call at most every `revalue_actions` (`_exits_at_call`), as a new opening node did; the HOUSE line
  counts every exit of the storey. `OpeningSettings.exits_as_nodes` (default False) keeps the 2026-10-04 design
  as a deprecated mode -- the opening-node regressions in `test_opening_nodes.py` (scan mode, unchanged) and
  `test_explore_visit.py` run it explicitly; `scan` and `sweep` visits keep exits as peek nodes regardless, and
  target landmarks are nodes in every mode. Visited-node decay (`entered_discount`, `lingered_discount`, the
  supervisor's cooldowns) and the footprint tracking are untouched. Nine regressions in
  `tests/test_room_level_gaps.py`; the room line's `exits=` field is pinned in `test_search_node_oracle.py`.
### Added (2026-10-19, HM3D under the Gibson-validated method)
- **`feat/objnav-benchmark-infra-nadav` merged into `feat/objnav-habitat-hm3d-nadav`**: the HM3D adapter now runs
  the method from `core/planning/object_search/` exactly as the Gibson val result did. `core/` differs from the
  infra branch by the HM3D label table and its registry line only; the HM3D branch's two algorithm-side edits were
  dropped on purpose (the `goal_changed` watchdog exemption in `route_memory.py`, a behaviour change the Gibson
  number was not measured with, and the `target_closing_enabled` switch, a no-op knob). A frozen HM3D
  configuration and a frozen Gibson one differ in the detector's deployment (URL, device, class order) and in
  nothing of the 548 method keys.
- `hm3d/run_benchmark.sh`: the one-command HM3D run -- services, a mesh-free preflight, then `hm3d/campaign.py`
  (one `hm3d.run` process per scene, resumable, aggregated into `campaign_summary.json`) or one `--scene`;
  `--smoke`, `--record`, `--status`, `--preflight`, `--version v1|v2`, `--agent falcon-yolo` as on Gibson.
- `objnav_benchmark_runtime/benchmark_services.sh`: the service orchestration both launchers share (GPU plan,
  Ollama, the YOLO-World detector started with the benchmark's own vocabulary, the segmenter, FFmpeg, teardown),
  lifted verbatim out of `gibson/run_benchmark.sh`, which now sources it; a detector already up at the port is
  checked to serve this benchmark's class order before it is reused (HM3D defaults to port 18092, Gibson to 18095).
### Changed
- `habitat/simulator.py` is the HM3D branch's superset (body height separate from the camera mount,
  `navmesh="published"|"agent"`, `pathfinder`, `navmesh_provenance()`), with the reset-time height
  reconciliation applied under `navmesh="agent"` only: under `"published"` the start is placed verbatim, as every
  Gibson run did. `HabitatRGBDSimulator` now requires `height_m=`; `gibson/env.py`, `smooth_replay.py` and
  `stair_diagnostic.py` pass `GibsonProtocol.agent_height_m` (0.88 m, the camera height -- no change in what runs).
- `provenance.episodes_of_category(ids, episodes, category, categories)` is the one category filter; Gibson's
  `run.episodes_of_category` delegates to it.
- The shared runtime keeps the HM3D branch's benchmark-neutral `dashboard.py` (titles from the summary's
  benchmark/split, unreachable DTGs omitted from the plot), `recording.py` (optional telemetry hook, encoder log
  closed on a failed start) and `run_evaluation`'s `progress` / `wrap_env` / `agent_class` seams.
- **The object-search method moved out of the benchmark runtime into `core`** (`core/planning/object_search/` new,
  `tasks/planning/objnav_benchmark_runtime/methods/` gone, 2026-10-19). A structural change only -- every module
  moved with `git mv`, no algorithm, setting or default changed; a run after this commit differs from one before
  it in `source_sha256` alone. The method (`policy.py`, `settings.py`) is now one package split by concern --
  `perception/`, `rooms/`, `world_model/`, `exploration/`, `multifloor/`, `closing/`, `falcon/`, `baselines/`,
  `route_memory.py` -- and its policy-free pieces went to the subsystem they belong to: `box_extent`, `box_occluder`,
  `detection_dedup`, `target_evidence`, `support_surface` in `core/mapping/objects`; `target_memory_3d` in
  `core/mapping/tracking`; `room_priors`, `revisable_room_labels` in `core/mapping/topology`; the camera-tilt
  controller and the box-driven pitch servo in `core/control/camera` (`pitch_controller`, `box_pitch_servo`: where to
  look is a control process beside where to move, merged in the `NavigationCommand`); the storey localiser
  `floor_atlas` in `core/localization` (from `core/planning/exploration`). Five inline ``atan2(sin, cos)`` wraps now
  call the shared `normalize_angle`; the box aliases use `core.common.math.bbox.BBox`. The seam to the
  services is `object_search/interfaces.py` (`Detection`, the `Detector` / `Segmenter` / `RoomReasoner` protocols,
  the two failure families, which the perception cycle now tells apart without importing `requests`); the HTTP
  clients moved to `objnav_benchmark_runtime/services/` (`http_detector`, `http_segmenter`, `llm`). The runtime
  shares one method factory across datasets (`policy_factory.build_policy`, as on the HM3D branch), lifts
  `detector_options.py` and `gpu_plan.py` to its root (Gibson keeps import shims), folds `select_episodes`,
  `explicit_episodes` and `gpu_plan_record` into `provenance.py` (the duplicate `source_fingerprint` in `gibson/run.py`
  is gone -- same digest), lifts `gibson/progress.py` to `progress.py` (shim kept), and moves the navmesh storey/stair
  reader to `habitat/stair_connectors.py`. The method's
  regression rig stays with the runtime (`tests/`); `core/planning/object_search/tests/` adds the boundary scan (no
  `tasks`, simulator or transport import anywhere in the method), a service-free smoke run and the failure
  contract. READMEs: `core/planning/object_search/README.md` (new), the runtime README's "Where the code is" and
  "Adding a dataset adapter", `core/control` (the camera branch), `core/mapping/{objects,tracking,topology}`,
  `core/planning/objnav`, `falcon`. Verified structural-only: every moved module is code-identical to its original
  (imports and docstrings aside) and four deterministic 140-step rollouts with fake services -- frontier and FALCON
  explorers, with and without a target, and the FALCON+YOLO baseline -- produce byte-identical traces, diagnostics
  and frozen configurations before and after.
### Added
- **Gibson validation result** (`objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md`, `gibson/results/gibson_val_20261018T105339Z/`, 2026-10-19): SR 0.788, SPL 0.465, DTG 0.96 m over 1000 of the 1,000 published SemExp v1.1 val episodes (5 scenes x 200), lean records, habitat-sim 0.2.4; per scene and per category, beside the paper's Table 2 rows, with the frozen configuration, in the file.
- **Gibson validation result** (`objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md`, `gibson/results/gibson_val_20261018_14b_sam/`, 2026-10-19): SR 0.784, SPL 0.480, DTG 1.00 m over 1000 of the 1,000 published SemExp v1.1 val episodes (5 scenes x 200), lean records, habitat-sim 0.2.4; per scene and per category, beside the paper's Table 2 rows, with the frozen configuration, in the file.
- **The FALCON + YOLO + closing baseline on the Gibson val split** (`gibson/results/gibson_val_falcon-yolo_20261018T173302Z/`, 2026-10-19): SR 0.665, SPL 0.337, DTG 1.75 m over the same 1,000 episodes as the full method's SR 0.792, SPL 0.464, DTG 0.94 m; per scene and per category in entry 032. `supervise_benchmark.sh` kills a run stalled for `STALL_MINUTES` (a habitat-sim segfault left a zombie it waited on for five hours) and calls an episode deterministic after three deaths.
- **The FALCON + YOLO + closing exploration baseline** (`objnav_benchmark_runtime/methods/falcon_yolo_baseline.py` new, `gibson/run.py --agent falcon-yolo`, `five_scene.py`, `run_benchmark.sh --agent`, `launch_detached.sh` `RUN_ARGS`, `supervise_benchmark.sh`, `floor_panels.py`, `visualization.py`, `search_panel.py`, `gibson/progress.py`, `gibson/FALCON_YOLO_BASELINE.md`, 2026-10-18): goal-free planar FALCON over the storey (bounded transactions in a 4 m disc about the agent, A* relocation to the nearest frontier when the disc is exhausted, a look-around and a STOP when the storey is) with the full method's YOLO perception cycle, `TargetClosing` takeover and legacy pursuit, camera controller and spawn-floor confinement unchanged, and no LLM, RPT*, scene graph, rooms, doors, openings, peeks or glances. The recorder draws the explorer's column in place of the search panel; lean records keep `exploration`. Twenty-one tests in `tests/test_falcon_yolo_baseline.py`. The swept-motion veto judges FALCON's own routes only (a relocation's or the pursuit's A* route is the planner's, as in the full method's frontier explorer; Wiconisco/000000 had spent 179 vetoes in its first doorway). The first fifteen recorded episodes (`runs/benchmark_falcon-yolo_15episodes_20261018T152349Z`): SR 0.800, SPL 0.433, DTG 0.94 m against the full method's 0.867 / 0.641 / 0.16 m on the same episodes.
- **Gibson validation result** (`objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md`, `gibson/results/gibson_val_20261016_small_llm/`, 2026-10-18): SR 0.792, SPL 0.464, DTG 0.94 m over 1000 of the 1,000 published SemExp v1.1 val episodes (5 scenes x 200), lean records, habitat-sim 0.2.4; per scene and per category, beside the paper's Table 2 rows, with the frozen configuration, in the file.
- **The box read around a surface in front of the object, and a standoff for a big object** (`objnav_benchmark_runtime/methods/box_occluder.py`
  new, `perception.py`, `target_path.py`, `target_closing.py`, 2026-10-15). Two Collierville tv failures that
  reproduced at the same actions in two 1,000-episode runs. (1) 000018: the television's box held 79 % television
  at 4.1 m and 20 % of a seat back's rail at 1.3 m in its lower middle; the 30th-percentile depth read the rail,
  the ±0.12 m support band kept 36 % of the centre patch against a 35 % gate, and the closing STOPped 3.3 m from the
  television after "approaching" the rail. `occluder_in_front` asks the whole box four measured things -- a second
  surface at least 0.5 m behind, spanning 80 % of the box's height AND width, covering more of the box than the
  near one, the near one leaving the top quarter of the box alone -- and `_supported_centroid` then sets the near
  pixels aside and reads the box again (`occluder_depth_m`, `occluder_px` in the row). A chair in front of a wall
  reaches the top of its box; a couch's backrest, a toilet's tank, a bed's headboard span the width only: none
  fires. The mask pipeline is untouched. (2) 000009: a 1.5 m television on its cabinet put every 0.70 m standoff
  point inside the cabinet, against it or behind it, in both planners, and three locks with the memory 0.02-0.14 m
  from the real television were released from 3 m as "no safe path from here"; a standoff 1.01 m from the refined
  memory with a seventeen-waypoint route was dropped as stale for a ring with none. `TargetApproachPath` widens the
  ring in the front arc out to the terminal radius when neither planner reaches the standoff ring (`widened`,
  `widened_standoffs`) and keeps a stale goal within terminal + tolerance when no ring point has a route
  (`stale_kept`, `stale_standoffs_kept`); `TargetClosing` begins the inspection from such a standoff with the fresh
  range within a forward step of the limit and STOPs when the inspection's steps are spent or blocked
  (`closest_approach_stop` on by default, `closest_approach_stops`; `note_blocked` / `note_action` tell the
  inspection whether its step moved). Re-run of the three recorded episodes
  (`runs/gibson_Collierville_tv_fix_20261015T064436Z`): 000018 SR 0 -> 1 (SPL 0.81, 45 actions, the occluder set
  aside on 10 frames, the lock at the real television), 000009 SR 0 -> 1 (SPL 1.00, 32 actions, one widened
  standoff, no boxed release); 000014, a detector false positive, unchanged. The toilet yardstick
  (`runs/benchmark_toilet_10episodes_20261015T064732Z`, the seeded ten): SR 0.90, SPL 0.530, DTG 0.23 m against the
  2026-10-14 reference's 0.90 / 0.456 / 0.59 -- the same nine successes and the same one timeout. Tests: `tests/test_box_occluder.py`
  (9) and six standoff/closest-approach tests in `tests/test_target_closing.py`; the runtime suite 860 pass. The
  per-episode ground-truth tools behind the diagnosis are `runs/_analysis/closing_vs_truth.py` and
  `replay_depth_dissection.py` (gitignored). LESSONS has the entry.
- **The STOP overshoot: decide, then walk the margin** (`objnav_benchmark_runtime/methods/target_closing.py`,
  `rpt_policy.py`, 2026-10-14 night). Every terminal reading is biased short of the object's footprint -- the near
  edge of a box with floor in it, a centroid fused from near surfaces -- and the evaluator measures to the
  footprint: Markleeville/000188 STOPped on a 0.84 m near edge 1.27 m from the toilet (DTG 0.27, SR 0) in the run the
  user chose, `runs/benchmark_toilet_10episodes_20261014T143259Z` (SR 0.80, SPL 0.366, DTG 0.62 m;
  Collierville/000036 129 actions, SPL 0.67, with the back-off fix). Once a STOP is decided, on whichever of the
  seven paths of `_inspect`, `TargetClosing._terminal` walks `ClosingSettings.stop_overshoot_steps` (1) more
  MOVE_FORWARD toward the 3-D memory -- a second, up to `stop_overshoot_max_steps` (2), only while the memory
  still stands beyond `stop_overshoot_within_m` (0.90 m) after the first -- and STOPs then: no frame is consulted (`observe` short-circuits), nothing is
  re-verified; a step is taken only while the memory would still stand `stop_overshoot_min_range_m` (0.45) beyond
  the agent and the deciding frame's near edge a body radius, the agent faces within a quarter turn of the memory
  (beyond one turn the converter turns toward it first), a step the converter reports blocked ends the stage
  (`note_blocked`, from the policy's `notify_blocked`; `note_action` counts the steps) and
  `stop_overshoot_max_actions` (5) ends it without the hooks. Phase `ADVANCE`; `stop_overshoot` in the command
  info and the diagnostics; counter `overshoot_commands`. 0 restores the STOP on the deciding frame -- the shared
  test fixture runs the decision tests so. Three fixes tried in between (`unvalued_prob` 0.05, a fused-memory
  range for small objects, a fold gate on the unknown behind a doorway; run `…150615Z`, SR 0.56) were reverted --
  the fused-memory range flipped CLOSE and INSPECT on alternate frames for 25 actions -- and are in `LESSONS.md`.
  Verification run with two steps, `runs/benchmark_toilet_10episodes_20261014T155701Z`: SR 0.90, SPL 0.450,
  DTG 0.59 m; every episode retraced `…143259Z` within two actions, Markleeville/000188 went from DTG 0.27 to SPL
  1.0, seven of nine overshoots took both steps and two stopped after one at the minimum range. Two steps were
  judged too dear for the SPL (every other STOP 0.01-0.07 lower); one step always (`…162832Z`) gave SR 0.80,
  SPL 0.357 with 000188 STOPped 3 cm outside the disc. The memory at the deciding frame tells the two apart
  (0.88-1.19 m in the eight successes, 1.25 m in 000188), hence the conditional second step -- verified in
  `…170224Z`: SR 0.90, SPL 0.456, DTG 0.59 m, the second step taken on 000188 and Collierville/000023 only.
- **A door beside the stairs, a camera at half a step, and a bump the rays erased** (`objnav_benchmark_runtime/methods/opening_nodes.py`,
  `room_search_loop.py`, `target_closing.py`, `observed_map.py`, 2026-10-14 evening). The openings' stair exclusion
  is directional -- only masked cells and portal points on the frontier's unknown side count (`near_stair_mask(...,
  heading)`, `beyond`): Darden/000034's bathroom door, 1.3-1.5 m beside the flight, was struck off the openings for a
  whole episode and the search walked 17 m to an "entered" living room instead. The closing's geometric pitch is
  quantised with hysteresis (`ClosingSettings.pitch_hysteresis` 0.15, `_quantise_pitch`): a wanted angle at half a
  tilt step chattered LOOK_DOWN / LOOK_UP with the memory's millimetre flicker and the agent never stepped. A second
  bump within `ObservedMap.repeat_bump_m` (0.5 m) of an earlier one leaves a sticky, body-wide bar in every export
  (`_touched`, `sticky_bumps`): the single `l_max` cell decays under the rays that look over a ledge the navmesh
  refuses, and Corozal/000019 bumped three times at one spot and lost its three nearby nodes to the releases.
  The back-off's retreat spot must show the target on the observed grid (`BackoffManeuver._sees`, a Bresenham
  ray stopped short of the target's cells): Collierville/000036 retreated 0.5 m from the goal viewpoint to a spot
  behind the door jamb, disproved the 0.90 toilet from there and lost 380 actions.
  Third verification run `runs/benchmark_toilet_10episodes_20261014T132341Z`: SR 0.80, SPL 0.477, DTG 0.88 m.
- **An exit the oracle has not valued is a node at a prior, not at 0.0** (`objnav_benchmark_runtime/methods/room_search_loop.py`,
  `opening_nodes.py`, `visualization.py`, 2026-10-14 evening). The oracle's numbers are replaced wholesale at every
  call and the fold is read against the numbers before it, so an exit folded into its room at the call and unfolded
  by the reply that dropped the room under `min_prob` came out with no number and read 0.0 -- Corozal/000142 lost
  the toilet's own doorway (12 m2 unseen, 6 m away) to a 12 m gap at 10 % with no SELECT in the 54 actions left.
  `RoomSearchLoop._exit_prob` hands such an exit to RPT* at the oracle's last number for it (`_exit_memory`) or at
  `OpeningSettings.unvalued_prob` (0.10); `prob_source` (`oracle`/`remembered`/`provisional`) in the estimate and
  the HUD record, event `opening_provisional`, stat `openings_provisional`. The node still buys the next call.
  Second verification run `runs/benchmark_toilet_10episodes_20261014T124355Z`: SR 0.90, SPL 0.522, DTG 0.30 m.
- **Five losses of one Collierville recording, and a flight the guard let through** (`objnav_benchmark_runtime/methods/exploration_fallback.py`,
  `opening_nodes.py`, `room_search_loop.py`, `path_glances.py`, `spawn_floor_guard.py`, `rpt_settings.py`;
  `core/planning/exploration/view_gain.py`, 2026-10-14 afternoon). The object-shadow rule no longer drops a frontier
  for a clock or a vase beside it (`SHADOWLESS_CLASSES`, `shadow_min_radius_m` 0.25) nor one with a room's floor
  right behind it (`shadow_exemption`: `shadow_exempt_m2` 2 within `shadow_exempt_range_m` 1.5, through the walls;
  `UnknownView.unknown_in_sector(max_range_m=...)`), and `detect_openings` applies the same rule -- Collierville/
  000023 lost the frontier onto its toilet's room as "the shadow of refrigerator 0" and a doorway as a clock's. A
  peek the supervisor's 0.6 m room-centre tolerance declared arrived short of its threshold walks the remainder and
  looks two turns to each side of the way it came through (`OpeningSettings.step_through`, `through_look_turns` 2,
  `through_steps` 8; `peek_step_through`). A transit whose room is missing from the registry keeps walking for
  `LoopSettings.room_missing_grace` (3) actions before the room is given up (the watershed flickers). The glance
  state records the gains at the agent's own position (`glance.here`). The spawn-floor guard's drop threshold is a
  knob, `RPTSettings.step_down_m`, kept at 0.30 m: 0.15 m masked Corozal/000142's own storey (the plane is the pose
  height). A floor-cut box is judged by its height, with an 8 % width floor (`clipped_box`): a toilet's tank at 1.3 m
  is 65-83 px wide and the 15 % bar refused four 0.91-0.95 frames. The floor-support gate puts an object on the
  storey when ONE terrain sample near it is the storey's floor (`PerceptionCycle._floor_support`): the lowest sample
  beside a toilet at a stairwell was the stairwell's. LESSONS 2026-10-14 (afternoon) has the traces.
- **The toilet re-run's nine behaviours, fixed from their traces** (`objnav_benchmark_runtime/methods/target_closing.py`,
  `target_path.py`, `room_search_loop.py`, `exploration_fallback.py`, `opening_nodes.py`, `perception_cycle.py`;
  `core/mapping/topology/search_node_oracle.py`, 2026-10-14). Target closing: a re-sighting of the ACTIVE candidate
  counts past the rejection memory (it judges a start, not a frame; `resighting_exemptions`); a depthless box on the
  anchor's bearing is a `glimpse` that carries the chain, is faced and stepped toward without running the verify clock
  (`max_verify_approach_steps` 24); the inspection takes one step when the fresh range is a step beyond the limit
  (`inspection_max_steps` 3) and reads a turn that would undo the last turn as aligned (`dead_band_alignments`);
  a locked target with no route on the known floor is walked to through the unknown before the spot is given up
  (`through_unknown`, `through_unknown_cost` 8, the policy's `optimistic_planner`); a candidate first read at
  `backoff_trust_confidence`, or one the detector had been firing on (`backoff_persistence_frames` 2 of
  `backoff_persistence_window` 6), is not backed off from; a fruitless sweep cycle releases the verification
  (`verify_sweep_max_actions` 6). The map: a target landmark seen once at `landmark_sighting_confidence` 0.50 is a
  `T` node, only a release that DISPROVED a candidate hides one (`landmark_refused`), and the floor-support band
  is 0.60 m. The loop: `second_pass` and `scanned_to_oracle` off (a room scanned once is finished and is no node of
  the oracle's), `lingered_discount` 0.2 for an entered room searched 20+ actions with at most one significant
  frontier (and the prompt says so), and a frontier-less room is entered at its doorstep, not its vantage point.
  The fallback: exits ranked by the unknown a look from each reveals, slivers under `explore_min_gain_m2`
  demoted, and the goal in force kept across a glance (`goal_memory_actions` 12). The same 10 toilet episodes
  (2 per scene, seed 17) went from SR 0.700 / SPL 0.425 / DTG 0.94 m to SR 0.900 / SPL 0.489 / DTG 0.23 m
  (`runs/benchmark_toilet_10episodes_20261014T085618Z`, before round 2); LESSONS 2026-10-14 has the traces.
- **Gibson validation result** (`objnav_benchmark_runtime/gibson/BENCHMARK_RESULTS.md`, `gibson/results/gibson_val_20261012T141500Z/`, 2026-10-14): SR 0.713, SPL 0.409, DTG 1.27 m over 1000 of the 1,000 published SemExp v1.1 val episodes (5 scenes x 200), lean records, habitat-sim 0.2.4; per scene and per category, beside the paper's Table 2 rows, with the frozen configuration, in the file.
- **The detached Gibson run as one command, and the same command after a reboot**
  (`objnav_benchmark_runtime/gibson/launch_detached.sh`, 2026-10-12): launcher in its own session, monitor and
  supervisor beside it, `PROGRESS_ENTRY` required so publishing closes the right entry, the supervisor's settings in
  `<output>/launch.env`; on an existing run it starts the Ollama container and a fresh supervisor that resumes from the
  replay snapshot; `--stop` ends a run for good. `--after-reboot` is for a card that is dead until the machine restarts
  (driver upgraded under the running kernel module): the launch's variables go to `launch.env` and a one-shot `@reboot`
  crontab entry launches once `nvidia-smi` and `docker` answer -- never on the CPU, since the device layout is frozen
  into the run. The supervisor's `alive()` now counts a PID only if its command line names the run (PID reuse after a
  reboot). BENCHMARK.md 2.3 documents both.
- **The box in the frame drives the camera pitch** (`objnav_benchmark_runtime/methods/bbox_pitch.py`,
  `TargetClosing._servo`, `TargetClosingSettings.bbox_pitch / bbox_pitch_distance_m 1.50 / bbox_pitch_region
  0.25`, 2026-10-11): within 1.5 m of the target, with the LOOK pair and a fresh box, the pitch is an
  image-space servo's and not the 3-D memory's -- a box in the lower quarter of the frame, or predicted to be
  cut by the lower edge on the next step toward it, is looked DOWN at by the rounded number of tilts; one in
  the upper quarter or about to be cut above, UP; one in the centre band leaves the camera where it is, even
  where the geometric controller would force a tilt (the two would hunt). No class, height or memory enters:
  a frame cut by the lower edge biases the memory exactly when the pitch matters, and every terminal STOP of
  the nine Gibson inspection recordings fired on such a frame. The geometry remains the fallback for a frame
  without a fresh box; without the LOOK pair (the published Gibson protocol) nothing changes. The band is
  never narrower than one tilt's rounding dead band, so the servo cannot hunt; pitch limits cut a tilt back.
- **The terminal STOP is taken on a whole, confident box** (`TargetClosing._before_stop`, `stop_centring`,
  `stop_confidence` 0.50, `weak_stop_max_holds` 2, 2026-10-11): with the LOOK pair, a fresh in-range box the
  servo can re-frame earns ONE centring tilt per inspection and the STOP is taken on the next frame -- on the
  whole box if the detector shows it, on the previous frame's evidence if it does not (the tilt may cost an
  action, never the STOP); a fresh in-range box under `stop_confidence` (the takeover's own start threshold:
  detect high, associate low, STOP high) is held for the one frame that keeps it whole at most twice, then the
  STOP goes ahead on the evidence there is. The STOP's info carries `confidence`, `bbox_region` and
  `depth_valid`; `bbox_pitch_corrections`, `stop_centring_tilts`, `weak_stop_holds`, `bbox_region`,
  `bbox_pitch_steps`, `box_drift_rows` and `goal_kept` join the diagnostics.
- **Regression check of the toilet closing rules on the nine inspection slots** (`runs/gibson_val_20261008T120112Z
  .inspection-fix`, progress entry 027, 2026-10-11): toilet 2/3 improved (Darden/000193 SR 1 at 129 actions,
  Wiconisco/000031 SR 1 at 253), couch 3/3 the same outcome (the Wiconisco false positives are the detector's,
  at 0.72-0.91; 000114 STOPs 31 actions earlier on the same wrong object via the blind-cone rule), bed the same,
  tv Collierville/000043 the same, tv Collierville/000194 regressed -- traced and fixed below.
- **Re-record named episodes of a run for failure inspection** (`objnav_benchmark_runtime/gibson/record_episodes.sh`,
  `BENCHMARK.md` 2.7, 2026-10-11): `record_episodes.sh <run> <out> [--after-run] label=Scene/NNNNNN[,fallback...]...`
  re-runs each slot with `--record` under the run's own frozen settings -- the `run.json` argv addressed to one
  episode (`--scene S --shards 200 --shard-index k --limit 1`), the snapshotted environment, the services reused
  or replayed -- keeps the first candidate whose re-run reproduces the original outcome (a GPU LLM is not
  bit-exact) and tries the next otherwise, and writes `videos/<NN>_<label>_<Scene>_<k>.mp4`, the final frame,
  for a wrong STOP the frames at the closing's first VERIFY / INSPECT / STOP decisions, and `index.md` with the
  original beside every attempt. `--after-run` queues it behind a benchmark still scoring; `--preflight` checks
  the slots without recording. First use: nine representative failures of the 1,000-episode validation run
  (toilet timeouts, Wiconisco couch false positives, circuitous tv/bed successes).
- **Detached Gibson run monitor** (`objnav_benchmark_runtime/gibson/monitor.py`, `tests/test_monitor.py`,
  `BENCHMARK.md` 2.3-2.5, 2026-10-08): the companion of a `nohup`/`setsid` launch of the 1,000-episode
  benchmark that nobody is watching. Every tick it appends one status line (UTC, done/total, running
  SR/SPL/DTG, ETA, the episode in progress, the last outcome) to `<output>/monitor.log` and keeps
  `<output>/episodes.csv` current -- the scalar columns of `episodes.jsonl` alone (id, scene, target,
  success, SPL, SoftSPL, path length, shortest path, start and final distance to goal, steps, STOP,
  termination, wall time, agent error), no diagnostics. Completion is judged by the launcher's PID
  (`launcher.pid`, written by the launcher itself because `setsid` forks), `exit_code.txt` supplies the
  code and is ignored when older than the PID file (a preflight into the same directory leaves one);
  the finished summary goes to the log and `FINAL_SUMMARY.txt`. `BENCHMARK.md` now gives the detached
  recipe, the stop command (`kill -TERM -- -PGID`, the lock being an advisory `flock` so a resume
  follows) and lists the new files.
- **LLM-first node oracle for the single-storey benchmark** (`core/mapping/topology/search_node_oracle.py`,
  `objnav_benchmark_runtime/methods/{house_context,room_search_loop,scene_graph}.py`, 2026-10-07): the 14B
  model is shown EVERY room of the storey -- finished and type-excluded ones included, each with
  `status=never_entered|entered|scanned(<how>, <ago>)` -- and a HOUSE line (room types found with their
  status, rooms unidentified, openings unlooked, whether frontier is reachable, `actions used about N of
  500`), and asked to reason as a person who knows homes does (where the target lives, incl. an en-suite
  toilet reached through a bedroom and a television in a living room then a bedroom, never a bathroom;
  what the rooms found imply for the unidentified ones -- one kitchen, one living room; and how the
  budget weighs: unexplored first, scanned home-type rooms late) before any number, writing `home`,
  `house`, `stage` and a `pass` verdict (`first`/`second`). Under `second` (or a floor with no unfinished
  room, exit or frontier) finished rooms are nodes again and a revisit scans from a spot
  `LoopSettings.revisit_standoff_m` (1.5 m) from the earlier scan points (`second_pass`,
  `second_pass_forced`, `revisits_*` stats). The three arithmetic floors (`unexplored_floor`,
  `unexplored_elsewhere`, `home_floor`) and the hard type exclusion (`type_prior`) are OFF by default; the
  storey/stairs rules became a `STAIRS_SUPPLEMENT` shown only when a staircase is a node. The reasoning
  model is not spent on a loop point with nothing offerable (`oracle_calls_skipped`), in `reconsider()`
  too. `tests/probe_node_oracle.py`: five single-storey scenarios, all five pass on `qwen2.5:14b-instruct`.
- **One-click Gibson benchmark** (`objnav_benchmark_runtime/gibson/run_benchmark.sh`, `BENCHMARK.md`,
  `gibson/progress.py`, `gibson.run --lean --progress-file`): services up (Ollama docker/native/external,
  the CPU YOLO-World detector), preflight, the 1,000 published episodes with LEAN records (the scores
  and the counters that explain an outcome, ~10 KB per episode instead of ~100 KB), a console progress
  bar per episode, `progress.json` rewritten atomically after every episode (done/total, ETA, the
  episode in progress, running SR/SPL/SoftSPL/DTG overall, per scene and per category), `--status DIR`
  and `--summary`, automatic resume into an existing `--output`, services down on exit. The README
  covers install, data, weights, one episode, the full split, sharding, reading the results.
- **The context check on a target candidate** (`objnav_benchmark_runtime/methods/target_context.py`,
  `TargetClosingSettings.context_*`, 2026-10-07): a box whose footprint lies in a room whose STRONG
  type the target is not searched for in (`room:kitchen`), or whose measured centroid height
  contradicts its class (`CLASS_HEIGHT_BANDS`, calibrated on the Allensville recordings: a bed's or
  a toilet's surface under 0.60 m above the agent base, `height:0.69`), counts 0.70 of its confidence
  toward the takeover's start threshold and needs four consecutive frames from two viewpoints
  0.30 m apart to lock. Allensville/2 and Newfields/2 of the 5x3 benchmark STOPped on a kitchen
  counter front read as `bed` from 0.55 m after two frames from one spot (`suspect`,
  `context_rejections`, `suspect_locks_held` in the diagnostics).
- **`override_confidence` (0.80)** on the takeover: a box at that confidence starts a takeover past
  the rejection memory of `unverified` and `contradicted by the map` releases and the release
  cooldown -- a 0.95 toilet in plain view at 2.4 m is not a flicker (the Allensville toilet run
  refused nine such frames) -- while a failed-inspection spot, a stalled approach, a boxed-in spot,
  border clipping and the map's own contradiction still hold; the rejection memory records `why`.
- **A verification frame that keeps the box whole, with the pitch following the target's
  elevation** (`verify_keep_in_frame`, `TargetClosing._elevation`, `elevation_band_m` 0.15 m): a
  fresh, aligned candidate too close to step toward used to be held, and a satisfied hold is executed
  as a TURN (the headless agent's idle action), which moved the centred Allensville toilet out of the
  frame at 1.18 m. The frame is now a LOOK in the direction of the target's elevation -- its 3-D
  centroid against the camera's height, or its box centre against the horizon -- DOWN at a toilet,
  UP at a wall-mounted television, never a fixed direction; level with the camera, the LOOK or TURN
  whose predicted pixel shift keeps the box in the frame.
- **Dynamic pitch on the close approach** (`look_down_distance_m` 1.30 m is now the band of the
  controller, both directions): CLOSE used to force a level camera until the terminal inspection;
  inside the band the approach carries the target's own geometric pitch (`_pitch`, at least one tilt
  toward its elevation, the same function the inspection uses), so a low toilet or a high television
  is held whole on the terminal frames. `_pitch` itself is symmetric now: a target above the camera
  within the band gets LOOK_UP where the rounded geometry said level.
- **Thresholds the agent walked through are cuts** (`room_watershed.trail_thresholds`,
  `WatershedRoomParams.threshold_*`, `ObservedSceneGraph.trail/thresholds`): a clearance dip of
  0.55 m or less with 0.30 m more clearance within 1.5 m of travel on both sides is a doorway,
  carved at the severing choke within 0.5 m exactly as a snapped door is (never a plain disk), so
  the room beyond separates the tick the agent is through it; a uniformly narrow corridor never dips.
- **A home object is a prior** (`LoopSettings.home_floor` 0.60, `SearchNode.home`,
  `SearchNodeOracle(home_floor=)`): a room holding a confirmed home object of the target (a bathtub
  for a toilet) is never read below the floor, whatever the model wrote or its storey verdict says,
  and is never finished by sight from outside it (`seen_from_scan` / `seen_through`) -- the
  Allensville toilet run finished the bathroom with the bathtub inside from the spawn point's spin
  and valued it 0 (`home_object_kept` events with `scanned`).
- Progress entry `docs/progress/entries/024-allensville-audit-perception-rooms-priors-closing.md`,
  regressions in `tests/test_perception_association.py`, `tests/test_target_context.py` and the
  registry/watershed/oracle/scan suites (1969 pass across the runtime, exploration, topology and
  objnav packages; 1891 before).

### Fixed
- **A refined target memory no longer throws away a reachable standoff goal** (`objnav_benchmark_runtime/
  methods/target_path.py`, `TargetApproachPath.kept`, 2026-10-11): a memory shift over `refine_distance_m`
  (0.15 m) used to drop the standoff goal and re-select one from a bearing-relative ring about the new
  position -- a different set of points, each needing a navmesh projection and a path. Collierville/000194:
  the along-ray association refined the television's memory by 0.11 m at 2.5 m, the ring found no path from
  a spot the old goal was reachable from, and the real television was released twice as "no safe path from
  here" (SR 1 -> 0 against the baseline, whose un-refined memory kept its goal). A goal that still ends
  inside `terminal_distance_m` of the refined memory is now kept, the reference is refreshed on every
  refinement (`goal_refinements` no longer climbs every action after a failed re-selection), and a kept goal
  a slow drift carries past the terminal radius is re-selected the action it falls out.
- **The full Gibson run died at episode 919 on a start 3.3 cm outside the navmesh; finishing it is
  explicit and recorded** (`objnav_benchmark/{kinematics,logger,run_info}.py`, `gibson/{protocol,run,
  report}.py`, `gibson/supervise_benchmark.sh`, 2026-10-11): habitat-sim's step filter runs for a turn
  too (`turn_left` is a body action), and `try_step(start, start)` pulls a start that lies *outside* the
  mesh horizontally onto its edge -- Wiconisco/000118, the only such start of the 1,000 (measured),
  "moved" 0.033 m on its warm-up turn and the 0.020 m turn-in-place bound ended the run, twice. New
  `KinematicTolerance.settle_xy_m` (Gibson: 0.05 m, one map cell) excuses that much horizontal placement
  on the first action that moves the agent, like `settle_m` does vertically. Because the fix changes the
  frozen configuration (`kinematics`, `source_sha256`), the resume that finished the run needed
  `gibson.run --resume --resume-accepting-changes kinematics source_sha256`: `MetricsLogger` accepts a
  difference in the named keys only (`RESUMABLE_CONFIG_CHANGES`: provenance or a self-check, never a
  decision or score input), keeps the recorded config, writes both values into the resume's `run.json`
  entry, and the report titles and discloses it. The supervisor forwards `RESUME_ARGS` on every replay.
- **The 1,000-episode run no longer dies on the kinematic contract** (`objnav_benchmark/{kinematics,
  env_contract,runner}.py`, `gibson/protocol.py`, 2026-10-07): published starts lie up to 0.28 m below
  the shipped navmesh and habitat-sim settles the agent on its first TRANSLATION (turns bypass the
  navmesh filter); `KinematicTolerance.settle_m` (0.30 for Gibson) takes that off the first moving
  action additively, and `climb_m` is 0.60 for Gibson because the evaluated storey itself steps up to
  0.50 m on one stride (treads, split levels; the 0.20 m bound refused ~1 step in 300). An agent position
  off the floor map reads as the unreachable sentinel instead of raising; `merge_runs` refuses
  diagnostic shards and compares `target_override`/`publishable`/`kinematics`.
- **Target closing** (`methods/target_closing.py`): a near candidate (inside 2 m) released from the spot
  the agent still stands on is refused until it moves (the rejection memory records where the agent
  stood; `same_spot_rejections`), a suspect candidate may step inside the terminal range to earn its
  second viewpoint, a locked target without a path is kept in frame instead of the idle hold that the
  headless agent executed as a turn (233 reversals in one Darden episode), and a frame the detector did
  not see is NO evidence: the chain is carried, the clocks pause (`blind_frames`) -- before, a transient
  detector failure released a correct lock with a 2 m rejection around the real target.
- **Perception**: a malformed detector reply for one frame is a frame failure (`DetectorFrameError`,
  back-off) rather than the end of the run; only identity drift (vocabulary/model) still is. The
  per-frame `/health` GET is gone (the `/detect` reply echoes the identity). An LLM request that runs
  into its timeout is never retried (one hung call cost 20 minutes).
- **Room-search loop**: a SEARCH-state `route_failed` (the scan's vantage unreachable with the agent let
  in by the arrival tolerance) now ends the turn `unreachable` instead of being dropped for 36 actions of
  blind fallback; a completed peek covers the cone it looked into, so the frontier it reveals through
  the same door is not a new opening (the "twenty openings" drain); the vantage-point EDT runs on the
  room's bounding box; the map-edge `ValueError` is caught inside the fallback guard.
- **Half-levels** (`methods/spawn_floor_guard.py`, `core/planning/exploration/floor_atlas.py`): a settled
  plateau 0.5-1.5 m off the spawn plane that the atlas read as a landing for ever (Klickitat: 361 idle
  turns) is adopted as a level of the spawn storey after 30 actions (`FloorAtlas.adopt_level`,
  `half_level_adopted`); the storey below is not a drop, goals on the level are allowed; a plateau a
  storey away stays confined.
- **Classes never merge in the landmark map, and same-class instances are told apart by size**
  (`core/mapping/objects/landmarks.py`, `floor_context.new`, `RPTSettings.landmark_dedupe_radius_m`
  0.35 / `landmark_footprint_iou` 0.25, 2026-10-07): the ObjectNav map no longer takes the class vote
  (`class_votes=False`) -- a cup on a table, a toilet beside a bathtub, a vase on a cabinet are separate
  instances whatever their proximity or footprint overlap -- and two observations of one class are one
  instance only within 0.35 m (the ported 0.70 m merged two dining chairs) or at a footprint-disc IoU
  of 0.25 (about one radius apart: a bed re-seen from its other side is one bed). The two vases on the
  cabinet in the first frame of the Allensville couch run were dropped as `same_frame_association` on
  every frame, and the toilet and the bathtub 0.5 m beside it were voted into one landmark that flipped
  to `bathtub` and released the toilet's lock as "contradicted by the map" with a 2 m rejection radius.
  The vote (`contradicted_by_map`, `map_rejections`, `map_releases`, `outvoted_detections`) stays in
  the library behind `class_votes=True` and reads zero in this runtime; the per-frame alias filter
  (a box of another class over the same pixels, IoU ≥ 0.9) is the one cross-class rule kept, on the
  detector's output, not the map. `RPTSettings.detection_confidence` 0.35 -> 0.30 (the takeover's
  tracking threshold; a landmark needs two observations, so a weak box costs nothing alone).
- **A box cut only by the bottom edge is not a sliver** (`perception.clipped_box`): a low object at
  1-2 m is cut by the bottom edge on every level frame; a box touching only that edge, with its top
  in the lower half and 15 % of the frame in both dimensions, counts. The Allensville toilet run
  refused 23 such frames of a 0.97 toilet at 1.2-1.7 m and released the candidate twice.
- **Room numbers follow the room, not the larger half** (`RoomRegistry(anchors=True)`,
  `TrackedRoom.anchor`, `memory_rooms`): a fresh mask holding a previous room's birth anchor is
  matched to it before any IoU score (oldest pid first on a merge), and the scene graph's registry
  remembers vanished rooms for the episode (`REGISTRY_MEMORY_TICKS` 1000, bounded to 48 masks). The
  Allensville spawn room was R0 -> R1 -> (merged) R0 -> R2 across steps 0-44 while the hallway the
  agent walked into carried R0 and its "entered" record.

### Added (2026-10-05)
- **The sight ledger** (`objnav_benchmark_runtime/methods/sightlines.py`, `SightSettings`,
  2026-10-05): every action the camera's cone is cast over the map it has just updated; an
  unknown cell a ray ends at inside the floor's visible band (beyond the blind radius, within
  2.5 m) from two distinct poses is **looked through** -- a railing, a window, a mesh hole -- and
  a connected unknown region under 3 m2 enclosed by known cells is a **pocket**. Both are
  settled: written OCCUPIED on the map the frontier logic reads (`ObservedSceneGraph.
  frontier_world` via `resolved_provider`; `accessible_frontiers(frontier_world=)`;
  `UnknownView(resolved=)`), so they are no frontier, no opening, no fallback exit and no
  glance gain; the planner and the display keep the real map (the panels draw settled unknown
  darker, `settled_cells`). Hanson 2026-10-05 walked to both ends of a balcony seen whole from
  its threshold, back to an island between the spawn point and the bed (O4, 31 actions) and
  behind a bed (O9, 42 actions). `sight` in the configuration and the episode record.
- **Two more finished-room verdicts** (`room_scans.py`, `LoopSettings.walkthrough_clearance_m`
  0.9 m / `fragment_max_m2` 3 m2): with no live frontier left, a room is `seen_through` when one
  recorded pose had half of it in the camera's cone with clear sight, or the agent stood in it
  and it is narrow (a corridor, a balcony); and a `fragment` when it is under 3 m2 with no door
  on it (the strip behind a bed). A room finished by what the walk to it showed is released
  before it is entered (`finished_in_transit`). The Hanson balcony was offered again at 0.25,
  "never entered", 158 actions after the agent had stood at its far end.
- **Cue glances** (`path_glances.py`, `GlanceSettings.cue_*`): a confident detection spanning
  30 % of the frame's height and cut off by its left or right edge turns the agent toward it -- enough turns to centre it and one
  more; the target's class and its home objects outright, any other class while that side still
  holds unknown floor; once per class and side within 2 m. Hanson action 101: a bathroom vanity
  (`cabinet 0.84`) on the left edge, two right glances taken, the toilet 250 actions later.
- **The footing sweep** (`camera_control.begin_inspection(reason="footing")`,
  `TargetClosingSettings.footing_*`, the exploration fallback's sixth rung): one LOOK_DOWN to 30
  degrees, a full circle and a LOOK_UP map the floor under the camera's blind radius, which an
  agent that has not moved cannot plan across (unknown is impassable); taken only while a quarter
  of the cells within 1.2 m are unknown. A LOCKED target with no
  safe path asks for it after two pathless actions and, six pathless actions after it, releases
  the lock **without a rejection** (`boxed_releases`; no far takeover from within 1 m of the
  spot) so the search can walk; the fallback takes it before its last-resort hold. Hanson/000002
  spawned beside a bed with the plant in view 4.4 m away and spun 159 actions on "no safe target
  path" to an agent error.
- **Signature objects** (`room_priors.SIGNATURE_OBJECTS`, `RoomLabelSettings.signature_objects`):
  a bed, a toilet, a shower, a bathtub, an oven, a stove or a refrigerator makes the classifier's
  agreeing label STRONG on its own, so the type prior rules the room out at once; the home-object
  and target-seen guards still hold. Hanson action 270: a bed and a television through the door,
  the "bedroom?" kept weak, the room entered and scanned for a toilet.
- **A start-clearance rule in the episode samplers** (`generate_development.
  MIN_START_CLEARANCE_M` 0.35 m, `start_clearance`; both `generate_start` and
  `cross_floor_starts`; `start_clearance_m` in the audits): no start within 0.35 m of the navmesh
  boundary. Frozen manifests keep their starts.
- **`generate_development --multistory --categories couch`** (`multifloor_generation.
  cross_floor_starts(categories=)`): a cross-floor campaign restricted to the goal categories named,
  skipping buildings without one on the reference floor; `generation.goal_categories` in the manifest.
- **Distractor prompts** in the Gibson context vocabulary (`toy`, `rocking horse`,
  `stuffed animal`) and `bathtub`: Hanson/000001 stopped at action 14 on a child's ride-on horse
  read as `chair` 0.9. A running detector service must be restarted with the new vocabulary.
### Changed
- **End-to-end review before the 5x3 benchmark (2026-10-05, evening)** -- small fixes only, found by
  reading every layer and replaying the Ranchester recording:
  - *Room ids* (`room_watershed.door_carve_mask`, `RoomRegistry(memory_ticks=10)`): a door snapped to a
    choke whose disk did not actually sever the floor had no separating power (the medial axis was
    parted, the floor was not; 47 % of the snapped disks in the Ranchester replay), so the room behind
    it merged into the hallway on the ticks the snap found a choke and split off under a new number on
    the ticks it did not -- R23, R26, R28, R52 for one room. Such a disk now falls back to the plain
    barring disk; and the registry keeps a vanished room's mask for ten updates, so a room absorbed
    for a tick re-adopts its own pid (ties broken by overlap, not raster order). The replay goes from
    36 pids for 10 rooms (26 deaths) to 11 pids for 11 rooms (3).
  - *Finished rooms re-judged on growth* (`RoomScanLedger(regrow_factor=1.5)`, `regrown` in the
    diagnostics): a sticky `fragment` / `seen_through` verdict reached on a sliver no longer finishes
    the room that grows out of it under the same number; a scan point still inside finishes it again.
  - *Generic objects alone classify nothing* (`RoomLabelSettings.generic_evidence_gate`): one cabinet
    made a "kitchen" at 0.9 on Ranchester's upper storey and the storey summary showed the oracle a
    kitchen found; weak labels are marked `kitchen?` in that summary as they are on the node lines.
  - *The node oracle prompt*: STEP 2 reads the storey from the rooms found AND its RANK in the building
    (`stair_nodes.storey_position`: "this storey is the highest of 2 known storeys, 1 above the lowest"
    instead of "(this one at +3.1 m)", which the 14B read as a ground floor; the rooms decide, and the
    line names no floor -- a first version that called every storey above the lowest "an upper floor"
    had the 14B say the living room lives elsewhere on Newfields' ground floor, whose lowest storey is
    a basement, with a living room found on it); an opening whose glimpsed objects
    name the home type is rule 1 (60 or more -- the 14B valued a sink-glimpsed gap at 0.20 for a
    toilet, "a bathroom, no toilet"); the worked example carries a toilet counter-example so a small
    model cannot copy its `elsewhere`; `parse_home_here` reads a negated "found" and a bare "no" as
    `missing`, never `elsewhere`.
  - *The room classifier* reads "Living Room" / "living-room" as `living_room` instead of `unknown`.
  - *The evaluator* (`MultiFloorDistance.success`): the height gate is the nearest goal sample's, not
    the region's median -- Klickitat's chair samples lie at +0.18 m and -0.41 m (a split-level floor),
    and a STOP beside the lower chair scored 0 four centimetres from the region. The campaign report
    writes `statistics.json` and `RESULTS.md` before it raises on a missing video (`missing_recordings`).
  - *Generation*: `generate_development --multistory --start-storey same` samples the start on the
    annotated storey (SemExp-style same-floor starts on the multistory harness, so SR is measurable),
    and a building with fewer annotated categories than episodes is ineligible (`distinct_categories`;
    Onaga's three episodes would all have been the couch).
  - *Target closing*: a fresh, off-centre sighting at a pitch other than the predicted one is centred
    at the pitch it was seen at (the converter tilts before it faces, so the predicted pitch LOOKed
    away from a target in view for the whole 24-action budget); an inspection the filtered estimate
    entered while every fresh frame measures the surface beyond the limit is ended and the approach
    resumed (`inspection_resumptions`) instead of releasing a target in plain view as "saw nothing";
    a glance in force is aborted the action a takeover starts; the landmark map keeps voting during
    the takeover so the documented map release can fire (the legacy target evidence stands down,
    `takeover_active`); the closing's own release turn is not charged to the room loop; a cue glance
    skips a box perception placed on another storey, outside the map or on a refused candidate; a
    detector whose vocabulary, model or configuration changed under the evaluation ends the run
    instead of entering the transport back-off.
  - *Exploration fallback*: a blind-radius demotion is sticky (the goal is retired and the goal in
    force cleared -- the step toward the next exit promoted it back and the fallback turned between
    two exits for ever, 0.25 m a step so no watchdog fired); the relocation target is kept until
    reached (the farthest cell from a moving agent flips between the two ends of a hall); routes are
    planned with every seen staircase written occupied (Pomaria 2026-10-04 walked down a flight it had
    not chosen behind a floor-wide frontier); the per-step record names this action's rung.
  - *After the 5x3 benchmark (same evening; not in its numbers)*: the closing releases a lock whose
    approach went nowhere -- `approach_stall_actions` (20) CLOSE actions inside `approach_stall_m`
    (0.20 m) -- with a rejection (`stall_releases`; Leonardo pushed into an unseen obstacle for 160
    actions and raised the closing bound); the search's blocked clock is reset by 60 % of a forward
    step, not a collision slide (Leonardo wedged for 350 actions); an opening released BLOCKED,
    UNREACHABLE or TRANSIT_TIMEOUT is retired for the storey instead of re-chosen by the escape hatch.
  - *Stairs*: a storey first reached by a ground-truth traversal is pinned to the connector's navmesh
    height (`storey_height_pinned`; the atlas had measured Hanson's upper storey 0.17 m high from
    poses on the eased last treads, so no staircase seen up there ever became a portal and the map's
    floor-clear mask never fired); `storey_position` uses the atlas's match tolerance; the fallback
    rule's "destination already searched out" verdict re-checks in 10 actions instead of hiding the
    stairs node from the oracle for 100; `_reapproach` re-snaps the entry by geometry alone; no arrival
    cooldown on the source portal in ground-truth mode (the grace and the rooms left hold the way back).
- **A strong room label needs a distinctive kind of object** (`RoomLabelSettings.distinctive_required`,
  `room_priors.GENERIC_OBJECTS`): a cabinet and a potted plant no longer make a strong "living_room".
- **The oracle's STEP 2 is a structured verdict** (`search_node_oracle.parse_home_here`, `home_here` in the
  reply schema and the worked example; `SearchNodeOracle(unexplored_elsewhere=)`, `LoopSettings.
  unexplored_elsewhere` 0.10, `NodeOracleResult.capped` / `home_here`): with `home_here="elsewhere"`
  every unexplored node is read at 0.10 exactly instead of the 0.25 floor. The Ranchester cross-floor
  couch episode (`runs/zson-couch-crossfloor-3x-20261005`) had the 3B model say "living rooms are
  downstairs" sixteen times and 25 on every upstairs gap sixteen times, and spent its 500 actions on
  thirteen peeks with the stairs (0.60) last in every order -- standing 0.9 m from them at action 171.
- **Target closing, after the second Hanson fly of 2026-10-05**: the footing sweep is cut short the
  action a path exists (the sweep's 30 degrees had stood for the whole approach); the inspection
  pitch follows the target's measured height, not its label (a plant in a tall planter is at eye
  level); a fresh, aligned, in-range sighting STOPs at whatever pitch it came at (the toilet
  projected at 60 degrees where its height predicted 30: 20 actions of LOOK_UP/LOOK_DOWN); a fresh
  unlocked candidate within one turn of the centre is stepped toward rather than centred (the
  centring turn dropped a far chair four cycles running); and the legacy target evidence refuses
  what the takeover refuses (`refuses_far_candidate`; nine actions walked toward a just-released
  chair, and the move restarted the warm-up).
- **Labels are read before the nodes are chosen** (`ObservedSceneGraph.refresh_labels`, called
  by the loop's `_reason` before the exclusions and the oracle): a room that became a strong
  bedroom on the loop point's own re-classification was shown to the oracle, valued 0.10 and
  kept in the order for one more loop point (Hanson action 150).
- **Room ids survive growth** (`RoomRegistry(containment_threshold=0.6)`): a pair under the IoU
  threshold matches when 60 % of the smaller mask lies in the larger; IoU matches are consumed
  first. One Hanson bedroom was R11 -> R14 -> R16 while the agent stood in it, and each number
  lost the record of having been entered.
- The exploration fallback's retired-frontier rung no longer retries a goal under the agent's
  feet (the converter has no action for it: an idle turn per action).
- The oracle's per-room `frontier=` counts, the openings and the fallback's exits are read off
  the sight ledger's overlay; `test_opening_nodes` turns the ledger off in its fixture because its
  two-room world's "patch by the door" is exactly the pocket the ledger settles.
### Added (2026-10-05, earlier the same day)
- **Informative glances along a route** (`objnav_benchmark_runtime/methods/path_glances.py`,
  `core/planning/exploration/view_gain.py`, 2026-10-05): the route in force is scored every few
  actions for the one point where a look to the left, to the right or all round would reveal the
  most unknown floor the walk itself will not (optimistic rays through the unknown, stopped by
  walls, the forward cones along the route subtracted); if the gain clears 3 m2 and 0.5 m2 per
  action, the look is performed when the agent gets there -- one in-place turn per action, a
  suspended phase like the warm-up (no loop charge, clocks paused, the route's watchdogs told the
  pause). A spot where a full rotation already stood is never glanced from; a target sighting
  or a stair traversal aborts a glance. `GlanceSettings` on the policy, `glances` in the method
  configuration, `episode_info` and the per-step `search` record, a HUD line.
- **An uncertainty floor on unexplored nodes** (`search_node_oracle.SearchNodeOracle(unexplored_floor=)`,
  `LoopSettings.unexplored_floor`, default 0.25): a never-entered room still `unknown`, or an
  opening nothing was glimpsed through, is read at no less than the floor whatever the model
  wrote (`floored` in the result and the estimate events). The room line says `entered=no`; the
  prompt gains rules 2b (unexplored places owe a look; never "too small") and 2c (a home-type
  object in a room of another type means the map merged two rooms); the worked example no
  longer writes a small unknown room off. Hanson 2026-10-04 left its storey at action 27 with
  two never-entered rooms and six doorways valued at 0-1% by the 3B model.
- **Door cuts snapped to the choke** (`core/mapping/topology/room_watershed.py`
  `WatershedRoomParams.door_snap_reach_m`, on in the ObjectNav scene graph at 0.9 m): a detected
  door is moved onto the nearest medial-axis constriction whose removal parts the skeleton into
  two substantial pieces, and the disk carved there is sized to the passage; basins on different
  sides of the carved mask never merge (`merge_basins_by_dynamics(basin_sides=)`), basins on one
  side still may; a door with no choke in reach keeps the plain disk. The Hanson bedroom (R0/R6)
  is one room again, R7 and the corridor stay apart; the other fourteen recorded floors move by
  at most one room.
- **Furniture belongs to the room around it** (`ObservedSceneGraph.object_room` / `room_near`):
  a landmark standing on occupied cells is credited to the nearest room floor within its
  footprint radius plus 0.6 m (the bed, the desks and the wardrobe of Hanson's bedroom were
  `room: null` and the room was a "living room?" from its chair).
- **Home objects** (`room_priors.HOME_OBJECTS`, `ruled_out`): a room holding an object of the
  kind the target lives with (a sink or a shower, for a toilet) is never ruled out by type.
### Changed
- **The type prior rules out STRONG labels only**: a weak label (one kind of object) never
  excludes a room; the oracle sees it as `type=bedroom?` and values it (`weak_type_kept`, once
  per room; `LoopSettings.weak_type_max_openings` is retained but no longer consulted). The
  relabel-ends-the-visit rule follows the same test.
- **The peek's approach bound follows the route the planner adopted**
  (`peek_approach_extended` events): Hanson's peek of action 166 was sized on a 3.5 m geodesic
  and cancelled 5.8 m short on a 12.5 m route still being followed.
- **Distances are measured the way A\* flies them**: `accessible_frontiers(preferred_cost=)` and
  `build_instance(preferred_cost=)` take the cost grid at the planner's preferred standoff and
  measure every reachable distance there, falling back to the body-radius graph where only a
  squeeze connects; `RPTSearchPolicy.preferred_cost` supplies it. Reachability is unchanged.
- **The room visit is a scan** (`objnav_benchmark_runtime/methods/room_search_loop.py`,
  `LoopSettings.visit="scan"`, 2026-10-04): transit to the room's vantage point -- the reachable
  interior cell of greatest clearance (`room_vantage.py`) -- turn a full circle on measured yaw,
  and leave; the room is finished for the episode. The **scan ledger** (`room_scans.py`) remembers
  where every rotation stood, not which room id it was credited to, and finishes a room when a scan
  point lies inside it or a scan saw more than half of it through observed free space; the verdict
  is sticky per `(floor, pid)`. Finished rooms and rooms whose identified type cannot hold the
  target (`room_priors.py`; a confirmed target object in the room outranks the prior) are **not
  nodes**: not shown to the oracle, not solved, not entered (`excluded` events/estimates/HUD).
  A clue mid-visit is deferred to the loop point; a relabel ends the visit only when the new type
  rules the room out. The former bounded sweep is kept as `visit="sweep"`.
- **The warm-up is one full rotation** (12 x 30 deg), recorded as the storey's first scan so the
  spawn room is finished before any room is chosen; repeated on every storey first entered.
- **Stairs are never a room**: seen stair footprints are removed from the free mask before the
  watershed (`ObservedSceneGraph.update(exclude=policy.room_exclusion(world))`).
- **Object identity is a per-frame vote** (`core/mapping/objects/landmarks.py` `class_votes=True`,
  wired in `floor_context.py`/`perception_cycle.py`): detections associate by position (dedupe
  radius or footprint-disc IoU >= 0.15, `perception.footprint_radius_m`) with a 0.35 m height
  check; each landmark tallies `votes` per class, one per frame; its class is the plurality,
  relabels are recorded, confirmation needs a clear plurality; target evidence is judged on the
  map's class (`contradicted_by_map`), so a `sofa` box on a five-times-confirmed bed is outvoted.
- Target closing on a big object at close range: a fresh box spanning the image centre column is
  aligned whatever its centre says; the terminal range is the near edge of the measured surface
  (`range_near_m`); an exhausted inspection after an in-range sighting STOPs
  (`stop_on_exhausted_inspection`) instead of raising. The border gate is shared with the legacy
  target-evidence path (`perception.clipped_box`). The rejection memory of a released candidate
  does not block a view from within `rejection_min_range_m` (2 m), and the lock's association
  radius grows with range (`association_range_gain`).
- Under `scan`, the staircase the agent arrived by is withheld while the storey still has a room
  that is neither finished nor ruled out (`way_back_held` loop events); a warm-up moved more than
  0.5 m mid-rotation starts its circle again.
- **Target closing starts only on a depth-projected candidate** and **tracks low once active**
  (`target_closing.track_confidence`, 0.30): a confident box the depth sensor cannot place no longer
  buys a twelve-step verification ("waiting for valid target depth", Ranchester attempts 4-5), and a
  re-sighting of the active candidate on its anchor counts from the lower threshold -- the agent's own
  centring turn dropped the Ranchester couch from 0.68 to 0.42 and reset the consecutive-frame run on
  every cycle (24 frames in view, two releases). `weak_resightings` in the diagnostics.
- **The exploration fallback ranks exits first** (`exploration_fallback.py`, `FallbackSettings.exits_first`):
  a frontier that is an object's shadow (beside a confirmed landmark's footprint, no longer than it
  could cast; `object_shadow`) or belongs to a room the loop ruled out by type waits in a new
  `frontier_demoted` rung behind every genuine opening and the stairs. The Ranchester recording's
  actions 61-102 chased ten strips of unknown between the beds while the passage to the stairs stood
  two metres away. `last_demoted` in the fallback record; `RoomSearchLoop.excluded` is public.
- **The fallback commits to its goal** (`goal_switch_gain` 2.0, `goal_match_m` 0.6): the frontier in
  force is driven until it is gone or refused unless another of its rung is worth twice as much; the
  greedy per-action order, re-ranked by the agent's own turning, produced a 180-degree turn toward one
  goal and a 210-degree turn back (Ranchester attempt 6, actions 42-55). Frontiers inside the camera's
  blind radius (`near_blind_m` 1.2) -- the floor under the agent that no step toward resolves -- are
  demoted with the shadows. The per-step record carries `method.fallback` (stage, goal, demotions).
- Per-storey loop events/stats/exclusions in `building.floor_contexts`; HUD shows the visit budget,
  scan phase and the rooms that are not nodes. Progress entry 019.
- **Openings are nodes** (`objnav_benchmark_runtime/methods/opening_nodes.py`, `LoopSettings.openings`,
  2026-10-04): every genuine exit of the mapped floor -- a frontier that is not an object's shadow,
  not under the agent, at least 0.8 m wide after merging within 1.5 m, not at the foot of a seen
  staircase -- is offered to the oracle and to RPT* beside the rooms and the stairs, with a sticky
  building-wide id (`O<n>`), the room it opens from, `doorway`/`gap` by a confirmed door frame, and
  the objects glimpsed through it; priced at a peek's 4 actions (`build_instance(service_s=...)`).
  Its visit is a **peek**: walk to the threshold (bound: the distance's worth of steps x 1.6 + 6, at
  least 20; one re-aim when the cell goes occupied), face the unknown, one look to each side, and the
  opening is retired for the storey (`peek_look`/`peek_complete`/`peek_abandoned`/`peek_reaimed`
  events, `OpeningRegistry` in the episode record). The heading into the unknown is from the known
  floor around the frontier cell toward the unknown; a glimpse is a landmark 0.3-3 m beyond the
  threshold inside a 35-degree cone (attempt 8 read the stair passage as a bedroom door from a
  hallway bed). A new opening while nothing is in force buys an oracle call at most every 6 actions.
  The oracle's prompt explains OPENING nodes and
  shows one in its worked example. Ranchester attempt 7, action 75: a toilet glimpsed through a door
  named the hallway a bathroom and every door off it was demoted with it.
- **A weak type label does not rule out a room with more than `weak_type_max_openings` (1) openings**
  (`weak_type_kept` events): the one object that named it may belong to the room behind a door. Openings
  count as "rooms left" for the way-back hold.
- **Room numbers are unique across the building**: a storey first entered numbers its rooms after the
  highest pid any storey has used (`RoomRegistry(first_pid=...)`, `FloorContextBank.new`), so `R0`
  names one room in a recording. Stair node ids are now a band (100 000-199 999); openings start at
  200 000. HUD: `OPENINGS (nodes)` section, `O<n>` markers with the heading into the unknown on the
  active floor map, the peek in force. Progress entry 020.
- **A confirmed landmark of the target's class is a node** (`opening_nodes.landmark_openings`,
  `OpeningSettings.landmarks_enabled` / `landmark_prob` 0.85 / `landmark_standoff_m` 1.5; `T<n>` on the
  HUD): offered to RPT* at a fixed probability without an oracle call, at a standoff on the agent's side,
  visited by the same peek (walk facing it, one look to each side); inspected landmarks and spots in the
  takeover's rejection memory are not offered again (`TargetClosing.rejected_near`). Ranchester attempt
  9 walked forty actions away from a confirmed `sofa` 3.8 m from the stair foot.
- **A locked target whose terminal inspection never saw it is released, not an episode error**
  (`target_closing.release_on_failed_inspection`, default on; `failed_inspection_radius_factor` 2.0): the
  anchor joins the rejection memory with twice the radius, the release action is one turn in place, and
  the search resumes. The Ranchester same-storey run ended as `agent_error` on a sofa-from-four-metres
  that was nothing from one, the real couch 2.85 m away. The release comes as soon as every inspection
  view has been tried once with nothing seen (not after the 24-action budget), and at once when the
  map's class vote at the anchor outvotes the lock (`map_releases`; the sofa from four metres is the bed
  from two). The rejection memory now records a radius per spot (`rejected[].radius_m`);
  `inspection_releases`, `map_releases` and `last_release` in the diagnostics.

### Changed
- `doorway_peek.enabled` and the room-coverage floor gate (`doorway_peek.gate_floor_departure`)
  default to **off**: a peek costs a scan and finishes nothing, and the gate held the Ranchester
  descent 14 times. Both remain available as ablations.
- Development runs use `qwen2.5:3b-instruct` as the reasoning model; the 14B stays the benchmark model.
- The Ranchester couch-downstairs development episode (`runs/zson-couch-downstairs-20261004-scan`)
  now succeeds in **150 actions (SR 1, SPL 0.71)** where the 2026-10-04 morning runs failed at 28 and
  500; the seven attempts of the day (185 / 215 / 167 / 150 actions on the successful ones) are kept
  under `attempt-*` with the diff each ran.
### Added (2026-10-01, HM3D: shared navigation core)
- HM3D ObjectNav now runs the same navigation core as Gibson. `feat/objnav-habitat-gibson-nadav`
  is merged into `feat/objnav-habitat-hm3d-nadav`: `objnav_benchmark_runtime/methods/` (RPT*,
  room-search loop, frontier sweep, exploration fallback, multi-floor coordinator with
  ground-truth stairs, YOLO-World default detector), `habitat/` and the shared
  evaluation/recording path are one core; `gibson/` and `hm3d/` are the two task modules
  that own their environments, loaders and protocols. The core imports neither task
  (`tests/test_runtime_boundaries.py`). `hm3d.run` gains `--explorer frontier|falcon`,
  `--detector-backend` and `--detector-timeout-s`; `HM3DEnv` attaches the navmesh storeys
  and stair connectors to the episode like `GibsonEnv` does. See
  `docs/progress/entries/013-*.md`.
- `hm3d/README.md` rewritten as the HM3D execution guide: environments, dataset and weight
  downloads, one recorded episode with the live dashboard, and the scene-per-process
  campaign, sharding and freeze-then-evaluate workflow for the full published split.
### Changed (2026-10-01, HM3D merge)
- `habitat/simulator.py` is the union of both branches: explicit `height_m` and the
  `navmesh="published"|"agent"` protocol choice with `navmesh_provenance()` (HM3D), plus
  `scene_structure()`, `last_collision` and RGB/depth extrinsic checks (Gibson). Gibson's
  protocol now states `agent_height_m = 0.88` explicitly instead of reusing the camera height.
- `gibson/stair_connectors.py` moved to `habitat/stair_connectors.py` (it reads any habitat-sim
  navmesh); the old path re-exports it.
- Target-closing recording HUD shows active/confirmed lock, phase, observed target range,
  elapsed time and FPS; per-frame LLM/classifier/RPT/A* counters permit freeze verification,
  and normally completed videos end with SR/SPL/DTG/runtime metrics.
- Single live toilet verification retained with failure evidence: takeover/lock worked, but
  route steering and reacquisition lost the target before LOOK_DOWN or STOP. No retry was run;
  the aborted run has a separately labeled diagnostic report, not a completed benchmark row.
- Irreversible ObjectNav target takeover: configurable confidence, two consecutive depth-consistent
  frames, bbox yaw centering, close low-target LOOK_DOWN and fresh verified explicit STOP; room/LLM/RPT*
  exploration cannot resume after takeover, including through failures or detection loss.
- Geometry-only local Habitat NavMesh projection for observed target standoff goals, followed by
  collision-qualified A*; availability is declared in run provenance, with no evaluator target/DTG access.
- Development evaluation now prints finalized video paths and SR/SPL/DTG/runtime/FPS immediately
  after each episode, supports timed inspection pauses, and saves completion and throughput summaries.
- Recorded HUD now includes evaluator-only DTG, scene, current room type/confidence and a
  coordinate-based active frontier ID, without exposing goal distance to the navigation policy.
- ObjectNav room-coverage floor gate: every observed room, including the spawn room, needs a
  completed interior peek and measured 180-degree scan before normal or fallback stair selection;
  new rooms revoke an approach, failures remain pending, and accidental early stair entry retreats safely.
- Regression coverage for real-converter interior scans, floor-local visit identity, failed and
  split rooms, stair-choice bypasses, adaptive floor-map scaling and persistent room partitions.
- **Stairs are known when they are SEEN** (`objnav_benchmark_runtime/methods/stair_sightings.py`):
  a perfect stair detector replaces the connector list the policy used to be handed at reset. Each
  navmesh connector's surface sample is projected into the current frame (`camera_geometry
  .project_to_image`) and depth-tested against the depth image; six visible samples make a
  sighting with a bounding box, drawn orange on the RGB overlay and carried in `detections` with
  `source: ground_truth`, `distance_m`, `visible_points`. Only a seen staircase becomes a portal
  (`portal_placed` with `seen_step`, `seen_from_m` and the sightings' `footprint`), an RPT* node
  or the explanation of a height departure -- a raised bathroom floor is a bathroom floor, and a
  staircase never in frame does not exist to the search (`height_departure_ignored: connector N
  underfoot has never been seen`). `GroundTruthStairs.touching/nearest(..., among=seen)`,
  `StairSightings` (first/last step, sightings, nearest distance, deduplicated footprint),
  `building.stairs_seen` in the diagnostics.
- Stair connectors rebuilt (`gibson/stair_connectors.py`): one connector per pair of ADJACENT
  storeys (a stairwell serving three storeys is two connectors, not one 5.4 m flight through the
  middle storey); the polyline is the flight's CENTRELINE -- one vertex per 0.2 m height band, the
  band component that touches the previous band's -- with anchors on the flight's own line and a
  verified flat EXIT point off each end (`bottom_exit_xyz`/`top_exit_xyz`); every leg is
  stride-checked with `PathFinder.try_step` (the simulator's `move_forward`) and repaired through
  the pathfinder only where a stride fails, legs under 0.3 m merged; `surface_xyz` (the detector's
  sample), `walkability` counts and `traversable` (False when the flight joins two navmesh islands:
  Hanson, Leonardo, Marstons, Shelbyville -- seen, never climbed) ride with each connector.
- `tests/rollout_stairs_navmesh.py`: every connector of every scene climbed both ways on the real
  navmesh by the real `GroundTruthTraversal` and `DiscreteActionConverter`, MOVE_FORWARD applied
  with `try_step` and sliding. 29/29 traversable climbs complete, 25 with zero blocked steps;
  island-split connectors reported SKIP. Exit 1 on any failed climb, so a campaign can gate on it.
- `tests/test_stair_sightings.py`: the detector (in frame, hidden behind a nearer wall, half
  hidden, behind the agent, beyond range, depth slack, polyline stand-in) and the sightings memory.
- Floor maps (`floor_panels.py`): every SEEN staircase drawn where its treads were seen (footprint
  dots, entry square, `S<n> up/down`); node labels only for the first six of the RPT* order plus the
  node in force and the next, every other room a dot; labels placed on the first free side and
  dropped rather than written over another; dark outline under every label; object dots smaller.
  Panel title counts the stairs.
### Changed
- Room peeks are now strictly one attempt per episode-local room: initial classifications,
  completed scans and consumed attempts bypass both repeat peeks and impossible floor-exit
  requirements. No cancellation refunds an attempt; overlap preserves the allowance through
  segmentation renumbering/splits/expansion without claiming a scan completed.
- Seen, current-floor stair connectors are excluded from peek viewpoints and peek-only A*
  routes; committed stair/atlas transitions cannot be interrupted by peeks. A coverage-vetoed
  room transit is released instead of repeatedly replanning into the same left/right loop.
- ObjectNav enables a bounded 1-degree path-heading hysteresis margin to prefer safe forward
  progress over reversing-turn chatter; generic converter behavior remains unchanged by default.
  1,206 core/runtime tests pass; headless Ranchester ascent/descent complete without blocked steps.
- Same-spawn Sofa verification (one Ranchester episode, unchanged policy/source) reached 500
  actions with SR=0/SPL=0/DTG=12.5852 m. Stairs were seen at 135, but 62 room-coverage vetoes
  kept the agent upstairs; descent, lower-floor target closing and STOP were not exercised.
  Full HUD video, exact-spawn proof and official metrics retained in `runs/sofa_same_spawn_20261001`.
- Target closing now retains its confirmed 3D target and NavMesh standoff path through occlusion;
  A* owns transit heading, fresh observations refine the goal, and stationary terminal inspection
  at 1.0 m (0.05 m confirmation tolerance) owns pitch/bbox servo and explicit STOP.
- Expected closing failures are recordable method errors rather than infrastructure aborts;
  the HUD distinguishes persistent lock from live visibility. Added occlusion/refinement/corner
  regressions; 645 runtime and converter tests pass.
- One recorded Ranchester toilet check verified unchanged LLM/RPT* counters, path continuation
  across three occluded decisions, LOOK_DOWN and policy STOP at 0.874 m, with no agent error.
  Official SR=0/SPL=0/DTG=10.8924 m remain: this manifest scores downstairs reference-floor targets,
  not the upstairs detection. This is closing-control verification, not benchmark success.
- Doorway peeks go a further metre inside the room with stopping-tolerance compensation and a
  deepest-safe-viewpoint fallback for small rooms; simply crossing a doorway no longer counts as scanned.
- Floor maps now show thin logical room outlines, short IDs and minimal route/robot/stair markers;
  object dots, probability text, frontier diamonds and visit-order chains stay off the map. A shared
  observed-house viewport and scale bars replace the fixed 32 m crop; room partitions persist per floor
  and are saved separately as numeric `room_partitions.npz`. Camera, depth and search column are unchanged.
- Development check of the room-peek gate: four recorded seed-17 episodes across Ranchester and
  Pomaria completed with no runtime failures, but **0/4 navigation successes**. Thirty measured
  half-turn scans completed; room repartitioning caused 37 peek cancellations and repeated scans,
  leaving rooms pending and preventing every floor change. The prior successful Ranchester couch
  episode regressed to the action limit. The 368 passing runtime tests validate behavior, not a
  performance gain; visit continuity across repartitioning remains an unresolved follow-up.
- `GroundTruthTraversal` never skips a vertex: a tight aim blocked twice sends the agent BACK to the
  last vertex it reached (`traversal_recovery`, `recoveries` in the diagnostics) and the route is
  resumed from there; the route handed to the converter ends at the next bend (`BEND_RAD`, 35
  degrees), so a landing corner is walked to, not cut by the half-metre lookahead; a vertex reached
  from off to one side is led back THROUGH (`VIA_M`); reached means 0.26 m in plan view and 0.45 m
  in height (`REACHED_XY_M`/`REACHED_Z_M`), replacing the 0.35 m sphere; the exit is the
  connector's verified exit point, then a stub swung 45/90 degrees either side and never behind.
  Pomaria's two flights: 40 actions each way, 0 blocked, in the probe and in the recorded run.
- `MultiFloorParams.arrival_grace_actions` (40): the staircase the agent arrived by is withheld from
  the loop's node list and from the fallback rule (`MultiFloorSearch.way_back_held`, `way_back_held`
  event) until the new storey has been looked at -- the node oracle chose the way back on the
  arrival action three times in one recording. Any other staircase is offered at once.
- `floor_decision`/`stair_nodes` see only seen, traversable connectors; `test_room_search_loop
  .stair_policy` marks the fixture's staircase seen.
### Earlier in this cycle
- ObjectNav node oracle (`core/mapping/topology/search_node_oracle.py`): the room LLM is now
  asked ONE question per loop point over every node the search could go to next -- each room
  (type or `unknown`, size, frontier left, time searched and how long ago, objects seen, whether
  the robot stands in it) and each staircase (up/down, whether the other storey was visited and
  what was found there, whether the robot arrived by it) -- for the probability, in percent,
  that going there NEXT finds the target, plus `elsewhere`. The system prompt states the
  judgement in full: an irrelevant type or a fully observed room reads ~0; a room searched long
  and recently reads low; an `unknown` room is an exploration node valued by size, frontier and
  the room types still missing; a hallway with frontier by what it leads to; a staircase by what
  this storey turned out to be; and never by distance, which RPT* charges. Code keeps only the
  contract (parse, drop invented ids, omitted nodes get a small share, rescale, clamp below 1,
  refuse a reply with no usable node, reuse a byte-identical prompt's reply). `RepairingNodeOracle`
  retries one malformed reply. `LLMClient` gains a REASONING route -- `LLM_REASONING_MODEL`
  (default `qwen2.5:14b-instruct`, 9 GB at Q4, chosen to fit beside Habitat in 30 GB of RAM),
  `LLM_REASONING_TIMEOUT_S` (600 s), `LLM_REASONING_MAX_TOKENS`, `LLM_REASONING_NUM_CTX` (8192,
  requested explicitly because Ollama truncates an outgrown window from the front) -- for this
  one call; the default model keeps the cheap, frequent ones; `VerifiedLLMClient` checks both are
  provisioned. Runtime is not the constraint: the search waits for the answer. Measured on the
  laptop's CPU container: ~25-30 s per call once loaded, ~75 s cold. `tests/probe_node_oracle.py`
  runs three scenarios against the live model; the prompt was tightened on its first run (size
  judged against the target's usual room -- a toilet's room is small; a known kitchen elsewhere
  outweighs a room that might be one; an example that cannot be parroted).
- Staircases as nodes of the RPT* order (`methods/stair_nodes.py`, `room_costs.build_instance
  (leaves=...)`): every reachable, uncooled portal of the floor is a node beside the rooms, with
  an id above every room pid, valued by the oracle, and sitting at the foot of its flight with a
  LEAF of the flight's length plus `MultiFloorParams.floor_change_cost_m` (8 m) charged on every
  arc into AND out of it -- so going upstairs puts every room down here that much further away
  and the expected-time-to-find objective sees the cost of changing floors. When the order puts a
  staircase first the loop commits it to the building coordinator (`MultiFloorSearch.commit`),
  the climb begins at the foot and the node's turn ends `traversed` (`ObjectSearchSupervisor
  .finish`); the way back down is a node like any other, marked `arrived_by`, with no cooldown.
- The clue rule: one confirmed object names a room (`RoomLabelSettings` 1/1/1; a single class is
  a *weak* label, shown to the oracle as `type=kitchen?`), the classifier is re-asked for the
  room in force or in transit the action a NEW KIND of object lands in it (a verdict for a set of
  kinds is reused across the watershed's re-partitions; 22 → 5 calls in the 60-action trace),
  and a changed name ends the room's turn with the new neutral verdict `RECLASSIFIED`
  (`room_reclassified` exit, in SEARCH or in TRANSIT before entering; not cooled, no attempt
  charged) so the estimate and the order are redone over the new fact.
- Committed ground-truth traversal (`methods/ground_truth_traversal.py`, `MultiFloorParams
  .commit_failures/commit_stall_actions/confirm_actions`, `FloorAtlas.settle`): a blocked step
  tightens the following to a straight line at the next vertex before any vertex is skipped;
  turning back takes 12 blocked steps or 30 stalled actions; the transition budget no longer
  turns the climb round; a storey the agent stands on at the far anchor is confirmed outright
  after 24 actions when the plateau test cannot fire (`destination_forced`); a failed approach
  is re-snapped and retried before the connector is deferred.
- Dashboard search column (`objnav_benchmark_runtime/search_panel.py`, `visualization
  .search_snapshot`, `floor_panels.render(search=...)`): the target and the oracle's split
  between the nodes and elsewhere, the RPT* visit order with its head (`S0` a staircase), the
  node in force or in transit, every room's type/probability/reason/frontier/time/objects, every
  staircase's probability, direction, climb cost and storey beyond, the objects by room and the
  loop's last events; rooms and stair nodes named and valued on the active floor map with the
  order drawn through them and the next node ringed. The same block rides every `steps.jsonl`
  row and `live.json`; `room_search_loop.estimate_events` records every oracle round.
- Tests: `core/mapping/topology/tests/test_search_node_oracle.py`, the leaf test in
  `test_room_costs.py`, `objnav_benchmark_runtime/tests/test_search_panel.py`, plus the
  reclassify, stair-node, fallback-rule, commitment and forced-confirmation regressions in the
  existing modules. `tests/trace_room_search_loop.py` drives the policy through the real
  `DiscreteActionConverter` on a point agent.
### Changed
- **No step budget decides a floor change in ground-truth mode.** `floor_search_actions` is the
  observed mode's allowance only; `floor_decision.py` is the exploration fallback's last resort
  (`rule: fallback`), asked when the floor is exhausted and the loop has nothing to offer, and it
  charges the climb through the same `stair_cost_m` as the RPT* leaf.
- `ObjectSearchSupervisor`: `RECLASSIFIED` (neutral) and `TRAVERSED` (productive) verdicts,
  `update(room_reclassified=...)`, `finish(verdict, note, now)`, `is_cooling()`; `NEUTRAL`
  verdicts charge no attempt.
- `ObservedSceneGraph.reason(..., extra_nodes, context, here_xy)`; `graph.probs` are the oracle's
  find-probabilities unnormalised (they sum with `stair_probs` to `p_present`); `last_inside`
  tracks when the agent was last in each room.
- `RoomTypeClassifier.cached(classes)` reads a held verdict without a call.
- `search_oracle.py` (the 3B-era per-room affinity oracle) is no longer used by the runtime; it
  remains for the ROS scene-graph stack.
### Earlier in this cycle
- ObjectNav exploration fallback (`objnav_benchmark_runtime/methods/exploration_fallback.py`):
  every failure of the decision pipeline -- an A* with no path, an RPT* instance that cannot
  be built, a room LLM that times out or answers badly, a bug in the loop -- now ends in a
  command that moves: the nearest reachable frontier anywhere on the floor (leaving the
  current room if that is where it is), then the stairs, then a retired frontier, then a
  relocation to the farthest reachable known cell; a hold only when the agent stands off the
  passable map. `RPTSearchPolicy.plan` wraps the decision and routes any exception there, so
  a model failure is no longer an agent error that ends the episode. Failed model services
  (room LLM, detector) get a doubling back-off (25 → 200 actions) during which they are not
  asked; failures are counted, kept with type/message/origin under `exploration_fallback` in
  the episode record, and logged once per type. See `docs/progress/entries/012-*.md`.
- Ground-truth stairs for the multi-story policy (`MultiFloorParams.stair_source`, default
  `ground_truth`; `observed` keeps the RGB-D discovery for comparison). The evaluator reads
  each scene's navmesh once (`gibson/stair_connectors.py`): area-supported storey heights and
  every off-level surface cluster as a connector with floor anchors and the navmesh shortest
  path, attached to the episode metadata (no goal, distance or target floor). The policy
  (`methods/stair_ground_truth.py`, `floor_decision.py`, `ground_truth_traversal.py`) makes
  each connector a portal, decides explicitly when to change floors -- eligible, reachable
  on the observed map now, destination unvisited or with frontier left, nearest first; up or
  down falls out of the connector -- with one recorded `floor_decision` event per distinct
  verdict, and walks the connector's own polyline with a FOLLOW command on every action.
  `configuration()["ground_truth_stairs"]` declares it. Verified on the real Ranchester
  navmesh: two storeys, one 5.5 m connector, the flight the recordings climbed.
- `objnav_benchmark_runtime/tests/{test_stair_ground_truth,test_ground_truth_floor_transitions,test_exploration_fallback}.py`
  (38 tests) plus the metadata/tripwire and YOLO-default regressions.
- `objnav_benchmark_runtime/QUICKSTART.md`: the end-to-end runbook for the Gibson ObjectNav
  simulation -- the three environments, the licensed dataset and weight downloads, then the
  services, frozen episodes, preflight, smoke run and campaign in strict order, with the
  refusals a newcomer meets and their fixes. Linked from the root README and the runtime and
  Gibson READMEs; its service and preflight steps were executed as written.
- `provision_grounded_vlm --model none --include-yolo` provisions YOLO-World X-v2 + CLIP alone
  (~0.5 GB) instead of forcing a Grounding DINO / BLIP-2 download for the YOLO-only default.

### Changed
- YOLO-World only is the default object-detection pipeline everywhere: `--detector-backend`
  defaults to `yolo_world` in `gibson.run`, `run_development`, `distinct_buildings`,
  `compare_explorers` and `stair_diagnostic` (`detector_options.DEFAULT_DETECTOR_BACKEND`),
  and `scene_graph/serve/configs/gibson_perception.json` selects `yolo_world`. Grounding
  DINO / BLIP-2 (`grounded_vlm`, `hybrid`) remain selectable but disabled until their stair
  verification is revisited; stairs no longer depend on the detector at all.
- In ground-truth stair mode the building coordinator runs no look-down inspections and
  applies no depth veto on the committed route; an unplanned height departure starts a
  transition only within 0.75 m of a known connector's polyline (`near_connector_m`) and
  beyond `departure_m`, and any other departure is logged once (`height_departure_ignored`)
  and left to the atlas. A spent retreat lets the atlas settle where the agent stands instead
  of halting the episode (`SAFE_HALT` is observed-mode only).
- `RoomSearchLoop._fallback` delegates to the exploration fallback; `_reason` turns a room-LLM
  failure into a recorded, backed-off `RoomReasoningUnavailable` instead of raising out of
  `plan`. `LoopSettings`/`RPTSettings` unchanged; `FallbackSettings` added to
  `configuration()`.

### Fixed
- The multi-story policy no longer starts a "stair traversal" on a raised bathroom floor or on
  a 12 cm height wobble at a real stair head and then turns in place "acquiring tread support"
  -- 120 + 12 actions of the 500 in `runs/room_search_loop_ranchester_20260928T093541Z_transitfix`.
  See LESSONS.
- The room-search loop's "nothing to work in" answer is no longer an empty hold that the
  headless agent spends as one idle `TURN_LEFT` per action until the step budget runs out.

### Added (earlier, same branch)
- ObjectNav frontier explorer now runs the room-search loop as specified
  (`objnav_benchmark_runtime/methods/room_search_loop.py`): bounded local exploration of the
  room in force -- `LoopSettings.local_steps` (10) actions or no frontier left -- with every
  in-room route planned on a copy of the map in which the other rooms are blocked, so no
  route crosses a door; then, on the same action the room ends, re-classify every room from
  every object so far, re-estimate P(target) per room, re-solve the RPT* order (local budget
  folded in as service time) and transit to the *nearest frontier inside* the chosen room
  (centroid when it has none); arrival is the agent's cell inside the room mask and resets
  the counter. The room LLM runs at loop points only (plus once at the start and when a room
  the estimate has never seen appears while nothing is in force); scene-graph geometry
  refreshes every action (`graph_period_steps` default 10 -> 1). The episode record carries
  `room_search_loop` (per-room estimates with the distance the order charged, and one event
  per transit/arrival/release). See `docs/progress/entries/011-objnav-room-search-loop.md`.
- `ObjectSearchSupervisor`: caller-driven `arrived` (TRANSIT -> SEARCH by the caller's own
  mask test) and `budget_spent` (SEARCH -> `BUDGET_SPENT` by the caller's action count);
  `frontier_exhausted` now also ends a TRANSIT whose room the map filled in from outside
  (`exhausted_in_transit` stat). New params `cooldown_verdicts` (which verdicts cool a room),
  `repeat_verdicts` (which cooling rooms the every-room-cooling escape hatch may repeat) and
  `resolve_on_release` (re-ask the solver after every release). Defaults keep the flown
  behaviour; the loop leaves `BUDGET_SPENT` out of the cooldown and `EXHAUSTED`/`MAPPED` out
  of the hatch.
- `frontier_ranking.frontier_goals_by_room`: every room's ranked, reachable frontier goals in
  one extraction and one Dijkstra, clusters credited to rooms by the same majority vote
  `count_frontier_clusters` uses -- so a room's goals and its reported cluster count are one
  population. `room_costs.frontier_clusters` exposes cluster member cells for it.
- `SearchOracle` reuses the model's last reply when the prompt it would show is
  byte-identical, re-applying the code-side time/frontier factors (`OracleResult.reused`,
  `oracle_reuses` in the episode record): a loop point over an unchanged map costs no call.
- `ObservedSceneGraph.labels` (the pid+1 room label image) and `room_at()`.
- Two full recorded Ranchester episodes of the room-search loop from two start positions
  (`runs/room_search_loop_ranchester_20260928T091752Z`: 000000 toilet from (2.97, -11.49),
  000001 couch from (-1.03, -19.97)) -- the first Ranchester episodes ever to run to
  completion on this workstation; every earlier attempt aborted on the room LLM. Plus
  `objnav_benchmark_runtime/tests/timeline_room_search_loop.py`, which prints a recording's
  `steps.jsonl` as a collapsed phase/room/command timeline with the oracle rounds and the
  detections that reached the target track.

### Changed
- A frontier cluster's goal is now its member cell nearest the centroid (on the boundary),
  not the centroid itself: for the arc-shaped clusters a depth range produces the centroid
  lies inside seen floor, so every goal was "no longer informative" on the action it was
  adopted and the room read exhausted with clusters still showing (see LESSONS).
- `frontier_sweep.py` is reduced to goal generation, committed-goal lifetime and the optional
  look-around; `SweepSettings.room_scan_turns` defaults to 0 (the loop's termination rule is
  *N steps or nothing left*) and `supervisor_rounds` moved to `LoopSettings` (default 5, the
  longest legitimate same-action chain; an overrun is counted and logged, not silent).
  In-room goals, the room entry point and the transit re-aim share one admissibility filter.

### Fixed
- Room-search loop transit no longer idles at a route end that the re-segmenting map moved out
  of the target room: arrival now also counts on the room's boundary (within the sweep's mask
  slack -- the eroded watershed mask stops short of the very frontier the transit aims at), and
  a route that ends where the room is not re-aims once at the room's current nearest frontier
  or releases the room as unreachable (`entry_lost` stat and event). The first recorded
  Ranchester run (`runs/room_search_loop_ranchester_20260928T091752Z`, episode 000000) spent
  31 of 299 actions turning in place at exactly such a spot until the route memory's 30-action
  no-progress clock released the room. See LESSONS.
- ObjectNav frontier explorer no longer spins in place for the supervisor's 30/90 s clocks
  once a room is swept: the new `frontier_sweep.py` gives a swept room one bounded
  look-around (a full rotation by default) and then ends it at once through two new
  `ObjectSearchSupervisor` exits, `frontier_exhausted` (verdict `EXHAUSTED`, productive) and
  `route_failed` (`UNREACHABLE` without the plan-grace wait). A released room is replaced by
  the next transit on the same action instead of by a throwaway floor-wide route. The
  Ranchester recording `e7c4f2ad5402` spent 24 % of its 472 actions on these idle turns. See
  `docs/progress/entries/010-objnav-exploration-efficiency.md`.
- Stair (look-down) inspections no longer fire every cooldown while following a path once the
  floor allowance is spent: they now start only for a named reason (`floor_exhausted`,
  `stair_detection`, `periodic`), the latter two only at an action with no committed route,
  and unprompted ones at most every `periodic_inspection_actions` (100). The same recording
  ran ten sweeps (17 % of actions) plus 7 % of turns recovering the heading each left behind.
- `CommittedRoute.reusable` no longer overwrites the recorded clear reason with `unplanned`,
  so `route_replaced` (new in every adopted route's info) and the per-reason
  `cleared:<reason>` stats say why a route was really dropped.

### Changed
- Frontier goals are ranked by utility (`core/planning/exploration/frontier_ranking.py`):
  sub-linear cluster size over geodesic distance on the planner's passable graph, with facing
  as a discount. Clusters with no known-free path are dropped instead of proposed, and up to
  three ranked goals are tried per action, so a refused A* no longer costs an idle turn. A
  committed frontier goal is kept while it is passable, still borders unknown space and is
  near the room being swept -- a re-segmented room mask alone no longer replaces it every
  ten actions, and a boundary the camera has already resolved no longer keeps it.
  `in_room_frontier_goals` (size order) remains for pose-free callers and the scene-graph
  count; both now share `frontier_cluster_cells`.

### Fixed (earlier)
- `rooster_twist_control_adapter.py`'s `max_yaw_rate` recalibrated from a never-validated
  0.5 rad/s to 1.8 rad/s, derived from a logged manual flight's actual turn-rate behavior
  (~4x too low previously — any planner-requested yaw rate was executed much faster than
  intended). See LESSONS.md and `docs/progress/entries/007-rooster-velocity-controller.md`.
- Broken `torch` import in the project venv (`venv/`): removed an orphaned cu12 NVIDIA package
  cluster conflicting with `torch==2.11.0`'s actual cu13 requirement, and force-reinstalled two
  cu13 packages (`nvidia-cudnn-cu13`, `nvidia-nvshmem-cu13`) whose library files were missing
  despite being reported installed (corrupted/interrupted earlier install). See LESSONS.md.

### Added
- Explicit, default-off shared-GPU authorization for the detector and frozen
  Gibson development campaigns; records the permission in model/run identity
  without changing navigation, scoring, CUDA checks or the detector memory cap.
  See `docs/progress/entries/009-grounded-vlm-gibson-evaluation.md`.
- Selectable Gibson perception modes: YOLO only, Grounding DINO Base + BLIP-2
  FLAN-T5 XL, or all three. One JSON `backend` setting controls model loading;
  raw/verified frame evidence is recorded without changing depth, floor or STOP
  safeguards. Added pinned/checksummed offline weight provisioning, explicit
  local CLIP loading, bounded verification and forwarded detector timeouts.
  See `docs/progress/entries/008-grounded-vlm-perception.md`; native staircase
  accuracy is not yet established.
- New `detector` container (`docker/Dockerfile.detector`, `bake.hcl` sibling target off
  `perception`, `docker-compose.detector.yml`, started persistently as `detector_dev`) running
  the YOLO-World detector sidecar — torch/ultralytics/CLIP kept out of `perception`/`robotican`
  since `yolo_world_trt` is shared task infra, not ROBOTICAN-specific. `run_object_mission_sphera.sh`
  now launches the sidecar via `docker exec` into it instead of the bare host venv, and passes
  `rgb_topic:=/R1/rgb_frame_path` explicitly (previously unset, silently defaulting to XTEND's
  topic). See LESSONS.md for three build issues hit and fixed along the way (numpy pin conflict,
  a mis-built CLIP wheel, TensorRT engine/version lock).
- Built the YOLO-World-`s` TensorRT engine (`yolo_world_trt/build_all.sh s`) end-to-end for
  the first time on this PC (installed `ultralytics`, downloaded `yolov8s-worldv2.pt`) —
  `./run_object_mission_sphera.sh --detector-only` now actually starts the detector sidecar.
  `m`/`l`/`x` are not built yet.
- Sphera/Rooster fork of the "pick an object, then fly to it and land" mission stack:
  `adapter/launch/object_mission_sphera.launch`, `config/mission_sphera.yaml`,
  `run_object_mission_sphera.sh`, and a placeholder `objects_sphera.json` (no real object
  catalog exists for `sphera_jail` yet). Mirrors XTEND's existing `object_mission.launch` stack
  the same way `sphera_drone.launch` mirrors `real_drone.launch` — additive, XTEND's originals
  untouched. Dry-validated (`mission_config.py` + `--help`); not yet flown live. See
  `docs/progress/entries/005-yolo-object-navigation.md`.
- Two new `mission_control.py` services (`rooster_jetson` group): `Rooster Frame Relay ->
  Jetson (R1)` and `Rooster Jetson Frame Watcher (R1)`, wiring the existing
  `dir_push_relay.py`/`dir_watch_path_publisher.py` mechanism (previously XTEND-only, never
  tested for Rooster) so captured frames can be forwarded to the Jetson over rsync/SSH,
  additive to (doesn't change) the existing local Rooster/Falcon vision pipeline. Verified
  end-to-end through the orchestrator's own start/stop/status code, not just ad-hoc commands —
  see `docs/progress/entries/006-rooster-frame-jetson-relay.md`.
- Closed-loop PD altitude hold for the ROBOTICAN Rooster (`rooster_unit.py`), replacing
  the previous open-loop throttle constant that reliably drifted to floor or ceiling.
- `demo_mode_topic`/`demo_mode_request_topic` params on `nav_stack.launch`, letting
  `sphera_drone.launch` route Rooster's demo-mode handshake through `/R1/...` topics
  instead of the XTEND-shaped `/xtend/...` defaults.
- `rooster_demo_mode_manager.py` (new, `falcon_adapter` ROS1 node): a minimal Rooster
  equivalent of `xtend_drone_demo_manager.py` — echoes a requested demo mode back as
  the authoritative current mode, with no other side effects (deliberately does not
  auto-land on FINISH the way the XTEND version does).

### Changed
- `sphera_drone.launch` now overrides FALCON's navigation controller to `multi_axis`
  for Rooster (a genuinely holonomic platform), instead of the default `roll_assist`.
- Every Y-axis bound in `maps/sphera_jail.yaml` (`init_y`, `map_min_y`/`map_max_y`,
  `box_min_y`/`box_max_y`, `vbox_min_y`/`vbox_max_y`) and the Y-axis launch args
  documented in the `fly-rooster-sphera` skill (`bev_ymin`/`bev_ymax`/`goal_y`) —
  negated and min/max-swapped to match the corrected localization sign (see Fixed).
- `map_max_z` raised `4.0` → `5.0` and `box_max_z`/`vbox_max_z` raised `1.8` → `4.0`
  in `maps/sphera_jail.yaml`, so RViz shows real room geometry up to the actual
  ceiling instead of truncating at an artificially low height, while keeping a
  real (1.0m) margin between the map and box/vbox bounds (see Fixed for why the
  margin matters).
- `cam_min_depth` raised `0.1` → `0.45` in `sphera_drone.launch` (see Fixed).
- `mapping_sync`'s `freeze_on_turning_mode` set to `false` in `sphera_drone.launch`
  (see Fixed) — turning-smear protection is temporarily disabled pending a real fix
  to the rotation-supervisor bug it exposed.
- Stale saved RViz camera position in `maps/sphera_jail.rviz` (`Focal Point Y: 14.66`)
  updated to `Y: -14.66` to match the corrected localization sign.

### Fixed
- `waypoint_follower_node.py`: `_publish_twist_multi` was defined twice in the same
  class; the second (older) definition silently shadowed the first and didn't accept
  the `vz` keyword the caller passed, so every navigation tick threw a `TypeError`
  before ever calling `.publish()`. FALCON's internal state showed `nav=RUN` the whole
  time, but `/cmd_vel` was never actually emitted — the drone never moved toward a
  clicked BEV goal regardless of controller/topic configuration.
- `rooster_ground_truth_localization.py`: `position.y` was passed straight through
  from Sphera/Unreal telemetry while yaw was already negated for the left-handed →
  right-handed conversion, leaving position and rotation handedness inconsistent.
  Completed the conversion by negating `position.y` too (see LESSONS.md for how this
  was verified).
- Click-to-fly deadlock for Rooster: the default `roll_assist` controller's demo-mode
  confirmation handshake never resolves because nothing publishes to `/xtend/demo_mode`
  for this platform — fixed via the `multi_axis` controller switch above (Rooster is
  holonomic and doesn't need the turn-then-forward handshake `roll_assist` requires).
- Noisy/speckled voxel map: the bottom ~25% of every RGB frame showed a near-constant
  `~0.17-0.35m` depth reading (the camera rig/mount itself, visible in its own FOV,
  not real environment) that was below `cam_min_depth` and so got fused as a permanent
  phantom wall directly ahead of the drone on every frame — very likely the real cause
  of "boxed in, no A* route" seen the same day. Fixed by raising `cam_min_depth`
  (see Changed).
- `exploration_node` crashing (`voxel_mapping::ESDF::getDistance` → glog FATAL
  "Address out of range") introduced by setting `vbox_max_z` exactly equal to
  `map_max_z` — ESDF's neighbor-cell queries need a real margin near a boundary, not
  just `map ⊇ vbox` ordering. Fixed by raising `map_max_z` instead (see Changed).
- RViz appearing completely empty (no error, just nothing rendered): a stale saved
  camera focal point in `sphera_jail.rviz` from before the Y-axis fix pointed the
  camera at the mirror-image empty location where the room used to be under the old
  sign convention (see Changed).

### Known issues (not yet fixed)
- `waypoint_follower_node.py`'s rotation supervisor gets stuck permanently requesting
  `"turning"` mode while its navigation loop targets the default startup goal with no
  real flight dynamics ever confirming a turn is complete — this permanently froze
  `mapping_sync`'s rotation-freeze mechanism once `rooster_demo_mode_manager.py` made
  it actually engage. Worked around by disabling `freeze_on_turning_mode` for now
  (see Changed); the supervisor bug itself is unresolved.
- Possible left/right (lateral) mirroring in the built map: reported once during a
  BEV-driven flight (drone next to the real left wall in Sphera; map showed it next
  to the right wall). Forward/back and altitude are both confirmed correct via
  quantitative ground-truth tests, so this is a different bug from the Y-axis fix
  above — most likely in how the camera's local lateral axis projects into world-frame
  points, not the drone's own tracked pose. Not yet conclusively confirmed or
  root-caused; a physical landmark was added to the test hallway to check this
  properly next session.

<!--
Example of a real entry once you have one:

## [Unreleased]
### Fixed
- Hover z-axis drift at hover_z=560 traced to accumulated integral windup in the altitude
  controller, not the sim's ranger noise as first suspected. See LESSONS.md.
-->
