# Changelog

All notable changes to this project are logged here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/) — one line per change, written for
future-you, not for a commit log.

## [Unreleased]
### Added
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
