# 035 - Room-level GAPs: exits are a room's attributes, the room's visit walks them

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** implemented; the rig's room/opening/oracle suites pass (190 + 9 new); not yet flown on a Gibson episode
**Roadmap item:** ObjectNav exploration efficiency (010) and the room-search loop (011); revises 020 (openings as nodes)

## Diagnosis
Since 2026-10-04 (entry 020) every doorway and gap to unseen space was a node of its own in the RPT\* scene
graph, with an LLM number of its own. A standalone probability per gap let the order chase open passes for
their own sake -- every gap of a living room valued and peeked one after another in a search for a toilet --
in place of the room-level prior the oracle was built to apply (a toilet lives in a bathroom; a bathroom's
unlooked door is worth more than three gaps off a living room). The exploration became spatial where it was
meant to be semantic.

## Goal
A two-level decision: the GLOBAL loop values rooms (and stairs) only, each room's line carrying its open
GAPs as attributes the model weighs against the room's type; a chosen room's LOCAL phase walks its GAPs until
none is left or the visit's bound runs out, then control returns to the global loop. Visited-node decay and
the footprint tracking unchanged; the change confined to the scene graph handed to the oracle/RPT\* and the
loop's state machine.

## Steps
- [x] `search_node_oracle.py`: `SearchNode.exits` / `doorways` / `glimpsed`; the ROOM line renders
      `exits=2 (1 doorway)  glimpsed through its exits: toilet`; the EXITS rule (exits raise a room only where
      its type fits the target's home; a hallway's or an unidentified room's by what the home still lacks; a
      living room's gaps never for a toilet); the worked example values rooms by their exits; the OPENING rules
      moved to `OPENINGS_SUPPLEMENT`, appended only when an exit node is offered.
- [x] `opening_nodes.py`: `OpeningSettings.exits_as_nodes` (default False, the deprecated 2026-10-04 mode),
      `RoomExits` / `room_exits`.
- [x] `scene_graph.py`: `reason(exits=...)` -> `nodes(exits=...)` on the room nodes.
- [x] `room_search_loop.py`: `LoopSettings.room_level_gaps`; every exit folded under `explore`; the exits
      attributed to the rooms and counted on the HOUSE line; a finished room with GAPs left to walk
      (`_gaps_left`) reopened as a node (`_reopened`: entered at a frontier, priced by its frontiers, discounted
      as entered, never a revisit scan); a mid-visit rotation no longer ends the visit while GAPs remain; a
      room in transit with GAPs left is not `finished_in_transit`; a new exit on an offered room buys a call at
      most every `revalue_actions`; the post-oracle `_exclusions` pass reads the unfolded exits.
- [x] `policy.py` configuration strings; the HUD's "folded into" line for an exit off no room.
- [x] Regressions: `tests/test_room_level_gaps.py` (9); the opening-node tests pinned to `exits_as_nodes=True`
      where they describe the selective fold; the oracle's line format.
- [x] README (step 3, the *Exits are the room's attributes* paragraph, the deprecated openings section),
      CHANGELOG.
- [x] Fly it: the five-scene campaign of 2026-10-21 (`runs/benchmark_15episodes_20261021T082233Z`, 12 of 15 scored
      before the user stopped it for the audit below): SR 0.917, SPL 0.611, DTG 0.57 m over the twelve, against the
      2026-10-18 run's SR 0.933 / SPL 0.639 / DTG 0.08 m on the same fifteen (one wrong STOP on a YOLO `bed`,
      Markleeville/000001). The oracle's reasoning read as intended ("living room scanned; its gaps may hide a bathroom
      door" at 0.2 under a bathroom-sized unknown room at 0.4); the reopen fired (the spawn room, 4 exits, step 10).
- [x] The audit the user asked for after watching the videos, same day -- five fixes, CHANGELOG "Fixed (2026-10-21)":
      the **doorway partition** (the sliver beyond an observed open door is a room node of its own, joined by a
      threshold edge: `sightline_thresholds`, the line cut, `stub_min_cells`); **side glances committed** (27
      scheduled / 4 taken in Collierville/000000 -- the plan re-planned away every two actions); **no redundant
      circles** (the rear must pay; none in a finished, frontier-less or being-walked room); the **BEARING approach**
      (a confident RGB box without depth is walked toward until the depth arrives); the **stair capsule** 0.6 -> 0.5 m
      drawn (`stair_margin_m`). A recorded verification episode: see the Notes.

## Notes
- 2026-10-21: what reopens a finished room is *significant* frontier (the explore visit's own population,
  `explore_min_gain_m2`) on a room with an exit credited to it -- the population the opening nodes covered,
  read on the room. A scanned room whose only frontier is the shadow behind its sofa stays finished; a
  scanned room with an unlooked doorway is offered again for what lies behind it.
- 2026-10-21: the local phase keeps its three exits -- GAPs exhausted, the bound, the clue rule -- and a
  rotation completed mid-visit finishes the room in the ledger without ending the visit while GAPs remain;
  under `exits_as_nodes` the rotation ends the visit as before, the openings being nodes.
- 2026-10-21: `scan` and `sweep` visits have no local walk to fold an exit into and keep exits as peek nodes;
  target landmarks are nodes in every mode.
- 2026-10-21, the recorded runs after the audit. Verification episode Collierville/000000 (toilet,
  `runs/gibson_Collierville_20261021T100529Z`): SR 1, SPL 0.764, 39 steps against the morning's 189 -- the floor
  segmented into four rooms where the morning saw two, threshold edges `t2`/`t4` through the observed doors, the
  bathroom-sized stub beyond the door valued 0.5 and chosen. The full fifteen with every fix
  (`runs/benchmark_15episodes_20261021T101046Z`, `benchmark_results.{json,csv}`, `summary.txt`, videos under
  `<Scene>/recordings/`): **SR 0.800, SPL 0.607, DTG 1.07 m**, 64 steps on average (the 12-of-15 morning run:
  0.917 / 0.611 / 0.57 m; 2026-10-18: 0.933 / 0.639 / 0.08 m). Twelve successes, eight of them at SPL >= 0.76;
  three failures: Corozal/000001 (tv, DTG 6.95 m -- the unbounded bearing walk, below, fixed and re-run to SR 1 /
  SPL 0.362 in `runs/gibson_Corozal_20261021T102540Z`), Darden/000000 (bed, DTG 3.96 m, 137 steps -- the closing
  released repeatedly, the historically flaky episode), and Markleeville/000001 (bed, DTG 5.20 m, 43 steps -- YOLO
  called the kitchen's dark dining table `bed` at 0.52 with the box filling the frame and the terminal confirmation
  stood; the same wrong STOP as the morning run's, pre-existing). With the Corozal fix counted the fifteen read
  SR 0.867 / SPL 0.631.
- 2026-10-21: the bearing walk's two further ends came from Corozal/000001: a television first seen at about
  5.5 m never projected at any range -- a screen is a hole in the Gibson mesh -- so the walk ran its whole bound
  (3.75 m) and left the agent beside a second television it then STOPped on. `bearing_max_walk_m` (3.0 m, a box
  first seen within the depth cap plus that distance is in range by then) and `bearing_near_box_frac` (0.30 of the
  frame's height without depth: within reach, unpositionable, no blind STOP). In the re-run both walks ended by
  the distance rule, the second within sight of the television the ordinary VERIFY then measured.

- 2026-10-21, afternoon, the Darden stair footprint. The user saw the stair oval closing the junction below the
  hallway in `Darden/000000` and asked for the stair inflation to come down there too. Measured on the recording
  and the navmesh: the margin drawn was 0.5 m (the same five cells as everywhere), the flight is 0.9 m wide on the
  navmesh so the margin is its half-width, not inflation -- but the capsule's round storey end put the margin's
  whole radius of plain floor in front of the first riser, 50 of the oval's 204 cells, across the junction's
  width (x -0.6..0.4, y 1.03..1.53; the flight begins at y 1.67). The storey end is now cut square
  (`peek_stairs._square_foot`); scene-agnostic, the sides and the far end unchanged. *The passage beyond is
  another matter:* the doorway from the junction into the east room, and the one from the living room into the
  east wing, are 0.2 m humps in the Gibson mesh (navmesh heights 0.17-0.22 m over a 0.02 m plane, continuous and
  walkable to Habitat's agent, whose climb limit is 0.2 m); the observed map's obstacle band begins 0.15 m above
  the plane, so both read as walls whenever the agent spawns in the west half (anchor 0.02; from the east half,
  anchor 0.13, the hump is floor). `Darden/000000` (bed at x 1.4-3.3, y 7.3-9.5, east wing) cannot be reached
  from its start on this mapping whatever the stairs do; `Darden/000001`, starting in the east wing, finds it at
  SPL 1.0.

- 2026-10-21, afternoon, four more changes on the user's reading of the Darden recording:
  **stairs observed, not read** -- `MultiFloorParams.stair_source` defaults to `observed`; the navmesh connector's
  footprint was the centreline, which begins 0.45 m before Darden's first riser (the observed portal placed itself at
  y 2.25, the riser; the GT entry was 1.67), and the published baselines have no stair oracle. Under the benchmark's
  confinement the observed candidates (`stair_terrain`, a reachable run of treads rising 0.40 m), the drops and the
  map's obstacle band are the whole stair handling. Darden/000000 re-run: SR 1 / SPL 0.597 in 162 steps -- through
  the junction into the east wing (the hump read free this time, anchor 0.03) -- where the campaign had SR 0; two
  observed portals, the staircase and a second one at (2.25, 0.45) in the east room that matches no connector,
  shrunk to 0.3 m by the corridor rule (an observed-mode false positive to watch: a 0.4 m rise over three
  step-connected levels is also what a bed frame or a counter can read as). Markleeville/000000: no stair
  candidate at the descending flight (its treads below the plane are never measured as support), 200 drop cells
  masked the stairwell, z stayed 0.18 all episode. Collierville/000000: no candidate either; the ascending flight is
  the obstacle band's; SR 1 at 82 steps.
  **The unvisited raised, the dwell decayed** (`unvisited_boost` 1.5, `unvisited_floor` 0.15,
  `dwell_half_life_steps` 20, `dwell_floor` 0.05; the oracle's STEP 4 says never below 15 for a never-entered unknown
  room and never lower for a low frontier count). **Wide glances** (`wide_rad` 120 degrees when the 130-160 band pays
  for the two extra turns): Markleeville scheduled four, took one. **The residential taxonomy, geometry in the
  classifier's prompt, shape-only passages, landmark retraction and conflicting signatures weak**: the first run with
  a flat four misses retracted 29 of Markleeville's 51 confirmed objects (sofas and chairs missed from the angles
  YOLO-World is poor at), so a miss now needs a distinct viewpoint and the misses must reach the landmark's own
  observation count; `hallway` by shape appeared in Darden (room 1 at step 50) and Markleeville (room 3 at 16), and
  the merged kitchen with two false beds beside an oven and a refrigerator is a WEAK bedroom now, not a strong one.
  Markleeville/000000 ended, in that first run, on the same wrong STOP as 000001 in the campaign -- the dark
  dining table as `bed`, confirmed by the closing's own two frames -- which no room label can prevent: the closing's
  terminal check against the room's type is the open item below. The second round, with the stricter retraction
  (`runs/gibson_{Markleeville_20261021T123311Z,Collierville_20261021T123840Z,Darden_20261021T124033Z}`): all three
  succeed -- Markleeville SR 1 / SPL 0.155 (361 steps, 12 retractions, the descending flight's head found as an
  observed portal at (2.05, -3.45) this time, 128 cells masked beside 251 of drops, z 0.18 throughout),
  Collierville SR 1 / 0.355 (82 steps, 3 retractions), Darden SR 1 / 0.210 (380 steps by the northern route, the
  hump reading as a wall this time; 6 retractions, all sofas and a cabinet; labels at the end: dining room, office,
  hallway x2, living room, bathroom, bedroom -- the taxonomy in use). SPL is low on the long routes; one episode
  each is too few to separate the four changes from the LLM's draw.

- 2026-10-22, the 25-episode campaign (`runs/benchmark_25episodes_20261022T044344Z`, five recorded episodes per
  scene, stopped by the user at 19 of 25 with Wiconisco not reached): **SR 0.895, SPL 0.631, DTG 0.76 m over the
  nineteen** -- Collierville 4/5 (tv: a STOP on a small kitchen screen at 0.34, under `stop_confidence`, accepted
  after the two extra frames the rule allows; DTG 5.9 m), Corozal 4/5 (toilet: the 500-step limit, 1168 s, DTG
  8.5 m), Darden 5/5 (both beds and the toilet at SPL 0.90, the episodes that failed in every earlier campaign),
  Markleeville 4/4 (000001 bed, the old wrong STOP, now SR 1 / SPL 0.51). Measured on the failures: the observed
  stair detector placed a "staircase" on Corozal's coffee table and sofa (and Darden's bed the day before), the
  classifier was asked 82-100 times per long episode because the geometry bucket keyed its cache, and the scene-graph
  update cost 1.4 s median on a 500-step map -- `_parts_skeleton` labelling the full 800x800 grid a hundred times per
  update. All three fixed the same morning (CHANGELOG "Fixed (2026-10-22)"); the stair profile, the objects-only key
  and the bounding-box crop (proved exact on 9,600 verdicts over three recorded maps).

## Open questions
- A low-confidence terminal box (`television 0.34` on a kitchen's wall screen, Collierville/000004) is accepted after
  two extra frames; whether a box under `stop_confidence` should also need the landmark's confirmation or a closer
  look is for the user to decide.
- A `bed` box on a dining table confirmed by the closing in a room whose evidence is ovens and cabinets
  (Markleeville 000000 and 000001): the room-type prior reaches the global loop and now the label's strength, but
  not the terminal confirmation. Whether a low-confidence terminal box contradicting a kitchen should need a
  second measured frame from another spot is for the user to decide.
- The observed stair candidate at Darden (2.25, 0.45): tighten `stair_terrain` to tread-like rises (0.12-0.24 m per
  step over 0.8 m of run) if it recurs; the corridor rule kept its mask to 0.3 m.
- Darden's 0.2 m thresholds: a climbable band in `observed_map.py` (floor within +0.20 of the plane, the obstacle
  band from there, matching Habitat's `agent_max_climb`) would open both doorways but changes the floor semantics
  in every scene -- a decision for the user, not a stair fix.
- With the hallway's doorways now the hallway's own, the agent walks every GAP of the hallway before the global
  loop can send it into the bathroom it glimpsed through the first one (the clue rule ends the turn only when
  the room in force is itself re-identified). The takeover still fires on the target itself; whether a home
  object glimpsed mid-visit should end the local phase early is for a recorded run to say.
- The exits counted on a room are the detector's exits (`min_cells`, the shadow test); a doorway misread as
  a shadow is nobody's GAP, as before.
- A target whose surface is a hole in the mesh (television screens) never projects: the bearing walk now stops
  short of it, but the ordinary VERIFY still needs a measured frame on the bezel or the stand. Whether the
  projection should tolerate a box whose centre is depthless when its rim measures is for a recorded run to say.
- Markleeville/000001: a detector `bed` at 0.52 on a dining table in a room the LLM had just called a kitchen
  ("beds are not in kitchens") was confirmed and STOPped on. The room-type prior reaches the global loop but not
  the terminal confirmation; whether a low-confidence box contradicting the room's type should need a second
  measured frame is open.
