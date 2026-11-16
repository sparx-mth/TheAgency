# object_search — the zero-shot object-search method

A `SearchPolicy` for the ObjectNav contract in
[`core/planning/objnav`](../objnav/README.md): given ground-truth RGB-D and
pose, find a named object category in an unseen home with a discrete action
set, and STOP inside the success region. Observed-only mapping, a room graph
with revisable LLM labels, a node oracle that values every room from the big
picture, RPT\* for the visit order, frontier or planar-FALCON local exploration,
A\*/WA\* routes, and a high-priority target closing. Training-free, and
given no simulator geometry: no navmesh, no pathfinder, no storey or stair
annotations (see "No privileged geometry" below).

```text
 ObjNavEnv (adapter, tasks/)                 RPTSearchPolicy (this package)
 ─────────────────────────────               ─────────────────────────────────────────────────────────
 ObjNavObservation ───────────────────────►  perception/   detector boxes → coherent 3-D centroids → landmarks, doors
                                             world_model/  per-storey log-odds occupancy, sight ledger, stair surfaces
                                             rooms/        watershed rooms → labels → node oracle → RPT* order → the loop
                                             exploration/  frontier sweep, glances, peeks, the fallback ladder, escape
                                             multifloor/   spawn-floor guard (default) or the building coordinator
                                             closing/      the target takeover: verify, approach, pitch, STOP
                                             navigation/   the route as the exact actions that walk it (lattice), doorway scrapes, the look-down
 NavigationCommand ◄──────────────────────  policy.py     one decision per frame, in world coordinates
```

The policy never names a simulator, a dataset or a wire format. It is built
with three services -- an object **detector**, an optional box-prompted
**segmenter** and a **room-reasoning LLM** -- behind the protocols of
[`interfaces.py`](interfaces.py); the HTTP clients that implement them, the
Habitat bridge, the Gibson files and the run CLI live in
[`tasks/planning/objnav_benchmark_runtime`](../../../tasks/planning/objnav_benchmark_runtime/README.md).
`tests/test_package_boundaries.py` enforces the line, and
`tests/test_no_privileged_geometry.py` the stricter one behind it: nothing
under `sparx_agency/core` can reach the simulator's navmesh, its pathfinder
or ground-truth storey data.

## Layout

| package | holds |
|---|---|
| `policy.py` | `RPTSearchPolicy`: composition, the per-frame decision order, `configuration()` and `episode_info()` |
| `settings.py` | `RPTSettings`, the one frozen-config record of the method's knobs (`--policy-config` overrides its fields) |
| `interfaces.py` | `Detection`, the `Detector` / `Segmenter` / `RoomReasoner` protocols, the frame-error classes and the failure contract |
| `perception/` | `cycle.py` (one synchronous prediction per frame, fusion, target evidence), `projection.py` (box → world centroid with coherent depth), `doors.py`, `target_context.py` (is this candidate at a height its class is seen at? -- a hand-written per-class height band; see "No hand-written semantic priors" below) |
| `rooms/` | `scene_graph.py` (observed rooms, doors, labels, the oracle's nodes; since 2026-11-16 every room's extents -- `core/mapping/topology/room_extent.py`: its walls by side, whether every side is located or the size is a LOWER BOUND, its covered floor in m2, as a share of the whole room and as what is left -- read off the storey's own occupancy grid and the partition cut out of it, recomputed on every update and shown to the oracle as `size=` and `covered=`; since 2026-11-16 the oracle's room line also carries the graph's edges, `connected to:` the types of the rooms a door or a walked threshold joins the room to, `ObservedSceneGraph.connections`, and the room's objects with their counts, and no longer says which room the agent stands in), `room_search_loop.py` (the loop), `room_scans.py`, `room_vantage.py`, `opening_nodes.py`, `house_context.py`, `oracle_retry.py`, `floor_context.py`, `discovery.py`. `stair_nodes.py` (a staircase as a node of the loop's RPT* instance) was deleted on 2026-11-16 with the loop's side of it, dormant since the ground-truth stair source went; it comes back with the multi-storey work. Since 2026-11-09, from the MP3D ground-truth run's graph and oracle bookkeeping: a room under `LoopSettings.tiny_room_m2` (1 m2) with no live frontier is a fragment whatever its doors (slivers of 0.2-0.8 m2 had been offered in 40% of the oracle calls and were the model's first choice in one call in eight); the HOUSE line's frontier statement is a tri-state -- reachable, only slivers, unreachable from here, none -- so a boxed-in agent no longer tells the model the home is covered (`house_context.FRONTIER_WORDS`); a mid-walk reconsideration is a clue -- a room on or off the offer, a room's type (`discovery.semantic_signature`, `semantic_change`) -- not any new object class; geometry alone waits for the loop point when the offer is the one valued at the last call or that call was under `reconsider_min_actions` ago, SELECT's unvalued-room rule waits the same window, and a release for an unreachable node re-selects on the standing numbers (QUCTc6BB5sX/000069 had asked 27 times in 230 actions inside a 5 m patch); and the room labeller holds a verdict that differs without a new kind of evidence until it repeats, and one on a kind lost to the partition for a refresh (`RoomLabelSettings.relabel_hysteresis`). |
| `world_model/` | the maps the decisions plan on: `observed_map.py` (RGB-D → per-storey occupancy on the core log-odds integrator, keyed by the storey `core/localization/floor_atlas` reports), `standing_floor.py` (since 2026-11-15: where the agent may stand -- a cell observed free, with the body's clearance of observed free floor around it, reachable from the agent through such cells; the whole replacement for the navmesh projection of the closing's standing points, see "No privileged geometry"), `sightlines.py` (since 2026-11-16 also the per-cell OBSERVED record: every free cell seen within `detect_range_m` = 4 m with a clear line of sight at a pitch that puts an object body in the frame -- what the room extents read each room's `covered=` share from), `stair_terrain.py`, `stair_grid.py`, `peek_stairs.py`, and `map_ledger.py` (since 2026-11-03: every action, how many planning cells changed, by observation or by the guard and by which guard layer -- drops, crossings, stair capsule, step overlay, trail, bump bars -- in or out of the camera's footprint, the frustum taken from the camera's actual transform, pitch included; `method.map_changes` in a recording, out-of-view jumps of 50+ cells as events; a cell changing where the camera is not looking is a map rewritten from something other than observation) |
| `exploration/` | `exploration_fallback.py`, `escape_probe.py`, `frontier_sweep.py`, `path_glances.py` (since 2026-10-31 geometric glances are off by default -- `RPTSettings.geometric_glances` turns them on -- and a geometric glance is scored on capped rays -- each credited with the first `unknown_run_m` = 1 m of unknown -- so a look through an open door beside the route keeps its value and a look at an unmapped wall loses it: on 232 recorded glances the old promise had r = -0.17 with what the map gained, the capped one r = 0.39), `doorway_peek.py`, `doorway_candidates.py`, `exploration_metrics.py` |
| `multifloor/` | `spawn_floor_guard.py` (the benchmark default: one storey; since 2026-10-31 the steps, sunken and raised parts of it the stair terrain can walk to within `step_surface_band_m` are free, never drops, and re-measure the storey plane; a drop is also every cell a depth ray crossed below the drop line on its way to it, `drop_rays.py`; since 2026-11-03 a cell's verdict is fixed when the cell is MEASURED: the guard keeps per cell the lowest landing, the lowest ray run through the column and the thresholds in force at that measurement, and nothing re-judges a cell the camera does not see again -- a storey plane re-measured on another level of a split-level house (00878/000000: 14 times in 500 actions) used to re-threshold the whole grid, 270-730 cells out of view each time, a sunken room painted occupied and released by turns; stair capsules are likewise drawn about the plane their portal was placed from, `anchor_z`, from the flight geometry the portal was placed with, `mask_path`, replaced only when a candidate reaches more than half a riser farther along the flight -- never from the terrain candidate's `path`, the route from the agent's own cell that changed with every step and redrew the capsules on 47 of 500 actions -- and with a margin that only narrows over the episode, `capsule_radius_cells`), `multifloor_policy.py` (the building coordinator; its stairs are the flights the robot's own depth measured, `world_model/stair_terrain.py` -- the `ground_truth` stair source that read the simulator's navmesh connectors from the episode metadata, with its perfect detector, its centreline traversal and its fallback rule, was removed on 2026-11-15 and asking for it raises), and `stair_traversal.py` (a flight walked on measured tread support); `stair_approach.py` (where a staircase offered as a node was entered from, and what the climb cost) went with the stair nodes on 2026-11-16, as did the coordinator's `commit`, `portal_by_id` and `way_back_held` and the two parameters that priced a stair node (`MultiFloorParams.floor_change_cost_m`, `arrival_grace_actions`). A spawn ON the stairs (174 of 2,000 HM3D v1 val starts) keeps its storey provisional until it stands on level support of floor size, walks the flight it saw to the nearer storey on purpose (bounded retries), and may switch once to the flight's other end along the climb the atlas measured when that storey is searched out (`spawn_switch.py`): two storeys at most, never back. `heading_commitment.py` holds the stair steering's heading per standing position, so a view-dependent tread edge cannot turn it back and forth |
| `closing/` | `target_closing.py` (since 2026-11-03 an inspection that saw nothing from a standoff the map shows obstructed -- something occupied between the agent and the memory, the test a memory STOP already needs -- releases the lock WITHOUT the 2x rejection memory and boxes the spot in instead, `blind_inspection_release`: in the 434-failure re-run every failed inspection was such a blind one, and two of the three ended with the agent at the goal refusing the target it could see); since 2026-11-05 (HM3D, the ten replayed regressions) it begins the bearing approach on a strong box CUT BY THE FRAME'S EDGE instead of dropping it: the camera is tilted toward the cut edge -- levelled when the doorway look-down held it down -- and the body faced to the box, so the next frame shows it whole (`bearing_clipped`; 00880/000009 walked past its television in ten border-rejected frames), counts a strong box of the class whose near edge measures within the terminal range plus one step as a sighting of the locked target although the anchor's association cylinder refused it (`terminal_resighting`, `terminal_resightings`: the cylinder is drawn about the first far frame's centroid, and the close frames of the 00891/000006 toilet -- bowl, not tank -- fell 3.5 cm outside it, twenty frames "saw nothing"), and leaves a centred frame measured within one forward step of the limit to the inspection's own step rule while that rule's budget lasts instead of cancelling the inspection for the approach (`step_out_inspections_kept`: the approach's route began with the turn back, forty-one turns inside the success region); since 2026-11-09 three rules from the MP3D ground-truth run's "seen, not converted" failures: an ARRIVED standoff with the fresh range at most one forward step beyond the limit is the inspection's whatever the step budget, whose step rule takes the one MOVE_FORWARD -- the arrived route's idle result was a TURN, and zsNo4HB9uLZ/000078 spun 90 degrees off a cushion 1.08 m away and lost it as "no safe path 1.1 m away"; the frame's re-sighting is the projection NEAREST the 3-D memory, a whole box before a sliver (`_best_projected`), not the most confident box -- with several instances of the class in frame at one confidence the range was measured on the neighbour; and the inspection resumes the approach at most `max_inspection_resumptions` times per lock and never on a side-clipped box, whose near range is the visible part's; **since 2026-11-16 the STOP is DECIDED on the closing's own evidence and TAKEN inside a stop criterion its caller states** (`stop_within_m`, `stop_measure`; `closing/stop_criterion.py`): how near the target the agent must stand is a benchmark's definition of its success zone, not the method's choice, and two benchmarks that both say "within 1 m" mean different things by it -- HM3D's metre is horizontal, MP3D's is 3-D from the floor -- so the range is read by the `planar` measure (the horizontal range to the target's nearest measured point) or the `euclidean` one (that range and the height of the target's lowest measured point above the floor under the agent, as the two legs of a triangle), and once the STOP is decided the closing walks straight on toward the memory until the range reads within the distance, consulting no frame (phase `ADVANCE`; it ends at a refused step, after four steps, or before it would stand inside 0.30 m of the memory); the default is `stop_within_m: 0`, the STOP on the deciding frame, and the benchmark adapters set the criterion (`tasks/.../objnav_benchmark_runtime/habitat/goal_zone.py`); what it replaced, removed that morning because it existed only for how an evaluator scores a STOP and was fitted on recorded validation STOPs: the STOP overshoot (forward steps after the decision: on the floor since 2026-10-14, switched off by the entrypoint under a view-point evaluator since 2026-11-05; for RAISED targets since 2026-11-10, `elevation_overshoot`) and the frontal standoff of 2026-11-10 (`frontal_standoff`, `perception/face_normal.py`: a plane fitted to a raised target's depth pixels and the standoff ring tried about its normal first, an in-range STOP from beside it deferred), which is not restored -- the standoff ring is walked to about the agent's own bearing; **known and open: a verification step refused by the simulator does not end the blind stepping** -- a box of the target class on the candidate's bearing without a usable depth (a glimpse) is stepped toward blind, and a step the simulator refuses leaves the pose and the frame as they were, so the same step is asked again until the paused verification clock runs out, 30 refusals in a row in zsNo4HB9uLZ/000015 and four more episodes of the 180-episode MP3D check (three before the navmesh removal); a fix ending it at the first refusal was measured on 2026-11-16 and withdrawn the same day, to be handled on its own (`LESSONS.md`); `target_path.py` (since 2026-11-15 every standoff is a cell of `world_model/standing_floor.py` -- floor the agent observed, with its body's clearance observed too, reachable on its own map -- and the route is planned to that cell exactly; the navmesh projection and the planner through unknown cells are gone, and while floor in front of the target is unobserved the standoffs beside it, behind it or out of the map's sight of it wait, as the detours do, until the closing has walked to the observed floor nearest the target and looked round; since 2026-11-05 each ring tries the FRONT arc -- the side the target was seen from -- with every planner before any point beside or behind the target with any: a toilet, a bed, a television stands against a wall, the point behind it is in the next room, and 00891/000006 and /000027 walked a 7.5 m known-floor route round the outside of the bathroom to it, inspected the wall from 0.9 m and STOPped 6 m from the goal; `back_arc_standoffs` counts the ones still taken; a standoff whose route is longer than `standoff_detour_factor` times the range plus a metre is the last choice, every other ring, arc, bearing and planner first -- the re-run of 00891/000006 projected a front-arc point through the bathroom wall onto a 6.6 m route for a 3.4 m range; `standoff_detours` counts the ones taken; the order in full: the agent's ring, the widened ones, each arc by arc, planner by planner -- the default planner at 0.25 m from every observed obstacle, then the retry planner at the body radius, 0.18 m, `TargetClosingSettings.standing_retry_clearance_m`, a planner of the closing's own since 2026-11-16), `target_backoff.py`, `approach_history.py` (since 2026-11-05 a STOP on the memory alone needs the last fresh box within `history_stop_max_unseen` = 15 actions -- the path must have TRACKED the target, not lost it for the length of the approach; `history_stops_unseen` counts the refusals), `target_pursuit.py` (a lock released for want of a path is kept and walked back to once a route exists, 2026-10-31; since 2026-11-11 a target glimpsed without depth is one remembered guess however often it is glimpsed, a bearing pursuit walks 48 actions at most and the episode's pursuits 150 together -- Z6MFQCViBuw's atrium chairs had eight episodes spend 275-453 actions walking back to a guess), `stop_criterion.py` (the two measures a stop criterion may name and the range each reads off the agent's own measurements, `stop_range_m`; pure functions) |
| `falcon/` | the planar FALCON explorer wired to the room search (`local_exploration: "falcon"`) |
| `route_memory.py` | the committed-route memory: when a planned path is kept, re-used or released |
| `navigation/` | the narrow-passage layer (2026-10-28): `kinematic_follower.py` (every committed route walked as the action sequence the discrete-action lattice plans, `core/planning/planners/lattice`), `bottlenecks.py` (doorways and corridors read off the clearance field), `scrape_guard.py` (a refused step in a doorway stays off the map), `doorway_look_down.py` (the camera pitched on the approach); see its README |
| `baselines/` | `falcon_yolo.py`: goal-free FALCON + the same perception and closing, no LLM (`--agent falcon-yolo`) |
| `tests/` | the package's contract tests: boundaries, a service-free smoke run, the failure contract; see below |

What the method shares with the rest of `core` it imports from there rather
than keeping a copy: the map vocabulary (`core/planning/environment`), the
A\*/WA\* planners and the discrete-action lattice (`core/planning/planners`),
the FALCON port, the RPT\* room solver, frontier ranking and
room costs (`core/planning/exploration`), the room watershed, registry and node
oracle (`core/mapping/topology`), the landmark map (`core/mapping/objects`),
the camera geometry and the action converter (`core/planning/objnav`), the
storey localiser (`core/localization/floor_atlas`) and the camera-tilt
controller and servo law (`core/control/camera`: where to *look* is a control
process beside where to *move*; the two meet in the `NavigationCommand`). Its
own policy-free pieces went the other way, into the subsystem they belong to:
box extent and occluders, detection dedup, target evidence and support
surfaces in `core/mapping/objects`; the fused 3-D target memory in
`core/mapping/tracking`; the revisable room labels in `core/mapping/topology`.

`world_model/` is mapping by nature and lives here by necessity: its products
are the planners' map vocabulary (`OccupancyGrid2D` of
`core/planning/environment`) and its inputs the contract's camera geometry,
and `core/mapping` imports nothing from `core/planning`. Moving the shared map
and pose vocabulary down into `core/common/types` would let it move; that is
a repository-wide change (over a hundred importers) and not this package's.

## No hand-written semantic priors
Since 2026-11-15 the method carries no table that tells it where a target is
likely to be. `core/mapping/topology/room_priors.py` was deleted with every
consumer: the `WHERE IT LIVES` line of the oracle's prompt (likely and rare
room types per target category), the hard `type_prior` exclusion of rooms a
table called implausible, the "home object" rules (a room kept a node, kept
from being finished by sight, a walk interrupted, a glance cued, a probability
floored, because a sink stands where a toilet lives), the "signature" and
"generic" object lists that decided how strong a room label is, and the
room-type half of the closing's context check. The semantics of the search
are the two LLM calls -- the room labeller and the node oracle -- and the
geometry is the map's. What is still written by hand, and must be said
wherever the method is described: the two prompts themselves (the oracle's
system prompt lists examples of typical homes and gives numeric bands -- since
2026-11-16 also for how much of a room's floor has been covered), the
code-side scaling of the oracle's numbers by exploration state
(`LoopSettings.entered_discount`, `lingered_discount`, `dwell_*`,
`unvisited_*`, `openings.landmark_prob`), the closing's per-class height bands
(`perception/target_context.CLASS_HEIGHT_BANDS`), its list of furniture that
carries other objects (`core/mapping/objects/support_surface.SUPPORT_SURFACE_CLASSES`)
and the fallback's list of classes that hide no floor
(`exploration_fallback.SHADOWLESS_CLASSES`). The last three are geometry about
objects, not beliefs about where a target is; they are still tables a paper
must name.

## No privileged geometry
Since 2026-11-15 the agent is given nothing of the scene but its observations.
Two things used to reach it from the simulator, and ASCENT, ApexNav, VLFM and
MFNP give their agents neither:

* **A navmesh projection of the closing's standing points.** The evaluator
  bound a `target_projector` to the policy, and every standoff, back-off spot
  and nearest-floor goal was snapped onto the simulator's navigation mesh --
  which knows where walkable floor lies whether or not the agent ever looked
  at it, and on which the benchmark's success region is defined. Removed, with
  its three call sites (`closing/target_path.py`, `target_backoff.py`,
  `target_closing.py`), the adapters' bindings and the
  `target_navmesh_projection` record.
* **Ground-truth storeys and stairs.** The adapters put the navmesh's storeys
  and stair connectors into `ObjNavEpisode.metadata`, and
  `MultiFloorParams.stair_source = "ground_truth"` read them. Removed: the
  episode carries no metadata, `stair_source` accepts `"observed"` only and
  raises on the old value, and `configuration()["ground_truth_stairs"]` is the
  constant `False`.

**What replaced the projection** is one module, `world_model/standing_floor.py`,
reading one thing, the occupancy grid the agent built from its own depth. A
cell is standing floor when (1) it was observed free, (2) every cell within
the body radius of it (0.18 m) was observed free too -- an occupied cell
there is a collision, an unknown one is the body standing on floor nobody has
seen -- and (3) it is joined to the agent by a chain of such cells. That is
one floor, whichever planner walks to it. The default planner is asked first,
of the standing cells that also keep its own standoff (`min_clearance_m`,
0.25 m) from every *observed obstacle* -- nearer than that it cannot end a
route -- and, **only when no route is found at that**, the retry planner of
the standing cells the body radius (0.18 m) from every observed obstacle
(`TargetClosingSettings.standing_retry_clearance_m`, whose default `None` is
the body radius; `policy.standing_retry_planner`,
`TargetClosing.standing_planners`; a planner of the closing's own since
2026-11-16, where the fallback's squeeze planner had served before). On the
0.1 m grid the nearest obstacle cell an end-of-route may have is 0.28 m away
at 0.25, 0.22 m at 0.20 and 0.20 m at 0.18. A retry at 0.20 m was measured
against the body radius on the 180-episode MP3D check that day: it was
refused fewer forward steps (5.92 against 6.79 an episode) and reached fewer
targets (SR 71.1 % against 73.3 %, SPL 0.322 against 0.325), so the body
radius stayed the default. The squeeze planner is still the exploration
fallback's and the transit's rung. Unknown cells cast no
standoff: beyond the body's own disc nothing has to have been observed. (The
first version asked the default planner only of cells whose whole 0.25 m
disc was observed free. A map of a few frames rarely has one, so the second
rung took a standoff in 32 of 85 paired episodes where the navmesh runs had
taken none.) A candidate point is
moved to the nearest standing cell within `standing_snap_m` (0.5 m), never
across an obstacle the map shows, or dropped; the route is planned to that
cell exactly (`PlanRequest.options["exact_goal"]`). Unobserved space is never
a standing point, never part of its clearance and never on the way to it:
`TargetClosingSettings.through_unknown` and the policy's `optimistic_planner`
went with the projection. What the agent cannot yet stand on it goes and looks
at -- the walk to the observed floor nearest the target, then the footing
sweep (`TargetClosing._no_path`).

**What the stairs lost.** With stair traversal forbidden (the benchmark
default) nothing: the flights masked were already the observed ones. With it
allowed, floor changes are the coordinator's allowance rule, as they were
under the observed source. The loop's "staircases as nodes of RPT\*"
(`rooms/stair_nodes.py`) offered a staircase only when its portal said which
storey it leads to, and only the navmesh ever said so: no staircase was a
node from 2026-11-15, and on 2026-11-16 the dormant code was deleted -- the
module, the loop's stair paths, `multifloor/stair_approach.py`, the
coordinator's commitment API, the oracle's `STAIRS` node with its storey
lines and its multi-storey supplement (which described a storey RANK that
only the navmesh's levels supplied). The search is single-storey; a
staircase as a node comes back with the multi-storey work, from the history
of this branch.

**Room sizes are the map's too (since 2026-11-16).** The `size=` and
`covered=` of a room on the oracle's line come from
`core/mapping/topology/room_extent.py`, which is handed the storey's
occupancy grid, the partition the watershed cut out of that grid, the sight
ledger's settled and observed cells and the agent's position -- and nothing
else: no navigation mesh, no path query, no room or region annotation of a
dataset, no table of how big a room of some type is. A room's size is only
ever what the camera has mapped of it: FINAL when all four walls are located,
otherwise a lower bound that is never presented as more.
`core/mapping/topology/tests/test_room_extent.py` pins the module's imports
(numpy, scipy, the standard library) and its inputs, beside the scans below.

**What keeps it out.** `tests/test_no_privileged_geometry.py` closes the three
ways such data can reach code: by import (the transitive import closure of
every `core` module may not contain the simulator, the benchmark harness or
its runtime), by name (no identifier in `core` may spell the navmesh, the
pathfinder's API, the projector seam or the storey structure -- an injected
handle needs no import, but the code using it must name the service), and by
the episode (no `core` module reads `ObjNavEpisode.metadata`, and a policy
handed the navmesh's structure there acts exactly as one handed nothing). Each
scanner is run against planted leaks. The adapters' half --
no episode built with metadata, nothing of an environment bound to a policy --
is `tasks/planning/objnav_benchmark_runtime/tests/test_agent_gets_no_scene_geometry.py`,
and the harness refuses such metadata at every reset
(`objnav_benchmark/env_contract.PRIVILEGED_KEY_PARTS`).

The navmesh stays where the benchmark needs it, outside the agent: the
simulator's physics, episode setup and start validation, the geodesic a score
is measured by, the harness's kinematics check, the after-the-fact analysis by
storey. The simulator's ground-truth SEMANTICS, when a run selects them
(`--perception ground-truth`), replace the detector and the segmenter at their
interfaces and are declared by the adapter; that is a separate, disclosed
upper bound and is unchanged.

## The committed-target rule and the cadence

Once the loop has chosen a room and a route to its entry point, the walk is
committed (`LoopSettings.transit_commitment`, since 2026-10-22): the room may
be renumbered by the watershed on the way -- merged, split, re-drawn -- and
the turn follows the place under the new number (`room_rebound`,
`ObjectSearchSupervisor.rebind`); a new name for the room, or the ledger
finishing it from outside, waits for the arrival; no classifier call is spent
in transit. A transit ends only by arrival, a refused or stalled route, the
follower's wedge, the supervisor's clocks, or the target takeover at the
policy level. A committed entry the planner refuses at its floor standoff
(`min_clearance_m`, 0.25 m) is asked once more of the `squeeze_planner` --
the body radius as its floor, the fallback's rung since 2026-10-25 (and the
closing's until 2026-11-16, when its standing points got a planner of their
own at the same clearance) -- before the room's other entries are tried and the room released
`unreachable` (`transit_squeezed`, since 2026-11-05: one jamb cell painted by a
look-down drew a 0.5 m way as 0.4 m, and 00843/000023 released the room 1.7 m
away, re-solved the order to the other end of the house and walked a 100-action
loop before coming back through that gap). While the committed route is walked the policy rebuilds the
scene graph's geometry every `RPTSettings.transit_graph_period_steps` actions
instead of every action (it was most of every decision's wall time), and
rebuilds it on the action the transit ends; a loop point always reasons over
geometry refreshed on its own action. Places a transit repeatedly failed to
reach are remembered by position (`entry_failure`, `unreachable:<n>`
exclusions) and left alone with a doubling back-off.

## How a component talks to the policy

The sub-components are constructed with the policy and read its shared state
through it (`self.policy.mapping`, `.episode`, `.route_memory`, `._navigate`):
one episode-scoped composition root, not a web of injected handles. The
convention is deliberate and the price is known -- a component is only
testable through a policy -- so the regression rig builds a real policy with
fake services on a synthetic map (`tests/fixtures.py` here; the benchmark
rig's `test_method.setup_policy`). Keep new components on the same pattern
rather than threading a fourth style through the package.

## Runtime

Python 3.8-compatible syntax, numpy and scipy; no ROS, no torch, and no
transport of its own -- the detector, segmenter and LLM clients are handed in
(`core/mapping/topology`'s facade still re-exports its HTTP `LLMClient`, so
`requests` is in the import closure; the method never calls it). It is not
imported by the FALCON Noetic container (which has no scipy) and is not part
of the `core/planning/objnav` import closure, so the contract's numpy-only
rule does not bind it. Importing the package root loads nothing; import the
policy by module path.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest sparx_agency/core/planning/object_search -q
```

The behavioural regression suite of the method -- the room loop, the closing,
the fallback ladder, several thousand assertions -- is the benchmark runtime's
rig: `tasks/planning/objnav_benchmark_runtime/tests/` (run it after every
change here). It lives there because it reasons about the Gibson camera, the
dashboard renders and the recorded traces, and because a rig belongs beside
the thing it exercises; the package's own tests check what the rig cannot --
that the method stays free of the adapter (`test_package_boundaries.py`) and
of the simulator's geometry (`test_no_privileged_geometry.py`), and the
standing floor's three rules on small grids and random maps
(`test_standing_floor.py`).
