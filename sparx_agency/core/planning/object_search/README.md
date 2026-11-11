# object_search — the zero-shot object-search method

A `SearchPolicy` for the ObjectNav contract in
[`core/planning/objnav`](../objnav/README.md): given ground-truth RGB-D and
pose, find a named object category in an unseen home with a discrete action
set, and STOP inside the success region. Observed-only mapping, a room graph
with revisable LLM labels, a node oracle that values every room from the big
picture, RPT\* for the visit order, frontier or planar-FALCON local exploration,
A\*/WA\* routes, and a high-priority target closing. Training-free; no
simulator geometry beyond what the adapter declares.

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
`tests/test_package_boundaries.py` enforces the line.

## Layout

| package | holds |
|---|---|
| `policy.py` | `RPTSearchPolicy`: composition, the per-frame decision order, `configuration()` and `episode_info()` |
| `settings.py` | `RPTSettings`, the one frozen-config record of the method's knobs (`--policy-config` overrides its fields) |
| `interfaces.py` | `Detection`, the `Detector` / `Segmenter` / `RoomReasoner` protocols, the frame-error classes and the failure contract |
| `perception/` | `cycle.py` (one synchronous prediction per frame, fusion, target evidence), `projection.py` (box → world centroid with coherent depth), `doors.py`, `target_context.py` (is this candidate plausible where it stands?) |
| `rooms/` | `scene_graph.py` (observed rooms, doors, labels, the oracle's nodes; since 2026-11-11 every room's extents -- `core/mapping/topology/room_extent.py`: its walls by side, its size as FINAL or a LOWER BOUND, its observed share -- recomputed on every update and shown to the oracle as `size=` and `observed=`), `room_search_loop.py` (the loop), `room_scans.py`, `room_vantage.py`, `stair_nodes.py`, `opening_nodes.py`, `house_context.py`, `oracle_retry.py`, `floor_context.py`, `discovery.py`. Since 2026-11-09, from the MP3D ground-truth run's graph and oracle bookkeeping: a room under `LoopSettings.tiny_room_m2` (1 m2) with no live frontier is a fragment whatever its doors (slivers of 0.2-0.8 m2 had been offered in 40% of the oracle calls and were the model's first choice in one call in eight); the HOUSE line's frontier statement is a tri-state -- reachable, only slivers, unreachable from here, none -- so a boxed-in agent no longer tells the model the home is covered (`house_context.FRONTIER_WORDS`); the oracle is briefed with a WHERE IT LIVES line from `core/mapping/topology/room_priors.TARGET_HOMES`, which covers every MP3D category (the 3B model had answered "a table is in the bathroom" for the fifteen it had no prior for); a mid-walk reconsideration is a clue -- a room on or off the offer, a room's type, a home object of the target new to the storey (`discovery.semantic_signature`, `semantic_change`) -- not any new object class; geometry alone waits for the loop point when the offer is the one valued at the last call or that call was under `reconsider_min_actions` ago, SELECT's unvalued-room rule waits the same window, and a release for an unreachable node re-selects on the standing numbers (QUCTc6BB5sX/000069 had asked 27 times in 230 actions inside a 5 m patch); and the room labeller holds a verdict that differs without a new kind of evidence until it repeats, and one on a kind lost to the partition for a refresh (`RoomLabelSettings.relabel_hysteresis`). |
| `world_model/` | the maps the decisions plan on: `observed_map.py` (RGB-D → per-storey occupancy on the core log-odds integrator, keyed by the storey `core/localization/floor_atlas` reports), `sightlines.py` (since 2026-11-11 also the per-cell OBSERVED record: every free cell seen within `detect_range_m` = 4 m with a clear line of sight at a pitch that puts an object body in the frame -- what the room extents read each room's `observed=` share from), `stair_terrain.py`, `stair_grid.py`, `peek_stairs.py`, and `map_ledger.py` (since 2026-11-03: every action, how many planning cells changed, by observation or by the guard and by which guard layer -- drops, crossings, stair capsule, step overlay, trail, bump bars -- in or out of the camera's footprint, the frustum taken from the camera's actual transform, pitch included; `method.map_changes` in a recording, out-of-view jumps of 50+ cells as events; a cell changing where the camera is not looking is a map rewritten from something other than observation) |
| `exploration/` | `exploration_fallback.py`, `escape_probe.py`, `frontier_sweep.py`, `path_glances.py` (since 2026-10-31 geometric glances are off by default -- `RPTSettings.geometric_glances` turns them on -- and a geometric glance is scored on capped rays -- each credited with the first `unknown_run_m` = 1 m of unknown -- so a look through an open door beside the route keeps its value and a look at an unmapped wall loses it: on 232 recorded glances the old promise had r = -0.17 with what the map gained, the capped one r = 0.39), `doorway_peek.py`, `doorway_candidates.py`, `exploration_metrics.py` |
| `multifloor/` | `spawn_floor_guard.py` (the benchmark default: one storey; since 2026-10-31 the steps, sunken and raised parts of it the stair terrain can walk to within `step_surface_band_m` are free, never drops, and re-measure the storey plane; a drop is also every cell a depth ray crossed below the drop line on its way to it, `drop_rays.py`; since 2026-11-03 a cell's verdict is fixed when the cell is MEASURED: the guard keeps per cell the lowest landing, the lowest ray run through the column and the thresholds in force at that measurement, and nothing re-judges a cell the camera does not see again -- a storey plane re-measured on another level of a split-level house (00878/000000: 14 times in 500 actions) used to re-threshold the whole grid, 270-730 cells out of view each time, a sunken room painted occupied and released by turns; stair capsules are likewise drawn about the plane their portal was placed from, `anchor_z`, from the flight geometry the portal was placed with, `mask_path`, replaced only when a candidate reaches more than half a riser farther along the flight -- never from the terrain candidate's `path`, the route from the agent's own cell that changed with every step and redrew the capsules on 47 of 500 actions -- and with a margin that only narrows over the episode, `capsule_radius_cells`), `multifloor_policy.py` and the stair modules the development protocols enable. A spawn ON the stairs (174 of 2,000 HM3D v1 val starts) keeps its storey provisional until it stands on level support of floor size, walks the flight it saw to the nearer storey on purpose (bounded retries), and may switch once to the flight's other end along the climb the atlas measured when that storey is searched out (`spawn_switch.py`): two storeys at most, never back. `heading_commitment.py` holds the stair steering's heading per standing position, so a view-dependent tread edge cannot turn it back and forth |
| `closing/` | `target_closing.py` (since 2026-11-03 an inspection that saw nothing from a standoff the map shows obstructed -- something occupied between the agent and the memory, the test a memory STOP already needs -- releases the lock WITHOUT the 2x rejection memory and boxes the spot in instead, `blind_inspection_release`: in the 434-failure re-run every failed inspection was such a blind one, and two of the three ended with the agent at the goal refusing the target it could see); since 2026-11-05 (HM3D, the ten replayed regressions) it begins the bearing approach on a strong box CUT BY THE FRAME'S EDGE instead of dropping it: the camera is tilted toward the cut edge -- levelled when the doorway look-down held it down -- and the body faced to the box, so the next frame shows it whole (`bearing_clipped`; 00880/000009 walked past its television in ten border-rejected frames), counts a strong box of the class whose near edge measures within the terminal range plus one step as a sighting of the locked target although the anchor's association cylinder refused it (`terminal_resighting`, `terminal_resightings`: the cylinder is drawn about the first far frame's centroid, and the close frames of the 00891/000006 toilet -- bowl, not tank -- fell 3.5 cm outside it, twenty frames "saw nothing"), and leaves a centred frame measured within one forward step of the limit to the inspection's own step rule while that rule's budget lasts instead of cancelling the inspection for the approach (`step_out_inspections_kept`: the approach's route began with the turn back, forty-one turns inside the success region); since 2026-11-09 three rules from the MP3D ground-truth run's "seen, not converted" failures: an ARRIVED standoff with the fresh range at most one forward step beyond the limit is the inspection's whatever the step budget, whose step rule takes the one MOVE_FORWARD -- the arrived route's idle result was a TURN, and zsNo4HB9uLZ/000078 spun 90 degrees off a cushion 1.08 m away and lost it as "no safe path 1.1 m away"; the frame's re-sighting is the projection NEAREST the 3-D memory, a whole box before a sliver (`_best_projected`), not the most confident box -- with several instances of the class in frame at one confidence the range was measured on the neighbour; and the inspection resumes the approach at most `max_inspection_resumptions` times per lock and never on a side-clipped box, whose near range is the visible part's; since 2026-11-10 two rules for RAISED targets, whose MP3D view points stand closer (0.84 m from the box at the median against 1.02 m for an object on the floor) and, for a thin wall-mounted object, only in front of it: the STOP overshoot reads the memory's range lifted by the target's lowest point above the base (`elevation_overshoot`, up to `stop_overshoot_elevated_max_steps` steps), and a lock whose sightings measure a standing plane keeps its facing normal (`perception/face_normal.py`) and has its standoff ring tried about the normal first, at the standoff radius and closer, with the agent's ring as the fallback, and an in-range STOP from more than 35 degrees off the normal deferred while the frontal standoff is to be walked (`frontal_standoff`) -- 2azQ1b91cZZ/000157 and two more pictures were STOPped on from beside them, 60-69 degrees off the wall, where no view point is; `target_path.py` (since 2026-11-05 each ring tries the FRONT arc -- the side the target was seen from -- with every planner, the unknown included, before any point beside or behind the target with any: a toilet, a bed, a television stands against a wall, the point behind it is in the next room, and 00891/000006 and /000027 walked a 7.5 m known-floor route round the outside of the bathroom to it, inspected the wall from 0.9 m and STOPped 6 m from the goal; `back_arc_standoffs` counts the ones still taken; a standoff whose route is longer than `standoff_detour_factor` times the range plus a metre is the last choice, every other ring, arc, bearing and planner first -- the re-run of 00891/000006 projected a front-arc point through the bathroom wall onto a 6.6 m route for a 3.4 m range; `standoff_detours` counts the ones taken; the order in full since the two were merged on 2026-11-11: the normal's rings, the agent's, the widened, each arc by arc, planner by planner), `target_backoff.py`, `approach_history.py` (since 2026-11-05 a STOP on the memory alone needs the last fresh box within `history_stop_max_unseen` = 15 actions -- the path must have TRACKED the target, not lost it for the length of the approach; `history_stops_unseen` counts the refusals), `target_pursuit.py` (a lock released for want of a path is kept and walked back to once a route exists, 2026-10-31; since 2026-11-11 a target glimpsed without depth is one remembered guess however often it is glimpsed, a bearing pursuit walks 48 actions at most and the episode's pursuits 150 together -- Z6MFQCViBuw's atrium chairs had eight episodes spend 275-453 actions walking back to a guess) |
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
`core/mapping/tracking`; the room priors and revisable room labels in
`core/mapping/topology`.

`world_model/` is mapping by nature and lives here by necessity: its products
are the planners' map vocabulary (`OccupancyGrid2D` of
`core/planning/environment`) and its inputs the contract's camera geometry,
and `core/mapping` imports nothing from `core/planning`. Moving the shared map
and pose vocabulary down into `core/common/types` would let it move; that is
a repository-wide change (over a hundred importers) and not this package's.

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
the body radius as its floor, the fallback's and the closing's rung since
2026-10-25 -- before the room's other entries are tried and the room released
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
that the method stays free of the adapter.
