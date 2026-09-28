# ObjectNav runtime and Gibson adapter

Observed RGB-D mapping → semantic room graphs → room LLM → RPT* room order →
local exploration and A*/WA* paths → existing discrete actions. Both explorers
retain persistent floor contexts; stair connectors come from the simulator's
navmesh by default (`RPTSettings.multifloor.stair_source`, see
[MULTISTORY.md](gibson/MULTISTORY.md)), the one declared ground-truth input.
This is an experimental multi-story algorithm, **not a claim of solved navigation**.

**To set up and run it end to end, start with [QUICKSTART.md](QUICKSTART.md)**: environments,
dataset and weight downloads, then the services, the frozen episodes and the campaign in
order. This file is the architecture reference.

## Exploration selection

- `--explorer frontier`: the existing observed-frontier baseline (default).
- `--explorer falcon`: bounded FALCON 2D/2.5D adaptation, not the unchanged ROS
  aerial binary. Local bursts, target interrupts and the episode ledger remain
  distinct; all emitted actions count toward the public 500-action cap.

See [FALCON component mapping](gibson/BOUNDED_FALCON.md),
[historical FALCON results](gibson/FALCON_RESULTS.md),
[distinct-building comparison](gibson/DISTINCT_BUILDINGS.md),
[Gibson protocol](gibson/PROTOCOL_AND_TUNING.md), and
[multi-story protocol](gibson/MULTISTORY.md). Earlier campaigns retain their
own frozen configurations; changed code does not relabel their outcomes.

## Reusable pieces

| Component | Responsibility |
|---|---|
| `methods/rpt_policy.py`, `rpt_settings.py` | Detector/mapper composition, RPT* and baseline selection |
| `methods/room_search_loop.py` | The seven-step room-search loop: bounded, room-confined local exploration; re-classify → re-estimate → re-order at each loop point; transit to the chosen room's nearest frontier |
| `methods/exploration_fallback.py` | Where every failed plan, model or decision lands: nearest floor-wide frontier, the stairs, a retired frontier, a relocation -- a move, never an idle spin; failure records and service back-off |
| `methods/frontier_sweep.py` | Frontier goal generation for a room or the floor, committed-goal lifetime, optional look-around |
| `methods/camera_control.py` | Sole pitch owner; bounded inspection, long unprompted cadence and safe restoration |
| `methods/perception.py`, `perception_cycle.py` | Fresh raw predictions, coherent pixel projection and floor-qualified fusion |
| `methods/observed_map.py`, `floor_context.py` | Independent occupancy, rooms, objects, association anchors and paused floor clocks |
| `methods/multifloor_policy.py` | The building coordinator: portals per floor, when to change floors, arrival bookkeeping, both stair sources |
| `methods/stair_ground_truth.py`, `floor_decision.py`, `ground_truth_traversal.py` | Ground-truth stairs (default): the navmesh connectors as the policy reads them, the explicit up/down choice with its recorded verdicts, and the polyline traversal |
| `methods/stair_traversal.py`, `stair_terrain.py` | Observed stairs (`stair_source: "observed"`): RGB-D support surfaces, step-connected paths, native-heading safety checks and the committed transition lifecycle |
| `gibson/stair_connectors.py` | Evaluator side: storeys and stair connectors read once per scene from the navmesh, attached to the episode metadata |
| `methods/falcon_policy.py`, `falcon_regions.py` | Bounded hierarchy and persistent local region history |
| `methods/falcon_motion.py`, `falcon_routes.py`, `route_memory.py` | Safe motion and interruption-aware committed routes |
| `methods/object_evidence.py` | Alias/frame deduplication, separated-view target evidence and bounded rejection |
| `methods/scene_graph.py`, `room_labels.py`, `doors.py` | Observed rooms/doors, the room label image, and revisable accumulated room reasoning |
| `methods/exploration_metrics.py` | Observed-area proxy, actions, revisits, stagnation and latency |
| `habitat/simulator.py` | Synchronous registered RGB-D, native extrinsic checks and ENU pose |
| `recording.py`, `floor_panels.py`, `visualization.py` | Exact commands/poses, persistent UNKNOWN panels, grids and video |
| `gibson/run.py`, `frozen_eval.py`, `run_development.py` | Protocol execution and frozen source/model/data checks |
| `tests/trace_room_search_loop.py` | Development probe: the loop on the production map size with fake services and a crude route-following executor; not a test |

The floor tracker remains ROS-free in `core/planning/exploration/floor_atlas.py`.
Host-side numpy/scipy terrain processing does not enter the Noetic exploration
facade. No robot's flight controller is changed.

## The room-search loop (frontier explorer)

`--explorer frontier` runs one explicit loop, with the scene graph maintained
in the background on every action (`RPTSettings.graph_period_steps`, default 1):
room geometry, object→room association, confirmed doors, cumulative search
time and remaining frontier clusters per room. No LLM call is part of that
refresh.

1. **Local exploration, bounded to the room in force.** Goals are the frontier
   clusters the room label image credits to the room (the same majority vote
   the scene graph's `frontier_clusters` count uses), ranked by
   `core/planning/exploration/frontier_ranking.py` -- gain over *geodesic*
   cost, facing as a discount. Each route is planned on a **copy of the map in
   which every other room's cells are written unknown**, so no in-room route
   crosses a door or takes the stairs (`LoopSettings.confine_routes`; lifted
   while the agent's own cell is outside the room, so a doorway or a moved
   mask cannot wall it out). The burst ends after `LoopSettings.local_steps`
   actions (10) or when nothing reachable is left -- whichever first -- and
   the supervisor is told so through its `budget_spent` / `frontier_exhausted`
   exits on that same action.
2. **Global re-classification.** At the loop point `graph.reason` re-labels
   every known room from every confirmed object observed so far.
3. **Per-room probability** -- the same call: the oracle's P(target in room ∧
   search ends there) from remaining frontiers, cumulative search time and
   room type. Distance enters through the RPT* objective, not the probability
   (both would charge travel twice); the per-room record in
   `room_search_loop.estimates` carries it. The oracle reuses the model's last
   reply when the prompt it would show is byte-identical, re-applying the
   code-side effort factors, so a loop point over an unchanged map costs no
   call (`oracle_reuses` in the episode record).
4. **Visit order** -- RPT* over the surviving rooms, with the local budget
   folded in as per-room service time; the order is re-solved at every loop
   point (`ObjectSearchParams.resolve_on_release`), never per action.
5. **A\*** -- the weighted planner behind `_navigate`.
6. **Direct transit to the closest frontier inside the chosen room**
   (`LoopSettings.entry_frontier`); a room with no frontier left keeps a
   centroid entry so the detector can still get a close look. A committed
   entry frontier that the camera resolves en route is re-aimed at the
   room's next one; a room with none left is released without the visit
   (`frontier_exhausted` in transit).
7. **Reset** -- arrival is the agent's own cell inside the room's mask
   (`arrived`), which restarts the local counter.

Steps 2–6 run on the action a room's turn ends, so a released room is
replaced by a transit at once -- never by a throwaway floor-wide route. A
room whose budget ran out is **not** put on the visit cooldown (the fresh
estimate may rightly send the agent straight back), while a room the live
map says is finished (exhausted or mapped) is never repeated by the
every-room-cooling escape hatch: the exploration fallback carries the search
instead. `room_search_loop.events` in the episode record lists every
transit, arrival and release with its verdict and local step count.

## The exploration fallback: a failure is a move, not a spin

Every way the decision pipeline can fail lands in one place,
`methods/exploration_fallback.py`, and comes out as a command that moves:

1. the **nearest reachable frontier anywhere on the floor** -- not confined to
   the room in force, so a room whose routes all fail is left rather than
   spun in;
2. the **stairs** (the building coordinator's decision), when the floor is
   exhausted or its allowance is spent;
3. a **retired frontier** -- a goal dropped for a transient plan failure is
   still unknown space;
4. a **relocation** to the farthest reachable known cell, for a vantage point
   the map may show a frontier from;
5. a single hold only when the agent stands off the observed passable map,
   where the idle turn is the one action that changes anything.

It is reached from the loop whenever no room is in force, and from
`RPTSearchPolicy.plan` whenever the decision itself raises -- an A* with no
path, an RPT* instance that cannot be built, a bug. A **room LLM** failure
(timeout, bad JSON, refused uniform prior) no longer ends the episode: the
loop records it, arms a doubling back-off (25 → 50 → 100 → 200 actions,
`FallbackSettings`) during which no model is asked, and the fallback carries
every action until the model answers again, at which point the loop resumes
its steps. A **detector** failure is handled the same way in
`perception_cycle.py`: the frame carries no detections, is counted as
skipped, and the service is left alone for the back-off. Every failure is
counted, kept with its type, message and origin under
`exploration_fallback.failures` in the episode record, and logged once per
type. Before this, an empty hold cost one idle `TURN_LEFT` per action until
the step budget ran out -- the "camera spinning in place" of the Ranchester
recordings -- and a model failure was an agent error.

## Frontier sweep and the action economy

Every action counts against the public 500-action cap, so the frontier explorer
is built to spend them moving toward unmapped space (the Ranchester recording
`e7c4f2ad5402` that motivated this spent 48 % of its actions on idle turns,
stair inspections and heading recovery):

- A boundary with no known-free path is never proposed. Up to
  `SweepSettings.plan_attempts` goals are tried per action, so a refused A*
  costs no idle action; a refused *transit* route ends the room through the
  supervisor's `route_failed` exit and the next room is chosen on the same
  action.
- A frontier cluster's goal is its **member cell nearest the centroid**, on
  the boundary. The centroid itself lies inside seen floor for the arc-shaped
  clusters a depth range produces, which made every goal "no longer
  informative" on the action it was adopted.
- A committed frontier goal is kept while it is passable, still has unknown
  space near it and still lies near the room being swept. A re-segmented room
  mask alone does not drop it; a boundary the camera has resolved does.
- The look-around after a swept room is off by default
  (`SweepSettings.room_scan_turns = 0`): the loop's termination rule is *N
  steps or nothing left*, and a swept room is released on the action its last
  goal resolves. The supervisor's 30 s stall and 90 s budget clocks remain as
  backstops behind the caller-driven exits.
- Every adopted route records `route_replaced`: why the route before it was
  dropped (`goal_changed`, `route_obstructed`, `frontier_completed_or_invalid`,
  `room_completed` ...). `route_commitment` stats count clears per reason.

## Camera, perception and transitions

`SEARCH → APPROACH_STAIRS → TRAVERSE → CONFIRM_DESTINATION → SEARCH` retains
connector, direction and progress across turns. Recovery uses the measured
reverse trail including the source-platform approach. Unconfirmed recovery
ends visibly in `SAFE_HALT`, not silent room search on a tread. Height
hysteresis alone cannot finish either traversal or retreat.

Positive camera pitch looks down. One controller arbitrates all pitch: a
location-deduplicated bounded inspection, a latched stair view, or ordinary
level viewing. A look-down inspection starts for one of three named reasons,
recorded in the building events: the floor has no frontier left
(`floor_exhausted`), a stair detection is pending (`stair_detection`), or the
floor allowance is spent and `periodic_inspection_actions` have passed since
the last sweep (`periodic`). The last two wait until no route is committed --
an inspection mid-route costs its own actions plus the turns to recover the
heading it leaves behind. STOP carries no camera action. Raw detection runs on
every observation; committed transitions and inspection views are quarantined
from room/object fusion and target confirmation. Coherent depth pixels are
projected at their actual image coordinates, with frame/floor/association
rejection reasons recorded. No bed/sofa proximity blacklist is used.

## Models and vocabulary

YOLO-World X-v2 remains the default; pretrained LLMDet Swin-L and explicit S/L
checkpoints remain selectable. The Gibson context vocabulary includes `stairs`
and `staircase`, neither a goal category nor a navigation oracle. Semantics can
request inspection; observed step geometry must independently support a portal.

Restart a dedicated detector with the exact `gibson.run --print-vocabulary`
output. A former 24-prompt service is intentionally rejected. HTTP checks pin
classes, checkpoint, configuration, runtime and prompt-feature provenance on
health and inference responses. See the [CPU detector runbook](../../mapping/scene_graph/serve/README.md).
Do not reconfigure another mission's service or place inference on Habitat's GPU.

## Persistent display and evaluation boundary

The recorder reserves **one occupancy map for each of K configured storeys**,
not K layers within every storey. All slots start UNKNOWN, gray; discovery IDs
bind slots without surveyed floor numbering. Each retains its full grid, objects
and trail. Panels use one fixed observed-origin viewport/scale; full arrays are
saved in `floor_maps.npz`. Display counts remain recorder-only, never in policy
settings or episode knowledge. Unvisited slots have no inferred elevation.

GT goals, topology, occupied cells, heights, distances and navmesh queries are
evaluator-only. The policy gets RGB, metric depth, pose and public task/action
information. Gibson reference-floor annotations are incomplete for full-building
ObjectNav; a real unannotated instance cannot be verified by its own detector.
The development-v2 native vertical validator remains **0.60 m**, distinct from
the policy's unchanged **0.24 m** maximum observed tread step.

Coverage is ever-observed occupancy area, not true coverage; it excludes initial
gain and transient terrain, and deduplicates revisits by stable floor ID.
Recordings and interpolated replays never feed evaluator data back to decisions.
Passing tests, encoded videos and completed episodes are not navigation success.
