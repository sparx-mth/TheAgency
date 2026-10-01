# ObjectNav runtime: shared navigation core and dataset tasks
Observed RGB-D mapping → semantic room graphs → room LLM → RPT* room order →
local exploration and A*/WA* paths → existing discrete actions. Both explorers
retain persistent floor contexts; stair connectors come from the simulator's
navmesh by default (`RPTSettings.multifloor.stair_source`, see
[MULTISTORY.md](gibson/MULTISTORY.md)). Habitat target closing additionally uses
declared local NavMesh projection (`target_navmesh_projection` in run configuration).
This is an experimental multi-story algorithm, **not a claim of solved navigation**.
## Layout: one navigation core, one task per dataset
| Layer | Where | Owns |
|---|---|---|
| **Shared navigation core** | `methods/` (RPT*, room-search loop, frontier sweep, exploration fallback, perception with YOLO-World default), `habitat/` (RGB-D bridge, navmesh stair connectors), `evaluation.py`, `provenance.py`, `recording.py`, `visualization.py`, `dashboard.py`, `replay.py` | The algorithm, its settings, and the dataset-neutral execution/recording path. **No dataset loader, episode list, category table, scoring rule or benchmark profile lives here**, and `tests/test_runtime_boundaries.py` fails if a core module imports `gibson/`, `hm3d/` or a label table. |
| **GibsonTask** | [`gibson/`](gibson/README.md) | SemExp v1.1 episodes, FMM scoring maps, planar path accounting, sliding, success without STOP, its CLI and campaigns. |
| **HM3DTask** | [`hm3d/`](hm3d/README.md) | HM3D ObjectNav v1/v2 episodes and scenes, habitat-lab's view-point geodesic metrics, STOP-required success, published-geodesic agreement preflight, its CLI and campaign runner. |
Both tasks construct the same `RPTSearchPolicy` from the same `RPTSettings`,
feed it the same `HabitatRGBDSimulator`, and hand it to the same
`run_evaluation`/`run_benchmark`; what differs is only the `ObjNavEnv` adapter,
the label mapper and the `EvaluationSettings` each protocol declares. Fix the
algorithm once, in `methods/`, and both benchmarks pick it up.
**To set up and run Gibson end to end, start with [QUICKSTART.md](QUICKSTART.md)**;
for HM3D, with [hm3d/README.md](hm3d/README.md). This file is the architecture
reference for the shared core.
## Exploration selection

### High-priority target closing

Both explorers yield **before any room/LLM/RPT* decision** when a target-category
YOLO detection reaches `target_closing.confidence` (default **0.50**). The first
candidate latches ownership in `VERIFY`; at least **two consecutive distinct
frames** with coherent depth and the same fixed 3D association anchor (0.50 m
tolerance) are required to lock `CLOSE`. Duplicate boxes and repeated calls for
the same frame do not count. Context detections below this threshold cannot enter
legacy target pursuit. Neither a timeout nor a rejected route can release ownership;
**only episode reset clears it**. Room reasoning, stair decisions, doorway peeks,
FALCON masks and exploration fallback remain suspended.

The closing sub-policy reuses coherent RGB-D backprojection with the actual camera
pitch. It samples standoff footpoints around the observed target, projects them
onto Habitat's local NavMesh when bound, and uses existing collision-qualified
observed-grid A* to approach. **The selected 3D footpoint and route persist through
occlusion**: missing detections neither cancel the path nor invoke a scan. Fresh
associated observations refine the stored target with a 50/50 update; an XY shift
over `refine_distance_m` (0.15 m) since the goal was selected triggers replanning.
Small refinements do not move the endpoint every frame. Collision, blockage and
progress validation remain active; persistence is not permission to cross obstacles.
Remote or other-floor snaps and unreachable paths
are rejected. The optional simulator callback contains no target annotations,
success flag or evaluator DTG; NavMesh geometry is nevertheless privileged and
is declared in the frozen configuration. Non-Habitat callers use observed-grid A*
without NavMesh projection, explicitly reported as such.

A* owns heading during transit, including doorway/corner turns that put the target
outside the FOV. Bbox yaw servoing is confined to terminal inspection so it cannot
fight the route. At `terminal_distance_m` (**1.0 m**), forward motion stops and
inspection remains stationary. Low objects request LOOK_DOWN; missed detections
trigger bounded pitch-up/down and yaw views referenced to the stored target bearing,
not accumulating turns from the current yaw. The camera owner is `TARGET_CLOSING`,
so level-view restoration cannot override inspection. The original no-tilt Gibson
protocol cannot emit LOOK_DOWN; multi-story development has tilt-enabled actions.
Turns retain the protocol's fixed increments (30 degrees in Habitat) and its
half-turn dead band, not arbitrary micro-turns.

STOP requires a **fresh**, associated detection, completed verification, centered
yaw, satisfied pitch and both filtered and freshly measured horizontal range
within the terminal radius plus `range_tolerance_m` (**0.05 m**, for pitch-dependent
projection noise). No STOP from remembered distance alone is allowed. This terminal
radius replaces legacy `stop_distance_m` for the closing sub-policy. Verification
is bounded at 12 actions, terminal inspection at 24 (`max_reacquire_steps`), and
total closing at 160 by default. Occlusion during transit has no separate timeout.
Exhaustion raises a recordable method error, **not** `ObjNavInternalError`: the harness
can finalize failed metrics. Its forced STOP is not a successful policy STOP and
appears as `termination=agent_error`. Global exploration never resumes.
All thresholds are configurable under `RPTSettings.target_closing`; the phase,
persistent lock, visibility, target xyz, projected goal, refinements, occluded path
steps, bbox and failures are included in episode/frame diagnostics and the HUD.

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
| `methods/target_closing.py` | Episode-local target takeover, consecutive-frame verification, bbox/depth servo, standoff A* and explicit STOP |
| `methods/target_path.py` | Persistent NavMesh standoff goal, meaningful target refinement and continuous collision-qualified A* execution through occlusion |
| `methods/room_search_loop.py` | The seven-step room-search loop: bounded, room-confined local exploration; re-classify → re-estimate → re-order at each loop point; transit to the chosen room's nearest frontier or to the foot of the chosen stairs; a room re-identified by a new kind of object ends its turn for a fresh solve |
| `core/mapping/topology/search_node_oracle.py` | The node oracle: one call per loop point, to the LLM client's REASONING model, over every room and staircase -- P(going there next finds the target) per node plus "elsewhere" |
| `methods/stair_nodes.py` | Staircases as nodes of the loop's RPT* instance: ids above every room pid, the facts the oracle values them by, the climb as a leaf charged on every arc |
| `methods/exploration_fallback.py` | Where every failed plan, model or decision lands: nearest floor-wide frontier, the stairs by the explicit fallback rule, a retired frontier, a relocation -- a move, never an idle spin; failure records and service back-off |
| `methods/frontier_sweep.py` | Frontier goal generation for a room or the floor, committed-goal lifetime, optional look-around |
| `methods/peek_stairs.py` | Floor-local seen-connector exclusion for peek viewpoints and A* routes; ordinary stair navigation is unchanged |
| `methods/camera_control.py` | Sole pitch owner; bounded inspection, long unprompted cadence and safe restoration |
| `methods/perception.py`, `perception_cycle.py` | Fresh raw predictions, coherent pixel projection and floor-qualified fusion |
| `methods/observed_map.py`, `floor_context.py` | Independent occupancy, rooms, objects, association anchors and paused floor clocks |
| `methods/multifloor_policy.py` | The building coordinator: portals per floor, the approach and climb of the staircase the loop's order chose, arrival bookkeeping, both stair sources; in ground-truth mode it decides nothing by a clock |
| `methods/floor_departure.py` | Mandatory room-peek coverage before any floor choice, approach revalidation, movement guard and safe recovery from accidental early stair entry |
| `methods/stair_ground_truth.py`, `floor_decision.py`, `ground_truth_traversal.py` | Ground-truth stairs (default): the navmesh connectors as the policy reads them, the fallback rule's up/down choice (worth per metre of approach, flight and fixed change cost), and the committed polyline traversal |
| `methods/stair_traversal.py`, `stair_terrain.py` | Observed stairs (`stair_source: "observed"`): RGB-D support surfaces, step-connected paths, native-heading safety checks and the committed transition lifecycle |
| `habitat/stair_connectors.py` | Evaluator side: storeys and stair connectors read once per scene from the navmesh in force, attached to the episode metadata by both tasks (`gibson/stair_connectors.py` is a re-export shim) |
| `methods/falcon_policy.py`, `falcon_regions.py` | Bounded hierarchy and persistent local region history |
| `methods/falcon_motion.py`, `falcon_routes.py`, `route_memory.py` | Safe motion and interruption-aware committed routes |
| `methods/object_evidence.py` | Alias/frame deduplication, separated-view target evidence and bounded rejection |
| `methods/scene_graph.py`, `room_labels.py`, `doors.py` | Observed rooms/doors, the room label image, one-object (weak) and two-class (strong) room labels revised the action a new kind of object lands, the nodes handed to the oracle |
| `methods/exploration_metrics.py` | Observed-area proxy, actions, revisits, stagnation and latency |
| `habitat/simulator.py` | Synchronous registered RGB-D, native extrinsic checks and ENU pose |
| `recording.py`, `floor_panels.py`, `search_panel.py`, `visualization.py` | Exact commands/poses, persistent UNKNOWN panels with the rooms and visit order drawn on the active floor, the search column, grids and video |
| `gibson/run.py`, `hm3d/run.py`, their `frozen_eval.py`, `run_development.py` | Per-task protocol execution and frozen source/model/data checks |
| `tests/trace_room_search_loop.py` | Development probe: the loop on the production map size with fake services and the real action converter on a point agent; not a test |

The floor tracker remains ROS-free in `core/planning/exploration/floor_atlas.py`.
Host-side numpy/scipy terrain processing does not enter the Noetic exploration
facade. No robot's flight controller is changed.

## Episode discovery and doorway inspection

Both explorers start with **`RPTSettings.warmup_steps=10`** emitted discovery
actions before normal room selection. RGB-D, YOLO, object confirmation and
scene-graph geometry remain synchronous on every frame. Warm-up uses reachable
floor-wide frontier gain/geodesic/heading utility; with no usable frontier yet,
a bounded turn acquires another view. Target pursuit and committed stair safety
take priority. Warm-up is episode-local, not repeated on each floor. Early peek
actions count toward it; the first global room decision waits for its completion.

**`RPTSettings.doorway_peek.enabled=true`** enables opportunistic inspection.
Disable opportunistic interruptions with `"doorway_peek": {"enabled": false}` in the JSON passed through
`--policy-config`; `"warmup_steps": 10` sets the mandatory positive warm-up budget.
This does not waive mandatory peeks before leaving a floor. These settings
and the floor-exit invariant are included in the frozen method configuration.

For a room with an entrance within **3 m of reachable travel**, the policy plans
**a further metre inside** (`doorway_peek.inset_m=1.0`), beyond the safe threshold
and allowing for the converter's early stopping tolerance. Door proximity
constrains the entrance, not the deeper viewpoint. Small rooms use the deepest
safe observed viewpoint available. A room-contained sightline and the normal
clearance-qualified planner reject viewpoints through walls or unknown space.
The robot aligns to one side and sweeps **at least 180 degrees of measured yaw**.
Duplicate plan calls and back-and-forth yaw jitter do not complete a scan.
The spawn room and a room merely walked through can receive their first peek
only when still unclassified and never previously peeked.

**At most ONE peek attempt per room per episode.** `room_peeked=True` is latched
when the attempt starts, not only after a successful scan. Cancellation, a failed
entry, a scan budget or target/stair preemption never refunds it. A room already
scanned, initially classified (including a single-object weak/provisional label,
with no extra confidence threshold), or previously peeked is exempt. Available
confirmed-object clues are classified before choosing a peek; a label obtained
during a peek cancels the rest of that inspection. `unknown` alone is not a label
exemption. `done=True` still means an actual measured scan completed, not simply
that the one-time allowance was consumed.

The floor-exit gate now asks whether any **eligible unclassified non-stair rooms**
still lack their one-time attempt, not whether exempt rooms completed a scan.
This is an eligibility policy, not a full semantic-coverage guarantee. Zero known
rooms remains incomplete. A genuinely new unknown room can revoke a stair approach;
classified/scanned/attempted rooms cannot force forbidden repeat peeks. A vetoed
ordinary room transit is released as unreachable rather than repeatedly replanned
through the same gated tread. Native collision checks and committed-traversal
safety remain in force.

Peeks cannot start or continue during a committed stair approach, traversal, atlas
transition, or target takeover. Seen connector geometry near the current elevation
is excluded from peek viewpoints and from a peek-only A* occupancy copy. A path
crossing the stair corridor is refused even if its endpoint lies in a room beyond
it; overhead/other-floor connector segments do not erase ordinary room space.
Stair-only regions are exempt from the peek gate. No scene geometry is added to
the normal map and unseen connectors do not create exclusions.

The previous route is isolated, not destroyed. Its local budget and timeout clocks
pause during inspection and any necessary return transit. After the scan, room
classification, node probabilities and RPT* run synchronously. If the old room
stays first, its **remaining** exploration budget resumes; if another room wins,
normal selection/arrival starts its local turn. New off-route rooms/object kinds
also trigger synchronous reconsideration with peeks disabled. A retained task
does not get a new ten-action allowance merely because evidence changed.

Peeks are non-nesting and remembered per floor and geometric region. Consumed
allowances follow overlapping persistent IDs and substantially overlapping
renumbered/split/expanded regions (60% of the smaller region), but never disjoint
rooms or another floor. This conservatively avoids repeating inspections after
segmentation changes; it does not certify each split child was physically scanned. Defaults:
approach budget 36 actions (extended by travel distance for mandatory distant
visits); return allowance 60 actions; scan budget at least 24 actions (scaled for small turn increments);
at most one attempt per region (`max_attempts` cannot exceed 1; legacy retry
timestamps remain diagnostic only). Invalidated
rooms, blocked entries and stalled scans resume the search rather than spin.
Target approach cancels inspection immediately; committed stair safety is never
abandoned. `doorway_peek.coverage`, `floor_change_held`, completed scan positions
and measured angles, warm-up, phase and event diagnostics are recorded.

ObjectNav's optional FALCON local burst also defaults to 10 actions; an explicit
`falcon.burst_actions` override remains supported. Its specialized local planner
and bounded region history remain distinct from the frontier explorer.

**Accessible frontiers are one shared inventory**: passable known free space,
robot clearance, connected reachability from the current pose, and safe frontier
goal association. No distance horizon is mistaken for inaccessibility. The room
LLM, room facts, entry selection, floor-wide discovery/fallback and dashboard use
it. Unassigned frontiers remain exploration choices and their representatives
remain in the recorded search snapshot, not cluttering the minimalist map. Retired/reached goals are a separate
action-selection filter, not falsely declared geometrically inaccessible. Zero
accessible frontiers does not prove complete semantic observation of a room.

CPU regressions in `tests/test_discovery.py` and `tests/test_doorway_actions.py`
cover startup, real action-converter entry/scan/return, budget preservation,
target preemption, reachability, map changes and synchronous reassessment.
`tests/test_one_time_peeks.py` covers exemptions, no-refund cancellations and stair
isolation. ObjectNav enables a 1-degree path-only angular hysteresis margin: after
forward progress or an opposite turn on the same path, prefer a distance-reducing
forward step within half a turn plus this margin. The generic converter defaults
to zero; STOP, pitch, final facing, blocked recovery and safety vetoes retain priority.
They are not measurements of detector/LLM accuracy or Habitat navigation success.

## The room-search loop (frontier explorer)

`--explorer frontier` runs one explicit loop, with the scene graph maintained
synchronously on every action (`RPTSettings.graph_period_steps`, default 1):
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
   actions (10), when nothing reachable is left, **or when the room is
   re-identified** -- whichever first -- and the supervisor is told so
   through its `budget_spent` / `frontier_exhausted` / `room_reclassified`
   exits on that same action. The third exit is **the clue rule**: one
   confirmed object names the room (`RoomLabelSettings.min_objects = 1`; a
   single class is a *weak* label, shown to the oracle as `type=kitchen?`,
   two or more confident classes a *strong* one), the classifier is re-asked
   for the room in force or in transit **the action a new kind of object
   lands in it** (`relabel` events; a second chair is not a clue, and a set
   of kinds already judged is never bought twice), and a changed name ends
   the room's turn as `reclassified` -- neutral: not cooled, no attempt
   charged -- so that steps 2-4 run over the new fact. Whether a bathroom
   is still worth sweeping for a frying pan is then the oracle's call, and
   it says so with a probability of about zero; the sink first taken for a
   kitchen is re-valued the action the toilet shows.
2. **Global re-classification.** At the loop point `graph.reason` re-labels
   every known room from every confirmed object observed so far.
3. **Probability per node** -- the same call, to
   `core/mapping/topology/search_node_oracle.py`. The model is shown every
   ROOM of the floor -- type or `unknown`, size, frontier clusters left,
   time searched and how long ago it was last inside, objects seen, whether
   the robot stands in it -- and every STAIRCASE off the floor -- up or
   down, whether the other storey was visited and what was found and
   searched there, whether the robot arrived by it -- plus one line on this
   storey and each other storey known. It answers, for each node, the
   probability in percent that **going there and searching finds the target**.
   These are independent search-success estimates, **not mutually exclusive
   location shares**, and are not restricted to the ten-action burst. Each is
   converted to a fraction and clamped below 1, without cross-node normalisation.
   RPT* uses multiplicative survival; the legacy `elsewhere` diagnostic now means
   the product of failure probabilities, and `p_present` its complement. This is
   an explicit independence approximation, not proof that real rooms are independent.
   The current room may remain the best choice for successive bursts. Everything the search
   needs weighed is the model's to weigh, and the system prompt says how:
   a type that rules the object out has low probability; a
   room searched long and recently reads low; a never-entered room of the
   right type is the best bet; an `unknown` room is an **exploration node**
   valued by its size, its frontier and the room types the target needs
   that are still missing on this storey; a hallway with frontier is valued
   by what it may lead to; a staircase is the whole set of rooms the other
   storey may hold, read from what this storey turned out to be (a kitchen
   and a living room make a ground floor, so the bedrooms are upstairs),
   discounted when that storey was searched, and not taken straight back
   when the robot has just come from it. It is told, twice, **never to
   reason about distance**: travel is the planner's. This is the one
   judgement per loop point the search cannot afford to get wrong, so the
   runtime routes it to the LLM client's reasoning model
   (`LLM_REASONING_MODEL`, default `qwen2.5:14b-instruct`, with its own
   timeout and context window -- a 3B double-counts effort into semantics
   and cannot weigh a staircase against an unknown room; the 14B follows
   the rules and answers in ~25 s on the laptop's CPU once loaded) and
   waits for it. The oracle
   reuses the model's last reply when the prompt it would show is
   byte-identical (effort numbers are shown in coarse steps for exactly
   that), so a loop point over an unchanged map costs no call
   (`oracle_reuses`). The per-node record in `room_search_loop.estimates`
   carries the probability, the model's reason and the distance charged.
4. **Visit order** -- RPT* over the surviving nodes, rooms and staircases
   together (`methods/stair_nodes.py`). A room enters the instance at its
   entry point; a staircase at the foot of the flight on this floor's map,
   with a **leaf** as long as the flight plus `MultiFloorParams
   .floor_change_cost_m` (8 m, about thirty actions) charged on every arc
   into and out of it -- so going upstairs puts every room down here that
   much further away, which is exactly what the expected-time-to-find
   objective must see. The local budget is folded in as per-node service
   time; the order is re-solved at every loop point
   (`ObjectSearchParams.resolve_on_release`) and on semantic discovery events,
    not merely on a timer. Floor choices become eligible only after all observed
    rooms have completed their interior peeks, then remain the LLM/RPT* decision;
    the existing 40-action arrival-return guard and fallback rule still apply.
   When the order puts a staircase first the loop commits it to the
   building coordinator, which approaches the foot of the flight and
   climbs; the node's turn ends `traversed` the action the climb begins,
   and the floor's search resumes from a fresh estimate when the robot
   comes back down -- with the way back up a node like any other, marked
   `arrived_by`. The order and its head are kept on the loop (`order`,
   `order_index`, `next_room`) for the recording and the dashboard.
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
estimate may rightly send the agent straight back), nor is a staircase
whose climb began or a room released `reclassified`; a room the live map
says is finished (exhausted or mapped) is never repeated by the
every-room-cooling escape hatch: the exploration fallback carries the
search instead. `room_search_loop.events` in the episode record lists every
transit, arrival, relabel, release and `stairs_taken` with its verdict and
local step count; `estimate_events` lists every oracle round with the nodes
shown, `p_present`, `elsewhere` and whether the reply was reused.
Runtime is not the constraint here, the judgement is: the oracle is asked
once per loop point and the search waits for the answer. The cheaper calls
stay bounded by construction: one classifier call per **new kind** of
object per room, with a verdict for a set of kinds reused across the
watershed's re-partitions rather than bought again (the trace probe went
from 22 classifier calls in 60 actions to 5 with identical decisions).
## The exploration fallback: a failure is a move, not a spin

Every way the decision pipeline can fail lands in one place,
`methods/exploration_fallback.py`, and comes out as a command that moves:

1. the **highest-utility reachable frontier anywhere on the floor** -- not confined to
   the room in force, so a room whose routes all fail is left rather than
   spun in;
2. the **stairs** by the explicit fallback rule (`floor_decision.py`:
   eligible, reachable now, storey worth visiting, worth per metre of
   approach, flight and fixed change cost), when the floor is exhausted.
   The normal way to the stairs is not this rung: the loop offers every
   staircase to the oracle and to RPT* as a node beside the rooms and climbs
   when the order says so. This is for the floor where that machinery has
   nothing left -- the room LLM is away, or no node is worth anything and
   the frontier is gone -- and a climb it makes is recorded
   `rule: fallback`;
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
- The optional extra look-around after a swept room is off by default
  (`SweepSettings.room_scan_turns = 0`): the loop's termination rule is *N
  steps or nothing left*, and a swept room is released on the action its last
  goal resolves. This is separate from the mandatory first-entry interior peek
  and its measured 180-degree scan. The supervisor's 30 s stall and 90 s budget clocks remain as
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
from global room/object fusion. Target closing has its own pitch-aware evidence
and takes priority even over committed stair tasks. Coherent depth pixels are
projected at their actual image coordinates, with frame/floor/association
rejection reasons recorded. No bed/sofa proximity blacklist is used.

## Models and vocabulary

YOLO-World X-v2 remains the default; pretrained LLMDet Swin-L and explicit S/L
checkpoints remain selectable. The Gibson context vocabulary includes `stairs`
and `staircase`, neither a goal category nor a navigation oracle. Semantics can
request inspection; observed step geometry must independently support a portal.

Restart a dedicated detector with the exact `gibson.run --print-vocabulary`
(or `hm3d.run --print-vocabulary`) output. A former 24-prompt service is intentionally rejected. HTTP checks pin
classes, checkpoint, configuration, runtime and prompt-feature provenance on
health and inference responses. See the [CPU detector runbook](../../mapping/scene_graph/serve/README.md).
Do not reconfigure another mission's service or place inference on Habitat's GPU.

## Persistent display and evaluation boundary

The recorder reserves **one occupancy map for each of K configured storeys**,
not K layers within every storey. All slots start UNKNOWN, gray; discovery IDs
bind slots without surveyed floor numbering. Each retains its full grid, room
partition, objects and trail. A shared viewport grows with the observed house
footprint in metre-sized increments; all floors use the same metric scale. Maps
stack or sit side-by-side to fit the house aspect ratio. Full occupancy arrays
are saved in `floor_maps.npz`, room labels in `room_partitions.npz`, without
pickled state. Display counts remain recorder-only, never in policy
settings or episode knowledge. Unvisited slots have no inferred elevation.

The dashboard frame (`visualization.render_dashboard`) reads, left to right:
the camera with the detector's boxes and the depth; minimalist floor plans with
thin logical room outlines on **every discovered floor**, short `R<n>` identifiers,
the room in force ringed amber and the next node green, a compact square for a
seen staircase, the robot, a subdued trail, the committed route and a scale bar.
Object dots, semantic names, probabilities, frontier diamonds and the visit-order
chain are deliberately absent from the maps. Those details remain in the **search column**
(`search_panel.py`): the target and the oracle's split between the listed
nodes and elsewhere, the RPT* visit order with its head (`S0` is a
staircase), the room in force with its local step count or the node in
transit, every room's type (a `?` marks a weak, one-object label), oracle
probability and the model's reason, frontier left, time spent and when it
was last inside, the objects seen in it, every staircase offered with its
probability, direction, the climb's cost in metres and the storey beyond,
the confirmed objects by room, and the loop's last events. All of it comes
from `method_snapshot(policy)["search"]`, which is also written to every
`steps.jsonl` row and, in short, to `live.json`, so the video and the trace
cannot disagree.

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

## Adapter boundary — keep benchmark rules out of shared code

Each simulator branch supplies:

1. An `ObjNavEnv` implementation: published episode loading, reset/step, episode
   termination and evaluator-only measurements. `measure()` is not called early.
2. Exact scene/split identities and a `TableLabelMapper` with that dataset's
   category spellings, detector prompts and accepted aliases. The infrastructure
   registry does not claim any dataset is implemented.
3. Explicit camera intrinsics, registered metric depth, body dimensions, action
   lengths/angles, sliding policy, step budget and protocol/version metadata.
4. Its published success/STOP/distance/path-accounting rules. Shared defaults
   remain **STOP required, 3D observed path length, zero initial accumulator**.
   Override `EvaluationSettings` only when the adapter's protocol requires it.
5. Data/model provisioning and checks, including GPU ownership. No weights,
   licensed scenes, containers or remote services are downloaded/launched by the
   shared evaluation entrypoint.
6. A manifest identifying the actual dataset/assets, model checkpoint/revision,
   package versions, seed and protocol. Keep credentials out of this metadata;
   the shared service URL validator rejects embedded credentials/query strings.

Gibson and both HM3D versions reuse the Habitat bridge. The HM3D versions
share one adapter -- [`hm3d/`](hm3d/README.md) -- with two versioned protocol
objects, because only their episodes and scene release differ. RoboTHOR needs an AI2-THOR environment bridge but reuses
the policy, action conversion, provenance and recording; neither it nor MP3D is
implemented here.

The Habitat bridge takes an explicit `navmesh` policy, because which navmesh an
episode runs on decides what is reachable and therefore sets `l` and every SPL.
`"published"` uses the shipped mesh verbatim; `"agent"` is habitat-sim's own
load-then-recompute-at-the-agent's-dimensions behaviour, which is what
habitat-lab's ObjectNav actually runs. Whichever an adapter picks,
`navmesh_provenance()` records what was in force.

### Explicit embodiment and method settings

`RPTSettings` (`methods/rpt_settings.py`) carries `body_height_m`,
`body_radius_m`, `preferred_clearance_m` and `stop_distance_m`; each task's
`run.py` sets them from its protocol's agent, never from a benchmark name. Physical
inflation is never relaxed below the supplied body radius. The stop distance
is a method choice, not an override of the evaluator's success definition.

`RPTSearchPolicy` records its planner, loop, fallback, sweep, door, room-label
and route settings in `configuration()`. `run_evaluation`
uses the same converter settings in the actual headless agent, so route
arrival/progress checks agree with the executor.

`HabitatRGBDSimulator` separately requires body height and radius; camera mount
height is not body height. Under `navmesh="published"` it loads the supplied
navmesh verbatim and never recomputes it; under `navmesh="agent"` it uses
habitat-sim's own recomputation and refuses a navmesh path that is not the
scene's sibling. It never snaps a published start. Only the explicitly synthetic
renderer smoke constructs its own fixture navmesh.

## Shared execution and locks

Construct the environment, policy, label mapper and adapter metadata, then call
`run_evaluation` with an explicit output directory. It uses the existing
`run_benchmark`/`MetricsLogger`, not a second scoring loop, and closes the
environment even on recording failures. Extra `evaluation_diagnostics()` is an
optional recorder-only hook; environments without it remain recordable.

`progress=` takes a `(index, total, record)` callback, so an adapter's CLI can
report each finished episode — a thousand-episode run that prints nothing until
it ends is unusable. It runs *after* the recorder's own hook, which is what
finishes writing an episode's artifacts, so a caller can never displace it.

For frozen evaluation:

1. Use `evaluation_configuration(config, episode_ids, settings, policy=policy)`
   to capture actual Python sources, the policy, converter, selected episode
   order and execution settings alongside adapter metadata.
2. Call `freeze_configuration` with a new lock path and an explicit
   `development_note` describing prior tuning/exposure. Existing locks are not
   overwritten.
3. Pass that lock to `run_evaluation(frozen_lock=...)`. It checks the actual
   execution configuration before output creation or reset, not merely an
   earlier preflight. To claim a complete published split, also supply the
   publisher's exact ordered `expected_episode_ids`.

There is no hardcoded scene count or episode count. A lock is a reproducibility
commitment, **not proof that validation was unseen**. It never creates a
held-out or SOTA claim. Full benchmark equivalence, tuning disclosure and
reference-version caveats remain the adapter's responsibility. Dashboards only
summarize recorded rows and do not label arbitrary subsets as complete splits.

## Recording and visual-only replay

Recording is optional and loads plotting/video dependencies only when requested.
It keeps images/maps out of per-step JSON; the JSON records compact observed
state, actions, predictions, paths and reasoning. Model service metadata pins
checkpoint bytes and vocabulary atomically with inference. Missing or changed
services raise rather than silently changing the evaluated method.

`render_replay(recording_dir, render_pose, ...)` takes a callback receiving an
`AgentPose` and returning uint8 RGB. The adapter owns scene loading, GPU safety
and camera-pitch handling. Replay interpolates translation/pitch and the shortest
yaw arc, includes the final pose, preserves aspect ratio, and writes only a
separate `smooth_rgb.mp4`. It does not alter original scores or observations.

## Dependencies and validation

`requirements.txt` lists the shared CPU/scientific extras. Core contracts and the
lightweight harness remain free of simulator/model imports. Install simulator
bindings in their dedicated environments; do not install torch into the
lightweight test environment. Recording requires a working FFmpeg/libx264;
`IMAGEIO_FFMPEG_EXE` may select an already installed executable.

From the checkout root with the test interpreter selected:

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests \
  sparx_agency/core/mapping/objects/tests sparx_agency/core/mapping/topology/tests \
  sparx_agency/core/planning/objnav sparx_agency/tasks/planning/objnav_benchmark \
  sparx_agency/core/planning/planners/common/tests \
  sparx_agency/core/planning/planners/astar/tests -q
```

Tests use explicit synthetic geometry and the existing fake/corridor fixtures,
not a real dataset's constants. They cover route lifecycle, physical clearance,
object/room evidence, optional imports, explicit protocol accounting, locks,
resumption, metadata drift, recording without GT telemetry and replay integrity.

For a provisioned Habitat environment and an idle rendering GPU:

```bash
python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.smoke
```

This creates a temporary license-free room and checks real RGB-D, ENU frame
conversion, left/right turns, forward movement and STOP. Its result is explicitly
synthetic, not a published benchmark score.
