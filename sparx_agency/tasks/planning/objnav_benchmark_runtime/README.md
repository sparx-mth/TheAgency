# ObjectNav runtime and Gibson adapter

Observed RGB-D mapping → semantic room graphs → room LLM → RPT* room order →
local exploration and A*/WA* paths → existing discrete actions. Both explorers
retain persistent floor contexts; stair connectors come from the simulator's
navmesh by default (`RPTSettings.multifloor.stair_source`, see
[MULTISTORY.md](gibson/MULTISTORY.md)). Habitat target closing additionally uses
declared local NavMesh projection (`target_navmesh_projection` in run configuration).
This is an experimental multi-story algorithm, **not a claim of solved navigation**.

**To set up and run it end to end, start with [QUICKSTART.md](QUICKSTART.md)**: environments,
dataset and weight downloads, then the services, the frozen episodes and the campaign in
order. This file is the architecture reference.

## Exploration selection

### High-priority target closing

Both explorers yield **before any room/LLM/RPT* decision** when a target-category
YOLO detection reaches `target_closing.confidence` (default **0.50**) **and projects
to a coherent 3-D position** -- a confident box the depth sensor cannot place has
nothing a second frame can be checked against, and the twelve-step verifications
such boxes bought ("waiting for valid target depth" from 3.8 m, Ranchester attempts
4 and 5) released every time. The first such candidate latches ownership in
`VERIFY`; at least **two consecutive distinct frames** with coherent depth and the
same fixed 3D association anchor (0.50 m tolerance) are required to lock `CLOSE`.
Duplicate boxes and repeated calls for the same frame do not count. Context
detections below this threshold cannot enter legacy target pursuit.

**Detect high, associate low.** A detector's confidence in one object swings with
the view: the Ranchester couch read 0.68 from the stair foot at one heading, 0.42
one 30-degree turn later with the same object centred, 0.52 clipped at the next --
and the agent's own centring turn broke the consecutive run on every cycle
(twenty-four frames with the couch in view, two releases). Once a candidate is
active, a box of the target's class that projects onto its anchor counts from
`track_confidence` (**0.30**); a weak box anywhere else, or before any takeover,
counts for nothing (`weak_resightings` in the diagnostics).

**What may start a takeover.** A box within `border_margin_px` (**8 px**) of any
image edge is a partial view — its depth centroid is unreliable, and the clipped
end of a bed reads as a sofa (the Ranchester couch-downstairs run of 2026-10-04
ended on exactly such an 84 px sliver). Clipped boxes cannot start or count
towards a lock; once locked, the approach tolerates spill-over. A box projecting
within `rejection_radius_m` (**1.0 m**) of a spot already released as unverified
on the same floor is ignored before a lock -- **unless it is seen from within
`rejection_min_range_m` (2.0 m)**: the memory stops the same far flicker from
restarting the takeover from the same place, and a close view is new evidence
(the Ranchester couch released at 3.8 m from the stair head was never
re-verified from 1 m, twice, until this). The 3-D association radius for the
consecutive-frame lock grows with range (`association_range_gain`, 0.15 m per
metre beyond 2 m: 0.77 m at 3.8 m), because the visible centroid of a long
object seen from far moves more than half a metre between two frames.

**Before the lock, ownership is provisional.** If `max_verify_steps` (12) pass
without the lock, `release_unverified` (default **true**) hands control back to
exploration in phase `RELEASED`, records the anchor in the rejection memory and
clears the legacy target hint; the room/stair search resumes where it was
interrupted. `release_unverified: false` restores the historical behaviour, where
the exhausted verification raises the recordable method error. **After the lock
nothing releases ownership**: neither a timeout nor a rejected route;
**only episode reset clears it**. Room reasoning, stair decisions, doorway peeks,
FALCON masks and exploration fallback remain suspended while the takeover owns
the action.

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
is bounded at 12 actions (`max_verify_steps`, released as above), terminal
inspection at 24 (`max_reacquire_steps`), and total closing at 160 by default.
Occlusion during transit has no separate timeout. **The terminal inspection on a
big object at close range** (the Ranchester couch, 2026-10-04, actions 172-196):
a fresh box that *spans the image centre column* counts as aligned whatever its
centre says (a clipped box's centre is not the object's centre, and centring it
turned the agent off the couch twelve times); the terminal range is the
**near edge** of the freshly measured surface (`range_near_m`, the 10th
percentile horizontal distance of the supported depth points) or the centroid,
whichever is nearer -- the benchmark's success radius is measured to the
object, not to its middle; and when the inspection budget runs out after the
locked target *was* seen fresh within terminal range from this spot, the
closing **STOPs** (`stop_on_exhausted_inspection`, default true) instead of
raising. Exhaustion of a **locked** target that was *never* seen from the
inspection spot means the lock was wrong -- a sofa from four metres that is
nothing from one (the Ranchester same-storey run of 2026-10-04 ended as an agent
error on exactly that, with the real couch 2.85 m away) -- so since 2026-10-04 the
candidate is **released** (`release_on_failed_inspection`, default true): its
anchor goes into the rejection memory with twice the radius
(`failed_inspection_radius_factor`), the release action is one turn in place,
and the search resumes on the next action with the spot excluded from the
takeover and from the target-landmark nodes. The release comes as soon as every
inspection view (three yaws, up to three pitches) has been tried once with
nothing seen -- not after the 24-action budget -- and *at once* when the map's
class vote at the anchor outvotes the lock (`map_releases`: the sofa from four
metres that the map has since confirmed as a bed from two). `release_on_failed_inspection:
false` restores the historical recordable method error (**not**
`ObjNavInternalError`: the harness finalizes failed metrics; its forced STOP is
not a successful policy STOP and appears as `termination=agent_error`).
All thresholds are configurable under `RPTSettings.target_closing`; the phase,
persistent lock, visibility, target xyz, projected goal, refinements, occluded path
steps, bbox, failures, release counts (`releases`, `inspection_releases`), border
rejections and the rejected spots with their radii
are included in episode/frame diagnostics and the HUD.

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
| `methods/room_search_loop.py` | The seven-step room-search loop: one scan visit per room (vantage point, full rotation, finished for the episode) or the bounded room-confined sweep; re-classify → re-estimate → re-order at each loop point over the rooms that are still nodes; transit to the chosen room's vantage point / nearest frontier or to the foot of the chosen stairs; a room re-identified by a new kind of object ends its turn for a fresh solve |
| `methods/room_scans.py` | The scan ledger: where every completed look-around stood, on every floor; a room is finished when a scan stood in it or saw more than half of it through observed free space -- sticky, id-independent |
| `methods/room_vantage.py` | Where to stand in a room to see it: the reachable interior cell of greatest clearance (distance transform of the room mask) |
| `methods/room_priors.py` | Where a target cannot be: the room types a search need not enter for it (exclusions, not permissions; `unknown` never excluded) |
| `core/mapping/topology/search_node_oracle.py` | The node oracle: one call per loop point, to the LLM client's REASONING model, over every room still a node and every staircase -- P(going there next finds the target) per node plus "elsewhere" |
| `methods/stair_nodes.py` | Staircases as nodes of the loop's RPT* instance: ids above every room pid, the facts the oracle values them by, the climb as a leaf charged on every arc |
| `methods/exploration_fallback.py` | Where every failed plan, model or decision lands: the best floor-wide exit (object shadows and frontiers of rooms the target cannot be in wait behind it), the stairs by the explicit fallback rule, the demoted frontiers, a retired frontier, a relocation -- a move, never an idle spin; failure records and service back-off |
| `methods/frontier_sweep.py` | Frontier goal generation for a room or the floor, committed-goal lifetime, optional look-around (the `sweep` ablation) |
| `methods/peek_stairs.py` | Floor-local seen-connector footprint: excluded from the room partition (stairs are never a room), from peek viewpoints and from a peek-only A* copy; ordinary stair navigation is unchanged |
| `methods/camera_control.py` | Sole pitch owner; bounded inspection, long unprompted cadence and safe restoration |
| `methods/perception.py`, `perception_cycle.py` | Fresh raw predictions, coherent pixel projection with a footprint radius, floor-qualified fusion, plan-view association with a height check, class votes per landmark |
| `core/mapping/objects/landmarks.py` | The landmark map: positional association (dedupe radius or footprint-disc IoU), per-class vote tally per landmark, plurality class with recorded relabels, confirmation by a clear plurality |
| `methods/observed_map.py`, `floor_context.py` | Independent occupancy, rooms, objects, association anchors and paused floor clocks |
| `methods/multifloor_policy.py` | The building coordinator: portals per floor, the approach and climb of the staircase the loop's order chose, arrival bookkeeping, both stair sources; in ground-truth mode it decides nothing by a clock |
| `methods/floor_departure.py` | The optional (`doorway_peek.gate_floor_departure`, off by default) room-peek coverage gate before a floor choice, with its approach revalidation, movement guard and safe recovery from accidental early stair entry |
| `methods/stair_ground_truth.py`, `floor_decision.py`, `ground_truth_traversal.py` | Ground-truth stairs (default): the navmesh connectors as the policy reads them, the fallback rule's up/down choice (worth per metre of approach, flight and fixed change cost), and the committed polyline traversal |
| `methods/stair_traversal.py`, `stair_terrain.py` | Observed stairs (`stair_source: "observed"`): RGB-D support surfaces, step-connected paths, native-heading safety checks and the committed transition lifecycle |
| `gibson/stair_connectors.py` | Evaluator side: storeys and stair connectors read once per scene from the navmesh, attached to the episode metadata |
| `methods/falcon_policy.py`, `falcon_regions.py` | Bounded hierarchy and persistent local region history |
| `methods/falcon_motion.py`, `falcon_routes.py`, `route_memory.py` | Safe motion and interruption-aware committed routes |
| `methods/object_evidence.py` | Alias/frame deduplication, separated-view target evidence and bounded rejection |
| `methods/scene_graph.py`, `room_labels.py`, `doors.py` | Observed rooms/doors, the room label image, one-object (weak) and two-class (strong) room labels revised the action a new kind of object lands, the nodes handed to the oracle |
| `methods/exploration_metrics.py` | Observed-area proxy, actions, revisits, stagnation and latency |
| `habitat/simulator.py` | Synchronous registered RGB-D, native extrinsic checks and ENU pose |
| `recording.py`, `floor_panels.py`, `search_panel.py`, `visualization.py` | Exact commands/poses, persistent UNKNOWN panels with the rooms and visit order drawn on the active floor, the search column, grids and video |
| `gibson/run.py`, `frozen_eval.py`, `run_development.py` | Protocol execution and frozen source/model/data checks |
| `tests/trace_room_search_loop.py` | Development probe: the loop on the production map size with fake services and the real action converter on a point agent; not a test |

The floor tracker remains ROS-free in `core/planning/exploration/floor_atlas.py`.
Host-side numpy/scipy terrain processing does not enter the Noetic exploration
facade. No robot's flight controller is changed.

## Episode discovery and doorway inspection

Both explorers start with **`RPTSettings.warmup_steps=12`** discovery actions
before normal room selection -- **one full rotation in place** at the
benchmark's 30-degree turn, taken where the agent stands. RGB-D, YOLO, object
confirmation and scene-graph geometry remain synchronous on every frame, so the
rotation gives the map its first rooms and the detector its first objects; when
it ends the agent's position is recorded in the **scan ledger**
(`methods/room_scans.py`), which finishes the spawn room before the first room
is ever chosen. Target pursuit and committed stair safety take priority; a
warm-up interrupted by one and resumed more than 0.5 m away starts its circle
again, so the ledger's record lands where the whole circle was turned. The
warm-up is repeated on every storey the agent first sets foot on (the stair
head is scanned before any room there is chosen). The first global room
decision waits for its completion.

**Doorway peeks are OFF by default since 2026-10-04.** Under the scan visit a
peek -- approach, a half turn from the threshold, the walk back -- costs about
as much as a room's own scan and finishes nothing (its half turn is not a
ledger scan), and the four peeks of the last Ranchester recording were all
cancelled. `"doorway_peek": {"enabled": true}` in the JSON passed through
`--policy-config` restores them with the behaviour below. **The floor-exit
gate they fed is OFF too** (`doorway_peek.gate_floor_departure`, default
false): a floor change is the RPT* order's or the fallback rule's to take
whenever it is chosen, with the 40-action arrival-return guard the only hold.
The Ranchester recording of 2026-10-04 held its descent fourteen times while
re-partitioning kept producing new "pending" rooms, and never reached the
storey the couch was on. `gate_floor_departure: true` restores the former
behaviour as an ablation. Both settings are included in the frozen method
configuration (`room_visit`, `floor_exit_gate`, `room_partition`).

When enabled: for a room with an entrance within **3 m of reachable travel**, the policy plans
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

With `gate_floor_departure` on, the floor-exit gate asks whether any **eligible
unclassified non-stair rooms** still lack their one-time attempt, not whether
exempt rooms completed a scan. This is an eligibility policy, not a full
semantic-coverage guarantee. Zero known rooms remains incomplete. A genuinely
new unknown room can revoke a stair approach; classified/scanned/attempted
rooms cannot force forbidden repeat peeks. A vetoed ordinary room transit is
released as unreachable rather than repeatedly replanned through the same
gated tread. Native collision checks and committed-traversal safety remain in
force whether the gate is on or off.

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

1. **One visit, bounded to the room in force** (`LoopSettings.visit`,
   default `scan`, specified 2026-10-04). **A visit is a look-around**: walk
   to the room's **vantage point** -- the interior cell of greatest clearance
   the observed map can reach (`methods/room_vantage.py`: the Euclidean
   distance transform of the room mask, so the middle of the open floor, not
   the doorway and not the far frontier) -- turn a **full circle**, and
   leave. The camera's ray ends at the nearest object, so a 360-degree scan
   from the open floor shows what a second viewpoint or a walk to the far
   frontier would; the room is then **finished for the episode**, whatever
   frontier its mask still shows. Finishing is the **scan ledger's**
   verdict (`methods/room_scans.py`): it remembers *where* every rotation was
   completed, not which room id it was credited to -- the watershed
   renumbers, splits and merges rooms on every update, and the Ranchester
   recordings re-entered the same room under a new number three times in a
   row. A room is finished when a completed scan point lies inside its mask
   (`scan_point_inside`) **or** when a completed scan *saw* more than
   `scan_seen_fraction` (0.5) of its cells within the camera's depth range
   through observed free space (`seen_from_scan`: the small room fully
   visible from the corridor, the half of a split room the scan stood in;
   unknown cells end the sightline, so nothing unresolved is credited). The
   verdict is sticky per `(floor, pid)`. The visit is bounded hard
   (`scan_visit_steps`, 36 -- released `budget_spent` beyond it) and the
   walk to the vantage point separately (`scan_approach_steps`, 20; an
   approach that runs out, or that A* refuses, scans from where the agent
   stands if that is inside the room -- `scans_in_place` -- and otherwise
   releases the room `unreachable`). The rotation ends on measured yaw
   (a full circle swept, or two turns past a circle's worth), the scan is
   recorded, and the turn ends `exhausted`: productive, cooled, never
   repeated by the escape hatch. **A clue mid-visit does not switch rooms**
   (`reconsiders_deferred`): the Ranchester recording switched at 57, 73
   and 121 and came back to each; the clue is kept for the loop point the
   visit ends on, where every node is re-valued anyway.

   **What is not a node.** At every loop point the loop withholds two kinds
   of room from the oracle, the solver and the transit, and records why
   (`excluded` in the events, estimates and HUD): a room the ledger has
   **finished**, and a room whose identified type **cannot hold the target**
   (`methods/room_priors.py`, `LoopSettings.type_prior`: a sofa is not
   searched for in a bedroom, a bathroom or a kitchen; a toilet not in a
   bedroom; the table is exclusions, not permissions, so a hallway stays
   searchable and `unknown` is never excluded -- unless an object of the
   target's own class has been confirmed inside the room, which outranks
   the prior). Excluded rooms carry probability 0 so nothing reads them as
   unvalued; when nothing is left to value and no staircase is offered, no
   model call is made and the exploration fallback carries the search.
   `LoopSettings.min_prob` (0.05) keeps the oracle's "1 -- too small for a
   couch" out of the order.

   **The clue rule holds in both modes**: one confirmed object names the
   room (`RoomLabelSettings.min_objects = 1`; a single class is a *weak*
   label, shown to the oracle as `type=kitchen?`, two or more confident
   classes a *strong* one), the classifier is re-asked for the room in
   force or in transit **the action a new kind of object lands in it**
   (`relabel` events; a second chair is not a clue, and a set of kinds
   already judged is never bought twice), and a changed name ends the
   room's turn as `reclassified` -- neutral: not cooled, no attempt charged
   -- so that steps 2-4 run over the new fact; the sink first taken for a
   kitchen is re-valued the action the toilet shows.

   `visit="sweep"` is the former behaviour, kept as an ablation: goals are
   the frontier clusters the room label image credits to the room, ranked
   by `core/planning/exploration/frontier_ranking.py`, each route planned
   on a **copy of the map in which every other room's cells are written
   unknown** (`LoopSettings.confine_routes`; lifted while the agent's own
   cell is outside the room); the burst ends after `local_steps` actions
   (10), when nothing reachable is left, or on the clue rule, and the same
   room may be chosen straight back.
2. **Global re-classification.** At the loop point `graph.reason` re-labels
   every known room from every confirmed object observed so far. **Stairs are
   never a room**: the footprint of every seen staircase near the storey's
   level is removed from the free mask before the watershed runs
   (`ObservedSceneGraph.update(..., exclude=policy.room_exclusion(world))`),
   so no "room" is carved on a landing, labelled, valued or waited on; a
   staircase enters the search only as a STAIRS node (below).
3. **Probability per node** -- the same call, to
   `core/mapping/topology/search_node_oracle.py`. The model is shown every
   ROOM of the floor **that is still a node** (not finished, not ruled out
   by type) -- type or `unknown`, size, frontier clusters left, time
   searched and how long ago it was last inside, objects seen, whether the
   robot stands in it -- and every STAIRCASE off the floor -- up or
   down, whether the other storey was visited and what was found and
   searched there, whether the robot arrived by it -- and every **OPENING**
   of the floor (below) -- a doorway or gap to space not yet seen, the
   mapped room it opens from and the objects glimpsed through it -- plus
   one line on this storey and each other storey known. It answers, for
   each node, the probability in percent that **going there and searching
   finds the target**.
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
   by what it may lead to; an opening is valued by the room likely **behind**
   it -- a toilet glimpsed through it makes it a bathroom's door, nothing
   glimpsed makes it the unknown room this storey may still owe the target;
   a staircase is the whole set of rooms the other
   storey may hold, read from what this storey turned out to be (a kitchen
   and a living room make a ground floor, so the bedrooms are upstairs),
   discounted when that storey was searched, and not taken straight back
   when the robot has just come from it. It is told, twice, **never to
   reason about distance**: travel is the planner's. This is the one
   judgement per loop point the search cannot afford to get wrong, so the
   runtime routes it to the LLM client's reasoning model
   (`LLM_REASONING_MODEL`, default `qwen2.5:14b-instruct`, with its own
   timeout and context window; the 14B follows the rules and answers in
   ~25 s on the laptop's CPU once loaded, and is the model for the
   benchmark runs. Development runs since 2026-10-04 set it to
   `qwen2.5:3b-instruct` for turnaround -- the type prior and the finished
   ledger now take the exclusions the 3B used to get wrong out of its
   hands, so what it is left to weigh is the unknown rooms against the
   stairs) and
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
   objective must see. The expected visit is folded in as per-node service
   time (`LoopSettings.service_steps()`: half the approach bound plus the
   rotation under `scan`, `local_steps` under `sweep`); the order is
   re-solved at every loop point
   (`ObjectSearchParams.resolve_on_release`) and on semantic discovery events,
   not merely on a timer. **A floor change is never gated on room coverage**
   (unless `doorway_peek.gate_floor_departure` is turned on): the staircase
   is a node like the rooms, taken when the order puts it first; the
   existing 40-action arrival-return guard and the fallback rule still apply.
   **The way back is the one exception**: under `scan` the staircase the
   agent arrived by is withheld while this storey still has a room that is
   neither finished nor ruled out (`way_back_held` events) -- the storey is
   looked at before "not here" means anything. The 3B oracle sent the
   Ranchester agent straight back up twice with "living room found; no
   couch here", three unscanned rooms on its list each time; once every
   room is finished the way back is offered, marked `arrived_by`.
   When the order puts a staircase first the loop commits it to the
   building coordinator, which approaches the foot of the flight and
   climbs; the node's turn ends `traversed` the action the climb begins,
   and the floor's search resumes from a fresh estimate when the robot
   comes back down -- with the way back up a node like any other, marked
   `arrived_by`. The order and its head are kept on the loop (`order`,
   `order_index`, `next_room`) for the recording and the dashboard.

   **Openings are nodes too** (`methods/opening_nodes.py`,
   `LoopSettings.openings`, since 2026-10-04). The room partition is a
   watershed over *observed* free space, so a room the agent has only
   glimpsed through its door is not a room yet: its few seen cells hang off
   the hallway's region, and whatever object stood in the glimpse names the
   hallway. In the Ranchester recording of 2026-10-04 (attempt 7, action 75)
   the agent stood in a hallway facing three doors; the toilet seen through
   one of them made the whole west half of the storey a "bathroom", every
   door off the hallway was demoted with it, and nothing was left to value
   but the stairs it happened to turn towards. The decision to descend was
   right; the reasoning that produced it was not, and in a search for
   something on *this* storey it would have been fatal. So every genuine
   **exit** of the mapped floor -- a frontier that is not an object's
   shadow, not the floor under the agent's feet, at least `min_cells` (8,
   0.8 m) wide after merging within `merge_m` (1.5 m), not at the foot of a
   seen staircase -- is an **opening**: a node with a sticky building-wide id
   (`O<n>` on the HUD, `OpeningRegistry`), described to the oracle by the
   room it opens from, whether a confirmed door frame stands at it
   (`doorway` / `gap`) and the objects glimpsed through it so far
   (confirmed landmarks 0.3-3 m beyond the threshold *inside a 35-degree
   cone* around its heading; the heading is the direction from the known
   floor around the frontier cell toward the unknown -- the mean of the
   unknown cells alone pointed the stair passage back into the hallway in
   attempt 8, and a half-plane glimpse then credited a hallway bed to it,
   so the passage read as a bedroom door for sixty actions), and priced
   to RPT* at a **peek's** few actions (`service_steps`, 4; `build_instance
   (service_s=...)`), not a room's scan. Its visit is the peek
   (`entry_peek`, `peek_look` / `peek_complete` events): walk to the
   threshold (bounded by `approach_bound`: the distance's worth of forward
   steps times 1.6 plus six turns, at least `approach_steps` 20; a walk
   that runs out looks from within `merge_m` or retires the opening; a
   refused threshold is re-aimed once at a passable cell on the agent's
   side), face the unknown, and sweep `2 * look_turns + 1` headings a turn
   apart from the side nearer the agent's yaw, one in-place turn per
   action; then the opening is retired for the storey and the node's turn
   ends `exhausted`.
   What the look saw is floor in the partition now: the room behind
   separates, is named by its objects, and is excluded by type or scanned
   like any other. **A weak type label does not rule out a room with more
   than `weak_type_max_openings` (1) openings** (`weak_type_kept` events):
   the one object that named it may belong to a room behind one of them.
   Openings count as "rooms left" for the way-back hold, so the agent does
   not go back upstairs with doors on this storey unlooked-into. A new
   opening appearing while nothing is in force buys the oracle a call at
   most every `revalue_actions` (6); a new room or staircase is valued at
   once. `openings.enabled=False` is the ablation in which the exits are the
   exploration fallback's alone.

   **A confirmed landmark of the target's class is a node too**
   (`kind="landmark"`, `T<n>` on the HUD, `landmarks_enabled`). Ranchester
   attempt 9's downstairs warm-up saw the couch three times at 0.41-0.45 --
   under the takeover's 0.50, the far frames without depth -- put a `sofa`
   landmark on the map 3.8 m away, and the search then walked *west for
   forty actions* chasing frontiers while the one object it was looking for
   stood confirmed on its own map. Such a landmark -- not yet looked at
   from close on this storey, not within the takeover's rejection memory --
   is offered to RPT* at a fixed `landmark_prob` (0.85; the detector's word
   is the evidence, the oracle is not asked) at a standoff point
   `landmark_standoff_m` (1.5 m) from it on the agent's side, and visited by
   the same peek: walk there facing it, one look to each side -- a few
   close frames for the detector, and the takeover does the rest, or the
   landmark is marked inspected (`landmarks_inspected`).
5. **A\*** -- the weighted planner behind `_navigate`.
6. **Direct transit to the chosen room's vantage point** under `scan`
   (`entry_vantage`; re-aimed once at the vantage point's new position, then
   the centroid, if the mask re-segments under the route), or **to the
   closest frontier inside the chosen room** under `sweep`
   (`LoopSettings.entry_frontier`; a room with no frontier left keeps a
   centroid entry; a committed entry frontier the camera resolves en route
   is re-aimed at the room's next one, and a room with none left is
   released without the visit -- `frontier_exhausted` in transit); or **to
   the threshold of the chosen opening**, where the peek begins.
7. **Reset** -- arrival is the agent's own cell inside the room's mask
   (`arrived`), which restarts the visit counter.

**Room numbers are unique across the building** (since 2026-10-04): each
storey keeps its own registry, and a storey first entered starts numbering
after the highest pid any storey has handed out
(`RoomRegistry(first_pid=...)`, `FloorContextBank.new`), so `R0` names one
room in a recording, not one per floor.

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

1. the **highest-utility reachable EXIT anywhere on the floor** -- not confined to
   the room in force, so a room whose routes all fail is left rather than
   spun in. Two kinds of frontier are not exits and wait for rung 3
   (`exits_first`, default **true**): an **object's shadow** -- the camera's
   ray ends at the nearest object, so behind every bed and cabinet a scan
   leaves a strip of unknown whose frontier sits beside the object
   (`shadow_margin_m` 0.30 beyond its footprint radius) and is no longer than
   the object could cast (`shadow_length_factor` 3 halo radii) -- and a
   frontier of a **room the target cannot be in** (the loop's `type:`
   exclusion). The 2026-10-04 Ranchester recording spent actions 61-102
   walking from one strip between the beds to the next (ten goal changes, a
   150-degree spin, back within 6 cm of where it started) while the passage
   to the stairs stood two metres away as a frontier the whole time,
   outranked because the strips were nearer. A finished room's opening is
   NOT demoted: it is how the next room is found. A frontier inside the
   camera's **blind radius** (`near_blind_m`, 1.2 m -- a level camera 0.88 m
   up sees the floor from about there) is demoted too: it appears under the
   agent wherever it stops, and no step toward it resolves it. Within a rung
   the fallback **commits**: the goal in force is driven until it is gone or
   refused unless another goal is worth `goal_switch_gain` (2x) its current
   utility -- the greedy per-action order, re-ranked by the agent's own
   turning, once produced a 180-degree turn toward one goal and a 210-degree
   turn back (`goal_kept` / `goal_switched` / `goal_outranked` in the stats);
2. the **stairs** by the explicit fallback rule (`floor_decision.py`:
   eligible, reachable now, storey worth visiting, worth per metre of
   approach, flight and fixed change cost), when the floor is exhausted.
   The normal way to the stairs is not this rung: the loop offers every
   staircase to the oracle and to RPT* as a node beside the rooms and climbs
   when the order says so. This is for the floor where that machinery has
   nothing left -- the room LLM is away, or no node is worth anything and
   the frontier is gone -- and a climb it makes is recorded
   `rule: fallback`;
3. the **demoted frontiers** of rung 1 (`frontier_demoted`): a shadow is still
   unknown space once every exit is spent, and so is the bathroom;
4. a **retired frontier** -- a goal dropped for a transient plan failure is
   still unknown space;
5. a **relocation** to the farthest reachable known cell, for a vantage point
   the map may show a frontier from;
6. a single hold only when the agent stands off the observed passable map,
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

**Object identity is a vote, per frame, per place** (since 2026-10-04;
`core/mapping/objects/landmarks.py` with `class_votes=True`, wired in
`methods/floor_context.py` and `methods/perception_cycle.py`). Every
projected detection carries a **footprint**: a disc around its world
centroid, half the box's width at its depth
(`perception.footprint_radius_m`, clamped to 0.15-1.5 m). A detection is
folded into an existing landmark **whatever its class** when its centroid
lies within the dedupe radius (0.70 m) of the landmark's or the two footprint
discs overlap by IoU ≥ 0.15 (`landmarks.disc_iou`, the lens formula -- a bed
seen from two sides overlaps itself even with the centroids a metre apart,
two cups 30 cm apart do not), and its height is within 0.35 m of the
landmark's first measured height (a television on a cabinet is two objects
stacked in plan view, never one vote). Each landmark keeps **`votes`: the
number of frames in which it was seen as each class**, at most one vote per
frame; its `class_name` is the plurality (a tie keeps the current class),
and a change of plurality is a recorded relabel (`landmark_relabels`,
`outvoted_detections` in the perception diagnostics). A landmark is
**confirmed** -- shown to the room classifier, the node oracle and the
HUD, and allowed to start target evidence -- only when its leading class
holds `min_observations` (2) votes **and strictly more than the runner-up**:
a bed seen five times and called a sofa once stays a bed and the "sofa" never
reaches the object list, the room label or the LLM; a bed/sofa tie is not an
object the search may act on yet. Target evidence is judged on the **map's**
class, not the box's: a `sofa` box projecting onto a confirmed `bed` is the
misidentification (`contradicted_by_map` in the target-closing pre-lock
check), and the box is counted as an outvoted detection rather than a
target. This is what makes the object list handed to the LLM a
high-confidence list rather than a log of every detector flicker.

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
