# ObjectNav runtime: shared navigation core and dataset tasks
Observed RGB-D mapping → semantic room graphs → room LLM → RPT* room order →
local exploration and A*/WA* paths → existing discrete actions. Both explorers
retain persistent floor contexts; stair connectors come from the simulator's
navmesh by default (`RPTSettings.multifloor.stair_source`, see
[MULTISTORY.md](gibson/MULTISTORY.md)), the one declared ground-truth input.
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
| `habitat/stair_connectors.py` | Evaluator side: storeys and stair connectors read once per scene from the navmesh in force, attached to the episode metadata by both tasks (`gibson/stair_connectors.py` is a re-export shim) |
| `methods/falcon_policy.py`, `falcon_regions.py` | Bounded hierarchy and persistent local region history |
| `methods/falcon_motion.py`, `falcon_routes.py`, `route_memory.py` | Safe motion and interruption-aware committed routes |
| `methods/object_evidence.py` | Alias/frame deduplication, separated-view target evidence and bounded rejection |
| `methods/scene_graph.py`, `room_labels.py`, `doors.py` | Observed rooms/doors, the room label image, and revisable accumulated room reasoning |
| `methods/exploration_metrics.py` | Observed-area proxy, actions, revisits, stagnation and latency |
| `habitat/simulator.py` | Synchronous registered RGB-D, native extrinsic checks and ENU pose |
| `recording.py`, `floor_panels.py`, `visualization.py` | Exact commands/poses, persistent UNKNOWN panels, grids and video |
| `gibson/run.py`, `hm3d/run.py`, their `frozen_eval.py` | Per-task protocol execution and frozen source/model/data checks |
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
(or `hm3d.run --print-vocabulary`) output. A former 24-prompt service is intentionally rejected. HTTP checks pin
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
