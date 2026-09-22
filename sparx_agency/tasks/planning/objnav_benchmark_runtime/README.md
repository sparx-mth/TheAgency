# ObjectNav runtime and Gibson adapter

Observed RGB-D mapping → semantic room graphs → room LLM → RPT* room order →
local exploration and A*/WA* paths → existing discrete actions. Both explorers
retain persistent floor contexts and observed stair connections. This is an
experimental multi-story algorithm, **not a claim of solved navigation**.

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
| `methods/frontier_sweep.py` | Room-by-room frontier sweep: ranked goals, one bounded look-around per swept room, release-and-reselect on the same action |
| `methods/camera_control.py` | Sole pitch owner; bounded inspection, long unprompted cadence and safe restoration |
| `methods/perception.py`, `perception_cycle.py` | Fresh raw predictions, coherent pixel projection and floor-qualified fusion |
| `methods/observed_map.py`, `floor_context.py` | Independent occupancy, rooms, objects, association anchors and paused floor clocks |
| `methods/multifloor_policy.py`, `stair_traversal.py` | RPT* portal scheduling and committed transition lifecycle |
| `methods/stair_terrain.py` | Measured support, step-connected paths and native-heading safety checks |
| `methods/falcon_policy.py`, `falcon_regions.py` | Bounded hierarchy and persistent local region history |
| `methods/falcon_motion.py`, `falcon_routes.py`, `route_memory.py` | Safe motion and interruption-aware committed routes |
| `methods/object_evidence.py` | Alias/frame deduplication, separated-view target evidence and bounded rejection |
| `methods/scene_graph.py`, `room_labels.py`, `doors.py` | Observed rooms/doors and revisable accumulated room reasoning |
| `methods/exploration_metrics.py` | Observed-area proxy, actions, revisits, stagnation and latency |
| `habitat/simulator.py` | Synchronous registered RGB-D, native extrinsic checks and ENU pose |
| `recording.py`, `floor_panels.py`, `visualization.py` | Exact commands/poses, persistent UNKNOWN panels, grids and video |
| `gibson/run.py`, `frozen_eval.py`, `run_development.py` | Protocol execution and frozen source/model/data checks |

The floor tracker remains ROS-free in `core/planning/exploration/floor_atlas.py`.
Host-side numpy/scipy terrain processing does not enter the Noetic exploration
facade. No robot's flight controller is changed.

## Frontier sweep and the action economy

Every action counts against the public 500-action cap, so the frontier explorer
is built to spend them moving toward unmapped space (the Ranchester recording
`e7c4f2ad5402` that motivated this spent 48 % of its actions on idle turns,
stair inspections and heading recovery):

- In-room and floor-wide goals come from
  `core/planning/exploration/frontier_ranking.py`: gain over *geodesic* cost on
  the planner's own passable graph, facing as a discount. A boundary with no
  known-free path is never proposed. Up to `SweepSettings.plan_attempts` goals
  are tried per action, so a refused A* costs no idle action.
- A committed frontier goal is kept while it is passable, still has unknown
  space near it and still lies near the room being swept. A re-segmented room
  mask alone does not drop it; a boundary the camera has resolved does.
- A room with nothing reachable left gets **one** look-around
  (`SweepSettings.room_scan_turns`, a full rotation by default, once per visit)
  and is then released through the supervisor's `frontier_exhausted` exit on the
  same action; a transit goal the planner refuses ends the room through
  `route_failed`. Neither waits on the 30 s stall or 90 s budget clocks, which
  remain as backstops. A released room is replaced by the next transit on the
  same action -- no throwaway floor-wide route in between.
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
