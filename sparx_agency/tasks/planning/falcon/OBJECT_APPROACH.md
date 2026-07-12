# Object approach — lock onto a named object and fly to it

Add-on mission for the FALCON stack: name an object (open-vocabulary), and the
drone flies its normal A*/NavDP route while scanning for it; once it is confirmed,
the drone **abandons the planner and visually servos up to the object**, holding
centred and very close — directly **in front of** it (a stable hover-lock). It
keeps tracking, so a moving object is followed; a lost track triggers an active
re-search; arriving at the goal without ever seeing the object starts a room sweep.
**No landing** — "close and in front, filling the view" *is* success.

Rebuilt cleanly from the reference `room_search_orchestrator` (a single 900-line
ROS2 node): the algorithm is ROS-free in `core`, the ROS glue is three small nodes.

## Pieces

Pure, ROS-free, unit-tested core (222 tests):

| Concern | Where | What |
|---|---|---|
| Detection | `core/mapping/detection/` | `DetectionModel` ABC + open-vocab YOLO-World backends; swap via the registry |
| Tracking | `core/mapping/tracking/` | `ObjectLockTracker` with two closure strategies (`~lock_mode`): `TargetTracker` (detector + box tracker, default) or `DetectionOnlyTracker` (detector box only). Box backend defaults to the robust `MedianFlowBoxTracker` (forward-backward + median consensus + appearance validation — fails honestly instead of tracking the background); `ConstantVelocityBoxModel` predicts through dropouts + gives the re-search velocity → `Track2D` |
| Control | `core/planning/visual_servo/` | `TargetConfirmationGate` (N-consecutive-frame acquisition, pose-free), `VisualServoController` (bbox[+depth] → body velocity), `ReSearchPolicy` (where to look when lost), `ScanSearchPolicy` (rotate-with-stops room sweep), `CommandForceShaper` (per-axis min/max force), `VisualApproachStateMachine` (SEARCH/SCAN/APPROACH/HOVER_LOCK/RECOVER) |
| 2D→range | `core/mapping/depth/depth_bbox_fusion.py` | robust metric range to the box from depth |

Wire format: `core/common/detection_message.py` — the single definition of the
detections JSON, shared by every producer and consumer.

ROS glue. The detector is a **ROS2 sidecar on the host**; the rest is ROS1 in the
FALCON container. They meet at two `std_msgs/String` topics on the bridge.

- **`tasks/mapping/ros2/yolo_detector_ros2_node.py`** (ROS2, host) — runs the
  **TensorRT** YOLO-World split (backbone on DLA + head on GPU, from
  `tasks/mapping/yolo_world_trt`) and publishes detections JSON; re-prompts on the
  `goal` topic. Inference is torch-free; only embedding a new prompt touches torch.
  It lives on the host because the FALCON image is `FROM ros:noetic-perception` and
  has no CUDA/TensorRT/pycuda, while the host already has the env that built the
  engines. It reads `/xtend/rgb_frame_path` natively, *upstream of the bridge*, so no
  image is ever bridged.
- **`adapter/scripts/yolo_detector_node.py`** (ROS1, container) — the same detector as
  an in-container node, started only with `detector:=internal`. Needs tensorrt +
  pycuda in the image; kept for a future JetPack-based build.
- **`adapter/scripts/object_approach_node.py`** (ROS1) — torch-free (Python-3.8-safe):
  tracks + servos + runs the state machine; force-shapes every command; takes over
  `/cmd_vel` via the `visual_servoing` demo-mode hand-off (so the follower goes
  passive), releases it in SEARCH; re-injects the goal on a give-up; renders the HUD.
- **`adapter/scripts/target_lock_viewer_node.py`** (ROS1) — displays the HUD Image.
  Separate so the GUI dependency and display loop stay off the control node
  (`viewer:=false` for headless).

That the detector was always decoupled by a JSON topic is what makes this split free:
moving it across a host boundary changed no algorithm and no other node.

## Data flow

```
        ROS2 (host, GPU)          ║ bridge ║        ROS1 (falcon container)
                                  ║        ║
RGB(frame_path) ─► yolo_detector ─╫─/object_approach/detections─► object_approach_node
   (read off disk)  (TensorRT)    ║        ║       │  confirmation gate (N frames)
                                  ║        ║       │  TargetTracker (LK + motion)
RGB  ─────────────────────────────╫────────╫────►  │
depth ────────────────────────────╫────────╫────►  │  → range via depth_bbox_fusion
pose  ────────────────────────────╫────────╫────►  │  → arrived_at_goal? (scan trigger)
                                  ║        ║       │  VisualServoController + FSM
                                  ║        ║       │  CommandForceShaper (min/max force)
      re-prompt ◄─/object_approach/goal◄───╫───────┤
                                  ║        ║       ├─► demo_mode_request=visual_servoing
                                  ║        ║       ├─► /cmd_vel (vx, vy, wz)
                                  ║        ║       ├─► /waypoint_nav/goal (re-inject on give-up)
                                  ║        ║       └─► /object_approach/overlay (HUD Image)
                                  ║        ║               └─► target_lock_viewer_node
```

Only the two String topics cross the bridge. The detector reads the frame-path topic
natively on the ROS2 side and loads the JPEG off disk, so no image is serialized.

## Run

See the **Object approach** section of [`README.md`](README.md) for the full recipe
(engine build, knobs, HUD, failure behaviour). The short version — three processes:

```bash
# 1. host: the TensorRT detector, in the env that has tensorrt + pycuda
PYTHONPATH=$PWD python3 sparx_agency/tasks/mapping/ros2/yolo_detector_ros2_node.py \
    --ros-args -p target_object:=monitor \
      -p backbone_engine:=/path/backbone.engine -p head_engine:=/path/head.engine \
      -p text_weights:=/path/yolov8s-worldv2.pt

# 2. host: the bridge (carries /object_approach/detections + /object_approach/goal)
cd bridge && ./run_bridge.sh

# 3. container: nav stack + servo + HUD + BEV window
roslaunch falcon_adapter real_drone_object_approach.launch \
    map_name:=office target_object:=monitor goal_x:=0.0 goal_y:=-3.0
# or add the mission to a nav stack that is already up:
roslaunch falcon_adapter object_approach.launch target_object:=monitor

rostopic pub -1 /object_approach/goal   std_msgs/String "data: 'hat'"   # retarget live
rostopic pub -1 /object_approach/enable std_msgs/Bool   "data: false"   # arm/disarm
```

Both launches default to `detector:=external` (no detector started in the container).

Intrinsics **must** match the live stream (raw K, not P). The TRT engines are not
portable — build them on the target. Tuning is all rosparams: see the footers of the
node files and the launch args.

**Two thresholds in series, both must be cleared to acquire.** `conf_thresh` is the
detector's floor: a weaker box is never emitted. `min_score` is the confirmation
gate's floor: an emitted box below it is drawn on the HUD but never counted toward
`n_confirm`. Lowering `conf_thresh` alone cannot make the mission lock on — the weak
boxes appear on screen and the gate silently drops them. Keep `min_score <= conf_thresh`.

## State machine

```
                    ┌──── arrived at goal, still unconfirmed ────► SCAN
                    │     (node sweeps: pause → rotate → pause)    │
                    │                                              │
SEARCH ─────────────┴─ confirmed(N) + lock ──► APPROACH ──centred&close──► HOVER_LOCK
  ▲  (planner flies; node passive)   ▲            │  (node drives /cmd_vel)   │ (hold + track)
  │                                  │            ▼ track lost                ▼ target moved away
  │                                  └── re-detect ──┐                        │
  └──────── recover timeout ──────────────────── RECOVER ◄─────────────────────┘
             (+ re-inject last goal)      (chase where it left / peek round occluder)
```

`drive_cmd_vel` is true in every state **except** SEARCH — SCAN owns `/cmd_vel` too,
because the sweep is the node's own motion. HOVER_LOCK is **not terminal**: a moving
object drops it back to APPROACH, so there is no stopping condition, only a condition.

**A brief loss stays in APPROACH, not RECOVER.** The tracker coasts (dead-reckons on
the last velocity) for `max_predict_s` through a few-frame dropout — blur, a thin
occluder, a fast target — so a 2-frame blip keeps servoing on the coasted box and
never leaves APPROACH. Only a loss that outlasts the coast enters RECOVER; only a
RECOVER that outlasts `recover_timeout_s` falls back to SEARCH/SCAN.

## Design notes

- **Closure strategy (`~lock_mode`).** `detector_tracker` (default) seeds the
  Median-Flow tracker from the detector and propagates the box every frame between
  detections. `detector` skips the tracker and closes on the detector's box alone
  (held for `~max_det_age_s`) — for when the detector already keeps up with the RGB
  stream, so tracking only adds a way to drift onto the background. Same servo/FSM.
- **Honest tracking.** The default box tracker is Median-Flow: forward-backward
  consistency + median-consensus box update + an appearance template, so an
  occluded / left-frame target is reported *lost* (→ RECOVER) instead of a
  confident box on the background — the failure the old plain-LK tracker had.
- **HUD colours = lock confidence.** `object_approach/overlay` colours the target:
  **green** box when the detector sees it, **orange** box when tracking only, a
  **red** whole-frame border while re-searching (RECOVER, box unknown), and a
  **grey** whole-frame border while searching from scratch (SEARCH/SCAN).
- **RECOVER manoeuvre reads where it went.** From the last valid track's position +
  image-plane velocity, RECOVER either **chases** a target that clearly left a side
  (yaw + gentle crab toward it) or, when it vanished from the frame *centre* (most
  likely occluded straight ahead), **peeks** around the occluder — a small forward
  nudge plus an oscillating sidestep+yaw that looks past both edges. Everything is
  small, the peek oscillates so the drone stays near where it lost the target (wall
  safety), and the whole episode is bounded by `recover_timeout_s`. Tune via the
  `~recover_*` params (footer of `object_approach_node.py`).
- **Minimum force.** The servo emits an analog velocity capped only at the top, but
  the platform will not move below a per-axis force floor. `CommandForceShaper` is the
  last stage before the wire, so servo / re-search / scan / brake commands all respect
  the same discipline. Default `fixed` (bang-bang: `0` or exactly `±level`); `snap`
  gives a proportional band above the floor; `none` disables it. Shaping `0 → 0`, so a
  stop stays a stop.
- **Exactly one publisher.** The node never publishes `/cmd_vel` until the platform
  echoes `demo_mode == visual_servoing`, and it releases (`fly_straight`) on every
  disabled/SEARCH tick. The follower goes fully passive while that mode is held.
- **Arrival is inferred.** There is no "route done" topic, so arrival is
  `‖pose − goal‖ ≤ arrive_radius_m` against the goal remembered from `goal_x/goal_y`
  and any live `/waypoint_nav/goal` click. No known goal or no pose → never scans.
- **The sweep stops to look.** The detector runs at a few Hz; rotating continuously
  would smear every frame, so the sweep alternates rotate bursts with stops and begins
  with a pause (a clean look straight ahead the instant the drone arrives).

## Limits / next

- No landing by design. Vertical centring is available (`use_vertical:=true`) but off
  by default (the platform holds altitude).
- Re-prompting the detector at runtime needs torch/ultralytics via `~text_weights`;
  precompute the text embeddings (`runtime.set_text_features`) for a fully torch-free
  deployment with a fixed target.
- The scan sweeps in place by default. Set `scan_forward_speed` / `scan_forward_s` to
  relocate a short step between bursts and search from a fresh vantage.
