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
| Control | `core/planning/visual_servo/` | `TargetConfirmationGate` (N-consecutive-frame acquisition, pose-free), `VisualServoController` (bbox[+depth] → body velocity), `ReSearchPolicy` (where to look when lost), `ScanSearchPolicy` (rotate-with-stops room sweep), `PulseShaper` (min-burst + coast/brake flight-command shaping), `VisualApproachStateMachine` (SEARCH/SCAN/APPROACH/HOVER_LOCK/RECOVER) |
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
                                  ║        ║       │  PulseShaper (min-burst + coast)
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

No detector runs in the container; both launches consume the host sidecar's detections.

Intrinsics **must** match the live stream (raw K, not P). The TRT engines are not
portable — build them on the target. Tuning is all rosparams: see the footers of the
node files and the launch args.

**Two floors that form a soft-confirm band.** `conf_thresh` is the detector's floor
(a weaker box is never emitted); keep it **low** (0.05) so weak boxes exist. `min_score`
is the confirmation gate's floor for HARD acquisition / the green HUD box (0.15,
**above** `conf_thresh`). A detection in `[conf_thresh, min_score)` is too weak to
acquire the target from scratch, but while tracking, one sitting *on* the tracked box
(IoU ≥ `confirm_iou`, score ≥ `soft_confirm_min_score`) **re-confirms** the lock and
resets the unconfirmed timer. So keep `conf_thresh ≤ soft_confirm_min_score < min_score`.

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
- **Don't track unconfirmed for long.** Even a good tracker can drift onto the
  background over seconds, so the tracker drops the lock after `~max_unconfirmed_s`
  (2 s) with no detection re-confirming the target → RECOVER. A *weak* detection
  sitting **on** the tracked box (score ≥ `~soft_confirm_min_score`, IoU ≥
  `~confirm_iou`) counts as a re-confirmation and resets that timer — so a genuinely
  tracked object (the detector just dipped below its acquire threshold) keeps its
  lock, while background drift (no overlapping detection) times out. The detector's
  `conf_thresh` must reach ≤ `~soft_confirm_min_score` for weak boxes to exist.
- **HUD colours = lock confidence.** `object_approach/overlay` colours the target:
  **green** box when the detector sees it, **orange** box when tracking only, a
  **red** whole-frame border while re-searching (RECOVER, box unknown), and a
  **grey** whole-frame border while searching from scratch (SEARCH/SCAN).
- **RECOVER manoeuvre reads where it went.** From the last valid track's position +
  image-plane velocity, RECOVER either **chases** a target that clearly left a side
  (yaw + gentle crab toward it) or, when it vanished from the frame *centre* (most
  likely occluded straight ahead), **peeks** around the occluder — a small forward
  nudge plus a sidestep+yaw **held to one side** (the last-seen bearing) for the
  whole episode, so the motion is a steady lean-and-look, not a jarring left↔right
  reversal. Everything is small and the held direction keeps the drone near where it
  lost the target (wall safety); the whole episode is bounded by `recover_timeout_s`.
  Tune via the
  `~recover_*` params (footer of `object_approach_node.py`).
- **Discrete + inertial flight commands (real drone).** The platform yaws/advances at
  a *fixed speed*, a lone control tick can't overcome its deadband (so a small
  correction needs ≥2 consecutive commands), and it *coasts* after a command stops —
  the same reality the A*/NavDP `waypoint_follower` handles. Closing runs at **10 Hz**
  (the follower's tick calibration) and every command (servo / re-search / scan /
  brake) passes through a stateful **`PulseShaper`** that: fixed-speed-quantises each
  axis (`force_mode` `fixed`/`snap`/`none`), **latches** any motion for ≥
  `min_burst_ticks` (2) so it actually registers, and can emit a brief opposite
  **brake** pulse (`brake_ticks`) to bleed off the coast. The servo also uses a
  **coarse yaw deadband** (`yaw_deadband` ≈0.35) that **grows as you close**
  (`yaw_close_deadband`), so a yaw doesn't sweep the (now large) target out of frame —
  fine centring is done by lateral crab, not yaw. Tune per airframe; a closed-loop
  inertial sim (`core/.../visual_servo/tests/test_closure_inertial.py`) shows this
  glides to centre with no overshoot where the analog servo oscillates. Preview it at
  home with the webcam rig's `--falcon-actuation`.
- **Standoff + acquisition angle.** The mission holds a **stop distance**
  `target_range_m` (**0.5 m** by default): the forward ramp reaches zero there and
  declares success (it never flies closer). Centring uses an **acquisition angle**,
  the angular analogue of a waypoint's acquisition *radius*: `center_tol` is the
  allowed centring deviation for hover-lock (a pulsed platform can't centre to a
  degree, so "centred" is a small band, not exact zero), `yaw_deadband` is the yaw
  acquisition angle (no yaw within it, so a burst can't overshoot 89°→97°), and
  `lateral_deadband` is the coast-aware crab band that does the fine centring.
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
