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
| Tracking | `core/planning/visual_tracking/` | `LucasKanadeBoxTracker` (detect-once/track-many, fast, GPU-free) + `ConstantVelocityBoxModel` (predict through dropouts + re-search velocity) + `TargetTracker` (composition + detector re-seed) → `Track2D` |
| Control | `core/planning/visual_servo/` | `TargetConfirmationGate` (N-consecutive-frame acquisition, pose-free), `VisualServoController` (bbox[+depth] → body velocity), `ReSearchPolicy` (where to look when lost), `ScanSearchPolicy` (rotate-with-stops room sweep), `CommandForceShaper` (per-axis min/max force), `VisualApproachStateMachine` (SEARCH/SCAN/APPROACH/HOVER_LOCK/RECOVER) |
| 2D→range | `core/mapping/depth/depth_bbox_fusion.py` | robust metric range to the box from depth |

ROS glue (`adapter/scripts/`):

- **`yolo_detector_node.py`** — runs the **TensorRT** YOLO-World split (backbone on
  DLA + head on GPU, from `tasks/mapping/yolo_world_trt`) on the RGB stream and
  publishes detections JSON; re-prompts on the `goal` topic. Inference is torch-free;
  only embedding a new prompt touches torch. *Separate* node so the heavy TRT/torch
  deps stay off the servo node.
- **`object_approach_node.py`** — torch-free (Python-3.8-safe): tracks + servos +
  runs the state machine; force-shapes every command; takes over `/cmd_vel` via the
  `visual_servoing` demo-mode hand-off (so the follower goes passive), releases it in
  SEARCH; re-injects the goal on a give-up; renders the live HUD.
- **`target_lock_viewer_node.py`** — displays the HUD Image. Separate so the GUI
  dependency and display loop stay off the control node (`viewer:=false` for headless).

## Data flow

```
RGB ─┬─► yolo_detector_node ──/detections(JSON)──► object_approach_node
     │      (TensorRT YOLO-World)                    │  confirmation gate (N frames)
     └────────────────────────────────────────────►  │  TargetTracker (LK + motion)
depth ──────────────────────────────────────────►    │  → range via depth_bbox_fusion
pose  ──────────────────────────────────────────►    │  → arrived_at_goal? (scan trigger)
                                                     │  VisualServoController + FSM
                                                     │  CommandForceShaper (min/max force)
                                                     ├─► demo_mode_request=visual_servoing
                                                     ├─► /cmd_vel (vx, vy, wz)
                                                     ├─► /waypoint_nav/goal (re-inject on give-up)
                                                     └─► /object_approach/overlay (HUD Image)
                                                             └─► target_lock_viewer_node
```

## Run

See the **Object approach** section of [`README.md`](README.md) for the full recipe
(engine build, knobs, HUD, failure behaviour). The short version:

```bash
# everything at once — nav stack + detector + servo + HUD + BEV window:
roslaunch falcon_adapter real_drone_object_approach.launch \
    map_name:=office target_object:=refrigerator goal_x:=0.0 goal_y:=-3.0

# or add the mission to a nav stack that is already up:
roslaunch falcon_adapter object_approach.launch target_object:=refrigerator

rostopic pub -1 /object_approach/goal   std_msgs/String "data: 'hat'"   # retarget live
rostopic pub -1 /object_approach/enable std_msgs/Bool   "data: false"   # arm/disarm
```

Intrinsics **must** match the live stream (raw K, not P). The TRT engines are not
portable — build them on the target. Tuning is all rosparams: see the footers of the
node files and the launch args.

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
             (+ re-inject last goal)      (yaw toward where it left)
```

`drive_cmd_vel` is true in every state **except** SEARCH — SCAN owns `/cmd_vel` too,
because the sweep is the node's own motion. HOVER_LOCK is **not terminal**: a moving
object drops it back to APPROACH, so there is no stopping condition, only a condition.

## Design notes

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
