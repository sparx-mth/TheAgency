# Object approach — lock onto a named object and fly to it

Add-on mission for the FALCON stack: name an object (open-vocabulary), and the
drone flies its normal A*/NavDP route while scanning for it; once it is confirmed,
the drone **abandons the planner and visually servos up to the object**, stopping
centred and very close — directly **in front of** it (a stable hover-lock). It
keeps tracking, so a moving object is followed; a lost track triggers an active
re-search. **No landing** — "close and in front, filling the view" *is* success.

Rebuilt cleanly from the reference `room_search_orchestrator` (a single 900-line
ROS2 node): the algorithm is ROS-free in `core`, the ROS glue is two small nodes.

## Pieces

Pure, ROS-free, unit-tested core (228 tests):

| Concern | Where | What |
|---|---|---|
| Detection | `core/mapping/detection/` | `DetectionModel` ABC + `YoloWorldDetector` (open-vocab YOLO-World, "OpenYOLO"); swap backends via the registry |
| Tracking | `core/planning/visual_tracking/` | `LucasKanadeBoxTracker` (detect-once/track-many, fast, GPU-free) + `ConstantVelocityBoxModel` (predict through dropouts + re-search velocity) + `TargetTracker` (composition + detector re-seed) → `Track2D` |
| Control | `core/planning/visual_servo/` | `TargetConfirmationGate` (N-consecutive-frame acquisition, pose-free), `VisualServoController` (bbox[+depth] → body velocity), `ReSearchPolicy` (where to look when lost), `VisualApproachStateMachine` (SEARCH/APPROACH/HOVER_LOCK/RECOVER) |
| 2D→range | `core/mapping/depth/depth_bbox_fusion.py` | robust metric range to the box from depth |

ROS glue (`adapter/scripts/`):

- **`yolo_detector_node.py`** — runs YOLO-World on the RGB stream, publishes
  detections JSON; re-prompts on the `goal` topic. *Separate* node so the heavy
  torch dep stays off the servo node. (Needs `ultralytics` in the runtime; can run
  as a sidecar publishing the same topic if Noetic lacks it.)
- **`object_approach_node.py`** — torch-free (Python-3.8-safe): tracks + servos +
  runs the state machine; takes over `/cmd_vel` via the `visual_servoing`
  demo-mode hand-off (so the follower goes passive), releases it in SEARCH.

## Data flow

```
RGB ─┬─► yolo_detector_node ──/detections(JSON)──► object_approach_node
     │                                               │  confirmation gate (N frames)
     └───────────────────────────────────────────►  │  TargetTracker (LK + motion)
depth ─────────────────────────────────────────►    │  → range via depth_bbox_fusion
                                                     │  VisualServoController + FSM
                                                     └─► demo_mode_request=visual_servoing
                                                         + /cmd_vel (holonomic vx,vy,wz)
```

## Run

```bash
# with a nav stack already up (nav_stack.launch / real_drone.launch):
roslaunch falcon_adapter object_approach.launch target_object:=refrigerator
rostopic pub -1 /object_approach/goal   std_msgs/String "data: 'hat'"     # retarget live
rostopic pub -1 /object_approach/enable std_msgs/Bool   "data: true"      # arm/disarm
```

Intrinsics **must** match the live stream (raw K, not P). Tuning is all rosparams —
see the footers of the two node files and the launch args.

## State machine

```
SEARCH ──confirmed(N)+lock──► APPROACH ──centred&close──► HOVER_LOCK
  ▲  (planner flies; node passive)   │  (node drives /cmd_vel)   │ (success; hold+track)
  │                                  ▼ track lost                ▼ track lost / moved away
  └──────── recover timeout ──── RECOVER ◄─────────────────────────┘
                               (yaw toward where it left; re-detect → APPROACH)
```


## Limits / next

- Detector runs a stock YOLO-World checkpoint (accuracy/speed unoptimised) — a
  TensorRT backend slots behind the same `DetectionModel` ABC (builder under
  `tasks/`, per the project's core-vs-TRT convention) with no servo/tracker change.
- No landing by design. Vertical centring is available (`use_vertical:=true`) but
  off by default (platform holds altitude).
