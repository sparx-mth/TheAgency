# Object-Approach Offline Target Lock

Visual sanity check for the "lock onto a named object and fly to it" mission
(`sparx_agency/tasks/planning/falcon/OBJECT_APPROACH.md`) **before** connecting to
the real drone and Falcon: name a target, and watch it get detected, confirmed,
tracked, and visually servoed — with the exact `/cmd_vel`-equivalent command shown
next to every frame, as ROLL/PITCH/YAW gauges plus the raw numbers. No ROS, no
Falcon, no drone connection required.

Two ways to run it:

- **`run_folder_target_lock.py`** — batch over a *finished* folder of RGB frames
  (e.g. a `compare_folder`/`detect_folder` capture); writes annotated `.jpg`s + a log.
- **`run_live_target_lock.py`** — live, on-screen window over a *growing* folder
  (e.g. the XTEND frame publisher's live stream) — always reacts to the newest frame.

## What it does

Both drive the same core stack `object_approach_node.py` wires to ROS, purely offline:

```
RGB frame (from a folder, finished or growing)
        │
   YoloTRTDetector.detect()            (open-vocab TensorRT YOLO-World)
        │
   TargetConfirmationGate              (acquire on N consecutive hits)
        │
   TargetTracker (LK + motion model)   (detect-once / track-many)
        │
   VisualServoController               (tracked box [+ depth] -> body velocity)
        │
   VisualApproachStateMachine          (SEARCH / APPROACH / HOVER_LOCK / RECOVER)
        │
   ReSearchPolicy                      (active re-search on a lost track)
```

Every frame is rendered as the camera image (raw detections in gray, the tracked
target in green/yellow/red for locked/predicted/lost — nothing else drawn on the
image) next to a status **panel** with:

- the mission state (`SEARCH`/`APPROACH`/`HOVER_LOCK`/`RECOVER`), confirmation streak,
- the tracked box's offsets/area/range and `at_target`,
- **ROLL** (arrow, lateral `vy`), **PITCH** (arrow, forward/back `vx`), and **YAW**
  (a circle with a point marking the commanded turn rate/direction) gauges,
- the raw `vx, vy, vz, yaw_rate` and where the command came from (`servo:...`,
  `recovery:...`, or "not driving" while the mission planner is in SEARCH).

## Setup

- Same TensorRT venv + engines as `tasks/mapping/yolo_world_trt` (see that
  package's `README.md` for building the backbone/head engines).
- A folder of RGB frames (`.jpg`/`.png`/...). Optionally a folder of aligned
  per-frame depth `.npy` (HxW, meters, same basename as the image) for
  range-gated approach/terminal logic — without it the servo uses the box
  area-fraction proxy.
- For the live viewer: a display. Run on the Jetson's own screen, or `ssh -X`/VNC
  into it — `cv2.imshow` needs somewhere to put the window.

## Run — finished folder (batch)

```bash
python -m sparx_agency.tasks.planning.object_approach_offline.run_folder_target_lock \
  --backbone sparx_agency/tasks/mapping/yolo_world_trt/engines/orin_sm87/yolo_world_s.backbone.fp16.dla0.engine \
  --head     sparx_agency/tasks/mapping/yolo_world_trt/engines/orin_sm87/yolo_world_s.head.fp16.gpu.engine \
  --text-weights /home/user/Downloads/yolov8s-worldv2.pt \
  --images /home/user/Downloads/walk_into/rgb \
  --out /home/user/Downloads/walk_into/target_lock \
  --target bottle \
  --distractors "chair, table, shelf" \
  --conf 0.4
```

Frames are copied to `<out>/raw` immediately (before the TensorRT engines and CLIP
text branch load) so a still-live source folder can't rotate frames out from under
the run; pass `--no-snapshot` if `--images` is already a finished, static capture.

### Expected outcome

- `<out>/annotated/*.jpg` — every frame, image + panel side by side. Watch `state`
  go `SEARCH` -> `APPROACH` -> `HOVER_LOCK` as the target is confirmed and
  centred/closed on, and the gauges/numbers show the command that would be
  published to `<drone_ns>/cmd_vel`.
- `<out>/target_lock.jsonl` — the same numbers per frame, for scripted checks.
- `<out>/summary.json` — how many frames landed in each mission state.

## Run — live stream

```bash
python -m sparx_agency.tasks.planning.object_approach_offline.run_live_target_lock \
  --backbone sparx_agency/tasks/mapping/yolo_world_trt/engines/orin_sm87/yolo_world_s.backbone.fp16.dla0.engine \
  --head     sparx_agency/tasks/mapping/yolo_world_trt/engines/orin_sm87/yolo_world_s.head.fp16.gpu.engine \
  --text-weights /home/user/Downloads/yolov8s-worldv2.pt \
  --images /tmp/xtend_frames \
  --target bottle \
  --distractors "chair, table, shelf" \
  --conf 0.4
```

Opens a window and keeps polling `--images` for its newest frame — if detection is
slower than the incoming frame rate it simply jumps to the latest one rather than
queuing a backlog, the same way the live ROS node always acts on the freshest
frame. Press `q` in the window (or Ctrl+C) to stop.

Camera intrinsics (`--fx --fy --cx --cy`) default to the values
`object_approach_node.py` uses for the live XTEND 504×294 stream — pass
`--img-width`/`--img-height` (and matching intrinsics) if your frames are a
different resolution, or the image-plane offsets/range will be wrong.

Once this looks right, the same core objects (`TargetTracker`,
`VisualServoController`, `VisualApproachStateMachine`, ...) are what
`object_approach_node.py` runs live against the real RGB/depth stream and
`<drone_ns>/cmd_vel` — see `OBJECT_APPROACH.md` for wiring that up against
Falcon and the drone.
