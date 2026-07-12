# Object-Approach Webcam Test Rig (test the target lock from home)

Run the **whole object-approach target-lock mission on a laptop webcam** — no drone,
no depth, no localization, no Jetson/TensorRT. It drives the *exact same*
`tasks/planning/object_approach_offline` pipeline (detect → confirm → track → visual
servo → SEARCH/APPROACH/HOVER_LOCK/RECOVER) and the *exact same* HUD the drone
renders, so you can exercise every new mechanism before flying:

- **detector-only vs detector+tracker** closure (`--lock-mode`),
- the **robust Median-Flow tracker** (hold an object, then cover it — the box does
  not run off onto the background),
- the **HUD colours**: 🟢 green box (detector sees it) · 🟠 orange box (tracking
  only) · 🔴 red whole-frame border (RECOVER) · ⬜ grey whole-frame border (SEARCH),
- the **RECOVER manoeuvres**: move the object off to a side (directional chase) vs
  hide it behind something in the centre (occluder peek).

## Two pieces

1. **`webcam_frame_publisher.py`** — the *drone RGB mock*. Captures the webcam,
   centre-crops/resizes to the drone resolution (504×294), and writes JPEGs to a
   rolling `/tmp/xtend_frames/` folder — byte-for-byte the interface the drone's
   XTEND publisher produces. With `--ros` it also publishes the real
   `/xtend/rgb_frame_path` `std_msgs/String` topic (`"<path> <sec> <nsec>"`).
2. **`run_webcam_target_lock.py`** — the *mission + HUD*. Reads that folder (or the
   camera directly), runs a laptop detector, drives `TargetLockPipeline`, and shows
   the target-lock HUD window.

## Requirements

- A webcam and OpenCV (`cv2`) — already in the project venv.
- **Detector** (`--detector`):
  - `yoloworld` (**default**; `yolo` is an alias) — the project's real
    **open-vocabulary YOLO-World** (`core/mapping/detection/YoloWorldDetector`), the
    torch analog of the drone's TensorRT YOLO-World. `--target` can be **any** word,
    not a fixed class list; `--distractors` add more prompts. Needs `torch` +
    `ultralytics` (in your active venv). Default weights `yolov8s-worldv2.pt`; on the
    **first run** ultralytics auto-installs the CLIP text deps (`clip`, `ftfy`) and
    downloads the YOLO-World weights + CLIP text model (~340 MB) — it may print a
    one-time "restart/rerun" notice; just run it again. The GPU is auto-selected.
  - `color` — a zero-dependency colour-blob **mock** (no model): hold up a coloured
    object (`--color red|orange|yellow|green|blue|purple`), reported as the target.
    For when you don't want to load YOLO-World.
- `--ros` on the publisher additionally needs `rclpy` (ROS2). Not needed for the
  offline HUD flow below.

> **Which Python?** Run from the repo root with the venv that has your detector
> deps **active** (YOLO-World needs `torch`/`ultralytics`; `--detector color`
> needs neither). `run_webcam_test.sh` and `python -m …` both use the active venv;
> `sparx_agency` resolves to this checkout when the repo root is the working dir.

## Run — two-process (faithful to the drone)

```bash
cd /path/to/TheAgency        # this checkout

# terminal 1 — the drone RGB mock (writes /tmp/xtend_frames):
python -m sparx_agency.tasks.planning.object_approach_webcam.webcam_frame_publisher

# terminal 2 — the mission + HUD (YOLO-World is the default; GPU auto-selected):
python -m sparx_agency.tasks.planning.object_approach_webcam.run_webcam_target_lock \
    --target person
```

Or a single command that starts both: `./sparx_agency/tasks/planning/object_approach_webcam/run_webcam_test.sh --target "fire extinguisher"` (any open-vocab prompt).

## Run — one process (simplest)

```bash
python -m sparx_agency.tasks.planning.object_approach_webcam.run_webcam_target_lock \
    --camera 0 --target cup --detector color --color red --lock-mode detector
```

## What to do in front of the camera

| To see… | Do this |
|---|---|
| 🟢 **green** (detected) | Hold the target in view. |
| 🟠 **orange** (tracking only) | Partly cover it / tilt it so the detector drops out but it's still visible — the tracker holds the box. |
| 🔴 **red border** (RECOVER, directional) | Move it briskly **off to one side** — the drone yaws/leans that way to chase it. |
| 🔴 **red border** (RECOVER, occluder peek) | Make it vanish from the **centre** (hide it behind your other hand) — the peek manoeuvre sidesteps to look around. |
| ⬜ **grey border** (SEARCH) | Take it away and wait past `--recover-timeout-s` (default 6 s). |

Watch the side panel: mission state, the box offsets/area, and the ROLL/PITCH/YAW
gauges showing the exact `/cmd_vel` the drone *would* be commanded.

## Lock mode & the colour mock

- `--lock-mode detector_tracker` (default): detector seeds the Median-Flow tracker,
  propagated every frame. Best with **textured / real objects** (as YOLO-World sees).
- `--lock-mode detector`: the detector's box alone, no tracking. Use this with the
  **colour mock on a plain, single-colour object** — the Median-Flow tracker honestly
  refuses a *flat, textureless* blob (that refusal is the anti-"lying" feature), so a
  perfectly uniform object won't hold in the tracker path but works fine detector-only.

## Troubleshooting

- **Window shows only the first frame (video is frozen, but the window is
  responsive).** The webcam handed OpenCV one frame and stalled — usually the
  GStreamer capture backend (`cap_gstreamer.cpp` warning). The publisher now forces
  the **V4L2** backend, which streams continuously. Confirm frames are flowing:
  the publisher prints `wrote N frames (~15.0 fps)` every ~3 s, and `ls /tmp/xtend_frames`
  should keep growing. To watch the raw camera directly, run the publisher alone with
  a preview: `python -m …webcam_frame_publisher --show`. If it's still one frame,
  try another device index (`--camera 1`, `--camera 2`) or free the webcam from other apps.
- **`waiting for new frames …` in the mission log while the publisher reports
  `wrote N frames`.** This was stale frames from an earlier run: a previous session
  leaves higher-numbered `frame_*.jpg` behind, and filename ordering then keeps the
  stale ones while deleting the fresh ones. The publisher now **clears the folder on
  startup** and the reader picks the newest by **modification time**, so it self-heals
  — just re-run. (Manual: `rm -f /tmp/xtend_frames/frame_*`.)
- **First YOLO-World run pauses.** It's downloading the weights + CLIP text model
  (~340 MB) and maybe auto-installing `clip`/`ftfy`; it may print a one-time
  "rerun" notice. Run the command again and it's cached.

## Preview the real drone's flight commands (`--falcon-actuation`)

By default the HUD shows the servo's smooth analog command. Add `--falcon-actuation`
to preview what the **real drone** actually gets: the platform yaws/advances at a
*fixed speed*, ignores a lone control tick (so motion is held for a ≥2-tick minimum
burst), and *coasts* after stopping. With the flag the closing runs the same
`PulseShaper` + coarse, closeness-growing **yaw deadband** the FALCON node uses — so
the gauges show discrete pulses and fine centring is done by lateral crab, not yaw.
Same tracking/RECOVER strategy; just the FALCON actuation layer on top.

## No depth / no localization

The drone supplies depth (metric range) and pose (arrival → room scan); at home you
have neither, and the stack degrades cleanly: the servo uses the **box-area
fraction** as the proximity proxy (`--target-area-frac`, the "close enough" size for
HOVER_LOCK), and with no pose it never enters the room-scan (SCAN) state. Everything
else — identification, tracking, lock modes, HUD colours, RECOVER — is fully live.
