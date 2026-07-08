# Visual Tracking

Detect-once / track-many: keep a lock on a single target between the slow detector
fires, emitting a `Track2D` every camera frame. ROS-free, GPU-free, Python-3.8-safe.

## Pipeline

```
detector (slow) ──► Detection2D ──► TargetTracker.on_detection()   (re-seed)
                                          │
RGB (fast) ──────────────────────► TargetTracker.on_frame() ──► Track2D ──► visual_servo
```

The detector re-fires occasionally; the tracker propagates the box every frame in
between and re-seeds on each fresh detection to bound drift.

## Layers

- **`LucasKanadeBoxTracker`** (`BoxTracker` impl) — the classic, fast, GPU-free
  core. Seed Shi-Tomasi corners in a detection box, propagate them frame-to-frame
  with pyramidal Lucas-Kanade; the bounding rect of the survivors *is* the new box,
  so scale is implicit (the box grows as the drone closes in — a free proximity
  signal). Percentile-trimmed rect (`outlier_k_mad`) so one jumped corner can't
  balloon the box; lock is lost below `min_matches` survivors. **Grayscale in,
  `BoxObservation` out** — raw box + surviving-feature count, nothing else.
  ~2-3 ms / 640×360 frame on a Jetson AGX Orin CPU.
- **`ConstantVelocityBoxModel`** — alpha-beta filter on the box centre (+ EMA
  size). Two jobs: (1) `predict()` dead-reckons the box through brief LK dropouts
  so the servo doesn't stall on one bad frame; (2) supplies the image-plane
  velocity the re-search policy needs to know *which way the target went*.
- **`TargetTracker`** — composes the above into the loop a node drives. Owns the
  RGB→gray conversion (once per frame), the detector re-seeds, and the short
  `max_predict_s` coast window. Emits `Track2D` carrying `valid`, `velocity_px`,
  and a `predicted` flag (measured vs dead-reckoned box), plus a last-known box on
  loss so recovery can read a re-search direction.

Config lives in `LKBoxTrackerConfig` / `MotionModelConfig` / `TargetTrackerConfig`
— see the dataclasses, not restated here.

## Swap seam

The propagation backend is injectable — swap the classic LK tracker for a
correlation/DNN tracker without touching the servo or FSM above:

```python
from sparx_agency.core.planning.visual_tracking import default_box_tracker_registry, TargetTracker

box_tracker = default_box_tracker_registry().create("lucas_kanade")
tracker = TargetTracker(box_tracker=box_tracker)   # inject any BoxTracker
```

Register new backends with `BoxTrackerFactory(name, create)` →
`BoxTrackerRegistry.register(...)`.

## Usage

```python
from sparx_agency.core.planning.visual_tracking import TargetTracker

tracker = TargetTracker()

# On a fresh detection (seeds / re-seeds the box):
tracker.on_detection(rgb, detection, stamp_s=t)

# Every camera frame thereafter:
track = tracker.on_frame(rgb, stamp_s=t)   # always returns a Track2D
if track.valid:
    feed_visual_servo(track)               # may be track.predicted during a dropout
```

The bbox→control law that consumes these tracks lives in
`core/planning/visual_servo`; `Track2D` is defined in
`core/common/types/perception.py`.
