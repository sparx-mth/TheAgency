# Tracking

Detect-once / track-many: keep a lock on a single target between the slow detector
fires, emitting a `Track2D` every camera frame. ROS-free, GPU-free, Python-3.8-safe.
Lives under `core/mapping` (a sibling of `detection/`) because tracking is a
perception concern; the control law that consumes its tracks is in
`core/planning/visual_servo`.

## Pipeline

```
detector ──► Detection2D ──► ObjectLockTracker.on_detection()   (seed / re-seed)
                                     │
RGB ─────────────────────► ObjectLockTracker.on_frame() ──► Track2D ──► visual_servo
```

## Closure strategy — two `ObjectLockTracker`s

Pick one with `make_lock_tracker(mode)`; both emit the same `Track2D`, so the
servo / FSM / recovery / HUD stack is identical either way.

- **`"detector_tracker"` (default) → `TargetTracker`** — the detector seeds a box
  tracker that propagates the box **every frame between detections** and re-seeds
  on each fresh detection to bound drift. Robust to a slow / intermittent detector.
- **`"detector"` → `DetectionOnlyTracker`** — the detector's box **alone** drives
  closure: no optical flow, the track is just the last detection held while fresh
  (`max_det_age_s`). Use it when the detector already keeps up with the RGB stream
  (as on the target edge hardware), so tracking adds nothing but a way to drift
  onto the background.

## Box-propagation backends (`BoxTracker`)

`TargetTracker` propagates the box with an injectable `BoxTracker`:

- **`MedianFlowBoxTracker` (the robust default).** Median-Flow (Kalal et al. 2010)
  on base OpenCV + numpy. Four ideas make it fail *honestly* instead of tracking
  the background: (1) **forward-backward** consistency drops any point that does
  not round-trip; (2) **median consensus** sets the box translation/scale from the
  median displacement / pairwise-distance ratio, so a few background points are
  out-voted instead of dragging the box; (3) an **appearance template** (NCC vs the
  seed) catches slow drift off a static object onto static background; (4) **honest
  loss** — lock is declared lost (not silently kept) when the median FB error is
  high, too few points survive, or the appearance drops. This is the fix for
  "tracking lies": when the object is occluded or leaves the frame, it reports loss
  rather than a confident box on nothing.
- **`LucasKanadeBoxTracker` (leaner / faster).** Shi-Tomasi corners + pyramidal LK;
  the bounding rect of the survivors is the new box. ~2-3 ms / 640×360 on a Jetson
  Orin CPU, but it *will* latch onto background once the object's own corners are
  lost — kept as the low-cost option, not the default.

`ConstantVelocityBoxModel` (alpha-beta on the box centre) smooths the box, supplies
the image-plane velocity the re-search policy needs, and dead-reckons through a
brief dropout for `max_predict_s`; the `predicted` flag on `Track2D` marks a
dead-reckoned box.

Config: `MedianFlowConfig` / `LKBoxTrackerConfig` / `MotionModelConfig` /
`TargetTrackerConfig` / `DetectionOnlyConfig` — see the dataclasses.

## Swap seam

```python
from sparx_agency.core.mapping.tracking import (
    make_lock_tracker, default_box_tracker_registry, TargetTracker,
)

# closure strategy:
tracker = make_lock_tracker("detector_tracker")   # or "detector"

# box-propagation backend (detector_tracker only), by name or injected:
box_tracker = default_box_tracker_registry().create("median_flow")  # or "lucas_kanade"
tracker = TargetTracker(box_tracker=box_tracker)                     # inject any BoxTracker
```

Register new backends with `BoxTrackerFactory(name, create)` →
`BoxTrackerRegistry.register(...)`.

## Usage

```python
from sparx_agency.core.mapping.tracking import make_lock_tracker

tracker = make_lock_tracker()          # detector_tracker + MedianFlow (defaults)

tracker.on_detection(rgb, detection, stamp_s=t)   # seed / re-seed on a detection
track = tracker.on_frame(rgb, stamp_s=t)          # every frame; always a Track2D
if track.valid:
    feed_visual_servo(track)                      # may be track.predicted mid-dropout
```

`Track2D` is defined in `core/common/types/perception.py`.
