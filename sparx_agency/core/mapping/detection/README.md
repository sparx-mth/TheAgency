# Detection

Open-vocabulary 2D object detection: RGB frame + prompt list → `Detection2D` boxes.
ROS-free, drone-agnostic. The perception front-end of the "lock onto a named
object and approach it" capability.

## Pipeline

```
RGB ─► DetectionModel.detect(prompts) ─► [Detection2D]  (pixel bbox_xyxy)
                                              │
depth ──────────────────────────────────────►│  bbox_to_xyz_cam_from_depth()
                                              ▼
                                        3D target (cam frame)
```

Downstream, the confirmation gate + tracker consume the boxes at frame rate; see
`core/planning/visual_servo` and `core/mapping/tracking`.

## Interface

`DetectionModel` (in `core/mapping/interfaces/detection_model.py`) is a two-method
ABC — the detection analog of `DepthModel`:

- `set_prompts(prompts)` — set/replace the open-vocab class strings. Cheap,
  idempotent, no retraining. Change the target object at runtime by re-calling.
- `detect(rgb) -> [Detection2D]` — numpy `HxWx3` in, pixel-space boxes out.

No camera intrinsics: 2D→3D lifting is a separate concern (see below).

## Backends

- **`YoloWorldDetector`** ("OpenYOLO") — the default, wrapping ultralytics
  `YOLOWorld`. The torch analog of `DepthAnythingV2DepthModel`: `ultralytics`/
  `torch` import **lazily** on the first `detect`, so this module imports cleanly
  in a GPU-free, Python-3.8, ROS-free environment (unit tests, the Noetic side).
  Config is `YoloWorldConfig` (checkpoint, device, conf/IoU/imgsz/max_det).
- A `YoloTRTDetector` (TensorRT runtime, the analog of `DepthEngineTRT`) can be
  added alongside, subclassing the same ABC. **The engine-build tooling belongs
  under `tasks/`**, per the project's core-vs-tasks TRT split — never in `core`.

## Registry

Pick a backend by name without importing its heavy deps until construction:

```python
from sparx_agency.core.mapping.detection import default_detection_registry

reg = default_detection_registry()      # constructs/lists with no torch present
reg.names()                             # ['yolo_world']
detector = reg.create("yolo_world")     # imports the backend now
```

Register new backends with `DetectorFactory(name, create)` →
`DetectionRegistry.register(...)`.

## Usage

```python
from sparx_agency.core.mapping.detection import YoloWorldDetector, YoloWorldConfig

det = YoloWorldDetector(YoloWorldConfig(device="cuda:0"))
det.set_prompts(["refrigerator", "chair"])   # open-vocab, no retraining
boxes = det.detect(rgb_hwc_uint8)            # -> [Detection2D(label, score, bbox_xyxy, ...)]
```

## 2D → 3D lifting

A detection is lifted to a camera-frame point by fusing its box with the depth
map — `core/mapping/depth/depth_bbox_fusion.py::bbox_to_xyz_cam_from_depth`. The
resulting metric range feeds the visual servo's approach/terminal logic.

See the `Detection2D` / `Track2D` dataclasses in
`core/common/types/perception.py` for the exact fields.
