# Detection

Open-vocabulary 2D object detection: RGB frame + category names → `Detection2D`
boxes. ROS-free and platform-independent; depth lifting and navigation remain
separate concerns.

## Interface

`DetectionModel` lives in `core/mapping/interfaces/detection_model.py`:

- `set_prompts(prompts)` stages/replaces the category vocabulary.
- `detect(rgb)` takes an RGB uint8 H×W×3 array and returns label, score and
  original-frame pixel XYXY boxes with frame dimensions.

Both backends import their model dependencies lazily. Importing the package,
constructing a detector, and staging prompts need no Torch or simulator.

## Backends

- **`YoloWorldDetector` / `YoloWorldConfig`** — default **YOLO-World X-v2**,
  `yolov8x-worldv2.pt`, through Ultralytics. Checkpoint, device, confidence, NMS
  IoU, image size and detection cap are configurable. Explicit S/L checkpoint
  paths remain available as baselines/rollback options.
- **`LlmDetDetector` / `LlmDetConfig`** — the selectable official pretrained
  LLMDet implementation through Transformers. Uses a local safetensors snapshot,
  exact category-token mapping, token-safe caption chunking and verified
  independent decoder heads; no silent model or precision substitution.

See the [service runbook and measured evidence](../../../tasks/mapping/scene_graph/serve/README.md)
for optional dependencies, local checkpoint provisioning and calibration.
The HTTP service defaults to the same X-v2 filename and fails if its local
checkpoint is missing; `--backend llmdet` requires explicit checkpoint/config.

## Registry

`default_detection_registry().names()` lists `llmdet` and `yolo_world`.
`registry.create("yolo_world")` constructs the lazy X-v2 default;
`registry.create("llmdet")` constructs the alternate backend.
`default_detection_registry(llmdet_config=..., yolo_world_config=...)` injects
explicit configs. Register extensions using `DetectorFactory` and
`DetectionRegistry.register`. TensorRT engine-build tooling belongs under
`tasks/`, never in this core package.

## 2D → 3D lifting

Fuse boxes with registered metric depth using
`core/mapping/depth/depth_bbox_fusion.py::bbox_to_xyz_cam_from_depth`.
The metric target feeds visual servo/approach logic; ObjectNav uses its existing
depth projection and landmark/evidence safeguards unchanged. See
`core/common/types/perception.py` for the `Detection2D`/`Track2D` fields.
