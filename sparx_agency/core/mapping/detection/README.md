# Detection

Open-vocabulary 2D object detection: RGB frame + category names → `Detection2D`
boxes. ROS-free and platform-independent; depth lifting and navigation remain
separate concerns.

## Interface

`DetectionModel` lives in `core/mapping/interfaces/detection_model.py`:

- `set_prompts(prompts)` stages/replaces the category vocabulary.
- `detect(rgb)` takes an RGB uint8 H×W×3 array and returns label, score and
  original-frame pixel XYXY boxes with frame dimensions.

All backends import their model dependencies lazily. Importing the package,
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
- **`GroundingDinoDetector` / `GroundingDinoConfig`** — official Grounding DINO
  Base, not DINOv2. Reuses token-safe category decoding, with a separate verified
  shared-decoder loader. Local safetensors only.
- **`GroundedVlmDetector` / `GroundedVlmConfig`** — `grounded_vlm` uses DINO
  proposals and BLIP-2 FLAN-T5 verification; `hybrid` additionally uses YOLO.
  Each selected detector sees every frame. Duplicates keep the first source's
  original score; no cross-model score averaging or artificial confidence boost.
  BLIP-2 checks a marked full frame and box crop, with both required to answer
  yes. Unknown answers, negative answers, budget overflow and conflicting
  overlapping stair/furniture labels are withheld. All labels are verified by
  default; selective verification is explicit. No VLM-generated boxes or floor
  connections are invented. Raw per-frame evidence remains available separately.

See the [service runbook and measured evidence](../../../tasks/mapping/scene_graph/serve/README.md)
for optional dependencies, local checkpoint provisioning and calibration.
The HTTP service defaults to the same X-v2 filename and fails if its local
checkpoint is missing; `--backend llmdet` requires explicit checkpoint/config.

## Registry

`default_detection_registry().names()` lists `yolo_world`, `llmdet`,
`grounding_dino`, `grounded_vlm` and `hybrid` (sorted).
`registry.create("yolo_world")` constructs the lazy X-v2 default;
`registry.create("llmdet")` constructs the alternate backend.
`default_detection_registry(llmdet_config=..., yolo_world_config=...)` injects
explicit configs. Register extensions using `DetectorFactory` and
`DetectionRegistry.register`. TensorRT engine-build tooling belongs under
`tasks/`, never in this core package.

`grounded_vlm_config` and `hybrid_config` inject composite configurations. The
former rejects a YOLO component; the latter requires one. YOLO-only constructs
neither DINO nor BLIP-2. Loaded composite vocabularies are immutable; restart a
dedicated service to change them, rather than partially reconfiguring models.
See the [grounded perception runbook](../../../tasks/mapping/scene_graph/serve/GROUNDED_VLM.md)
for the one-field JSON switch, provisioning, timing and limitations.

## 2D → 3D lifting

Fuse boxes with registered metric depth using
`core/mapping/depth/depth_bbox_fusion.py::bbox_to_xyz_cam_from_depth`.
The metric target feeds visual servo/approach logic; ObjectNav uses its existing
depth projection and landmark/evidence safeguards unchanged. See
`core/common/types/perception.py` for the `Detection2D`/`Track2D` fields.
