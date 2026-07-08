"""YOLO-World open-vocabulary detector backend ("OpenYOLO").

The torch analog of
:class:`sparx_agency.core.mapping.depth.depth_anything_v2.DepthAnythingV2DepthModel`:
a :class:`~sparx_agency.core.mapping.interfaces.detection_model.DetectionModel`
implementation that wraps ultralytics ``YOLOWorld``. ``ultralytics``/``torch`` are
imported **lazily** (on first :meth:`detect`) so this module imports cleanly in a
ROS-free, GPU-free, Python-3.8 environment for unit tests and on the Noetic side.

Open-vocabulary detection needs no retraining: the class list is set with
``model.set_classes(prompts)``; changing the target object at runtime
(:meth:`set_prompts`) is just a new prompt list.

A later ``YoloTRTDetector`` (TensorRT runtime, the analog of ``DepthEngineTRT``)
will subclass the same ABC; the engine-build tooling belongs under
``tasks/`` per the project's core-vs-tasks TRT split, never here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel


@dataclass(frozen=True)
class YoloWorldConfig:
    """Configuration for :class:`YoloWorldDetector`.

    Attributes:
        model_path: Ultralytics YOLO-World checkpoint (default the small variant).
        device: Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
        conf_thresh: Minimum detection confidence to keep.
        iou_thresh: NMS IoU threshold.
        imgsz: Inference image size (longest side).
        max_det: Cap on detections returned per frame.
    """

    model_path: str = "yolov8s-world.pt"
    device: str = "cuda:0"
    conf_thresh: float = 0.25
    iou_thresh: float = 0.5
    imgsz: int = 640
    max_det: int = 100

    def __post_init__(self) -> None:
        if not str(self.model_path).strip():
            raise ValueError("YoloWorldConfig.model_path must be set.")
        if not (0.0 <= self.conf_thresh <= 1.0):
            raise ValueError("conf_thresh must be in [0, 1].")
        if not (0.0 <= self.iou_thresh <= 1.0):
            raise ValueError("iou_thresh must be in [0, 1].")
        if int(self.imgsz) <= 0:
            raise ValueError("imgsz must be > 0.")


class YoloWorldDetector(DetectionModel):
    """Open-vocabulary detector backed by ultralytics YOLO-World.

    The model is loaded lazily on the first :meth:`detect` so the object can be
    constructed (and prompts staged) without ultralytics/torch present.

    Example:
        >>> det = YoloWorldDetector(YoloWorldConfig(device="cpu"))
        >>> det.set_prompts(["refrigerator", "chair"])       # doctest: +SKIP
        >>> boxes = det.detect(rgb_hwc_uint8)                 # doctest: +SKIP
    """

    def __init__(self, config: Optional[YoloWorldConfig] = None) -> None:
        self.cfg = config or YoloWorldConfig()
        self._model = None  # lazily created ultralytics YOLOWorld
        self._prompts: List[str] = []

    # ── prompts ──────────────────────────────────────────────────────
    def set_prompts(self, prompts: Sequence[str]) -> None:
        """Stage / apply the open-vocab class prompts (idempotent, cheap)."""
        cleaned = [str(p).strip() for p in prompts if str(p).strip()]
        if not cleaned:
            raise ValueError("set_prompts: at least one non-empty prompt required.")
        self._prompts = cleaned
        if self._model is not None:
            self._model.set_classes(self._prompts)

    @property
    def prompts(self) -> List[str]:
        """Current open-vocabulary prompts."""
        return list(self._prompts)

    # ── inference ────────────────────────────────────────────────────
    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        """Detect the current prompts in an RGB frame; see the ABC contract."""
        if not self._prompts:
            raise RuntimeError(
                "YoloWorldDetector.detect called before set_prompts(); set the "
                "target class list first."
            )
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"detect expects HxWx3 RGB, got shape {img.shape}.")
        h, w = int(img.shape[0]), int(img.shape[1])

        model = self._ensure_model()
        # ultralytics expects BGR numpy input; our contract is RGB.
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        result = model.predict(
            bgr,
            imgsz=int(self.cfg.imgsz),
            conf=float(self.cfg.conf_thresh),
            iou=float(self.cfg.iou_thresh),
            max_det=int(self.cfg.max_det),
            device=self.cfg.device,
            verbose=False,
        )[0]

        out: List[Detection2D] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        names = result.names
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            out.append(
                Detection2D(
                    label=str(names[int(k)]).strip().lower(),
                    score=float(c),
                    bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    frame_w=w,
                    frame_h=h,
                )
            )
        return out

    # ── internals ────────────────────────────────────────────────────
    def _ensure_model(self):
        """Load YOLO-World on first use; raises loudly if ultralytics is absent."""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLOWorld  # lazy: heavy torch dep
        except Exception as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "YoloWorldDetector requires the 'ultralytics' package "
                "(pip install ultralytics). Original error: %r" % (exc,)
            )
        model = YOLOWorld(str(self.cfg.model_path))
        model.to(self.cfg.device)
        model.set_classes(self._prompts)
        self._model = model
        return model
