"""Segment Anything prompted with detection boxes, through ultralytics.

The segmentation analog of
:class:`sparx_agency.core.mapping.detection.yolo_world.YoloWorldDetector`: a
:class:`~sparx_agency.core.mapping.interfaces.segmentation_model.SegmentationModel`
implementation that wraps ``ultralytics.SAM``. One class serves the whole family
-- the checkpoint decides the variant (``mobile_sam.pt``, ``sam_b.pt``,
``sam2.1_t.pt`` ...) -- and ``ultralytics``/``torch`` are imported **lazily**
(on first :meth:`segment`) so this module imports cleanly in a ROS-free,
GPU-free, Python-3.8 environment.

Why box prompts and not whole-image segmentation: the caller already has the
detector's boxes, and a box prompt makes SAM answer exactly the question the
depth read needs -- *the pixels of the thing inside this box* -- with the
occluder in front of it left out. Whole-image "segment everything" (FastSAM's
mode) would still need a box-to-mask matching step, which is where the
occluder gets back in.

Measured on the development laptop (4 CPU threads, 640 x 480, three boxes):
``mobile_sam.pt`` 0.29 s/frame, ``sam2.1_t.pt`` 0.60 s/frame; the image
encoder dominates, so more boxes in one call cost little.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from sparx_agency.core.mapping.interfaces.segmentation_model import BoxXYXY, SegmentationModel


@dataclass(frozen=True)
class SamPromptedConfig:
    """Configuration for :class:`SamPromptedSegmenter`.

    Attributes:
        model_path: Ultralytics SAM-family checkpoint. MobileSAM by default:
            the lightest encoder, the one a CPU beside a simulator can run per
            frame; ``sam2.1_t.pt`` is the stronger small model.
        device: Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
        imgsz: SAM's input side; the family is trained at 1024 and the
            ultralytics predictor letterboxes to it.
        max_boxes: Cap on prompts per frame; more raise rather than silently
            dropping boxes, because the caller lines masks up by position.
    """

    model_path: str = "mobile_sam.pt"
    device: str = "cpu"
    imgsz: int = 1024
    max_boxes: int = 64

    def __post_init__(self) -> None:
        if not str(self.model_path).strip():
            raise ValueError("SamPromptedConfig.model_path must be set.")
        if not str(self.device).strip():
            raise ValueError("SamPromptedConfig.device must be set.")
        if int(self.imgsz) <= 0:
            raise ValueError("imgsz must be > 0.")
        if int(self.max_boxes) <= 0:
            raise ValueError("max_boxes must be > 0.")


class SamPromptedSegmenter(SegmentationModel):
    """Box-prompted segmenter backed by ultralytics ``SAM``.

    The model is loaded lazily on the first :meth:`segment` so the object can
    be constructed without ultralytics/torch present.

    Example:
        >>> seg = SamPromptedSegmenter(SamPromptedConfig(device="cpu"))
        >>> masks = seg.segment(rgb_hwc_uint8, [(100, 200, 500, 400)])   # doctest: +SKIP
    """

    def __init__(self, config: Optional[SamPromptedConfig] = None) -> None:
        self.cfg = config or SamPromptedConfig()
        self._model = None  # lazily created ultralytics SAM

    # ── inference ────────────────────────────────────────────────────
    def segment(self, rgb: np.ndarray, boxes_xyxy: Sequence[BoxXYXY]) -> List[np.ndarray]:
        """One boolean ``HxW`` mask per box, in box order; see the ABC contract."""
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError("segment expects HxWx3 uint8 RGB, got shape %s dtype %s" % (img.shape, img.dtype))
        h, w = int(img.shape[0]), int(img.shape[1])
        boxes = [self._checked_box(box, w, h) for box in boxes_xyxy]
        if not boxes:
            return []
        if len(boxes) > int(self.cfg.max_boxes):
            raise ValueError("%d boxes exceed max_boxes=%d" % (len(boxes), self.cfg.max_boxes))
        model = self._ensure_model()
        # ultralytics expects BGR numpy input; our contract is RGB.
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        result = model.predict(bgr, bboxes=boxes, imgsz=int(self.cfg.imgsz),
                               device=self.cfg.device, verbose=False)[0]
        masks = getattr(result, "masks", None)
        if masks is None:
            return [np.zeros((h, w), dtype=bool) for _ in boxes]
        data = masks.data.detach().cpu().numpy()
        if data.ndim == 3 and data.shape[0] == 0:
            return [np.zeros((h, w), dtype=bool) for _ in boxes]
        if data.ndim != 3 or data.shape[0] != len(boxes) or data.shape[1:] != (h, w):
            raise RuntimeError("SAM returned masks of shape %s for %d boxes on a %dx%d frame"
                               % (data.shape, len(boxes), h, w))
        return [np.ascontiguousarray(mask > 0.5) for mask in data]

    # ── internals ────────────────────────────────────────────────────
    @staticmethod
    def _checked_box(box: BoxXYXY, w: int, h: int) -> List[float]:
        values = [float(v) for v in box]
        if len(values) != 4 or not all(np.isfinite(values)):
            raise ValueError("a box must be four finite numbers, got %r" % (box,))
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("box %r lies outside the %dx%d frame or is empty" % (box, w, h))
        return values

    def _ensure_model(self):
        """Load SAM on first use; raises loudly if ultralytics or the checkpoint is absent."""
        if self._model is not None:
            return self._model
        path = Path(self.cfg.model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("SAM checkpoint missing: %s (no download; provision it explicitly)" % path)
        try:
            from ultralytics import SAM  # lazy: heavy torch dep
        except Exception as exc:  # pragma: no cover - env-dependent
            raise ImportError("SamPromptedSegmenter requires the 'ultralytics' package "
                              "(pip install ultralytics). Original error: %r" % (exc,))
        model = SAM(str(path))
        model.to(self.cfg.device)
        self._model = model
        return model
