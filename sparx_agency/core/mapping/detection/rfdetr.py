"""RF-DETR (Roboflow, ICLR 2026) closed-set detector backend: COCO classes, answered through the prompt contract.

RF-DETR knows the 80 COCO classes and nothing else. ``set_prompts`` keeps the prompts that are exact COCO names and
``detect`` reports only those, under the prompt's own spelling; every other prompt is unanswered, for an open-vocabulary
detector beside it (see :mod:`~sparx_agency.core.mapping.detection.closed_open`). ``rfdetr``/``torch`` import lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel


# Lowercase COCO class names as RF-DETR spells them; kept static so setting prompts never imports the model.
COCO_NAMES = frozenset(
    name.replace("_", " ") for name in (
        "airplane", "apple", "backpack", "banana", "baseball_bat", "baseball_glove", "bear", "bed", "bench",
        "bicycle", "bird", "boat", "book", "bottle", "bowl", "broccoli", "bus", "cake", "car", "carrot", "cat",
        "cell_phone", "chair", "clock", "couch", "cow", "cup", "dining_table", "dog", "donut", "elephant",
        "fire_hydrant", "fork", "frisbee", "giraffe", "hair_drier", "handbag", "horse", "hot_dog", "keyboard",
        "kite", "knife", "laptop", "microwave", "motorcycle", "mouse", "orange", "oven", "parking_meter", "person",
        "pizza", "potted_plant", "refrigerator", "remote", "sandwich", "scissors", "sheep", "sink", "skateboard",
        "skis", "snowboard", "spoon", "sports_ball", "stop_sign", "suitcase", "surfboard", "teddy_bear",
        "tennis_racket", "tie", "toaster", "toilet", "toothbrush", "traffic_light", "train", "truck", "tv",
        "umbrella", "vase", "wine_glass", "zebra",
    ))


@dataclass(frozen=True)
class RfDetrConfig:
    """``model_size``: an ``rfdetr`` class name; ``weights_path``: its local checkpoint; ``conf_thresh``: the emission floor."""

    model_size: str = "RFDETRLarge"
    weights_path: str = "rf-detr-large-2026.pth"
    device: str = "cuda:0"
    conf_thresh: float = 0.05
    max_det: int = 100

    def __post_init__(self) -> None:
        if not str(self.weights_path).strip():
            raise ValueError("RfDetrConfig.weights_path must be set.")
        if not 0.0 <= self.conf_thresh <= 1.0:
            raise ValueError("conf_thresh must be in [0, 1].")
        if int(self.max_det) <= 0:
            raise ValueError("max_det must be positive.")


class RfDetrDetector(DetectionModel):
    """The COCO classes among the prompts, from RF-DETR; the model loads on first use."""

    def __init__(self, config: RfDetrConfig = None) -> None:
        self.cfg = config or RfDetrConfig()
        self._model = None
        self._prompts: List[str] = []
        self._names: Dict[int, str] = {}

    def set_prompts(self, prompts: Sequence[str]) -> None:
        cleaned = [str(p).strip() for p in prompts if str(p).strip()]
        if not cleaned:
            raise ValueError("set_prompts: at least one non-empty prompt required.")
        self._prompts = cleaned

    @property
    def prompts(self) -> List[str]:
        return list(self._prompts)

    @property
    def answered_prompts(self) -> Tuple[str, ...]:
        """The prompts this detector can answer: those that are exact COCO class names."""
        return tuple(p for p in self._prompts if p in COCO_NAMES)

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        if not self._prompts:
            raise RuntimeError("RfDetrDetector.detect called before set_prompts().")
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("detect expects HxWx3 RGB, got shape %s." % (img.shape,))
        from PIL import Image

        h, w = int(img.shape[0]), int(img.shape[1])
        names, answered = self._coco_names(), set(self.answered_prompts)
        found = self._ensure_model().predict(Image.fromarray(img.astype(np.uint8)), threshold=float(self.cfg.conf_thresh))
        out = []
        for box, conf, cid in zip(found.xyxy, found.confidence, found.class_id):
            label = names.get(int(cid))
            if label in answered:
                out.append(Detection2D(label=label, score=float(conf), bbox_xyxy=tuple(int(v) for v in box),
                                       frame_w=w, frame_h=h))
        out.sort(key=lambda d: -d.score)
        return out[:int(self.cfg.max_det)]

    def warmup(self) -> None:
        self.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    def preprocessing(self) -> Dict[str, object]:
        model = self._ensure_model()
        return {"input": "RGB uint8 HWC -> PIL", "model": self.cfg.model_size,
                "resolution": int(getattr(model.model_config, "resolution", 0)),
                "answers": "exact COCO class names among the prompts", "boxes": "original pixels XYXY, integer"}

    def _coco_names(self) -> Dict[int, str]:
        if not self._names:
            from rfdetr.assets.coco_classes import COCO_CLASSES

            self._names = {int(k): str(v).strip().lower() for k, v in COCO_CLASSES.items()}
        return self._names

    def _ensure_model(self):
        if self._model is None:
            try:
                import rfdetr
            except Exception as exc:  # pragma: no cover - environment-dependent
                raise ImportError("RfDetrDetector requires the 'rfdetr' package. Original error: %r" % (exc,))
            self._model = getattr(rfdetr, self.cfg.model_size)(pretrain_weights=str(self.cfg.weights_path),
                                                               device=self.cfg.device)
        return self._model
