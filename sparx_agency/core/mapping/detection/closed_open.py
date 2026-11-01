"""A closed-set detector for the classes it knows, an open-vocabulary one for the rest.

The six ObjectNav targets are COCO classes and a COCO detector is better at them than an open-vocabulary one; doors,
stairs, cabinets and stools are not COCO. Each prompt is answered by exactly one of the two: the closed-set detector
owns the prompts it can answer, the open one gets the others minus ``open_exclude`` (the target synonyms it must not
answer a second time). Optionally a closed-set ``suppress_label`` box is dropped when the open detector names a
``distractors`` word on the same object (a stool read as a chair), which is the open detector's strength.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.math.bbox import xyxy_area
from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.detection.rfdetr import RfDetrConfig
from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel


@dataclass(frozen=True)
class ClosedOpenConfig:
    """Both detectors' configs, the words the open one never answers and the optional distractor rule."""

    closed: RfDetrConfig = field(default_factory=RfDetrConfig)
    open_: YoloWorldConfig = field(default_factory=YoloWorldConfig)
    open_exclude: Tuple[str, ...] = ()
    suppress_label: str = "chair"
    distractors: Tuple[str, ...] = ()
    suppress_conf: float = 0.30
    suppress_iou: float = 0.50

    @property
    def device(self) -> str:
        return self.closed.device

    @property
    def conf_thresh(self) -> float:
        return self.closed.conf_thresh


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = xyxy_area(a) + xyxy_area(b) - inter
    return inter / union if union > 0 else 0.0


class ClosedOpenDetector(DetectionModel):
    """Prompts split between a closed-set detector and an open-vocabulary one; their detections are concatenated."""

    def __init__(self, config: ClosedOpenConfig = None, closed: DetectionModel = None, open_: DetectionModel = None) -> None:
        self.cfg = config or ClosedOpenConfig()
        if closed is None:
            from sparx_agency.core.mapping.detection.rfdetr import RfDetrDetector

            closed = RfDetrDetector(self.cfg.closed)
        if open_ is None:
            from sparx_agency.core.mapping.detection.yolo_world import YoloWorldDetector

            open_ = YoloWorldDetector(self.cfg.open_)
        self.closed, self.open_ = closed, open_
        self._prompts: List[str] = []
        self._open_prompts: List[str] = []

    def set_prompts(self, prompts: Sequence[str]) -> None:
        cleaned = [str(p).strip() for p in prompts if str(p).strip()]
        if not cleaned:
            raise ValueError("set_prompts: at least one non-empty prompt required.")
        self._prompts = cleaned
        self.closed.set_prompts(cleaned)
        owned = set(self.closed.answered_prompts)
        excluded = set(self.cfg.open_exclude)
        self._open_prompts = [p for p in cleaned if p not in owned and p not in excluded]
        if self._open_prompts:
            self.open_.set_prompts(self._open_prompts)

    @property
    def prompts(self) -> List[str]:
        return list(self._prompts)

    @property
    def open_prompts(self) -> List[str]:
        return list(self._open_prompts)

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        closed = self.closed.detect(rgb)
        opened = self.open_.detect(rgb) if self._open_prompts else []
        if self.cfg.distractors:
            closed = self._without_distractors(closed, opened)
        return sorted(closed + opened, key=lambda d: -d.score)

    def warmup(self) -> None:
        self.closed.warmup()
        if self._open_prompts:
            self.open_.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    def preprocessing(self):
        return {"closed": self.closed.preprocessing(), "closed_answers": list(self.closed.answered_prompts),
                "open_prompts": list(self._open_prompts), "open_exclude": list(self.cfg.open_exclude),
                "suppress": {"label": self.cfg.suppress_label, "distractors": list(self.cfg.distractors),
                             "conf": self.cfg.suppress_conf, "iou": self.cfg.suppress_iou}}

    def _without_distractors(self, closed, opened):
        named = [d for d in opened if d.label in self.cfg.distractors and d.score >= self.cfg.suppress_conf]
        return [d for d in closed if not (d.label == self.cfg.suppress_label and any(
            _iou(d.bbox_xyxy, n.bbox_xyxy) >= self.cfg.suppress_iou for n in named))]
