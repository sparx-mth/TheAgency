"""Selectable grounded perception; no cross-model score averaging or VLM boxes."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.common.label_match import normalize_label
from sparx_agency.core.common.math.bbox import iou
from sparx_agency.core.mapping.detection.blip2 import Blip2Config, Blip2Verifier
from sparx_agency.core.mapping.detection.grounding_dino import GroundingDinoConfig, GroundingDinoDetector
from sparx_agency.core.mapping.detection.llmdet_postprocess import validate_prompts
from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig, YoloWorldDetector
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel


STAIR_LABELS = frozenset(("stairs", "staircase"))
FURNITURE_LABELS = frozenset(("bed", "sofa", "couch"))


@dataclass(frozen=True)
class GroundedVlmConfig:
    grounding: GroundingDinoConfig = field(default_factory=GroundingDinoConfig)
    verifier: Blip2Config = field(default_factory=Blip2Config)
    yolo: Optional[YoloWorldConfig] = None
    # Empty means every class. Selective verification is an explicit opt-in;
    # otherwise a stair mislabelled as another category could bypass the VLM.
    verify_labels: Tuple[str, ...] = ()
    max_verifications: int = 8
    duplicate_iou: float = 0.6

    def __post_init__(self):
        if type(self.max_verifications) is not int or self.max_verifications <= 0:
            raise ValueError("max_verifications must be a positive integer")
        if not math.isfinite(self.duplicate_iou) or not 0 < self.duplicate_iou <= 1:
            raise ValueError("duplicate_iou must be in (0, 1]")
        if isinstance(self.verify_labels, str) or any(not isinstance(s, str) or not s.strip() for s in self.verify_labels):
            raise ValueError("verify_labels must be category names; empty means all")
        devices = {self.grounding.device, self.verifier.device}
        if self.yolo is not None:
            devices.add(self.yolo.device)
        if len(devices) != 1:
            raise ValueError("One service must use one explicitly configured device")

    @property
    def device(self):
        return self.grounding.device

    @property
    def conf_thresh(self):
        return min(self.grounding.conf_thresh, self.yolo.conf_thresh) if self.yolo else self.grounding.conf_thresh


class GroundedVlmDetector(DetectionModel):
    """Both detectors see every frame; DINO does not depend on YOLO finding stairs.

    Verification is a veto, not new geometric evidence. Budget-exhausted,
    ambiguous and negative candidates are withheld, never silently passed.
    Errors propagate. There is no cached answer from another frame.
    """

    def __init__(self, config=None):
        self.cfg = config or GroundedVlmConfig()
        self.grounding = GroundingDinoDetector(self.cfg.grounding)
        self.verifier = Blip2Verifier(self.cfg.verifier)
        self.yolo = YoloWorldDetector(self.cfg.yolo) if self.cfg.yolo else None
        self._prompts = []
        self._started = False
        self._diagnostics = {}

    @property
    def prompts(self):
        return list(self._prompts)

    def set_prompts(self, prompts):
        cleaned = validate_prompts(prompts)
        # A failed multi-model re-prompt must not leave half the service changed.
        # Once loaded, use a new dedicated service to change its vocabulary.
        if self._started and cleaned != self._prompts:
            raise ValueError("Restart the grounded-VLM service to change its vocabulary")
        if cleaned == self._prompts:
            return
        self.grounding.set_prompts(cleaned)
        if self.yolo is not None:
            self.yolo.set_prompts(cleaned)
        self._prompts = cleaned

    def warmup(self):
        """Check all selected weights at startup; do not classify dummy crops."""
        if not self._prompts:
            raise RuntimeError("Call set_prompts before warmup")
        self._started = True
        self.verifier.load()
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.grounding.detect(dummy)
        if self.yolo is not None:
            self.yolo.detect(dummy)

    def detect(self, rgb):
        if not self._prompts:
            raise RuntimeError("Call set_prompts before grounded-VLM inference")
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) == 0:
            raise ValueError("Expected non-empty HxWx3 uint8 RGB")
        self._diagnostics = {}
        self._started = True
        self.verifier.load()  # also fail at service startup if BLIP-2 cannot load
        raw = []
        if self.yolo is not None:
            raw.extend(("yolo_world", d) for d in self.yolo.detect(image))
        raw.extend(("grounding_dino", d) for d in self.grounding.detect(image))
        rows = [dict(source=source, cls=d.label, conf=float(d.score),
                     xyxy=list(d.bbox_xyxy), status="candidate") for source, d in raw]
        selected = self._candidates(raw, rows, image.shape[:2])
        accepted = self._verify(image, raw, rows, selected)
        self._diagnostics = {"mode": "hybrid" if self.yolo else "grounded_vlm",
                             "raw_detections": rows, "accepted_indices": accepted,
                             "verification_budget": self.cfg.max_verifications}
        return [raw[index][1] for index in accepted]

    def _candidates(self, raw, rows, frame_hw):
        """Source order wins duplicates, never the incomparable score magnitudes."""
        selected = []
        h, w = frame_hw
        for index, (_, detection) in enumerate(raw):
            x1, y1, x2, y2 = detection.bbox_xyxy
            if (detection.label not in self._prompts or not math.isfinite(detection.score)
                    or not 0 <= detection.score <= 1 or not 0 <= x1 < x2 <= w
                    or not 0 <= y1 < y2 <= h
                    or (detection.frame_h, detection.frame_w) != (h, w)):
                raise ValueError("Detector emitted an invalid label/score/box/frame")
            duplicate = next((old for old in selected
                              if normalize_label(raw[old][1].label) == normalize_label(detection.label)
                              and iou(raw[old][1].bbox_xyxy, detection.bbox_xyxy) >= self.cfg.duplicate_iou), None)
            if duplicate is not None:
                rows[index].update(status="duplicate", duplicate_of=duplicate)
            else:
                selected.append(index)
        # Verify structural proposals before furniture. Each source keeps its own
        # score order; scores from different models are not treated as calibrated.
        return sorted(selected, key=lambda i: normalize_label(raw[i][1].label) not in STAIR_LABELS)

    def _verify(self, image, raw, rows, selected):
        labels = {normalize_label(label) for label in self.cfg.verify_labels}
        accepted, calls = [], 0
        for index in selected:
            detection = raw[index][1]
            if labels and normalize_label(detection.label) not in labels:
                rows[index]["status"] = "not_requested"
                accepted.append(index)
                continue
            if calls >= self.cfg.max_verifications:
                rows[index]["status"] = "budget_exhausted"
                continue
            evidence = self.verifier.verify(image, detection)
            if evidence.get("verdict") not in ("yes", "no", "uncertain"):
                raise ValueError("Malformed visual-verification result")
            calls += 1
            rows[index].update(verification=evidence, status="verified" if evidence["verdict"] == "yes" else evidence["verdict"])
            if evidence["verdict"] == "yes":
                accepted.append(index)
        return self._reject_conflicts(raw, rows, accepted)

    def _reject_conflicts(self, raw, rows, accepted):
        """Contradictory overlapping stair/furniture confirmations both abstain."""
        rejected = set()
        stairs = [i for i in accepted if normalize_label(raw[i][1].label) in STAIR_LABELS]
        furniture = [i for i in accepted if normalize_label(raw[i][1].label) in FURNITURE_LABELS]
        for first in stairs:
            for second in furniture:
                if iou(raw[first][1].bbox_xyxy, raw[second][1].bbox_xyxy) >= self.cfg.duplicate_iou:
                    rejected.update((first, second))
        for index in rejected:
            rows[index]["status"] = "conflicting_labels"
        return [index for index in accepted if index not in rejected]

    def diagnostics(self):
        return self._diagnostics

    def preprocessing(self):
        return {"grounding_dino": self.grounding.preprocessing(),
                "blip2": self.verifier.preprocessing(),
                "merge": "YOLO-first exact-label IoU suppression; no score averaging",
                "verification": "configured labels only; unverified budget overflow withheld",
                "temporal_evidence": "one observation; no cross-frame answer reuse"}


