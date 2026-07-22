"""Colour-blob mock detector -- zero-dependency stand-in for the real detector.

The guaranteed-to-run backend: no torch, no TensorRT, no model download. It finds
the largest blob of a chosen colour and reports it as the target, so you can drive
the whole target-lock mission at home by holding up a coloured object -- a red cup,
a green sticky note, a blue lid. It is a *mock*: it identifies by colour, not by
class, but the detection it emits is a real
:class:`~sparx_agency.core.common.types.perception.Detection2D` labelled with the
mission target, which is all the confirmation gate / tracker / servo / FSM need to
exercise every new mechanism (tracking robustness, lock modes, the HUD colours, and
the RECOVER manoeuvres).

Hold the object -> green box (detected). Cover it or move it off-frame -> the
tracker coasts (orange) then the mission goes to RECOVER (red border) and finally
SEARCH (grey border): exactly the lifecycle the drone runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.tasks.planning.object_approach_webcam.detector_interface import WebcamDetector

HsvRange = Tuple[Tuple[int, int, int], Tuple[int, int, int]]

# Named colours -> one or more inclusive HSV ranges (OpenCV H in 0..179). Red wraps
# the hue circle, so it needs two ranges.
COLOR_RANGES: Dict[str, List[HsvRange]] = {
    "red": [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (179, 255, 255))],
    "orange": [((11, 120, 90), (22, 255, 255))],
    "yellow": [((23, 90, 90), (35, 255, 255))],
    "green": [((36, 70, 60), (85, 255, 255))],
    "blue": [((90, 90, 60), (128, 255, 255))],
    "purple": [((129, 70, 50), (158, 255, 255))],
}


@dataclass(frozen=True)
class ColorDetectorConfig:
    """Tuning for :class:`MockColorDetector`.

    Attributes:
        color: One of :data:`COLOR_RANGES`.
        min_area_frac: Ignore blobs smaller than this fraction of the frame (noise).
        open_ksize: Morphological-open kernel size (px) to clean speckle.
    """

    color: str = "red"
    min_area_frac: float = 0.004
    open_ksize: int = 5

    def __post_init__(self) -> None:
        if self.color not in COLOR_RANGES:
            raise ValueError("color must be one of %s, got %r"
                             % (sorted(COLOR_RANGES), self.color))
        if not (0.0 <= self.min_area_frac < 1.0):
            raise ValueError("min_area_frac must be in [0, 1).")


class MockColorDetector(WebcamDetector):
    """Report the largest blob of a chosen colour as the mission target."""

    def __init__(self, config: Optional[ColorDetectorConfig] = None) -> None:
        self.cfg = config or ColorDetectorConfig()
        self._label = "target"
        self._ranges = COLOR_RANGES[self.cfg.color]
        k = int(self.cfg.open_ksize)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k > 0 else None

    def set_prompts(self, prompts: Sequence[str]) -> None:
        self._label = (list(prompts)[0] if prompts else "target").strip().lower()

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        h, w = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = None
        for lo, hi in self._ranges:
            part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        if self._kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < self.cfg.min_area_frac * w * h:
            return []
        x, y, bw, bh = cv2.boundingRect(largest)
        # Score = how solidly the blob fills its box (a clean blob -> high score),
        # nudged up by size so a big, solid target reads as confident.
        fill = area / float(max(1, bw * bh))
        score = float(min(0.99, 0.35 + 0.6 * fill))
        return [Detection2D(label=self._label, score=score,
                            bbox_xyxy=(int(x), int(y), int(x + bw), int(y + bh)),
                            frame_w=int(w), frame_h=int(h))]
