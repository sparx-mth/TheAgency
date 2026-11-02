"""What the closing needs of a vision-language model's yes/no on a target candidate, asked once before STOP.

The detector proposes and the model only vetoes: ``no`` releases the candidate, ``yes`` and ``unsure`` leave the STOP
as it was. The model service itself is an adapter under ``tasks`` (``vlm_confirmer.py``); a failure is ``unsure``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

YES, NO, UNSURE = "yes", "no", "unsure"
MIN_CROP_PX = 224

DEFINITIONS = {
    "chair": "chair (a seat for one person with a back; not a stool, not a sofa)",
    "armchair": "chair (a seat for one person with a back; not a stool, not a sofa)",
    "couch": "sofa (a couch for several people)",
    "sofa": "sofa (a couch for several people)",
    "tv": "television or computer monitor",
    "television": "television or computer monitor",
    "monitor": "television or computer monitor",
    "potted plant": "potted plant",
}


@dataclass(frozen=True)
class Verdict:
    """``answer`` is yes, no or unsure; ``detail`` the model's own words or the error."""

    answer: str
    ms: float
    detail: str


class CandidateConfirmer(Protocol):
    def confirm(self, rgb: np.ndarray, box_xyxy: Sequence[float], label: str) -> Verdict: ...


def crop_around(rgb: np.ndarray, box_xyxy: Sequence[float], margin: float) -> np.ndarray:
    """The box grown by ``margin`` of its size on every side, at least ``MIN_CROP_PX`` wide and tall, inside the frame."""
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    half_w = max((x2 - x1) * (0.5 + margin), MIN_CROP_PX / 2)
    half_h = max((y2 - y1) * (0.5 + margin), MIN_CROP_PX / 2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, right = int(max(0, cx - half_w)), int(min(width, cx + half_w))
    top, bottom = int(max(0, cy - half_h)), int(min(height, cy + half_h))
    return rgb[top:max(bottom, top + 1), left:max(right, left + 1)]


def question_for(label: str) -> str:
    kind = DEFINITIONS.get(label.lower(), label)
    return ('A robot detector marked the object in the middle of this image as a %s. Is it really? '
            'Answer with JSON only: {"answer": "yes"}, {"answer": "no"} or {"answer": "unsure"}.' % kind)


def parse_answer(content: str) -> str:
    try:
        answer = str(json.loads(content).get("answer", "")).strip().lower()
    except (ValueError, AttributeError):
        return UNSURE
    return answer if answer in (YES, NO) else UNSURE
