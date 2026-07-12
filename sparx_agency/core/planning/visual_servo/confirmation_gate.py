"""Localization-free target acquisition gate: confirm on N consecutive detections.

The trigger that flips the mission from "fly the A*/NavDP route while scanning"
to "abandon the planner and visually approach". Deliberately *simple* and
pose-free — unlike the reference stack, which confirmed via world-frame object
aggregation, observation counts, and an LLM matcher (all needing localization).

Here a target is acquired once the detector reports it in ``n_confirm`` consecutive
frames above ``min_score`` (a small ``miss_tolerance`` bridges a dropped frame so
one flicker doesn't reset the streak). Label matching is fuzzy-lite: exact,
substring, or shared token — enough for prompts like ``"refrigerator"`` /
``"hat"`` / ``"weapon"`` that are already in the detector's vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.common.math.bbox import iou


def label_matches(target: str, label: str) -> bool:
    """Fuzzy-lite class match: exact, substring, or shared whitespace/underscore token."""
    t = str(target).strip().lower()
    c = str(label).strip().lower()
    if not t or not c:
        return False
    if t == c or t in c or c in t:
        return True
    t_tokens = set(t.replace("_", " ").split())
    c_tokens = set(c.replace("_", " ").split())
    return bool(t_tokens & c_tokens)


def select_target_detection(detections: Sequence[Detection2D], target: str,
                            min_score: float) -> Optional[Detection2D]:
    """Highest-scoring detection matching ``target`` with score >= ``min_score``."""
    best: Optional[Detection2D] = None
    for d in detections:
        if float(d.score) < min_score:
            continue
        if not label_matches(target, d.label):
            continue
        if best is None or float(d.score) > float(best.score):
            best = d
    return best


def select_overlapping_target_detection(
        detections: Sequence[Detection2D], target: str, ref_bbox, min_iou: float,
        min_score: float) -> Optional[Detection2D]:
    """Best ``target`` detection overlapping ``ref_bbox`` (IoU >= ``min_iou``).

    The "look harder near the box we're tracking" check: while tracking, a *weak*
    detection (one too low-confidence to acquire the target from scratch) that sits
    right on the tracked box is strong evidence the object is still there, so it can
    keep the lock alive. Background drift has no such overlapping detection, so it is
    not kept alive. Candidates are ranked by overlap first, then score.
    """
    ref = tuple(float(v) for v in ref_bbox)
    best: Optional[Detection2D] = None
    best_key = (-1.0, -1.0)
    for d in detections:
        if float(d.score) < min_score or not label_matches(target, d.label):
            continue
        ov = iou(tuple(float(v) for v in d.bbox_xyxy), ref)
        if ov < min_iou:
            continue
        key = (ov, float(d.score))
        if key > best_key:
            best_key, best = key, d
    return best


@dataclass(frozen=True)
class ConfirmationGateConfig:
    """Tuning for :class:`TargetConfirmationGate`.

    Attributes:
        n_confirm: Consecutive matching detector frames required to confirm.
        min_score: Minimum detection confidence to count a frame as a hit.
        miss_tolerance: Consecutive misses allowed without resetting the streak
            (0 = strict consecutive; 1 bridges a single dropped detection).
    """

    n_confirm: int = 3
    min_score: float = 0.30
    miss_tolerance: int = 1

    def __post_init__(self) -> None:
        if self.n_confirm < 1:
            raise ValueError("n_confirm must be >= 1.")
        if self.miss_tolerance < 0:
            raise ValueError("miss_tolerance must be >= 0.")


@dataclass(frozen=True)
class ConfirmationState:
    """Result of one gate update.

    Attributes:
        confirmed: True once the streak has reached ``n_confirm``.
        streak: Current consecutive-hit count.
        best: The matching detection this frame (to seed the tracker), or None.
    """

    confirmed: bool
    streak: int
    best: Optional[Detection2D]


class TargetConfirmationGate:
    """Confirm a target over consecutive detector frames (pose-free)."""

    def __init__(self, target: str,
                 config: Optional[ConfirmationGateConfig] = None) -> None:
        self.cfg = config or ConfirmationGateConfig()
        self.set_target(target)

    def set_target(self, target: str) -> None:
        """Set / change the target class; resets the streak."""
        self._target = str(target).strip().lower()
        self.reset()

    @property
    def target(self) -> str:
        return self._target

    def reset(self) -> None:
        """Clear the streak (e.g. when the mission drops back to SEARCH)."""
        self._streak = 0
        self._misses = 0

    def update(self, detections: Sequence[Detection2D]) -> ConfirmationState:
        """Feed one detector frame's detections; advance the streak.

        Returns the current :class:`ConfirmationState`. ``best`` is the matching
        detection this frame (highest score), suitable for seeding the tracker.
        """
        if not self._target:
            return ConfirmationState(confirmed=False, streak=0, best=None)
        best = select_target_detection(detections, self._target, self.cfg.min_score)
        if best is not None:
            self._streak += 1
            self._misses = 0
        else:
            self._misses += 1
            if self._misses > self.cfg.miss_tolerance:
                self._streak = 0
        confirmed = self._streak >= self.cfg.n_confirm
        return ConfirmationState(confirmed=confirmed, streak=self._streak, best=best)
