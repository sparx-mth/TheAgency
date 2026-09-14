"""Alias-aware frame deduplication and multi-view support for target acquisition."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from sparx_agency.core.common.math.bbox import iou

ALIASES = {"couch": "sofa", "tv": "television"}


def deduplicate_detections(detections):
    """Suppress duplicate prompts without broad fuzzy category matching."""
    kept = []
    for detection in sorted(detections, key=lambda d: -d.conf):
        canonical = ALIASES.get(detection.cls, detection.cls)
        candidate = replace(detection, cls=canonical)
        if any((old.cls == canonical and iou(old.xyxy, candidate.xyxy) >= 0.5)
               or iou(old.xyxy, candidate.xyxy) >= 0.9 for old in kept):
            continue
        kept.append(candidate)
    return kept


@dataclass(frozen=True)
class TargetEvidenceSettings:
    min_observations: int = 3
    min_view_translation_m: float = 0.2
    min_view_turn_deg: float = 20.0
    max_unseen_steps: int = 8
    verification_turns: int = 2
    rejected_cooldown_steps: int = 20


class TargetEvidence:
    """Require repeated support from one associated landmark, not duplicate boxes.

    Confirmation is still fallible visual evidence, not ground truth. A target
    hypothesis can guide approach before it is confirmed, but cannot trigger
    STOP. Turning to verify is bounded; it is not a mandatory initial sweep.
    """

    def __init__(self, settings=None):
        self.settings = settings or TargetEvidenceSettings()
        self._support = {}
        self._rejected = {}

    def observe(self, landmark, pose, step):
        if self.is_suppressed(landmark.id, step):
            return False
        state = self._support.get(landmark.id)
        if state is None or step - state["last_step"] > self.settings.max_unseen_steps:
            state = {"count": 0, "first": (pose.x, pose.y, pose.yaw),
                     "last_step": -1, "separated": False, "verification_turns": 0}
            self._support[landmark.id] = state
        if state["last_step"] != step:
            state["count"] += 1
            state["last_step"] = step
        first = state["first"]
        turn = abs(math.atan2(math.sin(pose.yaw - first[2]), math.cos(pose.yaw - first[2])))
        state["separated"] |= (math.dist((pose.x, pose.y), first[:2]) >= self.settings.min_view_translation_m
                               or turn >= math.radians(self.settings.min_view_turn_deg))
        return state["count"] >= self.settings.min_observations and state["separated"]

    def verification_yaw(self, landmark_id, current_yaw, turn_rad):
        """Spend the selected hypothesis's turn budget, never another object's."""
        state = self._support.get(landmark_id)
        if state is None or state["verification_turns"] >= self.settings.verification_turns:
            return None
        state["verification_turns"] += 1
        return current_yaw + turn_rad

    def reject(self, landmark_id, step):
        self._rejected[landmark_id] = step + self.settings.rejected_cooldown_steps
        self._support.pop(landmark_id, None)

    def is_suppressed(self, landmark_id, step):
        return step < self._rejected.get(landmark_id, -1)

    def verification_started(self, landmark_id):
        return self._support.get(landmark_id, {}).get("verification_turns", 0) > 0

    def diagnostics(self):
        return {str(key): {name: value for name, value in item.items() if name != "first"}
                for key, item in self._support.items()}

