"""Multi-view support for a target hypothesis: repeated sightings of ONE landmark from separated viewpoints.

A target hypothesis is confirmed when the same associated landmark (the
plan-view association is the landmark map's) has been observed
``min_observations`` times in distinct frames, from viewpoints at least
``min_view_translation_m`` apart or ``min_view_turn_deg`` turned -- duplicate
boxes in one frame count once. Confirmation is still fallible visual
evidence, not ground truth: it may guide an approach before it is confirmed
and never triggers a STOP by itself. A bounded verification turn budget and a
rejection cooldown per landmark keep one object's evidence from being spent
on another.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from sparx_agency.core.common.types import normalize_angle


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
        turn = abs(normalize_angle(pose.yaw - first[2]))
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

