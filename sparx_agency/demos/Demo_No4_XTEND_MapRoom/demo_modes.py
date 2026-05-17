#!/usr/bin/env python3
from __future__ import annotations

from enum import Enum


class DemoMode(str, Enum):
    """Shared XTEND demo modes.

    Keep this file importable by planner, localization, and the demo manager.
    """

    IDLE = "idle"
    FLY_STRAIGHT = "fly_straight"
    TURNING = "turning"
    VISUAL_SERVOING = "visual_servoing"
    FINISH = "finish"

    @classmethod
    def from_text(cls, text: str) -> "DemoMode | None":
        key = str(text).strip().lower()

        aliases = {
            "idle": cls.IDLE,
            "fly": cls.FLY_STRAIGHT,
            "forward": cls.FLY_STRAIGHT,
            "straight": cls.FLY_STRAIGHT,
            "fly_straight": cls.FLY_STRAIGHT,
            "turn": cls.TURNING,
            "turning": cls.TURNING,
            "visual_servoing": cls.VISUAL_SERVOING,
            "visual-servoing": cls.VISUAL_SERVOING,
            "servo": cls.VISUAL_SERVOING,
            "servoing": cls.VISUAL_SERVOING,
            "object_servo": cls.VISUAL_SERVOING,
            "track_object": cls.VISUAL_SERVOING,
            "center_object": cls.VISUAL_SERVOING,
            "finish": cls.FINISH,
            "arrived": cls.FINISH,
            "done": cls.FINISH,
            "land": cls.FINISH,
        }

        return aliases.get(key)
