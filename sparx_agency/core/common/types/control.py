# core/common/types/control.py
"""
Control-related types.

Defines generic, control commands and kinematic limits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict
from .primitives import _assert_finite


class ControlMode(str, Enum):
    VELOCITY = "velocity"
    POSITION = "position"
    ACCELERATION = "acceleration"


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """
    Generic control command (ROS-free).
    """
    mode: ControlMode
    x: float
    y: float
    z: float = 0.0
    yaw_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _assert_finite("ControlCommand.x", self.x)
        _assert_finite("ControlCommand.y", self.y)
        _assert_finite("ControlCommand.z", self.z)
        _assert_finite("ControlCommand.yaw_rate", self.yaw_rate)

    @staticmethod
    def velocity(vx: float, vy: float, vz: float = 0.0, yaw_rate: float = 0.0, **meta):
        return ControlCommand(ControlMode.VELOCITY, vx, vy, vz, yaw_rate, dict(meta))


@dataclass(frozen=True, slots=True)
class KinematicLimits:
    """
    Conservative default kinematic limits.

    These values are intended as safe fallbacks.
    Real robots should override them via configuration.
    """
    max_speed_xy: float = 0.5
    max_speed_z: float = 0.3
    max_yaw_rate: float = 0.5
