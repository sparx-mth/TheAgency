"""Control-related types: commands and kinematic limits."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .primitives import _assert_finite


class ControlMode(str, Enum):
    """Control command mode."""
    VELOCITY = "velocity"
    POSITION = "position"
    ACCELERATION = "acceleration"


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """Generic control command output."""
    mode: ControlMode
    x: float
    y: float
    z: float = 0.0
    yaw_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_finite("ControlCommand.x", self.x)
        _assert_finite("ControlCommand.y", self.y)
        _assert_finite("ControlCommand.z", self.z)
        _assert_finite("ControlCommand.yaw_rate", self.yaw_rate)

    @staticmethod
    def velocity(
        vx: float, vy: float, vz: float = 0.0, yaw_rate: float = 0.0, **meta: Any
    ) -> ControlCommand:
        """Factory for velocity commands."""
        return ControlCommand(ControlMode.VELOCITY, vx, vy, vz, yaw_rate, dict(meta))


@dataclass(frozen=True, slots=True)
class KinematicLimits:
    """
    Kinematic constraints for trajectory generation.

    Used by smoothers and trackers to respect physical limits.
    Configure per-robot; defaults are conservative.

    Attributes:
        max_speed_xy: Maximum planar speed (m/s).
        max_speed_z: Maximum vertical speed (m/s).
        max_yaw_rate: Maximum angular velocity (rad/s).
        max_accel_xy: Maximum planar acceleration (m/s²).
        max_accel_z: Maximum vertical acceleration (m/s²).
    """
    max_speed_xy: float = 0.5
    max_speed_z: float = 0.3
    max_yaw_rate: float = 0.5
    max_accel_xy: Optional[float] = 1.0
    max_accel_z: Optional[float] = 0.5

    def __post_init__(self) -> None:
        if self.max_speed_xy <= 0:
            raise ValueError(f"max_speed_xy must be > 0, got {self.max_speed_xy}")
        if self.max_speed_z <= 0:
            raise ValueError(f"max_speed_z must be > 0, got {self.max_speed_z}")
        if self.max_yaw_rate <= 0:
            raise ValueError(f"max_yaw_rate must be > 0, got {self.max_yaw_rate}")
        if self.max_accel_xy is not None and self.max_accel_xy <= 0:
            raise ValueError(f"max_accel_xy must be > 0, got {self.max_accel_xy}")
        if self.max_accel_z is not None and self.max_accel_z <= 0:
            raise ValueError(f"max_accel_z must be > 0, got {self.max_accel_z}")