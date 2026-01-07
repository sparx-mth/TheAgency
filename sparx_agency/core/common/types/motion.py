# core/common/types/motion.py
"""
Motion-related data types.

Defines velocities, accelerations, and full robot state representations.
All types are used by planning, control, and tracking modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from .geometry import Pose2D, Pose3D
from .primitives import _assert_finite


@dataclass(frozen=True, slots=True)
class Twist2D:
    """
    2D velocity in world frame.

    vx, vy are linear velocities (m/s),
    yaw_rate is angular velocity (rad/s).
    """
    vx: float
    vy: float
    yaw_rate: float = 0.0

    def __post_init__(self):
        _assert_finite("Twist2D.vx", self.vx)
        _assert_finite("Twist2D.vy", self.vy)
        _assert_finite("Twist2D.yaw_rate", self.yaw_rate)

    def speed(self) -> float:
        """Return planar speed sqrt(vx^2 + vy^2)."""
        return hypot(self.vx, self.vy)


@dataclass(frozen=True, slots=True)
class Twist3D:
    """
    3D velocity in world frame.
    """
    vx: float
    vy: float
    vz: float
    yaw_rate: float = 0.0

    def __post_init__(self):
        _assert_finite("Twist3D.vx", self.vx)
        _assert_finite("Twist3D.vy", self.vy)
        _assert_finite("Twist3D.vz", self.vz)
        _assert_finite("Twist3D.yaw_rate", self.yaw_rate)


@dataclass(frozen=True, slots=True)
class Accel2D:
    """
    2D linear acceleration in world frame.
    """
    ax: float
    ay: float

    def __post_init__(self):
        _assert_finite("Accel2D.ax", self.ax)
        _assert_finite("Accel2D.ay", self.ay)


@dataclass(frozen=True, slots=True)
class Accel3D:
    """
    3D linear acceleration in world frame.
    """
    ax: float
    ay: float
    az: float

    def __post_init__(self):
        _assert_finite("Accel3D.ax", self.ax)
        _assert_finite("Accel3D.ay", self.ay)
        _assert_finite("Accel3D.az", self.az)


@dataclass(frozen=True, slots=True)
class State2D:
    """
    Full 2D robot state used by trackers and controllers.
    """
    pose: Pose2D
    twist: Twist2D = field(default_factory=lambda: Twist2D(0.0, 0.0, 0.0))


@dataclass(frozen=True, slots=True)
class State3D:
    """
    Full 3D robot/drone state used by trackers and controllers.
    """
    pose: Pose3D
    twist: Twist3D = field(default_factory=lambda: Twist3D(0.0, 0.0, 0.0, 0.0))
