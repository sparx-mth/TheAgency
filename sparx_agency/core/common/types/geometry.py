"""Geometry types: poses and angle utilities."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, hypot, pi
from .primitives import Vec2, Vec3, _assert_finite


def normalize_angle(angle: float) -> float:
    """Normalize angle to (-π, π]."""
    while angle <= -pi:
        angle += 2 * pi
    while angle > pi:
        angle -= 2 * pi
    return angle


@dataclass(frozen=True, slots=True)
class Pose2D:
    """2D pose: position (x, y) and orientation (yaw) in world frame."""
    x: float
    y: float
    yaw: float = 0.0

    def __post_init__(self) -> None:
        _assert_finite("Pose2D.x", self.x)
        _assert_finite("Pose2D.y", self.y)
        _assert_finite("Pose2D.yaw", self.yaw)

    def position(self) -> Vec2:
        return Vec2(self.x, self.y)

    def heading(self) -> Vec2:
        return Vec2(cos(self.yaw), sin(self.yaw))

    def distance_to(self, other: Pose2D) -> float:
        return hypot(other.x - self.x, other.y - self.y)

    def bearing_to(self, other: Pose2D) -> float:
        return atan2(other.y - self.y, other.x - self.x)


@dataclass(frozen=True, slots=True)
class Pose3D:
    """3D pose with yaw-only orientation."""
    x: float
    y: float
    z: float
    yaw: float = 0.0

    def __post_init__(self) -> None:
        _assert_finite("Pose3D.x", self.x)
        _assert_finite("Pose3D.y", self.y)
        _assert_finite("Pose3D.z", self.z)
        _assert_finite("Pose3D.yaw", self.yaw)

    def position(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)

    def distance_to(self, other: Pose3D) -> float:
        return hypot(other.x - self.x, other.y - self.y, other.z - self.z)