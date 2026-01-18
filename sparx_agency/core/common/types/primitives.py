"""
Primitive type aliases and small numeric/vector helpers.

This module is safe to use across core planning/mapping/localization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt, isfinite
from typing import Union, Tuple

# ---------------------------------------------------------------------
# Scalar / index aliases
# ---------------------------------------------------------------------

Number = Union[int, float]

Coord2D = Tuple[Number, Number]
Coord3D = Tuple[Number, Number, Number]

Index2D = Tuple[int, int]
Index3D = Tuple[int, int, int]


# ---------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------

def _assert_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


# ---------------------------------------------------------------------
# Vector primitives
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __post_init__(self):
        _assert_finite("Vec2.x", self.x)
        _assert_finite("Vec2.y", self.y)

    def norm(self) -> float:
        return hypot(self.x, self.y)

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __post_init__(self):
        _assert_finite("Vec3.x", self.x)
        _assert_finite("Vec3.y", self.y)
        _assert_finite("Vec3.z", self.z)

    def norm(self) -> float:
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z
