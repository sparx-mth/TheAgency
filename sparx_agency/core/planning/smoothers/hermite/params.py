"""Parameters for Hermite spline trajectory smoother (2D and 3D)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HermiteParams:
    """
    Configuration for cubic Hermite spline smoothing (2D).

    Attributes:
        dt: Time step for trajectory sampling (seconds).
        min_point_spacing: Minimum distance between waypoints (meters).
        tangent_scale: Tangent magnitude scaling factor.
        nominal_speed_xy: Default cruise speed when no limits provided (m/s).
        arc_lut_samples: Resolution of arc-length lookup table.
        zero_endpoint_velocity: If True, enforce zero velocity at start/end.
    """
    dt: float = 0.02
    min_point_spacing: float = 0.05
    tangent_scale: float = 0.5
    nominal_speed_xy: float = 0.4
    arc_lut_samples: int = 600
    zero_endpoint_velocity: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.min_point_spacing < 0:
            raise ValueError(f"min_point_spacing must be >= 0, got {self.min_point_spacing}")
        if self.tangent_scale <= 0:
            raise ValueError(f"tangent_scale must be > 0, got {self.tangent_scale}")
        if self.nominal_speed_xy <= 0:
            raise ValueError(f"nominal_speed_xy must be > 0, got {self.nominal_speed_xy}")
        if self.arc_lut_samples < 50:
            raise ValueError(f"arc_lut_samples must be >= 50, got {self.arc_lut_samples}")


@dataclass(frozen=True)
class HermiteParams3D:
    """
    Configuration for cubic Hermite spline smoothing (3D).

    Same as 2D with additional nominal_speed_z for vertical motion.
    """
    dt: float = 0.02
    min_point_spacing: float = 0.05
    tangent_scale: float = 0.5
    nominal_speed_xy: float = 0.4
    nominal_speed_z: float = 0.3
    arc_lut_samples: int = 600
    zero_endpoint_velocity: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.min_point_spacing < 0:
            raise ValueError(f"min_point_spacing must be >= 0, got {self.min_point_spacing}")
        if self.tangent_scale <= 0:
            raise ValueError(f"tangent_scale must be > 0, got {self.tangent_scale}")
        if self.nominal_speed_xy <= 0:
            raise ValueError(f"nominal_speed_xy must be > 0, got {self.nominal_speed_xy}")
        if self.nominal_speed_z <= 0:
            raise ValueError(f"nominal_speed_z must be > 0, got {self.nominal_speed_z}")
        if self.arc_lut_samples < 50:
            raise ValueError(f"arc_lut_samples must be >= 50, got {self.arc_lut_samples}")