"""Parameters for Hermite spline trajectory smoother."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HermiteParams:
    """
    Configuration for cubic Hermite spline smoothing.

    The smoother interpolates waypoints with G1 (tangent) continuity,
    then samples at fixed time intervals assuming constant cruise speed.

    Attributes:
        dt: Time step for trajectory sampling (seconds).
        min_point_spacing: Minimum distance between waypoints (meters).
            Closer points are merged to avoid numerical issues.
        tangent_scale: Tangent magnitude scaling factor.
            Larger values produce smoother curves but may overshoot corners.
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