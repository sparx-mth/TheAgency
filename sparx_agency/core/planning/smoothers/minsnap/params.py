"""Parameters for minimum-snap trajectory smoother."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MinSnapParams:
    """
    Configuration for minimum-snap trajectory smoothing.

    The smoother generates polynomial trajectories that minimize snap
    (4th derivative), producing smooth accelerations suitable for
    quadrotors and other agile robots.

    Attributes:
        dt: Time step for trajectory sampling (seconds).
        min_point_spacing: Minimum distance between waypoints (meters).
        nominal_speed_xy: Default cruise speed when no limits provided (m/s).
        v_eff_ratio: Fraction of max speed used for time allocation.
        min_segment_time: Minimum time per path segment (seconds).
        segment_time_scale: Safety factor for segment time estimates.
        degree: Polynomial degree (7 for snap minimization).
        idx_minimized_orders: Derivative orders to minimize (4 = snap).
        num_continuous_orders: Continuity order at waypoints.
        algorithm: Solver algorithm ("closed-form" or "constrained").
        zero_endpoint_velocity: If True, enforce v=0 at start/end.
    """
    dt: float = 0.02
    min_point_spacing: float = 0.05
    nominal_speed_xy: float = 0.5
    v_eff_ratio: float = 0.6
    min_segment_time: float = 1.2
    segment_time_scale: float = 1.8
    degree: int = 7
    idx_minimized_orders: Tuple[int, ...] = (4,)
    num_continuous_orders: int = 4
    algorithm: str = "closed-form"
    zero_endpoint_velocity: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.min_point_spacing < 0:
            raise ValueError(f"min_point_spacing must be >= 0, got {self.min_point_spacing}")
        if self.nominal_speed_xy <= 0:
            raise ValueError(f"nominal_speed_xy must be > 0, got {self.nominal_speed_xy}")
        if not 0 < self.v_eff_ratio <= 1:
            raise ValueError(f"v_eff_ratio must be in (0, 1], got {self.v_eff_ratio}")
        if self.min_segment_time <= 0:
            raise ValueError(f"min_segment_time must be > 0, got {self.min_segment_time}")
        if self.segment_time_scale <= 0:
            raise ValueError(f"segment_time_scale must be > 0, got {self.segment_time_scale}")
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1, got {self.degree}")
        if self.num_continuous_orders < 0:
            raise ValueError(f"num_continuous_orders must be >= 0, got {self.num_continuous_orders}")