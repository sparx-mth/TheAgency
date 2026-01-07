"""
Minimum-snap parameters.

This module defines the algorithm-level knobs used by the MinSnap smoother.
These parameters control:
- waypoint filtering (removing nearly-duplicate points)
- segment time allocation (based on distance and limits)
- sampling resolution (dt)
- library settings (polynomial degree, continuity, minimized derivative order)

Per-call overrides:
- The pipeline can pass overrides via `SmootherRequest.options` without
  reconstructing the smoother instance.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinSnapParams:
    """
    Parameters for minimum-snap smoothing.

    Notes:
    - dt is the sampling period used to generate discrete TrajectoryPoint samples.
    - time allocation is a heuristic; it aims to produce trajectories that are
      likely to respect limits, but does not enforce constraints analytically.
    """
    dt: float = 0.02
    min_point_spacing: float = 0.05

    # Time allocation heuristic (trapezoidal estimate + margins)
    nominal_speed_xy: float = 0.5
    v_eff_ratio: float = 0.6
    min_segment_time: float = 1.2
    segment_time_scale: float = 1.8

    # minsnap_trajectories settings
    degree: int = 7
    idx_minimized_orders: tuple[int, ...] = (4,)  # minimize snap
    num_continuous_orders: int = 4
    algorithm: str = "closed-form"

    # Boundary constraints
    constrain_endpoints_velocity_zero: bool = True

    def validate(self) -> None:
        """Validate parameter ranges to avoid silent runtime issues."""
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.min_point_spacing < 0:
            raise ValueError("min_point_spacing must be >= 0")
        if self.nominal_speed_xy <= 0:
            raise ValueError("nominal_speed_xy must be > 0")
        if not (0.0 < self.v_eff_ratio <= 1.0):
            raise ValueError("v_eff_ratio must be in (0, 1]")
        if self.min_segment_time <= 0:
            raise ValueError("min_segment_time must be > 0")
        if self.segment_time_scale <= 0:
            raise ValueError("segment_time_scale must be > 0")
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if self.num_continuous_orders < 0:
            raise ValueError("num_continuous_orders must be >= 0")
