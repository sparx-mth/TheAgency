from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BezierParams:
    """
    Parameters for heading-aware Hermite smoothing (Bezier-like).

    The algorithm generates discrete TrajectoryPoint samples at dt,
    assuming approximately constant cruise speed along arc length.
    """
    dt: float = 0.02
    min_point_spacing: float = 0.05

    # Tangent shaping (larger -> smoother but may overshoot corners)
    tangent_scale: float = 0.5

    # Time allocation via constant-speed assumption
    nominal_speed_xy: float = 0.4

    # Arc-length LUT resolution
    arc_lut_samples: int = 600

    # If True: enforce v=0 at endpoints (may help some trackers)
    constrain_endpoints_velocity_zero: bool = False

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.min_point_spacing < 0:
            raise ValueError("min_point_spacing must be >= 0")
        if self.tangent_scale <= 0:
            raise ValueError("tangent_scale must be > 0")
        if self.nominal_speed_xy <= 0:
            raise ValueError("nominal_speed_xy must be > 0")
        if self.arc_lut_samples < 50:
            raise ValueError("arc_lut_samples must be >= 50")
