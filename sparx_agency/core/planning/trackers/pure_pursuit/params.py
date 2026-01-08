"""
Pure Pursuit parameters.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class PurePursuitParams:
    """
    Pure Pursuit configuration.

    Based on Coulter (1992) with modern extensions for adaptive lookahead
    and speed profiling.
    """
    # Robot model
    holonomic: bool = True  # True for omnidirectional, False for diff-drive
    wheelbase: Optional[float] = None  # For Ackermann steering (meters)

    # Lookahead (m)
    base_lookahead: float = 0.6
    min_lookahead: float = 0.3
    max_lookahead: float = 1.5
    lookahead_speed_gain: float = 0.5  # L_d += speed × gain

    # Speed (m/s)
    cruise_speed: float = 0.4
    min_speed: float = 0.1
    max_speed: float = 0.5
    curvature_speed_factor: float = 0.3  # slow on curves
    slow_down_distance: float = 1.0  # ramp down near goal

    # Tolerances (m)
    goal_tolerance: float = 0.15
    path_tolerance: float = 0.8  # max cross-track error

    # Angular velocity limits
    max_yaw_rate: float = 0.5  # rad/s

    # Altitude control (3D)
    altitude_kp: float = 1.2
    max_vertical_speed: float = 0.3

    # Smoothing (low-pass filter α)
    speed_smoothing: float = 0.3
    yaw_rate_smoothing: float = 0.3

    # Trajectory sampling
    sample_dt: float = 0.05  # seconds

    # Closest point search window (indices)
    closest_search_back: int = 10
    closest_search_forward: int = 120