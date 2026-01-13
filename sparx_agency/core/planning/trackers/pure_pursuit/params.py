"""Pure Pursuit parameters (2D and 3D)."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class PurePursuitParams:
    """
    Pure Pursuit configuration (2D).

    Based on Coulter (1992) with modern extensions for adaptive lookahead
    and speed profiling.
    """
    # Robot model
    holonomic: bool = True
    wheelbase: Optional[float] = None

    # Lookahead (m)
    base_lookahead: float = 0.6
    min_lookahead: float = 0.3
    max_lookahead: float = 1.5
    lookahead_speed_gain: float = 0.5

    # Speed (m/s)
    cruise_speed: float = 0.4
    min_speed: float = 0.1
    max_speed: float = 0.5

    # Curvature adaptation
    curvature_speed_factor: float = 0.5
    curvature_lookahead_factor: float = 0.8

    slow_down_distance: float = 1.0

    # Tolerances (m)
    goal_tolerance: float = 0.15
    path_tolerance: float = 0.8

    # Angular velocity limits
    max_yaw_rate: float = 0.5

    # Altitude control (3D)
    altitude_kp: float = 1.2
    max_vertical_speed: float = 0.3

    # Smoothing
    speed_smoothing: float = 0.3
    yaw_rate_smoothing: float = 0.3

    # Trajectory sampling
    sample_dt: float = 0.05

    # Search window
    closest_search_back: int = 10
    closest_search_forward: int = 120


@dataclass(frozen=True, slots=True)
class PurePursuitParams3D:
    """
    Pure Pursuit configuration (3D).

    Simplified for holonomic 3D robots (drones): velocity points
    directly toward lookahead point in 3D space.
    """
    # Lookahead (m)
    base_lookahead: float = 0.6
    min_lookahead: float = 0.3
    max_lookahead: float = 1.5
    lookahead_speed_gain: float = 0.5

    # Speed (m/s)
    cruise_speed: float = 0.4
    min_speed: float = 0.1
    max_speed: float = 0.5
    max_speed_z: float = 0.3

    # Curvature adaptation (based on xy curvature)
    curvature_speed_factor: float = 0.5
    curvature_lookahead_factor: float = 0.8

    slow_down_distance: float = 1.0

    # Tolerances (m)
    goal_tolerance: float = 0.15
    path_tolerance: float = 0.8

    # Angular velocity limits
    max_yaw_rate: float = 0.5

    # Yaw control
    yaw_mode: str = "velocity"  # "velocity" = face direction of motion, "path" = follow path yaw

    # Smoothing
    speed_smoothing: float = 0.3
    yaw_rate_smoothing: float = 0.3

    # Trajectory sampling
    sample_dt: float = 0.05

    # Search window
    closest_search_back: int = 10
    closest_search_forward: int = 120