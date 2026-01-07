from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PurePursuitParams:
    """
    Pure Pursuit tracker parameters (ROS-free).

    This tracker outputs a velocity ControlCommand in BODY frame:
        cmd = (vx_body, vy_body, vz, yaw_rate)
    """

    # Lookahead (meters)
    base_lookahead: float = 0.6
    min_lookahead: float = 0.3
    max_lookahead: float = 1.5
    lookahead_speed_gain: float = 0.5

    # Speed profile (m/s)
    cruise_speed: float = 0.4
    min_speed: float = 0.1
    max_speed: float = 0.5

    # Curvature-based speed reduction
    curvature_speed_factor: float = 0.3

    # Slow down near goal (meters)
    slow_down_distance: float = 1.0

    # Tolerances (meters)
    goal_tolerance: float = 0.15
    path_tolerance: float = 0.8

    # Altitude control (simple P, optional)
    altitude_kp: float = 1.2
    max_vertical_speed: float = 0.3

    # Yaw control (smooth)
    yaw_kp: float = 0.5
    max_yaw_rate: float = 0.35
    yaw_deadband: float = 0.15
    yaw_speed_threshold: float = 0.05
    yaw_rate_smoothing: float = 0.15

    # Resampling for internal discrete representation
    sample_dt: float = 0.05  # seconds, used for trajectory.sample_by_time(sample_dt)

    # Closest-point search window (indices in sampled list)
    closest_search_back: int = 10
    closest_search_forward: int = 120

    # Internal speed smoothing
    speed_smoothing_alpha: float = 0.3
