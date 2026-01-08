"""
Pure Pursuit path tracking algorithm.

Based on Coulter (1992), CMU-RI-TR-92-01:
"Implementation of the Pure Pursuit Path Tracking Algorithm"

Core idea: Compute a circular arc from robot position to a lookahead point,
then derive the steering command from the arc's curvature.

Key formula:
    κ = 2·sin(α) / L_d

where:
    α   = angle to lookahead point in robot frame
    L_d = lookahead distance
    κ   = curvature of the arc to follow

For differential drive / unicycle:
    ω = v · κ = v · 2·sin(α) / L_d

For Ackermann steering:
    δ = arctan(L · κ)  where L = wheelbase
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from sparx_agency.core.common.types import TrajectoryPoint
from sparx_agency.core.common.types.geometry import normalize_angle


def trajectory_to_xy(points: List[TrajectoryPoint]) -> np.ndarray:
    """Extract (N, 2) position array from trajectory points."""
    return np.array([[p.x, p.y] for p in points], dtype=np.float64)


def get_curvature(point: TrajectoryPoint) -> float:
    """Extract curvature from trajectory point (default 0)."""
    return float(point.curvature) if point.curvature is not None else 0.0


def find_closest_index(
    xy: np.ndarray,
    pos: np.ndarray,
    current_idx: int,
    search_back: int,
    search_forward: int,
) -> Tuple[int, float]:
    """
    Find closest point index within a window around current_idx.

    Returns:
        (closest_index, cross_track_error)
    """
    n = len(xy)
    current_idx = max(0, min(current_idx, n - 1))
    i0 = max(0, current_idx - search_back)
    i1 = min(n, current_idx + search_forward)

    if i0 >= i1:
        return current_idx, 0.0

    window = xy[i0:i1]
    dists = np.linalg.norm(window - pos, axis=1)
    local_idx = int(np.argmin(dists))

    return i0 + local_idx, float(dists[local_idx])


def find_lookahead_point(
    xy: np.ndarray,
    pos: np.ndarray,
    start_idx: int,
    lookahead_dist: float,
) -> Tuple[int, np.ndarray]:
    """
    Find lookahead point on path at distance L_d from robot.

    Classic method: find intersection of circle (radius L_d, centered at robot)
    with path segments. Falls back to arc-length if no intersection.

    Returns:
        (index, point_xy)
    """
    n = len(xy)
    if start_idx >= n - 1:
        return n - 1, xy[-1]

    L_d_sq = lookahead_dist ** 2

    # Search forward for segment intersecting the lookahead circle
    for i in range(start_idx, n - 1):
        p1 = xy[i]
        p2 = xy[i + 1]

        # Check if either endpoint is beyond lookahead
        d1_sq = np.sum((p1 - pos) ** 2)
        d2_sq = np.sum((p2 - pos) ** 2)

        # If segment straddles the lookahead circle, interpolate
        if d1_sq <= L_d_sq <= d2_sq:
            # Linear interpolation to find point at exactly L_d
            d1 = np.sqrt(d1_sq)
            d2 = np.sqrt(d2_sq)
            t = (lookahead_dist - d1) / max(d2 - d1, 1e-6)
            t = np.clip(t, 0, 1)
            point = p1 + t * (p2 - p1)
            return i + 1, point

        # If we've passed the lookahead distance, use this point
        if d2_sq > L_d_sq:
            return i + 1, p2

    # Fallback: return last point
    return n - 1, xy[-1]


def compute_pure_pursuit_curvature(
    pos: np.ndarray,
    yaw: float,
    target: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Classic Pure Pursuit: compute curvature to reach lookahead point.

    The robot follows a circular arc from its current position to the target.

    Args:
        pos: Robot position (x, y).
        yaw: Robot heading (radians).
        target: Lookahead point (x, y).

    Returns:
        (curvature, alpha, lookahead_dist)
        - curvature: κ = 2·sin(α) / L_d
        - alpha: angle to target in robot frame
        - lookahead_dist: actual distance to target
    """
    # Vector to target in world frame
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    L_d = np.sqrt(dx * dx + dy * dy)

    if L_d < 1e-6:
        return 0.0, 0.0, 0.0

    # Angle to target in world frame
    target_angle = np.arctan2(dy, dx)

    # Angle to target in robot frame (alpha)
    alpha = normalize_angle(target_angle - yaw)

    # Pure Pursuit curvature formula: κ = 2·sin(α) / L_d
    curvature = 2.0 * np.sin(alpha) / L_d

    return float(curvature), float(alpha), float(L_d)


def compute_steering_commands(
    curvature: float,
    speed: float,
    max_yaw_rate: float,
    wheelbase: Optional[float] = None,
) -> Tuple[float, Optional[float]]:
    """
    Convert curvature to steering commands.

    Args:
        curvature: Path curvature κ (1/m).
        speed: Linear velocity (m/s).
        max_yaw_rate: Maximum angular velocity (rad/s).
        wheelbase: If provided, also compute Ackermann steering angle.

    Returns:
        (yaw_rate, steering_angle)
        - yaw_rate: ω = v · κ (for diff-drive / unicycle)
        - steering_angle: δ = arctan(L · κ) (for Ackermann, or None)
    """
    # Differential drive / unicycle: ω = v · κ
    yaw_rate = speed * curvature
    yaw_rate = float(np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate))

    # Ackermann steering angle (optional)
    steering_angle = None
    if wheelbase is not None and wheelbase > 0:
        steering_angle = float(np.arctan(wheelbase * curvature))

    return yaw_rate, steering_angle


def compute_body_velocity(
    speed: float,
    alpha: float,
    holonomic: bool = True,
) -> Tuple[float, float]:
    """
    Compute body-frame velocity components.

    Args:
        speed: Desired speed magnitude.
        alpha: Angle to target in robot frame.
        holonomic: If True, allow lateral velocity (omnidirectional).
                   If False, only forward velocity (diff-drive).

    Returns:
        (vx_body, vy_body)
    """
    if holonomic:
        # Omnidirectional: can move directly toward target
        vx = speed * np.cos(alpha)
        vy = speed * np.sin(alpha)
    else:
        # Non-holonomic: only forward motion, steering handles direction
        vx = speed
        vy = 0.0

    return float(vx), float(vy)


def adaptive_lookahead(
    base: float,
    speed: float,
    curvature: float,
    speed_gain: float,
    bounds: Tuple[float, float],
) -> float:
    """
    Adaptive lookahead distance.

    L_d = base + speed × gain, reduced on tight curves.
    """
    lookahead = base + speed_gain * max(0.0, speed)

    # Reduce on tight curves (curvature > 0.5 means radius < 2m)
    if curvature > 0.5:
        lookahead *= 0.7

    return float(np.clip(lookahead, bounds[0], bounds[1]))


def compute_target_speed(
    cruise: float,
    dist_to_goal: float,
    curvature: float,
    slow_down_dist: float,
    curvature_factor: float,
    bounds: Tuple[float, float],
) -> float:
    """
    Speed profile: slows near goal and on curves.
    """
    speed = cruise

    # Slow down near goal
    if slow_down_dist > 0 and dist_to_goal < slow_down_dist:
        ratio = dist_to_goal / slow_down_dist
        speed *= 0.3 + 0.7 * ratio

    # Slow down on curves: v_max ∝ 1/√κ (centripetal acceleration limit)
    # Simplified: 1 / (1 + k × curvature)
    curve_factor = 1.0 / (1.0 + curvature_factor * max(0.0, curvature))
    speed *= curve_factor

    return float(np.clip(speed, bounds[0], bounds[1]))