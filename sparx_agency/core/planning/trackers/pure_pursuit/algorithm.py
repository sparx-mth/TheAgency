"""
Pure Pursuit path tracking algorithm (2D and 3D).

Based on Coulter (1992), CMU-RI-TR-92-01.

2D: Classic curvature-based steering (κ = 2·sin(α) / L_d)
3D: Direct velocity toward lookahead point (holonomic)
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from sparx_agency.core.common.types import TrajectoryPoint
from sparx_agency.core.common.types.geometry import normalize_angle


# =============================================================================
# Shared utilities
# =============================================================================

def get_curvature(point: TrajectoryPoint) -> float:
    """Extract curvature from trajectory point (default 0)."""
    return float(point.curvature) if point.curvature is not None else 0.0


def adaptive_lookahead(
    base: float,
    speed: float,
    curvature: float,
    speed_gain: float,
    curvature_gain: float,
    bounds: Tuple[float, float],
) -> float:
    """
    Adaptive lookahead distance based on speed and curvature.

    L_d = (base + speed × speed_gain) / (1 + curvature_gain × |κ|)
    """
    lookahead = base + speed_gain * max(0.0, speed)
    curve_factor = 1.0 / (1.0 + curvature_gain * abs(curvature))
    lookahead *= curve_factor
    return float(np.clip(lookahead, bounds[0], bounds[1]))


def compute_target_speed(
    cruise: float,
    dist_to_goal: float,
    curvature: float,
    slow_down_dist: float,
    curvature_factor: float,
    bounds: Tuple[float, float],
) -> float:
    """Speed profile based on distance to goal and path curvature."""
    speed = cruise

    if slow_down_dist > 0 and dist_to_goal < slow_down_dist:
        ratio = dist_to_goal / slow_down_dist
        speed *= 0.3 + 0.7 * ratio

    curve_factor = 1.0 / (1.0 + curvature_factor * abs(curvature))
    speed *= curve_factor

    return float(np.clip(speed, bounds[0], bounds[1]))


# =============================================================================
# 2D Pure Pursuit
# =============================================================================

def trajectory_to_xy(points: List[TrajectoryPoint]) -> np.ndarray:
    """Extract (N, 2) position array from trajectory points."""
    return np.array([[p.x, p.y] for p in points], dtype=np.float64)


def find_closest_index(
    xy: np.ndarray,
    pos: np.ndarray,
    current_idx: int,
    search_back: int,
    search_forward: int,
) -> Tuple[int, float]:
    """Find closest point index within a window (2D)."""
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
    """Find lookahead point on path at distance L_d from robot (2D)."""
    n = len(xy)
    if start_idx >= n - 1:
        return n - 1, xy[-1]

    L_d_sq = lookahead_dist ** 2

    for i in range(start_idx, n - 1):
        p1, p2 = xy[i], xy[i + 1]
        d1_sq = np.sum((p1 - pos) ** 2)
        d2_sq = np.sum((p2 - pos) ** 2)

        if d1_sq <= L_d_sq <= d2_sq:
            d1, d2 = np.sqrt(d1_sq), np.sqrt(d2_sq)
            t = np.clip((lookahead_dist - d1) / max(d2 - d1, 1e-6), 0, 1)
            return i + 1, p1 + t * (p2 - p1)

        if d2_sq > L_d_sq:
            return i + 1, p2

    return n - 1, xy[-1]


def compute_pure_pursuit_curvature(
    pos: np.ndarray,
    yaw: float,
    target: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Classic Pure Pursuit: compute curvature to reach lookahead point (2D).

    Returns: (curvature, alpha, lookahead_dist)
    """
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    L_d = np.sqrt(dx * dx + dy * dy)

    if L_d < 1e-6:
        return 0.0, 0.0, 0.0

    target_angle = np.arctan2(dy, dx)
    alpha = normalize_angle(target_angle - yaw)
    curvature = 2.0 * np.sin(alpha) / L_d

    return float(curvature), float(alpha), float(L_d)


def compute_steering_commands(
    curvature: float,
    speed: float,
    max_yaw_rate: float,
    wheelbase: Optional[float] = None,
) -> Tuple[float, Optional[float]]:
    """Convert curvature to steering commands (2D)."""
    yaw_rate = float(np.clip(speed * curvature, -max_yaw_rate, max_yaw_rate))

    steering_angle = None
    if wheelbase is not None and wheelbase > 0:
        steering_angle = float(np.arctan(wheelbase * curvature))

    return yaw_rate, steering_angle


def compute_body_velocity(
    speed: float,
    alpha: float,
    holonomic: bool = True,
) -> Tuple[float, float]:
    """Compute body-frame velocity components (2D)."""
    if holonomic:
        return float(speed * np.cos(alpha)), float(speed * np.sin(alpha))
    return float(speed), 0.0


# =============================================================================
# 3D Pure Pursuit
# =============================================================================

def trajectory_to_xyz(points: List[TrajectoryPoint]) -> np.ndarray:
    """Extract (N, 3) position array from trajectory points."""
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)


def find_closest_index_3d(
    xyz: np.ndarray,
    pos: np.ndarray,
    current_idx: int,
    search_back: int,
    search_forward: int,
) -> Tuple[int, float]:
    """Find closest point index within a window (3D)."""
    n = len(xyz)
    current_idx = max(0, min(current_idx, n - 1))
    i0 = max(0, current_idx - search_back)
    i1 = min(n, current_idx + search_forward)

    if i0 >= i1:
        return current_idx, 0.0

    window = xyz[i0:i1]
    dists = np.linalg.norm(window - pos, axis=1)
    local_idx = int(np.argmin(dists))

    return i0 + local_idx, float(dists[local_idx])


def find_lookahead_point_3d(
    xyz: np.ndarray,
    pos: np.ndarray,
    start_idx: int,
    lookahead_dist: float,
) -> Tuple[int, np.ndarray]:
    """Find lookahead point on path at distance L_d from robot (3D)."""
    n = len(xyz)
    if start_idx >= n - 1:
        return n - 1, xyz[-1]

    L_d_sq = lookahead_dist ** 2

    for i in range(start_idx, n - 1):
        p1, p2 = xyz[i], xyz[i + 1]
        d1_sq = np.sum((p1 - pos) ** 2)
        d2_sq = np.sum((p2 - pos) ** 2)

        if d1_sq <= L_d_sq <= d2_sq:
            d1, d2 = np.sqrt(d1_sq), np.sqrt(d2_sq)
            t = np.clip((lookahead_dist - d1) / max(d2 - d1, 1e-6), 0, 1)
            return i + 1, p1 + t * (p2 - p1)

        if d2_sq > L_d_sq:
            return i + 1, p2

    return n - 1, xyz[-1]


def compute_velocity_3d(
    pos: np.ndarray,
    target: np.ndarray,
    speed: float,
    max_speed_z: float,
) -> Tuple[float, float, float, float]:
    """
    Compute 3D velocity vector toward lookahead point.

    For holonomic robots (drones), velocity points directly at target.
    Z velocity is clamped separately for safety.

    Args:
        pos: Current position (x, y, z).
        target: Lookahead point (x, y, z).
        speed: Desired total speed (m/s).
        max_speed_z: Maximum vertical speed (m/s).

    Returns:
        (vx, vy, vz, yaw) - velocity components and heading
    """
    delta = target - pos
    dist = np.linalg.norm(delta)

    if dist < 1e-6:
        return 0.0, 0.0, 0.0, 0.0

    # Unit direction
    direction = delta / dist

    # Scale by speed
    vx = direction[0] * speed
    vy = direction[1] * speed
    vz = direction[2] * speed

    # Clamp vertical speed
    vz = float(np.clip(vz, -max_speed_z, max_speed_z))

    # Yaw from xy velocity
    yaw = float(np.arctan2(vy, vx)) if abs(vx) + abs(vy) > 1e-6 else 0.0

    return float(vx), float(vy), float(vz), yaw


def compute_yaw_rate_3d(
    current_yaw: float,
    target_yaw: float,
    max_yaw_rate: float,
    dt: float = 0.05,
) -> float:
    """
    Compute yaw rate to reach target yaw.

    Simple proportional control with rate limiting.
    """
    yaw_error = normalize_angle(target_yaw - current_yaw)
    # P-control with implicit gain (reach target in ~0.5s)
    yaw_rate = yaw_error / 0.5
    return float(np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate))