from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from sparx_agency.core.common.types.geometry import normalize_angle
from sparx_agency.core.common.types.planning import TrajectoryPoint


@dataclass(frozen=True, slots=True)
class TrackingDebug:
    """
    Optional debug outputs you might want to log upstream.
    """
    closest_index: int
    lookahead_index: int
    cross_track_error: float
    dist_to_goal: float
    curvature: float
    lookahead_m: float
    speed_target: float
    speed_cmd: float
    desired_yaw: float


def world_to_body_velocity(vx_world: float, vy_world: float, yaw: float) -> Tuple[float, float]:
    """Rotate a world-frame planar velocity into body frame."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx_body = vx_world * c + vy_world * s
    vy_body = -vx_world * s + vy_world * c
    return vx_body, vy_body


def _dist2(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    return dx * dx + dy * dy


def distance_to_goal(points: Sequence[TrajectoryPoint], x: float, y: float) -> float:
    """Distance to final point in the sampled trajectory."""
    if not points:
        return float("inf")
    g = points[-1]
    return math.hypot(float(g.x) - x, float(g.y) - y)


def get_curvature(p: TrajectoryPoint) -> float:
    """Curvature accessor with safe default."""
    return float(p.curvature) if p.curvature is not None else 0.0


def find_closest_index(
    points: Sequence[TrajectoryPoint],
    x: float,
    y: float,
    current_index: int,
    search_back: int,
    search_forward: int,
) -> Tuple[int, float]:
    """
    Find closest point index within a limited window around current_index.
    Returns (closest_index, cross_track_error_m).
    """
    if not points:
        raise ValueError("Trajectory has no points")

    n = len(points)
    current_index = max(0, min(current_index, n - 1))
    i0 = max(0, current_index - max(0, search_back))
    i1 = min(n - 1, current_index + max(0, search_forward))

    best_i = current_index
    best_d2 = float("inf")

    for i in range(i0, i1 + 1):
        p = points[i]
        d2 = _dist2(x, y, float(p.x), float(p.y))
        if d2 < best_d2:
            best_d2 = d2
            best_i = i

    return best_i, math.sqrt(best_d2)


def compute_lookahead(
    base_lookahead: float,
    min_lookahead: float,
    max_lookahead: float,
    lookahead_speed_gain: float,
    current_speed: float,
    curvature: float,
) -> float:
    """
    Adaptive lookahead:
      lookahead = base + k*speed
      reduce on tight curves (simple heuristic)
    """
    lookahead = base_lookahead + lookahead_speed_gain * max(0.0, current_speed)
    if curvature > 0.5:
        lookahead *= 0.7
    return max(min_lookahead, min(lookahead, max_lookahead))


def compute_speed(
    cruise_speed: float,
    min_speed: float,
    max_speed: float,
    slow_down_distance: float,
    curvature_speed_factor: float,
    dist_to_goal: float,
    curvature: float,
    clearance_factor: float,
) -> float:
    """
    Speed policy:
    - slow down near goal
    - slow down on curves using 1/(1+k*curvature)
    - multiply by clearance factor (0..1)
    """
    speed = cruise_speed

    if slow_down_distance > 1e-6 and dist_to_goal < slow_down_distance:
        alpha = max(0.0, min(1.0, dist_to_goal / slow_down_distance))
        speed *= (0.3 + 0.7 * alpha)

    curve_factor = 1.0 / (1.0 + curvature_speed_factor * max(0.0, curvature))
    speed *= curve_factor
    speed *= max(0.0, min(1.0, clearance_factor))

    return max(min_speed, min(speed, max_speed))


def compute_yaw_rate(
    current_yaw: float,
    desired_yaw: float,
    current_speed: float,
    yaw_kp: float,
    max_yaw_rate: float,
    yaw_deadband: float,
    yaw_speed_threshold: float,
    yaw_rate_smoothing: float,
    prev_yaw_rate: float,
) -> float:
    """
    Smoothed yaw-rate control:
    - no yaw correction at low speeds
    - deadband to avoid jitter
    - exponential smoothing on output
    """
    if current_speed < yaw_speed_threshold:
        return 0.8 * prev_yaw_rate

    yaw_error = normalize_angle(desired_yaw - current_yaw)
    if abs(yaw_error) < yaw_deadband:
        return 0.7 * prev_yaw_rate

    target = yaw_kp * yaw_error
    target = max(-max_yaw_rate, min(target, max_yaw_rate))

    a = max(0.0, min(1.0, yaw_rate_smoothing))
    return a * target + (1.0 - a) * prev_yaw_rate


def pick_lookahead_index(
    points: Sequence[TrajectoryPoint],
    closest_index: int,
    lookahead_m: float,
) -> int:
    """
    Choose a lookahead point by accumulating distances forward from closest_index.
    """
    n = len(points)
    if n == 0:
        raise ValueError("Trajectory has no points")

    closest_index = max(0, min(closest_index, n - 1))
    if closest_index >= n - 1:
        return n - 1

    accum = 0.0
    i = closest_index
    while i < n - 1 and accum < lookahead_m:
        p0 = points[i]
        p1 = points[i + 1]
        ds = math.hypot(float(p1.x) - float(p0.x), float(p1.y) - float(p0.y))
        accum += ds
        i += 1

    return min(i, n - 1)


def clearance_to_factor(
    clearance_m: float,
    min_clearance_for_full_speed: float,
    min_clearance_threshold: float,
) -> float:
    """
    Map clearance (meters) -> [0.3..1.0] factor.
    """
    if clearance_m >= min_clearance_for_full_speed:
        return 1.0
    if clearance_m <= min_clearance_threshold:
        return 0.3

    denom = (min_clearance_for_full_speed - min_clearance_threshold)
    if denom <= 1e-9:
        return 1.0

    ratio = (clearance_m - min_clearance_threshold) / denom
    ratio = max(0.0, min(1.0, ratio))
    return 0.3 + 0.7 * ratio
