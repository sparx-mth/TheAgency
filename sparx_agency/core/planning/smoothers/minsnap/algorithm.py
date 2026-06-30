"""
Minimum-snap trajectory generation.

Produces smooth polynomial trajectories that minimize snap (4th derivative),
ideal for quadrotors and other dynamically agile robots.

Uses the minsnap_trajectories library for polynomial optimization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
try:
    from numpy.typing import NDArray
except ImportError:  # numpy < 1.20 (e.g. ROS Noetic system numpy lacks numpy.typing)
    from typing import Any as NDArray  # used only in (stringized) annotations

from sparx_agency.core.common.types import Path2D, TrajectoryPoint, KinematicLimits
from .params import MinSnapParams

try:
    import minsnap_trajectories as ms
    _MS_AVAILABLE = True
except ImportError as e:
    ms = None  # type: ignore
    _MS_AVAILABLE = False
    _MS_IMPORT_ERROR = e


@dataclass(frozen=True)
class MinSnapSolution:
    """Result of minimum-snap trajectory generation."""
    samples: Tuple[TrajectoryPoint, ...]
    total_length: float
    total_time: float


def solve(
    path: Path2D,
    params: MinSnapParams,
    limits: Optional[KinematicLimits] = None,
) -> MinSnapSolution:
    """
    Generate minimum-snap trajectory from path.

    Args:
        path: Input waypoints.
        params: Algorithm configuration.
        limits: Optional kinematic constraints.

    Returns:
        Discrete trajectory samples.

    Raises:
        ImportError: If minsnap_trajectories is not installed.
        ValueError: If fewer than 2 waypoints after filtering.
    """
    if not _MS_AVAILABLE:
        raise ImportError(
            "minsnap_trajectories is required for MinSnapSmoother. "
            "Install with: pip install minsnap-trajectories"
        ) from _MS_IMPORT_ERROR

    # Extract and filter waypoints
    pts = np.array([(p.x, p.y, 0.0) for p in path.points])
    pts = _filter_close_points(pts, params.min_point_spacing)

    if len(pts) < 2:
        raise ValueError("MinSnap requires at least 2 distinct waypoints")

    # Get velocity/acceleration limits
    max_v, max_a = _get_limits(params, limits)

    # Allocate segment times
    times = _allocate_times(pts, max_v, max_a, params)

    # Build minsnap waypoints
    waypoints = _build_waypoints(pts, times, params.zero_endpoint_velocity)

    # Generate trajectory
    traj = ms.generate_trajectory(
        waypoints,
        degree=params.degree,
        idx_minimized_orders=params.idx_minimized_orders,
        num_continuous_orders=params.num_continuous_orders,
        algorithm=params.algorithm,
    )

    # Sample trajectory
    total_time = float(traj.time_reference[-1])
    samples = _sample_trajectory(traj, total_time, params.dt)

    total_length = _compute_arc_length(samples)

    return MinSnapSolution(
        samples=tuple(samples),
        total_length=total_length,
        total_time=total_time,
    )


def _filter_close_points(pts: NDArray, min_spacing: float) -> NDArray:
    """Remove consecutive points closer than min_spacing."""
    if len(pts) <= 2:
        return pts

    kept = [pts[0]]
    for p in pts[1:-1]:
        if np.linalg.norm(p - kept[-1]) >= min_spacing:
            kept.append(p)

    # Always include endpoint
    if np.linalg.norm(pts[-1] - kept[-1]) >= min_spacing * 0.5:
        kept.append(pts[-1])
    else:
        kept[-1] = pts[-1]

    return np.array(kept)


def _get_limits(
    params: MinSnapParams,
    limits: Optional[KinematicLimits]
) -> Tuple[float, float]:
    """Extract velocity and acceleration limits."""
    if limits is None:
        return params.nominal_speed_xy, 1.0

    max_v = min(limits.max_speed_xy, params.nominal_speed_xy)
    max_a = limits.max_accel_xy if limits.max_accel_xy else 1.0

    return max_v, max_a


def _allocate_times(
    pts: NDArray,
    max_v: float,
    max_a: float,
    params: MinSnapParams,
) -> List[float]:
    """
    Allocate cumulative waypoint times using trapezoidal motion estimate.

    Uses triangular profile for short segments, trapezoidal for longer ones.
    """
    times = [0.0]
    v_eff = max_v * params.v_eff_ratio

    for i in range(1, len(pts)):
        dist = float(np.linalg.norm(pts[i] - pts[i-1]))

        if dist < 0.01:
            seg_time = params.min_segment_time
        else:
            # Time to accelerate to v_eff
            t_accel = v_eff / max_a
            d_accel = 0.5 * max_a * t_accel ** 2

            if dist < 2 * d_accel:
                # Triangular profile (can't reach full speed)
                seg_time = 2 * math.sqrt(dist / max_a)
            else:
                # Trapezoidal profile
                seg_time = 2 * t_accel + (dist - 2 * d_accel) / v_eff

            seg_time = max(seg_time * params.segment_time_scale, params.min_segment_time)

        times.append(times[-1] + seg_time)

    return times


def _build_waypoints(
    pts: NDArray,
    times: List[float],
    zero_endpoints: bool,
) -> List:
    """Build minsnap Waypoint objects."""
    waypoints = []

    for i, (pt, t) in enumerate(zip(pts, times)):
        is_endpoint = (i == 0 or i == len(pts) - 1)

        if zero_endpoints and is_endpoint:
            waypoints.append(ms.Waypoint(
                time=t,
                position=pt,
                velocity=np.zeros(3)
            ))
        else:
            waypoints.append(ms.Waypoint(time=t, position=pt))

    return waypoints


def _sample_trajectory(traj, total_time: float, dt: float) -> List[TrajectoryPoint]:
    """Sample trajectory at uniform time intervals."""
    t_samples = np.arange(0.0, total_time + dt/2, dt)
    if t_samples[-1] < total_time - 1e-6:
        t_samples = np.append(t_samples, total_time)

    # Get position, velocity, acceleration
    pva = ms.compute_trajectory_derivatives(traj, t_samples, order=3)
    pos, vel, acc = pva[0], pva[1], pva[2]

    samples = []
    prev_yaw: Optional[float] = None
    prev_t: Optional[float] = None

    for i, t in enumerate(t_samples):
        vx, vy = float(vel[i, 0]), float(vel[i, 1])

        # Compute yaw and yaw rate from velocity
        yaw: Optional[float] = None
        yaw_rate: Optional[float] = None

        speed_xy = math.hypot(vx, vy)
        if speed_xy > 1e-6:
            yaw = math.atan2(vy, vx)

            if prev_yaw is not None and prev_t is not None:
                dt_actual = float(t) - prev_t
                if dt_actual > 1e-9:
                    # Normalize angle difference to [-π, π]
                    dyaw = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
                    yaw_rate = dyaw / dt_actual

            prev_yaw = yaw
            prev_t = float(t)

        # Compute curvature from velocity and acceleration
        curvature = _compute_curvature(vel[i], acc[i])

        samples.append(TrajectoryPoint(
            t=float(t),
            x=float(pos[i, 0]),
            y=float(pos[i, 1]),
            z=float(pos[i, 2]),
            vx=vx,
            vy=vy,
            vz=float(vel[i, 2]),
            ax=float(acc[i, 0]),
            ay=float(acc[i, 1]),
            az=float(acc[i, 2]),
            yaw=yaw,
            yaw_rate=yaw_rate,
            curvature=curvature,
        ))

    # Compute arc lengths
    _add_arc_lengths(samples)

    return samples


def _compute_curvature(v: NDArray, a: NDArray) -> float:
    """Compute curvature as |v × a| / |v|³."""
    speed = float(np.linalg.norm(v))
    if speed < 1e-6:
        return 0.0

    cross = np.cross(v, a)
    return float(np.linalg.norm(cross)) / (speed ** 3)


def _add_arc_lengths(samples: List[TrajectoryPoint]) -> None:
    """Add cumulative arc length to samples (modifies in place via replacement)."""
    if not samples:
        return

    s = 0.0
    for i in range(len(samples)):
        if i > 0:
            dx = samples[i].x - samples[i-1].x
            dy = samples[i].y - samples[i-1].y
            dz = samples[i].z - samples[i-1].z
            s += math.sqrt(dx*dx + dy*dy + dz*dz)

        # Replace sample with updated s value
        old = samples[i]
        samples[i] = TrajectoryPoint(
            t=old.t, x=old.x, y=old.y, z=old.z,
            vx=old.vx, vy=old.vy, vz=old.vz,
            ax=old.ax, ay=old.ay, az=old.az,
            yaw=old.yaw, yaw_rate=old.yaw_rate,
            s=s, curvature=old.curvature,
        )


def _compute_arc_length(samples: List[TrajectoryPoint]) -> float:
    """Get total arc length from samples."""
    if not samples:
        return 0.0
    return samples[-1].s or 0.0