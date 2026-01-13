"""
Cubic Hermite spline smoothing algorithm.

Produces G1-continuous trajectories from waypoint paths by:
1. Building cubic Hermite splines with computed tangents
2. Parameterizing by arc length for uniform-speed sampling
3. Sampling at fixed time intervals
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline

from sparx_agency.core.common.types import Path2D, TrajectoryPoint, KinematicLimits
from .params import HermiteParams


@dataclass(frozen=True)
class HermiteSolution:
    """Result of Hermite smoothing: discrete trajectory samples."""
    samples: Tuple[TrajectoryPoint, ...]
    total_length: float
    total_time: float


def solve(
    path: Path2D,
    params: HermiteParams,
    limits: Optional[KinematicLimits] = None,
) -> HermiteSolution:
    """
    Smooth path using cubic Hermite splines.

    Args:
        path: Input waypoints with poses.
        params: Algorithm configuration.
        limits: Optional kinematic constraints.

    Returns:
        Smoothed trajectory as discrete samples.

    Raises:
        ValueError: If fewer than 2 distinct waypoints after filtering.
    """
    # Extract and filter waypoints
    pts = np.array([(p.x, p.y) for p in path.points])
    pts = _filter_close_points(pts, params.min_point_spacing)

    if len(pts) < 2:
        raise ValueError("Hermite smoothing requires at least 2 distinct waypoints")

    # Compute spline parameter (cumulative chord length)
    u = _cumulative_chord_length(pts)

    # Compute tangents for G1 continuity
    tangents = _compute_tangents(pts, u, params.tangent_scale)

    # Build Hermite splines
    spline_x = CubicHermiteSpline(u, pts[:, 0], tangents[:, 0])
    spline_y = CubicHermiteSpline(u, pts[:, 1], tangents[:, 1])

    # Build arc-length lookup table
    arc_s, arc_u = _build_arc_length_table(spline_x, spline_y, u, params.arc_lut_samples)
    total_length = float(arc_s[-1])

    # Compute speed and total time
    speed = _select_speed(params, limits)
    total_time = total_length / speed if speed > 0 and total_length > 0 else 0.0

    # Sample trajectory
    samples = _sample_trajectory(
        spline_x, spline_y, arc_s, arc_u,
        total_length, total_time, speed, params.dt,
        params.zero_endpoint_velocity
    )

    return HermiteSolution(
        samples=tuple(samples),
        total_length=total_length,
        total_time=total_time
    )


def _filter_close_points(pts: NDArray, min_spacing: float) -> NDArray:
    """Remove points closer than min_spacing, keeping first and last."""
    if len(pts) <= 2:
        return pts

    kept = [pts[0]]
    for p in pts[1:-1]:
        if np.linalg.norm(p - kept[-1]) >= min_spacing:
            kept.append(p)

    # Always include last point if sufficiently far
    if np.linalg.norm(pts[-1] - kept[-1]) >= min_spacing * 0.5:
        kept.append(pts[-1])
    elif len(kept) > 1:
        kept[-1] = pts[-1]  # Replace last kept with actual endpoint
    else:
        kept.append(pts[-1])

    return np.array(kept)


def _cumulative_chord_length(pts: NDArray) -> NDArray:
    """Compute cumulative chord-length parameterization."""
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.zeros(len(pts))
    u[1:] = np.cumsum(diffs)
    return u


def _compute_tangents(pts: NDArray, u: NDArray, scale: float) -> NDArray:
    """
    Compute tangent vectors for G1 continuity.

    Endpoints: direction to/from neighbor.
    Interior: average of normalized incoming/outgoing directions.
    Magnitude scaled by local segment length.
    """
    n = len(pts)
    tangents = np.zeros_like(pts)

    for i in range(n):
        if i == 0:
            direction = pts[1] - pts[0]
            seg_len = u[1] - u[0]
        elif i == n - 1:
            direction = pts[-1] - pts[-2]
            seg_len = u[-1] - u[-2]
        else:
            incoming = pts[i] - pts[i-1]
            outgoing = pts[i+1] - pts[i]

            # Normalize and average
            in_norm = np.linalg.norm(incoming)
            out_norm = np.linalg.norm(outgoing)
            if in_norm > 1e-9:
                incoming = incoming / in_norm
            if out_norm > 1e-9:
                outgoing = outgoing / out_norm

            direction = incoming + outgoing
            seg_len = (u[i+1] - u[i-1]) / 2

        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0])

        tangents[i] = direction * seg_len * scale

    return tangents


def _build_arc_length_table(
    spline_x: CubicHermiteSpline,
    spline_y: CubicHermiteSpline,
    u: NDArray,
    num_samples: int
) -> Tuple[NDArray, NDArray]:
    """Build lookup table mapping arc length to spline parameter."""
    u_samples = np.linspace(u[0], u[-1], num_samples)
    x_vals = spline_x(u_samples)
    y_vals = spline_y(u_samples)

    ds = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
    arc_s = np.zeros(len(u_samples))
    arc_s[1:] = np.cumsum(ds)

    return arc_s, u_samples


def _select_speed(params: HermiteParams, limits: Optional[KinematicLimits]) -> float:
    """Select cruise speed respecting limits."""
    if limits is None:
        return params.nominal_speed_xy
    return min(limits.max_speed_xy, params.nominal_speed_xy)


def _sample_trajectory(
    spline_x: CubicHermiteSpline,
    spline_y: CubicHermiteSpline,
    arc_s: NDArray,
    arc_u: NDArray,
    total_length: float,
    total_time: float,
    speed: float,
    dt: float,
    zero_endpoints: bool
) -> List[TrajectoryPoint]:
    """Sample trajectory at fixed time intervals."""
    if total_time <= 0 or total_length <= 0:
        # Degenerate case
        return [
            TrajectoryPoint(t=0.0, x=float(spline_x(arc_u[0])), y=float(spline_y(arc_u[0]))),
            TrajectoryPoint(t=0.0, x=float(spline_x(arc_u[-1])), y=float(spline_y(arc_u[-1])))
        ]

    n_samples = int(total_time / dt) + 1
    times = np.linspace(0, total_time, n_samples)

    samples = []
    for t in times:
        s = (t / total_time) * total_length
        u = float(np.interp(np.clip(s, 0, total_length), arc_s, arc_u))

        x = float(spline_x(u))
        y = float(spline_y(u))

        # First derivative → tangent direction
        dx_du = float(spline_x(u, 1))
        dy_du = float(spline_y(u, 1))
        speed_du = math.hypot(dx_du, dy_du)

        if speed_du > 1e-9:
            tx, ty = dx_du / speed_du, dy_du / speed_du
        else:
            tx, ty = 1.0, 0.0

        vx, vy = tx * speed, ty * speed

        # Second derivative → curvature
        ddx = float(spline_x(u, 2))
        ddy = float(spline_y(u, 2))
        curvature = abs(dx_du * ddy - dy_du * ddx) / (speed_du ** 3) if speed_du > 1e-9 else 0.0

        yaw = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-9 else 0.0

        samples.append(TrajectoryPoint(
            t=float(t), x=x, y=y, z=0.0,
            vx=vx, vy=vy, vz=0.0,
            s=s, curvature=curvature, yaw=yaw
        ))

    # Enforce zero endpoint velocities if requested
    if zero_endpoints and samples:
        samples[0] = _with_zero_velocity(samples[0])
        samples[-1] = _with_zero_velocity(samples[-1])

    return samples


def _with_zero_velocity(pt: TrajectoryPoint) -> TrajectoryPoint:
    """Return copy of point with zero velocity."""
    return TrajectoryPoint(
        t=pt.t, x=pt.x, y=pt.y, z=pt.z,
        vx=0.0, vy=0.0, vz=0.0,
        ax=pt.ax, ay=pt.ay, az=pt.az,
        yaw=pt.yaw, yaw_rate=pt.yaw_rate,
        s=pt.s, curvature=pt.curvature
    )