"""
Cubic Hermite spline smoothing algorithm (2D and 3D).

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
try:
    from numpy.typing import NDArray
except ImportError:  # numpy < 1.20 (e.g. ROS Noetic system numpy lacks numpy.typing)
    from typing import Any as NDArray  # used only in (stringized) annotations

from sparx_agency.core.common.types import Path2D, TrajectoryPoint, KinematicLimits
from .params import HermiteParams, HermiteParams3D

# Try importing Path3D, define fallback if not available
try:
    from sparx_agency.core.common.types import Path3D
except ImportError:
    from dataclasses import field
    from typing import Any, Dict
    from sparx_agency.core.common.types import Pose3D

    @dataclass(frozen=True)
    class Path3D:
        points: Tuple[Pose3D, ...]
        frame_id: str = "map"
        metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Solution types
# =============================================================================

@dataclass(frozen=True)
class HermiteSolution:
    """Result of Hermite smoothing: discrete trajectory samples."""
    samples: Tuple[TrajectoryPoint, ...]
    total_length: float
    total_time: float


# =============================================================================
# Cubic Hermite spline (numpy-only; no scipy dependency)
# =============================================================================

class _CubicHermiteSpline:
    """Numpy-only cubic Hermite spline — a drop-in for the subset of
    ``scipy.interpolate.CubicHermiteSpline`` this module uses.

    Construct with strictly-increasing knots ``x``, values ``y`` and slopes
    ``dydx`` (dy/dx at the knots); call ``spline(xq)`` for the value and
    ``spline(xq, 1)`` / ``spline(xq, 2)`` for the 1st / 2nd derivative. ``xq`` may
    be a scalar or a 1-D array (queries are assumed within ``[x[0], x[-1]]``).

    Per segment this is the SAME unique cubic scipy constructs (matching endpoint
    value and first derivative), so results agree with scipy to float precision.
    It exists so ``core`` imports under ROS Noetic, whose system Python ships no
    scipy.
    """

    def __init__(self, x: NDArray, y: NDArray, dydx: NDArray) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.dydx = np.asarray(dydx, dtype=float)
        if self.x.ndim != 1 or self.x.shape[0] < 2:
            raise ValueError("CubicHermiteSpline needs >= 2 knots")
        self.h = np.diff(self.x)
        if np.any(self.h <= 0.0):
            raise ValueError("CubicHermiteSpline knots must be strictly increasing")

    def __call__(self, xq, nu: int = 0):
        q = np.asarray(xq, dtype=float)
        scalar = q.ndim == 0
        q = np.atleast_1d(q)
        # Segment index per query, clamped into the valid knot range.
        idx = np.searchsorted(self.x, q, side="right") - 1
        idx = np.clip(idx, 0, self.x.shape[0] - 2)
        hi = self.h[idx]
        t = (q - self.x[idx]) / hi
        p0, p1 = self.y[idx], self.y[idx + 1]
        # Endpoint tangents in t-space are h * (dy/dx).
        m0, m1 = self.dydx[idx] * hi, self.dydx[idx + 1] * hi
        if nu == 0:
            t2 = t * t
            t3 = t2 * t
            val = ((2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0
                   + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1)
        elif nu == 1:
            t2 = t * t
            val = ((6 * t2 - 6 * t) * p0 + (3 * t2 - 4 * t + 1) * m0
                   + (-6 * t2 + 6 * t) * p1 + (3 * t2 - 2 * t) * m1) / hi
        elif nu == 2:
            val = ((12 * t - 6) * p0 + (6 * t - 4) * m0
                   + (-12 * t + 6) * p1 + (6 * t - 2) * m1) / (hi * hi)
        else:
            raise ValueError("nu must be 0, 1 or 2")
        return float(val[0]) if scalar else val


# Name the rest of the module constructs/annotates by (drop-in replacement).
CubicHermiteSpline = _CubicHermiteSpline


# =============================================================================
# Shared utilities
# =============================================================================

def _filter_close_points(pts: NDArray, min_spacing: float) -> NDArray:
    """Remove points closer than min_spacing, keeping first and last."""
    if len(pts) <= 2:
        return pts

    kept = [pts[0]]
    for p in pts[1:-1]:
        if np.linalg.norm(p - kept[-1]) >= min_spacing:
            kept.append(p)

    if np.linalg.norm(pts[-1] - kept[-1]) >= min_spacing * 0.5:
        kept.append(pts[-1])
    elif len(kept) > 1:
        kept[-1] = pts[-1]
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
    """Compute tangent vectors for G1 continuity (works for 2D and 3D)."""
    n = len(pts)
    dim = pts.shape[1]
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

            in_norm = np.linalg.norm(incoming)
            out_norm = np.linalg.norm(outgoing)
            if in_norm > 1e-9:
                incoming = incoming / in_norm
            if out_norm > 1e-9:
                outgoing = outgoing / out_norm

            direction = incoming + outgoing
            seg_len = (u[i+1] - u[i-1]) / 2

        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            direction = direction / norm
        else:
            direction = np.zeros(dim)
            direction[0] = 1.0

        tangents[i] = direction * seg_len * scale

    return tangents


def _with_zero_velocity(pt: TrajectoryPoint) -> TrajectoryPoint:
    """Return copy of point with zero velocity."""
    return TrajectoryPoint(
        t=pt.t, x=pt.x, y=pt.y, z=pt.z,
        vx=0.0, vy=0.0, vz=0.0,
        ax=pt.ax, ay=pt.ay, az=pt.az,
        yaw=pt.yaw, yaw_rate=pt.yaw_rate,
        s=pt.s, curvature=pt.curvature
    )


# =============================================================================
# 2D Hermite (unchanged)
# =============================================================================

def _build_arc_length_table_2d(
    spline_x: CubicHermiteSpline,
    spline_y: CubicHermiteSpline,
    u: NDArray,
    num_samples: int
) -> Tuple[NDArray, NDArray]:
    """Build lookup table mapping arc length to spline parameter (2D)."""
    u_samples = np.linspace(u[0], u[-1], num_samples)
    x_vals = spline_x(u_samples)
    y_vals = spline_y(u_samples)

    ds = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
    arc_s = np.zeros(len(u_samples))
    arc_s[1:] = np.cumsum(ds)

    return arc_s, u_samples


def _select_speed_2d(params: HermiteParams, limits: Optional[KinematicLimits]) -> float:
    """Select cruise speed respecting limits (2D)."""
    if limits is None:
        return params.nominal_speed_xy
    return min(limits.max_speed_xy, params.nominal_speed_xy)


def _sample_trajectory_2d(
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
    """Sample 2D trajectory at fixed time intervals."""
    if total_time <= 0 or total_length <= 0:
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

        dx_du = float(spline_x(u, 1))
        dy_du = float(spline_y(u, 1))
        speed_du = math.hypot(dx_du, dy_du)

        if speed_du > 1e-9:
            tx, ty = dx_du / speed_du, dy_du / speed_du
        else:
            tx, ty = 1.0, 0.0

        vx, vy = tx * speed, ty * speed

        ddx = float(spline_x(u, 2))
        ddy = float(spline_y(u, 2))
        curvature = abs(dx_du * ddy - dy_du * ddx) / (speed_du ** 3) if speed_du > 1e-9 else 0.0

        yaw = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-9 else 0.0

        samples.append(TrajectoryPoint(
            t=float(t), x=x, y=y, z=0.0,
            vx=vx, vy=vy, vz=0.0,
            s=s, curvature=curvature, yaw=yaw
        ))

    if zero_endpoints and samples:
        samples[0] = _with_zero_velocity(samples[0])
        samples[-1] = _with_zero_velocity(samples[-1])

    return samples


def solve(
    path: Path2D,
    params: HermiteParams,
    limits: Optional[KinematicLimits] = None,
) -> HermiteSolution:
    """Smooth 2D path using cubic Hermite splines."""
    pts = np.array([(p.x, p.y) for p in path.points])
    pts = _filter_close_points(pts, params.min_point_spacing)

    if len(pts) < 2:
        raise ValueError("Hermite smoothing requires at least 2 distinct waypoints")

    u = _cumulative_chord_length(pts)
    tangents = _compute_tangents(pts, u, params.tangent_scale)

    spline_x = CubicHermiteSpline(u, pts[:, 0], tangents[:, 0])
    spline_y = CubicHermiteSpline(u, pts[:, 1], tangents[:, 1])

    arc_s, arc_u = _build_arc_length_table_2d(spline_x, spline_y, u, params.arc_lut_samples)
    total_length = float(arc_s[-1])

    speed = _select_speed_2d(params, limits)
    total_time = total_length / speed if speed > 0 and total_length > 0 else 0.0

    samples = _sample_trajectory_2d(
        spline_x, spline_y, arc_s, arc_u,
        total_length, total_time, speed, params.dt,
        params.zero_endpoint_velocity
    )

    return HermiteSolution(samples=tuple(samples), total_length=total_length, total_time=total_time)


# =============================================================================
# 3D Hermite (new)
# =============================================================================

def _build_arc_length_table_3d(
    spline_x: CubicHermiteSpline,
    spline_y: CubicHermiteSpline,
    spline_z: CubicHermiteSpline,
    u: NDArray,
    num_samples: int
) -> Tuple[NDArray, NDArray]:
    """Build lookup table mapping arc length to spline parameter (3D)."""
    u_samples = np.linspace(u[0], u[-1], num_samples)
    x_vals = spline_x(u_samples)
    y_vals = spline_y(u_samples)
    z_vals = spline_z(u_samples)

    ds = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2 + np.diff(z_vals)**2)
    arc_s = np.zeros(len(u_samples))
    arc_s[1:] = np.cumsum(ds)

    return arc_s, u_samples


def _select_speed_3d(params: HermiteParams3D, limits: Optional[KinematicLimits]) -> float:
    """Select cruise speed respecting limits (3D)."""
    if limits is None:
        return params.nominal_speed_xy
    return min(limits.max_speed_xy, params.nominal_speed_xy)


def _sample_trajectory_3d(
    spline_x: CubicHermiteSpline,
    spline_y: CubicHermiteSpline,
    spline_z: CubicHermiteSpline,
    arc_s: NDArray,
    arc_u: NDArray,
    total_length: float,
    total_time: float,
    speed: float,
    dt: float,
    zero_endpoints: bool
) -> List[TrajectoryPoint]:
    """Sample 3D trajectory at fixed time intervals."""
    if total_time <= 0 or total_length <= 0:
        return [
            TrajectoryPoint(t=0.0, x=float(spline_x(arc_u[0])), y=float(spline_y(arc_u[0])), z=float(spline_z(arc_u[0]))),
            TrajectoryPoint(t=0.0, x=float(spline_x(arc_u[-1])), y=float(spline_y(arc_u[-1])), z=float(spline_z(arc_u[-1])))
        ]

    n_samples = int(total_time / dt) + 1
    times = np.linspace(0, total_time, n_samples)

    samples = []
    for t in times:
        s = (t / total_time) * total_length
        u = float(np.interp(np.clip(s, 0, total_length), arc_s, arc_u))

        x = float(spline_x(u))
        y = float(spline_y(u))
        z = float(spline_z(u))

        # First derivatives
        dx_du = float(spline_x(u, 1))
        dy_du = float(spline_y(u, 1))
        dz_du = float(spline_z(u, 1))
        speed_du = math.sqrt(dx_du**2 + dy_du**2 + dz_du**2)

        if speed_du > 1e-9:
            tx, ty, tz = dx_du / speed_du, dy_du / speed_du, dz_du / speed_du
        else:
            tx, ty, tz = 1.0, 0.0, 0.0

        vx, vy, vz = tx * speed, ty * speed, tz * speed

        # 3D curvature: |r' × r''| / |r'|³
        ddx = float(spline_x(u, 2))
        ddy = float(spline_y(u, 2))
        ddz = float(spline_z(u, 2))

        # Cross product r' × r''
        cross_x = dy_du * ddz - dz_du * ddy
        cross_y = dz_du * ddx - dx_du * ddz
        cross_z = dx_du * ddy - dy_du * ddx
        cross_mag = math.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

        curvature = cross_mag / (speed_du ** 3) if speed_du > 1e-9 else 0.0

        # Yaw from xy velocity
        yaw = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-9 else 0.0

        samples.append(TrajectoryPoint(
            t=float(t), x=x, y=y, z=z,
            vx=vx, vy=vy, vz=vz,
            s=s, curvature=curvature, yaw=yaw
        ))

    if zero_endpoints and samples:
        samples[0] = _with_zero_velocity(samples[0])
        samples[-1] = _with_zero_velocity(samples[-1])

    return samples


def solve_3d(
    path: Path3D,
    params: HermiteParams3D,
    limits: Optional[KinematicLimits] = None,
) -> HermiteSolution:
    """Smooth 3D path using cubic Hermite splines."""
    pts = np.array([(p.x, p.y, p.z) for p in path.points])
    pts = _filter_close_points(pts, params.min_point_spacing)

    if len(pts) < 2:
        raise ValueError("Hermite smoothing requires at least 2 distinct waypoints")

    u = _cumulative_chord_length(pts)
    tangents = _compute_tangents(pts, u, params.tangent_scale)

    spline_x = CubicHermiteSpline(u, pts[:, 0], tangents[:, 0])
    spline_y = CubicHermiteSpline(u, pts[:, 1], tangents[:, 1])
    spline_z = CubicHermiteSpline(u, pts[:, 2], tangents[:, 2])

    arc_s, arc_u = _build_arc_length_table_3d(spline_x, spline_y, spline_z, u, params.arc_lut_samples)
    total_length = float(arc_s[-1])

    speed = _select_speed_3d(params, limits)
    total_time = total_length / speed if speed > 0 and total_length > 0 else 0.0

    samples = _sample_trajectory_3d(
        spline_x, spline_y, spline_z, arc_s, arc_u,
        total_length, total_time, speed, params.dt,
        params.zero_endpoint_velocity
    )

    return HermiteSolution(samples=tuple(samples), total_length=total_length, total_time=total_time)