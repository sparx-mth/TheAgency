"""
MinSnap trajectory generation.

This module is the only place that integrates with the external library
`minsnap_trajectories`. It performs:

1) Waypoint preparation:
   - Extract x/y from Path2D points
   - Optionally filter points that are too close

2) Time allocation:
   - Assign cumulative waypoint times using a trapezoidal/triangular motion
     estimate with safety margins.

3) Trajectory generation:
   - Build `ms.Waypoint` objects and call `ms.generate_trajectory(...)`.

4) Sampling:
   - Sample position/velocity/acceleration on a uniform time grid (dt).
   - Compute auxiliary scalars:
       * s (cumulative arc length)
       * curvature (|v x a| / |v|^3)
       * optional yaw / yaw_rate from velocity direction

The output of this module is a tuple of core `TrajectoryPoint` samples plus a
small debug dictionary (useful for evaluation or visualization elsewhere).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.common.types import Path2D
from core.common.types.planning import TrajectoryPoint

from .params import MinSnapParams

try:
    import minsnap_trajectories as ms
except Exception as e:  # pragma: no cover
    ms = None
    _MINSNAP_IMPORT_ERROR = e
else:
    _MINSNAP_IMPORT_ERROR = None


@dataclass(frozen=True)
class MinSnapRawResult:
    """
    Raw algorithm output.

    `samples` are time-ordered `TrajectoryPoint` objects.
    `debug` contains non-essential metadata (counts, times, effective limits).
    """
    samples: Tuple[TrajectoryPoint, ...]
    debug: Dict[str, Any]


def _filter_points(xs: List[float], ys: List[float], zs: List[float], min_spacing: float) -> Tuple[List[float], List[float], List[float]]:
    """Drop consecutive waypoints closer than `min_spacing`."""
    if not xs:
        return [], [], []
    fx, fy, fz = [xs[0]], [ys[0]], [zs[0]]

    for x, y, z in zip(xs[1:], ys[1:], zs[1:]):
        dist = math.sqrt((x - fx[-1]) ** 2 + (y - fy[-1]) ** 2 + (z - fz[-1]) ** 2)
        if dist >= min_spacing:
            fx.append(x); fy.append(y); fz.append(z)

    # Always include last point (tolerant)
    if len(xs) > 1:
        dist = math.sqrt((xs[-1] - fx[-1]) ** 2 + (ys[-1] - fy[-1]) ** 2 + (zs[-1] - fz[-1]) ** 2)
        if dist >= min_spacing * 0.5:
            fx.append(xs[-1]); fy.append(ys[-1]); fz.append(zs[-1])

    return fx, fy, fz


def _extract_limits(limits: Optional[Any], fallback_v: float) -> Tuple[float, float]:
    """
    Best-effort extraction of (max_v, max_a) from an optional limits object.

    The smoother interface uses an abstract `limits` type, so we support multiple
    common attribute names. If a field is missing, a conservative fallback is used.
    """
    if limits is None:
        return fallback_v, 1.0

    max_v = None
    for k in ("max_velocity", "max_speed", "max_speed_xy"):
        if hasattr(limits, k):
            max_v = float(getattr(limits, k))
            break
    if max_v is None or max_v <= 0:
        max_v = fallback_v

    max_a = None
    for k in ("max_acceleration", "max_acc", "max_acc_xy"):
        if hasattr(limits, k):
            max_a = float(getattr(limits, k))
            break
    if max_a is None or max_a <= 0:
        max_a = 1.0

    return max_v, max_a


def _allocate_times(xs: List[float], ys: List[float], zs: List[float], *, max_v: float, max_a: float, p: MinSnapParams) -> List[float]:
    """
    Allocate cumulative waypoint times with a simple motion estimate.

    The segment time is computed using a triangular/trapezoidal approximation,
    then scaled and clamped to ensure conservative timing.
    """
    times = [0.0]
    v_eff = max_v * p.v_eff_ratio
    a_max = max_a

    for i in range(1, len(xs)):
        dist = math.sqrt((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2 + (zs[i] - zs[i - 1]) ** 2)

        if dist < 0.01:
            times.append(times[-1] + p.min_segment_time)
            continue

        t_accel = v_eff / a_max
        d_accel = 0.5 * a_max * t_accel ** 2

        if dist < 2 * d_accel:
            seg_time = 2 * math.sqrt(dist / a_max)
        else:
            seg_time = 2 * t_accel + (dist - 2 * d_accel) / v_eff

        seg_time = max(seg_time * p.segment_time_scale, p.min_segment_time)
        times.append(times[-1] + seg_time)

    return times


def _curvature(v: np.ndarray, a: np.ndarray) -> float:
    """Curvature via |v x a| / |v|^3 (3D)."""
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])

    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if speed <= 1e-6:
        return 0.0

    cx = vy * az - vz * ay
    cy = vz * ax - vx * az
    cz = vx * ay - vy * ax
    cross_mag = math.sqrt(cx * cx + cy * cy + cz * cz)
    return cross_mag / (speed ** 3)


def _cum_arc_length(pos: np.ndarray) -> np.ndarray:
    """Cumulative arc length for sampled positions shaped (N, 3)."""
    s = np.zeros((pos.shape[0],), dtype=float)
    for i in range(1, pos.shape[0]):
        dp = pos[i] - pos[i - 1]
        s[i] = s[i - 1] + float(np.linalg.norm(dp))
    return s


class MinSnapAlgorithm:
    """Generate discrete minimum-snap samples from a Path2D."""

    def __init__(self, *, params: MinSnapParams) -> None:
        self.params = params

    def solve(
        self,
        path: Path2D,
        *,
        limits: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
        world: Any = None,
    ) -> MinSnapRawResult:
        """
        Build and sample a minimum-snap trajectory.

        Args:
            path: geometric waypoints (2D; z is set to 0)
            limits: optional dynamics limits object
            options: per-call overrides (e.g., dt, min_point_spacing)
            world: reserved for future environment-aware rules

        Returns:
            MinSnapRawResult with `TrajectoryPoint` samples and debug metadata.
        """
        if ms is None:  # pragma: no cover
            raise ImportError("minsnap_trajectories is not installed") from _MINSNAP_IMPORT_ERROR

        opts = dict(options or {})
        dt = float(opts.get("dt", self.params.dt))
        min_spacing = float(opts.get("min_point_spacing", self.params.min_point_spacing))

        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if min_spacing < 0:
            raise ValueError("min_point_spacing must be >= 0")

        xs = [p.x for p in path.points]
        ys = [p.y for p in path.points]
        zs = [0.0 for _ in path.points]

        xs, ys, zs = _filter_points(xs, ys, zs, min_spacing)
        if len(xs) < 2:
            raise ValueError("Need at least 2 distinct waypoints after filtering")

        max_v, max_a = _extract_limits(limits, fallback_v=self.params.nominal_speed_xy)
        times = _allocate_times(xs, ys, zs, max_v=max_v, max_a=max_a, p=self.params)

        refs: List[ms.Waypoint] = []
        for i in range(len(xs)):
            pos = np.array([xs[i], ys[i], zs[i]], dtype=float)
            if self.params.constrain_endpoints_velocity_zero and (i == 0 or i == len(xs) - 1):
                refs.append(ms.Waypoint(time=times[i], position=pos, velocity=np.zeros(3)))
            else:
                refs.append(ms.Waypoint(time=times[i], position=pos))

        traj = ms.generate_trajectory(
            refs,
            degree=self.params.degree,
            idx_minimized_orders=self.params.idx_minimized_orders,
            num_continuous_orders=self.params.num_continuous_orders,
            algorithm=self.params.algorithm,
        )

        total_time = float(traj.time_reference[-1])
        t_samples = np.arange(0.0, total_time + 1e-9, dt, dtype=float)
        if t_samples.size == 0 or t_samples[-1] < total_time - 1e-6:
            t_samples = np.append(t_samples, total_time)

        pva = ms.compute_trajectory_derivatives(traj, t_samples, order=3)
        pos = pva[0]
        vel = pva[1]
        acc = pva[2]

        s_cum = _cum_arc_length(pos)

        samples: List[TrajectoryPoint] = []
        prev_yaw: Optional[float] = None
        prev_t: Optional[float] = None

        for i, t in enumerate(t_samples):
            x, y, z = float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])
            vx, vy, vz = float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2])
            ax, ay, az = float(acc[i, 0]), float(acc[i, 1]), float(acc[i, 2])

            yaw: Optional[float] = None
            yaw_rate: Optional[float] = None
            if math.hypot(vx, vy) > 1e-6:
                yaw = math.atan2(vy, vx)
                if prev_yaw is not None and prev_t is not None:
                    dt_y = float(t - prev_t)
                    if dt_y > 1e-9:
                        dyaw = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
                        yaw_rate = dyaw / dt_y
                prev_yaw = yaw
                prev_t = float(t)

            samples.append(
                TrajectoryPoint(
                    t=float(t),
                    x=x, y=y, z=z,
                    vx=vx, vy=vy, vz=vz,
                    ax=ax, ay=ay, az=az,
                    yaw=yaw,
                    yaw_rate=yaw_rate,
                    s=float(s_cum[i]),
                    curvature=float(_curvature(vel[i], acc[i])),
                )
            )

        debug = {
            "n_waypoints_in": len(path.points),
            "n_waypoints_used": len(xs),
            "dt": dt,
            "total_time": total_time,
            "total_length": float(s_cum[-1]) if len(s_cum) else 0.0,
            "max_v_used": max_v,
            "max_a_used": max_a,
            "waypoint_times": times,
        }

        return MinSnapRawResult(samples=tuple(samples), debug=debug)
