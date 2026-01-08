from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from sparx_agency.core.common.types.planning import Path2D, TrajectoryPoint
from sparx_agency.core.common.types.control import KinematicLimits
from .params import BezierParams


@dataclass(frozen=True)
class BezierSolution:
    samples: Tuple[TrajectoryPoint, ...]


class BezierAlgorithm:
    """
    Heading-aware smoothing using cubic Hermite splines.

    Steps:
      1) Filter near-duplicate waypoints
      2) Parameterize by chord length
      3) Compute tangents for G1 heading continuity
      4) Build arc-length lookup table
      5) Sample in time at dt -> TrajectoryPoint(t,x,y,vx,vy,s,curvature,yaw)
    """

    def __init__(self, *, params: BezierParams) -> None:
        self.params = params

    def solve(
        self,
        *,
        path: Path2D,
        limits: Optional[KinematicLimits],
        options: Dict[str, Any],
        world: Any = None,
    ) -> BezierSolution:
        p = self._merge_params(options)

        xs, ys, yaws = self._extract_xy_yaw(path)
        xs, ys, yaws = self._filter_points(xs, ys, yaws, p.min_point_spacing)
        if len(xs) < 2:
            raise ValueError("BezierAlgorithm requires at least 2 distinct waypoints")

        xs_np = np.array(xs, dtype=float)
        ys_np = np.array(ys, dtype=float)

        # parameter u = cumulative chord length
        u = self._compute_parameters(xs_np, ys_np)

        dx, dy = self._compute_tangents(xs_np, ys_np, u, p.tangent_scale)

        spline_x = CubicHermiteSpline(u, xs_np, dx)
        spline_y = CubicHermiteSpline(u, ys_np, dy)

        arc_s, arc_u = self._build_arc_length_lut(spline_x, spline_y, u, p.arc_lut_samples)
        total_len = float(arc_s[-1])

        cruise_speed = self._choose_speed_xy(p, limits)
        total_time = (total_len / cruise_speed) if (cruise_speed > 0 and total_len > 0) else 0.0

        samples = self._sample_time(
            spline_x=spline_x,
            spline_y=spline_y,
            arc_s=arc_s,
            arc_u=arc_u,
            total_len=total_len,
            total_time=total_time,
            cruise_speed=cruise_speed,
            dt=p.dt,
            endpoints_zero_v=p.constrain_endpoints_velocity_zero,
        )

        return BezierSolution(samples=tuple(samples))

    # -----------------------
    # params / options
    # -----------------------

    def _merge_params(self, options: Dict[str, Any]) -> BezierParams:
        if not options:
            return self.params
        data = {**self.params.__dict__}
        for k, v in options.items():
            if k in data:
                data[k] = v
        merged = BezierParams(**data)
        merged.validate()
        return merged

    def _choose_speed_xy(self, p: BezierParams, limits: Optional[KinematicLimits]) -> float:
        if limits is None:
            return p.nominal_speed_xy
        # Conservative: clamp by limits.max_speed_xy
        return min(float(limits.max_speed_xy), float(p.nominal_speed_xy))

    # -----------------------
    # path extraction
    # -----------------------

    @staticmethod
    def _extract_xy_yaw(path: Path2D) -> Tuple[List[float], List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        yaws: List[float] = []
        for p in path.points:
            xs.append(float(p.x))
            ys.append(float(p.y))
            yaws.append(float(p.yaw))
        return xs, ys, yaws

    # -----------------------
    # geometry helpers
    # -----------------------

    @staticmethod
    def _filter_points(
        xs: Sequence[float],
        ys: Sequence[float],
        yaws: Sequence[float],
        min_spacing: float,
    ) -> Tuple[List[float], List[float], List[float]]:
        if len(xs) == 0:
            return [], [], []

        out_x = [float(xs[0])]
        out_y = [float(ys[0])]
        out_yaw = [float(yaws[0])]

        for x, y, yaw in zip(xs[1:], ys[1:], yaws[1:]):
            if math.hypot(float(x) - out_x[-1], float(y) - out_y[-1]) >= min_spacing:
                out_x.append(float(x))
                out_y.append(float(y))
                out_yaw.append(float(yaw))

        # Ensure final point present (unless almost identical)
        if len(xs) > 1 and math.hypot(float(xs[-1]) - out_x[-1], float(ys[-1]) - out_y[-1]) >= (min_spacing * 0.5):
            out_x.append(float(xs[-1]))
            out_y.append(float(ys[-1]))
            out_yaw.append(float(yaws[-1]))

        return out_x, out_y, out_yaw

    @staticmethod
    def _compute_parameters(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        u = np.zeros(len(xs), dtype=float)
        u[1:] = np.cumsum(diffs)
        return u

    @staticmethod
    def _compute_tangents(
        xs: np.ndarray,
        ys: np.ndarray,
        u: np.ndarray,
        tangent_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tangents for G1 continuity:
          - endpoints: direction to/from neighbor
          - interior: normalized(incoming) + normalized(outgoing)
        Tangent magnitude scaled by local segment length.
        """
        n = len(xs)
        dx = np.zeros(n, dtype=float)
        dy = np.zeros(n, dtype=float)

        for i in range(n):
            if i == 0:
                direction = np.array([xs[1] - xs[0], ys[1] - ys[0]], dtype=float)
            elif i == n - 1:
                direction = np.array([xs[-1] - xs[-2], ys[-1] - ys[-2]], dtype=float)
            else:
                incoming = np.array([xs[i] - xs[i - 1], ys[i] - ys[i - 1]], dtype=float)
                outgoing = np.array([xs[i + 1] - xs[i], ys[i + 1] - ys[i]], dtype=float)

                in_len = float(np.linalg.norm(incoming))
                out_len = float(np.linalg.norm(outgoing))
                if in_len > 1e-9:
                    incoming /= in_len
                if out_len > 1e-9:
                    outgoing /= out_len

                direction = incoming + outgoing

            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                direction /= norm
            else:
                direction = np.array([1.0, 0.0], dtype=float)

            if i == 0:
                seg_len = float(u[1] - u[0])
            elif i == n - 1:
                seg_len = float(u[-1] - u[-2])
            else:
                seg_len = float((u[i + 1] - u[i - 1]) / 2.0)

            dx[i] = float(direction[0] * seg_len * tangent_scale)
            dy[i] = float(direction[1] * seg_len * tangent_scale)

        return dx, dy

    @staticmethod
    def _build_arc_length_lut(
        spline_x: CubicHermiteSpline,
        spline_y: CubicHermiteSpline,
        u: np.ndarray,
        num_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        u_samples = np.linspace(float(u[0]), float(u[-1]), int(num_samples), dtype=float)
        x_vals = spline_x(u_samples)
        y_vals = spline_y(u_samples)

        ds = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
        arc_s = np.zeros(len(u_samples), dtype=float)
        arc_s[1:] = np.cumsum(ds)
        return arc_s, u_samples

    @staticmethod
    def _arc_length_to_u(arc_s: np.ndarray, arc_u: np.ndarray, s: float) -> float:
        s_clamped = float(np.clip(s, 0.0, float(arc_s[-1])))
        return float(np.interp(s_clamped, arc_s, arc_u))

    @staticmethod
    def _curvature_from_derivatives(dx: float, dy: float, ddx: float, ddy: float) -> float:
        speed = math.hypot(dx, dy)
        if speed <= 1e-9:
            return 0.0
        return abs(dx * ddy - dy * ddx) / (speed ** 3)

    def _sample_time(
        self,
        *,
        spline_x: CubicHermiteSpline,
        spline_y: CubicHermiteSpline,
        arc_s: np.ndarray,
        arc_u: np.ndarray,
        total_len: float,
        total_time: float,
        cruise_speed: float,
        dt: float,
        endpoints_zero_v: bool,
    ) -> List[TrajectoryPoint]:
        if total_time <= 0.0 or total_len <= 0.0:
            # Degenerate: two points at t=0
            u0 = float(arc_u[0])
            u1 = float(arc_u[-1])
            p0 = TrajectoryPoint(t=0.0, x=float(spline_x(u0)), y=float(spline_y(u0)))
            p1 = TrajectoryPoint(t=0.0, x=float(spline_x(u1)), y=float(spline_y(u1)))
            return [p0, p1]

        n = int(total_time / dt)
        out: List[TrajectoryPoint] = []

        for k in range(n + 1):
            t = min(k * dt, total_time)
            s = (t / total_time) * total_len
            u = self._arc_length_to_u(arc_s, arc_u, s)

            x = float(spline_x(u))
            y = float(spline_y(u))

            # First derivative wrt u -> tangent direction
            dx_du = float(spline_x(u, 1))
            dy_du = float(spline_y(u, 1))
            speed_du = math.hypot(dx_du, dy_du)

            if speed_du > 1e-9:
                tx = dx_du / speed_du
                ty = dy_du / speed_du
            else:
                tx, ty = 1.0, 0.0

            vx = tx * cruise_speed
            vy = ty * cruise_speed

            # Second derivative for curvature estimate
            ddx_du2 = float(spline_x(u, 2))
            ddy_du2 = float(spline_y(u, 2))
            curvature = self._curvature_from_derivatives(dx_du, dy_du, ddx_du2, ddy_du2)

            yaw = math.atan2(vy, vx) if (abs(vx) + abs(vy)) > 1e-9 else None

            out.append(
                TrajectoryPoint(
                    t=float(t),
                    x=float(x),
                    y=float(y),
                    z=0.0,
                    vx=float(vx),
                    vy=float(vy),
                    vz=0.0,
                    s=float(s),
                    curvature=float(curvature),
                    yaw=yaw,
                )
            )

        if endpoints_zero_v and out:
            # enforce start/end v=0 without changing positions/timestamps
            out[0] = TrajectoryPoint(**{**out[0].__dict__, "vx": 0.0, "vy": 0.0, "vz": 0.0})
            out[-1] = TrajectoryPoint(**{**out[-1].__dict__, "vx": 0.0, "vy": 0.0, "vz": 0.0})

        # Ensure last sample is exactly at total_time
        if out and abs(out[-1].t - total_time) > 1e-9:
            u_end = float(arc_u[-1])
            out.append(
                TrajectoryPoint(
                    t=float(total_time),
                    x=float(spline_x(u_end)),
                    y=float(spline_y(u_end)),
                    z=0.0,
                    vx=0.0 if endpoints_zero_v else out[-1].vx,
                    vy=0.0 if endpoints_zero_v else out[-1].vy,
                    vz=0.0,
                    s=float(total_len),
                )
            )

        return out
