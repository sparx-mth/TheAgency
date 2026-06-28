"""
Discrete trajectory adapter.

Wraps a sequence of TrajectoryPoint samples as a Trajectory interface.
Used by all smoothers that produce discrete samples rather than continuous representations.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import List, Sequence, Tuple

from sparx_agency.core.common.types import TrajectoryPoint


class DiscreteTrajectory:
    """
    Trajectory implementation backed by discrete samples.

    Provides linear interpolation between samples for continuous queries.

    Attributes:
        points: Ordered sequence of trajectory samples.
    """

    def __init__(self, points: Sequence[TrajectoryPoint]) -> None:
        if len(points) < 2:
            raise ValueError("DiscreteTrajectory requires at least 2 points")

        self._points = tuple(points)
        self._times = tuple(p.t for p in self._points)

        # Validate monotonic time
        for i in range(1, len(self._times)):
            if self._times[i] < self._times[i-1]:
                raise ValueError(f"Trajectory times must be monotonic, got t[{i-1}]={self._times[i-1]}, t[{i}]={self._times[i]}")

    @property
    def total_time(self) -> float:
        return self._times[-1] - self._times[0]

    @property
    def start(self) -> Tuple[float, float, float]:
        p = self._points[0]
        return (p.x, p.y, p.z)

    @property
    def end(self) -> Tuple[float, float, float]:
        p = self._points[-1]
        return (p.x, p.y, p.z)

    def sample(self, t: float) -> TrajectoryPoint:
        """
        Sample trajectory at time t using linear interpolation.

        Times outside [t_start, t_end] are clamped to endpoints.
        """
        if t <= self._times[0]:
            return self._points[0]
        if t >= self._times[-1]:
            return self._points[-1]

        # Find bracketing samples
        idx = bisect_right(self._times, t) - 1
        idx = max(0, min(idx, len(self._points) - 2))

        p0, p1 = self._points[idx], self._points[idx + 1]
        dt = p1.t - p0.t

        if dt < 1e-9:
            return p0

        alpha = (t - p0.t) / dt
        return self._lerp(p0, p1, alpha)

    def sample_by_time(self, dt: float) -> List[TrajectoryPoint]:
        """Sample trajectory at uniform time intervals."""
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        samples = []
        t = self._times[0]
        while t <= self._times[-1]:
            samples.append(self.sample(t))
            t += dt

        # Ensure final point is included
        if samples and abs(samples[-1].t - self._times[-1]) > 1e-9:
            samples.append(self._points[-1])

        return samples

    @staticmethod
    def _lerp(p0: TrajectoryPoint, p1: TrajectoryPoint, alpha: float) -> TrajectoryPoint:
        """Linear interpolation between two trajectory points."""
        def interp(a: float, b: float) -> float:
            return a + alpha * (b - a)

        def interp_opt(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return a if alpha < 0.5 else b
            return interp(a, b)

        return TrajectoryPoint(
            t=interp(p0.t, p1.t),
            x=interp(p0.x, p1.x),
            y=interp(p0.y, p1.y),
            z=interp(p0.z, p1.z),
            vx=interp(p0.vx, p1.vx),
            vy=interp(p0.vy, p1.vy),
            vz=interp(p0.vz, p1.vz),
            ax=interp(p0.ax, p1.ax),
            ay=interp(p0.ay, p1.ay),
            az=interp(p0.az, p1.az),
            yaw=interp_opt(p0.yaw, p1.yaw),
            yaw_rate=interp_opt(p0.yaw_rate, p1.yaw_rate),
            s=interp_opt(p0.s, p1.s),
            curvature=interp_opt(p0.curvature, p1.curvature),
        )