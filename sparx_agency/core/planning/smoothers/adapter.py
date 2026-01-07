"""
Discrete trajectory adapter.

Wraps time-ordered `TrajectoryPoint` samples into an object implementing the
`Trajectory` protocol expected by the planning stack.

Design goals:
- Minimal logic (no smoothing/optimization)
- Only time-based sampling utilities
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from core.common.types.planning import Trajectory, TrajectoryPoint


@dataclass(frozen=True)
class DiscreteTrajectory(Trajectory):
    """
    Trajectory backed by discrete samples.

    Sampling behavior:
    - `sample(t)` returns the nearest sample in time.
    - `sample_by_time(dt)` returns a uniform time grid using `sample(t)`.
    """
    points: Tuple[TrajectoryPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("DiscreteTrajectory requires at least 2 points")
        ts = [p.t for p in self.points]
        if any(ts[i] > ts[i + 1] for i in range(len(ts) - 1)):
            raise ValueError("Points must be sorted by increasing t")

    @property
    def total_time(self) -> float:
        return float(self.points[-1].t - self.points[0].t)

    @property
    def start(self) -> Tuple[float, float, float]:
        p = self.points[0]
        return float(p.x), float(p.y), float(p.z)

    @property
    def end(self) -> Tuple[float, float, float]:
        p = self.points[-1]
        return float(p.x), float(p.y), float(p.z)

    def sample(self, t: float) -> TrajectoryPoint:
        """
        Return the closest sample to the requested time.

        Convention:
        - `t` is absolute in the same timeline as the stored samples.
        """
        if t <= self.points[0].t:
            return self.points[0]
        if t >= self.points[-1].t:
            return self.points[-1]

        lo, hi = 0, len(self.points) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.points[mid].t < t:
                lo = mid
            else:
                hi = mid

        a = self.points[lo]
        b = self.points[hi]
        return a if (t - a.t) <= (b.t - t) else b

    def sample_by_time(self, dt: float) -> List[TrajectoryPoint]:
        """Sample the trajectory at fixed time steps (absolute time grid)."""
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        eps = 1e-9
        t0 = float(self.points[0].t)
        t_end = float(self.points[-1].t)

        out: List[TrajectoryPoint] = []
        t = t0

        # Sample from t0 to t_end
        if t_end <= t0 + eps:
            return [self.points[0], self.points[-1]]

        n = int((t_end - t0) / dt)
        for _ in range(n + 1):
            out.append(self.sample(t))
            t += dt

        # Ensure final point included
        if not out or abs(float(out[-1].t) - t_end) > eps:
            out.append(self.points[-1])

        return out
