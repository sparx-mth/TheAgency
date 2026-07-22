"""Per-trajectory safety metrics scored against a fused :class:`JudgeField`.

Two things here are deliberately stricter than ``train/evaluate.py``:

* **Densification.** NavDP emits 24 waypoints that can be tens of centimetres
  apart, so sampling only at the waypoints lets a path step straight over a thin
  obstacle and still score as clear. Every path is resampled to
  ``step_m`` before scoring.
* **Observability.** A waypoint in never-observed space is not "safe", it is
  unmeasured. Those samples are excluded from the clearance statistics and
  reported separately as ``frac_unobserved`` -- a route that scores well only by
  flying into unmapped space is not a good route.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np

from .judge_map import JudgeField


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Safety and quality summary of one trajectory.

    Attributes:
        min_clearance_m: smallest observed distance to an obstacle; the primary
            safety number, since safety is a worst-case property.
        p5_clearance_m: 5th-percentile clearance -- the sustained tight part of
            the route, less sensitive to a single stray sample than the min.
        mean_clearance_m: average clearance over observed samples.
        frac_below_safe: fraction of observed samples tighter than ``d_safe_m``.
        collides: True if any observed sample sits inside an obstacle.
        frac_unobserved: fraction of samples in never-seen space.
        path_len_m: arc length of the densified path.
        goal_gap_m: distance from the final waypoint to the goal.
        bending: sum of second-difference magnitudes; higher is kinkier.
    """

    min_clearance_m: float
    p5_clearance_m: float
    mean_clearance_m: float
    frac_below_safe: float
    collides: bool
    frac_unobserved: float
    path_len_m: float
    goal_gap_m: float
    bending: float

    def as_dict(self) -> dict:
        """Plain-dict view, for CSV/JSON writing."""
        return asdict(self)


def densify(xy: np.ndarray, step_m: float = 0.02) -> np.ndarray:
    """Resample a polyline to roughly uniform ``step_m`` spacing.

    Args:
        xy: ``(N, 2)`` polyline vertices.
        step_m: target spacing in meters.

    Returns:
        ``(M, 2)`` resampled points, ``M >= N``. Degenerate (zero-length) inputs
        are returned unchanged.
    """
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-6:
        return xy
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(total / step_m)) + 1, len(xy))
    want = np.linspace(0.0, total, n)
    return np.stack([np.interp(want, dist, xy[:, 0]),
                     np.interp(want, dist, xy[:, 1])], axis=1)


def bending(xy: np.ndarray) -> float:
    """Total second-difference magnitude of a polyline (kinkiness)."""
    if len(xy) < 3:
        return 0.0
    d2 = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    return float(np.sum(np.linalg.norm(d2, axis=1)))


def score(traj_world: np.ndarray, goal_world: np.ndarray, field: JudgeField,
          d_safe_m: float = 0.30, step_m: float = 0.02) -> TrajectoryMetrics:
    """Score one world-frame trajectory against the fused judge field.

    Args:
        traj_world: ``(N, 2)`` waypoints in world XY.
        goal_world: ``(2,)`` goal in world XY.
        field: the fused clearance field.
        d_safe_m: clearance below which a sample counts as too tight.
        step_m: densification spacing.

    Returns:
        The :class:`TrajectoryMetrics` for this trajectory. If no sample falls in
        observed space the clearance fields are NaN and ``collides`` is False --
        the caller should drop such samples rather than treat them as safe.
    """
    dense = densify(traj_world, step_m)
    clear, seen = field.sample(dense)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)

    obs = clear[seen]
    if obs.size == 0:
        return TrajectoryMetrics(
            min_clearance_m=float("nan"), p5_clearance_m=float("nan"),
            mean_clearance_m=float("nan"), frac_below_safe=float("nan"),
            collides=False, frac_unobserved=1.0, path_len_m=float(seg.sum()),
            goal_gap_m=float(np.linalg.norm(traj_world[-1] - goal_world)),
            bending=bending(traj_world))

    return TrajectoryMetrics(
        min_clearance_m=float(obs.min()),
        p5_clearance_m=float(np.percentile(obs, 5)),
        mean_clearance_m=float(obs.mean()),
        frac_below_safe=float((obs < d_safe_m).mean()),
        collides=bool((obs <= 0.0).any()),
        frac_unobserved=float(1.0 - seen.mean()),
        path_len_m=float(seg.sum()),
        goal_gap_m=float(np.linalg.norm(traj_world[-1] - goal_world)),
        bending=bending(traj_world),
    )


#: Metric name -> whether a *higher* value is better. Used by the stats layer to
#: orient paired deltas so "positive delta = improvement" always holds.
HIGHER_IS_BETTER = {
    "min_clearance_m": True,
    "p5_clearance_m": True,
    "mean_clearance_m": True,
    "frac_below_safe": False,
    "frac_unobserved": False,
    "path_len_m": False,
    "goal_gap_m": False,
    "bending": False,
}
