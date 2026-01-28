"""
Reference utilities for local replanning.

We want a small local goal "ahead" on the provided reference. To keep this module
simple and robust, we select the goal from a discrete polyline extracted from:
- Path2D / Path3D: use their points directly
- Trajectory protocol: sample it at a coarse dt

The local goal selection is intentionally simple for the first iteration:
- find nearest reference point to current position
- march forward until we reach lookahead distance
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from sparx_agency.core.common.types.geometry import Pose2D, Pose3D
from sparx_agency.core.common.types.planning import Path2D, Path3D, Trajectory, TrajectoryPoint


def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _polyline_length_2d(pts: Sequence[Tuple[float, float]]) -> float:
    return sum(_dist2(a, b) for a, b in zip(pts[:-1], pts[1:])) if len(pts) >= 2 else 0.0


def _polyline_length_3d(pts: Sequence[Tuple[float, float, float]]) -> float:
    return sum(_dist3(a, b) for a, b in zip(pts[:-1], pts[1:])) if len(pts) >= 2 else 0.0


def extract_reference_points_2d(ref: Union[Path2D, Trajectory], *, sample_dt: float = 0.2) -> List[Tuple[float, float]]:
    if isinstance(ref, Path2D):
        return [(p.x, p.y) for p in ref.points]

    # Trajectory protocol
    pts: List[Tuple[float, float]] = []
    total = float(getattr(ref, "total_time", 0.0))
    if total <= 0:
        return pts
    t = 0.0
    while t <= total + 1e-9:
        tp: TrajectoryPoint = ref.sample(t)
        pts.append((tp.x, tp.y))
        t += sample_dt
    return pts


def extract_reference_points_3d(ref: Union[Path3D, Trajectory], *, sample_dt: float = 0.2) -> List[Tuple[float, float, float]]:
    if isinstance(ref, Path3D):
        return [(p.x, p.y, p.z) for p in ref.points]

    # Trajectory protocol
    pts: List[Tuple[float, float, float]] = []
    total = float(getattr(ref, "total_time", 0.0))
    if total <= 0:
        return pts
    t = 0.0
    while t <= total + 1e-9:
        tp: TrajectoryPoint = ref.sample(t)
        pts.append((tp.x, tp.y, tp.z))
        t += sample_dt
    return pts


def select_goal_on_reference_2d(
    ref_pts: Sequence[Tuple[float, float]],
    pos_xy: Tuple[float, float],
    lookahead_m: float,
    min_sep_m: float,
) -> Optional[Tuple[float, float]]:
    if len(ref_pts) < 2:
        return None

    # nearest point index
    best_i = 0
    best_d = float("inf")
    for i, p in enumerate(ref_pts):
        d = _dist2(p, pos_xy)
        if d < best_d:
            best_d = d
            best_i = i

    # march forward by arc length
    acc = 0.0
    prev = ref_pts[best_i]
    for p in ref_pts[best_i + 1 :]:
        step = _dist2(prev, p)
        acc += step
        prev = p
        if acc >= lookahead_m:
            if _dist2(pos_xy, p) >= min_sep_m:
                return p
            # if too close, keep going
    # fallback: last point if it is not too close
    if _dist2(pos_xy, ref_pts[-1]) >= min_sep_m:
        return ref_pts[-1]
    return None


def select_goal_on_reference_3d(
    ref_pts: Sequence[Tuple[float, float, float]],
    pos_xyz: Tuple[float, float, float],
    lookahead_m: float,
    min_sep_m: float,
) -> Optional[Tuple[float, float, float]]:
    if len(ref_pts) < 2:
        return None

    best_i = 0
    best_d = float("inf")
    for i, p in enumerate(ref_pts):
        d = _dist3(p, pos_xyz)
        if d < best_d:
            best_d = d
            best_i = i

    acc = 0.0
    prev = ref_pts[best_i]
    for p in ref_pts[best_i + 1 :]:
        step = _dist3(prev, p)
        acc += step
        prev = p
        if acc >= lookahead_m:
            if _dist3(pos_xyz, p) >= min_sep_m:
                return p
    if _dist3(pos_xyz, ref_pts[-1]) >= min_sep_m:
        return ref_pts[-1]
    return None
