"""Corner rounding for stop-and-turn path followers (ROS-free, 2D).

Line-of-sight smoothing leaves a path with a few sharp, any-angle corners. A
follower that cannot translate and rotate at once must stop, settle and yaw in
place at every corner sharper than it can glide through. This module makes such
paths gentler in two cheap, geometry-only passes:

* :func:`merge_collinear_2d` drops near-straight interior vertices so the
  follower is not handed spurious micro-corners to stop at.
* :func:`chamfer_corners_2d` cuts *moderate* corners into two half-angle turns,
  so each is shallow enough for the follower to glide through (no stop) — while
  leaving genuinely sharp corners as real, deliberate turns.

Obstacle awareness is injected via a ``clear_fn`` so this stays map/ROS-free.
"""
from __future__ import annotations

from math import atan2, hypot
from typing import Callable, List, Optional, Sequence

from sparx_agency.core.common.types import Pose2D

ClearFn = Callable[[Pose2D, Pose2D], bool]


def _turn_angle(prev: Pose2D, v: Pose2D, nxt: Pose2D) -> float:
    """Heading deviation at vertex ``v`` on ``prev -> v -> nxt`` (rad, 0=straight)."""
    ux, uy = v.x - prev.x, v.y - prev.y
    wx, wy = nxt.x - v.x, nxt.y - v.y
    lu, lw = hypot(ux, uy), hypot(wx, wy)
    if lu < 1e-9 or lw < 1e-9:
        return 0.0
    return abs(atan2(ux * wy - uy * wx, ux * wx + uy * wy))


def _pull_back(v: Pose2D, toward: Pose2D, dist: float) -> Pose2D:
    """Point at ``dist`` from ``v`` along the ray ``v -> toward``."""
    dx, dy = toward.x - v.x, toward.y - v.y
    seg = hypot(dx, dy)
    if seg < 1e-9:
        return Pose2D(v.x, v.y)
    return Pose2D(v.x + dx / seg * dist, v.y + dy / seg * dist)


def merge_collinear_2d(
    points: Sequence[Pose2D], angle_thresh: float
) -> List[Pose2D]:
    """Drop interior vertices whose turn angle is below ``angle_thresh`` (rad).

    Collapses near-straight runs to their endpoints. Endpoints are preserved.
    """
    if len(points) <= 2 or angle_thresh <= 0.0:
        return list(points)
    out = [points[0]]
    for i in range(1, len(points) - 1):
        if _turn_angle(out[-1], points[i], points[i + 1]) >= angle_thresh:
            out.append(points[i])
    out.append(points[-1])
    return out


def chamfer_corners_2d(
    points: Sequence[Pose2D],
    max_turn: float,
    chamfer_max: float,
    chamfer_dist: float,
    min_runup: float,
    clear_fn: Optional[ClearFn] = None,
) -> List[Pose2D]:
    """Cut moderate corners into two half-angle turns so they become glide-able.

    A corner whose turn angle is in ``(max_turn, chamfer_max]`` and has at least
    ``min_runup`` of leg on both sides is replaced by two points pulled back from
    the vertex, halving the per-vertex heading change. Gentler corners are
    already glide-able and kept; sharper ones are genuine turns and kept. If
    ``clear_fn`` rejects the cut segment (it would clip an obstacle), the original
    sharp vertex is kept. Endpoints are preserved.
    """
    if len(points) <= 2:
        return list(points)
    out = [points[0]]
    for i in range(1, len(points) - 1):
        prev, v, nxt = points[i - 1], points[i], points[i + 1]
        turn = _turn_angle(prev, v, nxt)
        leg = min(hypot(v.x - prev.x, v.y - prev.y),
                  hypot(nxt.x - v.x, nxt.y - v.y))
        if not (max_turn < turn <= chamfer_max) or leg < min_runup:
            out.append(v)
            continue
        d = min(chamfer_dist, 0.5 * leg)
        c1, c2 = _pull_back(v, prev, d), _pull_back(v, nxt, d)
        if clear_fn is not None and not clear_fn(c1, c2):
            out.append(v)        # cutting the corner would clip an obstacle
            continue
        out.append(c1)
        out.append(c2)
    out.append(points[-1])
    return out


def round_corners_2d(
    points: Sequence[Pose2D], params, clear_fn: Optional[ClearFn] = None
) -> List[Pose2D]:
    """Merge near-collinear vertices, then chamfer moderate corners.

    ``params`` is a ``WeightedAStarParams`` supplying the angle/length knobs.
    Returns the input unchanged for trivial paths.
    """
    if len(points) <= 2:
        return list(points)
    pts = merge_collinear_2d(points, params.corner_merge_rad)
    return chamfer_corners_2d(
        pts,
        params.corner_max_turn_rad,
        params.corner_chamfer_max_rad,
        params.corner_chamfer_dist_m,
        params.corner_min_runup_m,
        clear_fn=clear_fn,
    )
