"""
Path manipulation utilities for behaviors.

This module provides helper functions for working with Path2D objects,
including trimming already-traversed segments and selecting subgoals
along a path.

Functions:
    trim_path_prefix: Remove leading points near the current pose
    pick_subgoal_along_path: Select an intermediate waypoint from a path

Example:
    >>> from sparx_agency.core.planning.behaviors.utils import trim_path_prefix
    >>> trimmed = trim_path_prefix(planned_path, robot_pose)
    >>> tracker.follow(trimmed)
"""

from __future__ import annotations

from typing import Optional

from sparx_agency.core.common.types import Path2D, Pose2D


def trim_path_prefix(path: Path2D, pose: Pose2D, *, tol_m: float = 1e-6) -> Path2D:
    """
    Remove leading path points that are at or behind the current pose.

    When a robot has already traversed part of a path, this function
    removes the traversed prefix to avoid backtracking. The function
    ensures the returned path remains valid (at least 2 points).

    Args:
        path: The original Path2D to trim.
        pose: Current robot pose. Points within `tol_m` of this pose
            are considered "at" the pose and will be removed.
        tol_m: Position tolerance in meters. Points where both
            |x - pose.x| <= tol_m and |y - pose.y| <= tol_m are removed.
            Defaults to 1e-6 (effectively exact match).

    Returns:
        A new Path2D with leading points removed. If trimming would
        result in fewer than 2 points, returns a degenerate path
        [pose, pose] to maintain validity.

    Example:
        >>> path = Path2D(points=(p0, p1, p2, p3), frame_id="map")
        >>> # Robot is at p1
        >>> trimmed = trim_path_prefix(path, p1, tol_m=0.01)
        >>> # trimmed.points == (p1, p2, p3) or (p2, p3)
    """
    pts = list(path.points)
    while len(pts) >= 2 and abs(pts[0].x - pose.x) <= tol_m and abs(pts[0].y - pose.y) <= tol_m:
        pts.pop(0)
    if len(pts) < 2:
        pts = [pose, pose]
    return Path2D(points=tuple(pts), frame_id=path.frame_id, metadata=dict(path.metadata))


def pick_subgoal_along_path(path: Path2D, *, step_idx: int = 1) -> Optional[Pose2D]:
    """
    Select an intermediate waypoint from a path.

    Useful for behaviors that need a short-horizon subgoal rather than
    following the entire path. The returned point is typically a few
    steps ahead on the path.

    Args:
        path: The Path2D to select from. Must have at least 2 points.
        step_idx: Index of the point to select (1-based lookahead).
            Clamped to valid range [1, len(path.points) - 1].
            Defaults to 1 (the second point in the path).

    Returns:
        The Pose2D at the selected index, or None if the path has
        fewer than 2 points.

    Example:
        >>> path = Path2D(points=(start, p1, p2, p3, goal), frame_id="map")
        >>> subgoal = pick_subgoal_along_path(path, step_idx=2)
        >>> # subgoal == p2
    """
    if len(path.points) < 2:
        return None
    idx = max(1, min(int(step_idx), len(path.points) - 1))
    return path.points[idx]