"""Polyline and frame arithmetic, shared by label generation and scoring.

Small, pure, and separated out because three very different consumers need the
same handful of operations and none of them should own them: the expert builds
labels with these, the metrics score trajectories with them, and the evaluation
decodes stored actions with them.

Frames are the repo-wide convention throughout: **body FLU** (`x` forward, `y`
left) and world ENU, related by the aircraft's `(x, y, yaw)`.

numpy only; Python-3.8-safe idioms so nothing here blocks reuse from `core`.
"""
from __future__ import annotations

from math import cos, degrees, sin
from typing import Sequence, Tuple

import numpy as np


def arclength(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length of an ``(N, 2)`` polyline, starting at 0."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample an ``(N, 2)`` polyline to uniform ``spacing``, endpoints kept."""
    along = arclength(points)
    total = float(along[-1])
    if total <= 1e-9:
        return points[:1].repeat(2, axis=0)
    count = max(2, int(np.ceil(total / max(spacing, 1e-6))) + 1)
    targets = np.linspace(0.0, total, count)
    return np.stack([np.interp(targets, along, points[:, 0]),
                     np.interp(targets, along, points[:, 1])], axis=1)


def truncate(points: np.ndarray, length_m: float) -> Tuple[np.ndarray, float]:
    """First ``length_m`` of a polyline, with an interpolated final point.

    Returns:
        ``(polyline, length_used_m)``. A route shorter than ``length_m`` is
        returned whole, which is what makes a near goal decelerate to a stop
        once it is resampled onto a fixed number of steps.
    """
    along = arclength(points)
    total = float(along[-1])
    if total <= length_m:
        return points, total
    keep = int(np.searchsorted(along, length_m))
    tip = np.array([[np.interp(length_m, along, points[:, 0]),
                     np.interp(length_m, along, points[:, 1])]])
    return np.concatenate([points[:keep], tip], axis=0), length_m


def to_body(points: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    """Express a world ``(N, 2)`` polyline in the body FLU frame at ``pose``."""
    c, s = cos(float(pose[2])), sin(float(pose[2]))
    dx = points[:, 0] - float(pose[0])
    dy = points[:, 1] - float(pose[1])
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=1)


def to_world(points_body: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`to_body`."""
    c, s = cos(float(pose[2])), sin(float(pose[2]))
    return np.stack([
        float(pose[0]) + points_body[:, 0] * c - points_body[:, 1] * s,
        float(pose[1]) + points_body[:, 0] * s + points_body[:, 1] * c,
    ], axis=1)


def decode_action(action: np.ndarray, scale: float = 4.0) -> np.ndarray:
    """NavDP action ``(T, 3)`` -> body-frame waypoints ``(T, 2)``.

    NavDP's trajectory is ``cumsum(action / scale)``; the division happens
    *before* the cumulative sum, which is the detail most reimplementations of
    this get wrong.
    """
    return np.cumsum(np.asarray(action, dtype=np.float64)[:, :2] / scale, axis=0)


def turn_magnitude_deg(waypoints_body: np.ndarray) -> float:
    """Largest heading deviation from straight ahead over a body-frame path.

    Zero for a straight run; ~90 for a route that turns a right-angle corner
    inside the horizon. Stored per sample so evaluation can be reported
    separately for turns, which is where a navigation policy actually fails.
    """
    steps = np.diff(np.concatenate([np.zeros((1, 2)), waypoints_body]), axis=0)
    keep = np.linalg.norm(steps, axis=1) > 1e-6
    if not keep.any():
        return 0.0
    headings = np.arctan2(steps[keep, 1], steps[keep, 0])
    return float(degrees(np.max(np.abs(headings))))
