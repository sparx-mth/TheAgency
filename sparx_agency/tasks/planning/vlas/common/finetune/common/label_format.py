"""Convert a body-frame target polyline into each model's native action label.

The correctors emit a variable-length ``Path2D`` in the body FLU frame
(``x=forward``, ``y=left``, meters). NavDP and FlowNav each expect a fixed-horizon
action tensor in their own encoding. This module does the arc-length resampling
and the encoding, and *only* that -- it does not know about the network.

NavDP  : ``(24, 3)`` = per-step ``(dx, dy, dyaw)`` scaled x4, clamped to [-1, 1]
         (``trajectory = cumsum(action / 4)``; ``clip_sample=True`` -> ~0.25 m/step).
FlowNav: ``(8, 2)``  = absolute egocentric ``(x, y)`` waypoints in *waypoint units*
         (meters / ``metric_waypoint_spacing``); the trainer applies ``get_delta``
         + min-max ``normalize_data`` on top.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from sparx_agency.core.common.types import Path2D
from sparx_agency.core.common.types.geometry import normalize_angle

PolyLine = Union[np.ndarray, Path2D]


def _as_fwd_left(poly: PolyLine) -> np.ndarray:
    """Return an ``(M, 2)`` ``[fwd, left]`` float32 array from a Path2D or array."""
    if isinstance(poly, Path2D):
        arr = np.array([[p.x, p.y] for p in poly.points], dtype=np.float32)
    else:
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] < 2:
        raise ValueError("polyline needs >= 2 points")
    return arr


def resample_arclength(points: np.ndarray, n: int) -> np.ndarray:
    """Resample an ``(M, 2)`` polyline to ``n`` points evenly spaced by arc length.

    The endpoints are preserved. A degenerate (zero-length) polyline returns ``n``
    copies of the first point.
    """
    pts = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-9:
        return np.repeat(pts[:1], n, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, n)
    x = np.interp(targets, s, pts[:, 0])
    y = np.interp(targets, s, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def to_navdp_label(
    poly: PolyLine,
    horizon: int = 24,
    scale: float = 4.0,
    clamp: float = 1.0,
    with_yaw: bool = True,
) -> np.ndarray:
    """Encode a body-frame target as a NavDP action tensor ``(horizon, 3)``.

    The polyline is resampled to ``horizon + 1`` points (origin first), first-
    differenced into per-step displacements, scaled by ``scale`` and clamped -- the
    inverse of NavDP's ``cumsum(action / scale)`` reconstruction.

    Args:
        poly: Body-frame ``[fwd, left]`` polyline (``Path2D`` or ``(M, 2)`` array).
            Its first point should be the robot origin.
        horizon: Number of action steps (NavDP ``predict_size`` = 24).
        scale: The ``x4`` factor NavDP applies before ``cumsum`` divides by it.
        clamp: Clip the scaled action to ``[-clamp, +clamp]`` (``clip_sample``).
        with_yaw: If ``True``, the 3rd channel is the per-step heading change
            (scaled/clamped); else it is left 0.

    Returns:
        ``(horizon, 3)`` float32 ``(dx, dy, dyaw)`` action, ready as the diffusion
        target ``x0``.
    """
    pts = _as_fwd_left(poly)
    wp = resample_arclength(pts, horizon + 1)          # includes origin at [0]
    deltas = np.diff(wp, axis=0)                        # (horizon, 2)
    action = np.zeros((horizon, 3), dtype=np.float32)
    action[:, :2] = deltas * scale

    if with_yaw:
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])  # per-segment heading
        prev = np.concatenate([[0.0], headings[:-1]])
        dyaw = np.array([normalize_angle(h - p) for h, p in zip(headings, prev)],
                        dtype=np.float32)
        action[:, 2] = dyaw * scale

    np.clip(action, -clamp, clamp, out=action)
    return action


def to_flownav_label(
    poly: PolyLine,
    horizon: int = 8,
    metric_waypoint_spacing: float = 0.25,
) -> np.ndarray:
    """Encode a body-frame target as a FlowNav action tensor ``(horizon, 2)``.

    FlowNav's ``actions`` are *absolute* egocentric waypoints in waypoint units;
    the trainer applies ``get_delta`` + min-max normalization afterwards. So we
    resample to ``horizon + 1`` (drop the origin) and divide by the metric spacing.

    Args:
        poly: Body-frame ``[fwd, left]`` polyline; first point is the origin.
        horizon: Number of waypoints (FlowNav ``len_traj_pred`` = 8; must be
            divisible by 4 for the ``ConditionalUnet1D`` down-sampling).
        metric_waypoint_spacing: Meters-per-"waypoint-unit" for the drone dataset;
            keep it consistent between labels and the ESDF penalty grid.

    Returns:
        ``(horizon, 2)`` float32 ``(x=fwd, y=left)`` waypoints in waypoint units.
    """
    if horizon % 4 != 0:
        raise ValueError("FlowNav horizon must be divisible by 4 (UNet down_dims)")
    pts = _as_fwd_left(poly)
    wp = resample_arclength(pts, horizon + 1)[1:]      # drop origin -> absolute wps
    return (wp / float(metric_waypoint_spacing)).astype(np.float32)
