"""Pure, stateless axis-aligned bounding-box geometry (image plane).

Cross-cutting helpers shared by the visual tracker
(:mod:`sparx_agency.core.mapping.tracking`) and the visual-servo control
law (:mod:`sparx_agency.core.planning.visual_servo`). Data *types* (``Detection2D``,
``Track2D``) live in :mod:`sparx_agency.core.common.types`; this module only holds
the arithmetic that operates on their boxes.

Box convention
--------------
All boxes are ``(x1, y1, x2, y2)`` in image pixels, origin **top-left**,
``+x`` right, ``+y`` down, with ``x2 >= x1`` and ``y2 >= y1``. The ``cxcywh``
form is ``(centre_x, centre_y, width, height)``.

ROS-free and Python-3.8-safe.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


def xyxy_center(bbox: BBox) -> Tuple[float, float]:
    """Return the box centre ``(cx, cy)`` in pixels."""
    x1, y1, x2, y2 = bbox
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def xyxy_size(bbox: BBox) -> Tuple[float, float]:
    """Return ``(width, height)`` in pixels, clamped to be non-negative."""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def xyxy_area(bbox: BBox) -> float:
    """Return the box area in px^2."""
    w, h = xyxy_size(bbox)
    return w * h


def area_frac(bbox: BBox, frame_w: int, frame_h: int) -> float:
    """Box area as a fraction of the image area.

    A flat target viewed head-on through a pinhole fills the image as ``~1/d^2``,
    so this is a smooth, monotone proxy for proximity when true depth is absent.
    """
    denom = float(max(1, int(frame_w)) * max(1, int(frame_h)))
    return xyxy_area(bbox) / denom


def center_offset_norm(bbox: BBox, frame_w: int, frame_h: int) -> Tuple[float, float]:
    """Normalised centre offset from the image centre, each component in ``[-1, 1]``.

    ``+x`` means the box centre is to the **right** of the image centre, ``+y``
    means **below** it. Used by the servo to derive a centring command.
    """
    cx, cy = xyxy_center(bbox)
    half_w = max(1e-6, 0.5 * float(frame_w))
    half_h = max(1e-6, 0.5 * float(frame_h))
    ox = (cx - 0.5 * float(frame_w)) / half_w
    oy = (cy - 0.5 * float(frame_h)) / half_h
    return _clamp(ox, -1.0, 1.0), _clamp(oy, -1.0, 1.0)


def rescale_xyxy(bbox: BBox, src_w: int, src_h: int, dst_w: int, dst_h: int) -> BBox:
    """Rescale a box from one frame size to another (same box, different pixel grid)."""
    if (src_w, src_h) == (dst_w, dst_h):
        return bbox
    sx, sy = dst_w / src_w, dst_h / src_h
    x1, y1, x2, y2 = bbox
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def clip_xyxy(bbox: BBox, frame_w: int, frame_h: int) -> BBox:
    """Clip a box to the image bounds ``[0, frame_w] x [0, frame_h]``."""
    x1, y1, x2, y2 = bbox
    x1 = _clamp(x1, 0.0, float(frame_w))
    x2 = _clamp(x2, 0.0, float(frame_w))
    y1 = _clamp(y1, 0.0, float(frame_h))
    y2 = _clamp(y2, 0.0, float(frame_h))
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes; 0 when they do not overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = xyxy_area(a) + xyxy_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def xyxy_to_cxcywh(bbox: BBox) -> Tuple[float, float, float, float]:
    """Convert ``(x1, y1, x2, y2)`` -> ``(cx, cy, w, h)``."""
    cx, cy = xyxy_center(bbox)
    w, h = xyxy_size(bbox)
    return cx, cy, w, h


def cxcywh_to_xyxy(cxcywh: Tuple[float, float, float, float]) -> BBox:
    """Convert ``(cx, cy, w, h)`` -> ``(x1, y1, x2, y2)``."""
    cx, cy, w, h = cxcywh
    hw, hh = 0.5 * max(0.0, w), 0.5 * max(0.0, h)
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def bounds_rect(points_xy: np.ndarray, k_mad: float = 0.0) -> BBox:
    """Axis-aligned bounding rectangle of a set of 2D points.

    With ``k_mad <= 0`` this is the exact min/max rectangle. With ``k_mad > 0`` a
    per-axis robust outlier rejection runs first: points whose coordinate is more
    than ``k_mad`` robust standard deviations (``1.4826 * MAD``) from the median
    are dropped, then the exact min/max of the survivors is returned. This drops a
    feature that jumped onto the background *without* systematically shrinking the
    box (unlike percentile trimming), so the area stays a faithful proximity
    signal. If rejection would leave too few points it falls back to the full set.

    Args:
        points_xy: ``(N, 2)`` array of ``(x, y)`` pixel coordinates.
        k_mad: Robust-sigma multiplier for outlier rejection (e.g. ``3.0`` keeps
            ~99% of a Gaussian cloud). ``0`` disables rejection.

    Returns:
        ``(x1, y1, x2, y2)``.

    Raises:
        ValueError: If ``points_xy`` is empty or not ``(N, 2)``.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        raise ValueError("bounds_rect: no points")
    if k_mad > 0.0 and pts.shape[0] >= 4:
        keep = np.ones(pts.shape[0], dtype=bool)
        for axis in (0, 1):
            c = pts[:, axis]
            med = np.median(c)
            mad = np.median(np.abs(c - med))
            sigma = 1.4826 * mad
            if sigma > 1e-6:
                keep &= np.abs(c - med) <= k_mad * sigma
        if int(keep.sum()) >= max(2, pts.shape[0] // 4):
            pts = pts[keep]
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return (x1, y1, x2, y2)


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp scalar ``v`` to ``[lo, hi]``."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v
