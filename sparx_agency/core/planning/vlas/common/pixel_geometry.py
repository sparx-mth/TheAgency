"""A pixel and a depth are a place in the world; keep the two directions honest.

Every VLA here that draws anything needs the same two conversions, and they are
each other's inverse:

* :func:`pixel_to_body` -- a pixel plus its metric depth is a point in front of
  the aircraft. What a System-2 pixel goal, a click, or a detection *means*.
* :func:`body_to_pixel` -- a point in front of the aircraft is a pixel. What it
  takes to draw that meaning back onto a **later** frame.

The second one is the reason this module exists. A pixel goal is a pixel in the
frame the model saw, and an overlay that redraws it at the same coordinate on
every subsequent frame is not showing a target -- it is showing a sticker. The
aircraft moves and turns, the scene slides past, and the marker sits still: on
screen the goal "never updates", however often the model actually changes it.
Converted to a world point once and re-projected per frame, it stays on the
thing it was pointing at.

Frame conventions, matching the rest of ``core``
------------------------------------------------
body (FLU, REP-103)
    ``+x`` forward, ``+y`` left, ``+z`` up; the aircraft at the origin.
camera (OpenCV optical)
    ``u`` right, ``v`` down, optical axis along body ``+x``, so::

        forward = d
        left    = -(u - cx) * d / fx
        up      = -(v - cy) * d / fy

    The mount offset between the camera and the body origin is neglected, the
    same simplification :mod:`..navdp.geometry` documents: both directions make
    it, so it cancels in anything that goes one way and back.

Numpy-only at import and Python 3.8 clean -- the FALCON Noetic adapter imports
``core``.
"""
from __future__ import annotations

from math import atan2
from typing import Optional, Tuple

import numpy as np


def patch_median_depth(depth, px, py, half=10, min_valid=0.1, max_valid=50.0):
    # type: (np.ndarray, float, float, int, float, float) -> Optional[float]
    """Median of the valid depth in a ``2*half`` box around ``(px, py)``.

    A single pixel's depth is the one sample most likely to be a miss, an edge,
    or the sky; a small median over its neighbourhood is what makes a clicked or
    predicted pixel usable at all.

    Args:
        depth: ``(H, W)`` metric depth (optical Z). NaN and zero are allowed.
        px, py: pixel column and row at the centre of the box.
        half: half box size, pixels.
        min_valid, max_valid: a reading counts only if strictly between these.

    Returns:
        The median valid depth in metres, or ``None`` when the patch holds no
        valid reading -- the caller decides the fallback, because a sensible
        fallback for a navigation goal and for an overlay are different numbers.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("patch_median_depth expects (H, W); got %r" % (depth.shape,))
    h, w = depth.shape
    patch = depth[max(0, int(py) - half):min(h, int(py) + half),
                  max(0, int(px) - half):min(w, int(px) + half)]
    valid = patch[np.isfinite(patch) & (patch > min_valid) & (patch < max_valid)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_to_body(u, v, depth_m, intr):
    # type: (float, float, float, object) -> Tuple[float, float, float]
    """Back-project one pixel at a known depth into the body frame.

    Args:
        u, v: pixel column and row.
        depth_m: metric depth (optical Z) at that pixel.
        intr: camera :class:`~sparx_agency.core.common.types.Intrinsics`.

    Returns:
        ``(forward, left, up)`` in metres, body FLU.
    """
    d = float(depth_m)
    return (d,
            -(float(u) - intr.cx) * d / intr.fx,
            -(float(v) - intr.cy) * d / intr.fy)


def body_to_pixel(forward, left, up, intr, min_forward_m=0.05):
    # type: (float, float, float, object, float) -> Optional[Tuple[float, float]]
    """Project a body-frame point onto the image. The inverse of :func:`pixel_to_body`.

    Unlike :func:`~sparx_agency.core.planning.vlas.navdp.geometry.body_point_to_pixel`,
    which assumes its point lies on a ground plane a fixed height below the
    camera, this uses the point's **own** height -- so a goal on a table, a
    doorway, or a wall lands where it actually is rather than on the floor
    beneath it.

    Args:
        forward, left, up: body-frame point, metres, FLU.
        intr: camera intrinsics.
        min_forward_m: points at or behind this are not on the image at all.

    Returns:
        ``(u, v)`` as floats, or ``None`` when the point is at or behind the
        camera plane. The result is **not** clamped to the frame: a caller that
        wants to show an off-screen goal as an edge marker needs to know how far
        off it is, and a clamped pixel cannot say.
    """
    if float(forward) < float(min_forward_m):
        return None
    return (intr.cx - intr.fx * float(left) / float(forward),
            intr.cy - intr.fy * float(up) / float(forward))


def body_to_world(forward, left, up, pose):
    # type: (float, float, float, Tuple[float, float, float, float]) -> Tuple[float, float, float]
    """Place a body-frame point in the world, given the pose it was seen from.

    Args:
        forward, left, up: body-frame point, metres.
        pose: ``(x, y, z, yaw)`` the aircraft was at when the point was seen.

    Returns:
        ``(x, y, z)`` in the world frame.
    """
    x, y, z, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return (float(x + c * forward - s * left),
            float(y + s * forward + c * left),
            float(z + up))


def world_to_body(point, pose):
    # type: (Tuple[float, float, float], Tuple[float, float, float, float]) -> Tuple[float, float, float]
    """Express a world point in the body frame. The inverse of :func:`body_to_world`.

    Args:
        point: ``(x, y, z)`` in the world frame.
        pose: ``(x, y, z, yaw)`` the aircraft is at **now**.

    Returns:
        ``(forward, left, up)`` in metres, body FLU.
    """
    px, py, pz = point
    x, y, z, yaw = pose
    dx, dy = float(px) - x, float(py) - y
    c, s = np.cos(yaw), np.sin(yaw)
    return (float(c * dx + s * dy), float(-s * dx + c * dy), float(pz - z))


def bearing_to(forward, left):
    # type: (float, float) -> float
    """Bearing to a body-frame point, radians CCW from the nose."""
    return float(atan2(float(left), float(forward)))
