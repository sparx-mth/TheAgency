"""Depth-pixel -> body-frame point goal (the deployed NavDP convention).

When the user clicks a pixel ``(u, v)`` on the image, the goal NavDP is asked to
reach is the 3-D point that pixel sees, expressed as a body-FLU ``(forward, left)``
target. This matches the live drone code (``navdp_drone_live.pixel_to_pointgoal``)
byte-for-byte so the tool feeds NavDP exactly what it sees at deployment:

* ``forward`` = a robust (median) depth over a small patch around the click, and
* ``left``    = ``-(u - cx) * depth / fx`` (horizontal bearing at that range).

The vertical pixel coordinate ``v`` and the camera pitch are intentionally NOT
used for the *goal* (NavDP's point goal is a horizontal target); the occupancy
grid built for the correction still uses the full deprojection with pitch.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from sparx_agency.core.common.types import Intrinsics


def pixel_to_goal(
    u: int,
    v: int,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    patch: int = 10,
) -> Tuple[float, float, float]:
    """Convert a clicked depth pixel to a body-FLU ``(forward, left)`` goal.

    Args:
        u: Pixel column (x) of the click.
        v: Pixel row (y) of the click.
        depth_m: ``(H, W)`` float32 depth in meters (0 / non-finite = invalid).
        intrinsics: Pinhole intrinsics matching ``depth_m``'s resolution.
        patch: Half-size of the square window whose median depth is the range
            (robust to a single bad pixel), matching the deployed 20x20 window.

    Returns:
        ``(forward, left, depth)`` in meters. ``forward``/``left`` are the goal;
        ``depth`` is the sampled range (equal to ``forward``).

    Raises:
        ValueError: if the patch around the click has no valid depth.
    """
    h, w = depth_m.shape
    y0, y1 = max(0, v - patch), min(h, v + patch)
    x0, x1 = max(0, u - patch), min(w, u + patch)
    win = depth_m[y0:y1, x0:x1]
    valid = win[np.isfinite(win) & (win > 0.0)]
    if valid.size == 0:
        raise ValueError(f"no valid depth around pixel ({u}, {v})")

    depth = float(np.median(valid))
    forward = depth
    left = -(float(u) - intrinsics.cx) * depth / intrinsics.fx
    return forward, left, depth
