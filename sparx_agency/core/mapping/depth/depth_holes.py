"""Depth holes: no-return pixels filled from their valid neighbours, when -- and only when -- a surface surrounds them.

Habitat renders depth from the scanned mesh, and a Gibson scan has no
geometry where the scanner saw glass, a mirror or a glossy black screen:
those pixels come back as ``0``. Measured on Markleeville/000000's spawn
view (2026-10-08): holes of 5-10 thousand pixels in most headings and
voids of 30-125 thousand (up to 44 % of the frame) where a wall section
is missing altogether.

The zero-shot ObjectNav baselines all **drop** such pixels: SemExp, L3MVN
and SG-Nav push them to 100 m so the map splat gives them zero weight,
home-robot (OSG-Nav) zero-weights them at the camera origin, ApexNav culls
them as radius outliers; VLFM alone inpaints (IP-Basic, then
``fill_small_holes`` under 100 000 px) and then *excludes* the filled pixels
from its obstacle cloud. Dropping is what this runtime did too, and a
dropped hole is unknown: free beside it is a frontier, and a frontier at a
mirror is a walk to a mirror.

This module takes the one step the baselines do not: a hole that is
**enclosed** -- at least ``min_rim`` of its rim is valid depth rather than
image border -- and no larger than ``max_fraction`` of the frame is filled
with the depth of its nearest valid pixel (spatial interpolation of the
surface around it: the frame of a screen, the wall around a window, the
floor around a missing patch). The hole then projects as the boundary it
is. A hole that is not enclosed -- open to the image edge on most of its
rim, so the surface around it is not known -- or too large is left as
no reading, dropped as before and left to the sightline ledger.

Host-only: numpy, scipy and OpenCV; never on a FALCON import path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DepthHoleFill:
    """The rule for filling no-return pixels.

    Attributes:
        enabled: Off restores the baselines' behaviour: every hole dropped.
        max_fraction: Largest hole filled, as a fraction of the frame's
            pixels; a void larger than this is a missing wall, not a surface.
        min_rim: Smallest share of a hole's rim (its 8-connected boundary
            pixels) that must be valid depth, not image border, for the
            surface around it to be known.
    """

    enabled: bool = True
    max_fraction: float = 0.5
    min_rim: float = 0.5

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("DepthHoleFill.enabled must be a bool")
        for name in ("max_fraction", "min_rim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("DepthHoleFill.%s must lie in [0, 1]" % name)


_EIGHT = np.ones((3, 3), dtype=bool)


def fill_depth_holes(depth: np.ndarray, holes: np.ndarray, rule: DepthHoleFill) -> Tuple[np.ndarray, Dict[str, int]]:
    """Fill the enclosed holes of ``depth`` from their nearest valid pixels.

    Args:
        depth: ``(H, W)`` float depth in metres; the values at ``holes`` are
            ignored, every other finite positive value is a valid reading.
        holes: ``(H, W)`` bool, the no-return pixels.
        rule: Which holes qualify.

    Returns:
        ``(filled depth, stats)`` -- a copy with the qualifying holes filled,
        and ``{"holes": n, "filled": n, "pixels_filled": n, "pixels_left": n}``.
    """
    holes = np.asarray(holes, dtype=bool)
    stats = {"holes": 0, "filled": 0, "pixels_filled": 0, "pixels_left": int(holes.sum())}
    if not rule.enabled or not holes.any():
        return depth, stats
    valid = ~holes & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return depth, stats
    count, labels, table, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    stats["holes"] = int(count - 1)
    limit = rule.max_fraction * depth.size
    border = np.zeros_like(holes)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    fill = np.zeros_like(holes)
    for index in range(1, count):
        if table[index, cv2.CC_STAT_AREA] > limit:
            continue
        component = labels == index
        rim = ndimage.binary_dilation(component, structure=_EIGHT) & ~component
        # The rim is what surrounds the hole inside the frame plus the hole's own pixels on the
        # image edge: a window open to the top of the frame has sky, not surface, on that side.
        rim_total = int(rim.sum()) + int((component & border).sum())
        if rim_total == 0 or valid[rim].sum() < rule.min_rim * rim_total:
            continue                                        # open to the image edge or to other holes: no surface around it
        fill |= component
        stats["filled"] += 1
    if not fill.any():
        return depth, stats
    _, (rows, cols) = ndimage.distance_transform_edt(~valid, return_indices=True)
    filled = np.array(depth, copy=True)
    filled[fill] = depth[rows[fill], cols[fill]]
    stats["pixels_filled"] = int(fill.sum())
    stats["pixels_left"] = int((holes & ~fill).sum())
    return filled, stats
