"""A surface standing in front of the detected object, inside its box, found from the depth alone.

The box pipeline reads a detection's range as a low percentile of the depth inside the shrunken box
(``geometry.robust_bbox_depth``: the near side of the object, robust to the background a box overhangs
onto). The opposite error had no defence: a surface IN FRONT of the detected object inside its box wins
the percentile the moment it covers a third of the box's centre. Collierville/000018 (2026-10-15): the
top rail of a seat back at camera height stood 1.3 m from the agent between it and a television 4.1 m
away; the television's box held 79 % television and 20 % rail, the rail was read as the range, the
closing walked to 0.8 m of the rail and STOPped 3.3 m from the television -- the same five episodes,
at the same actions, in two 1,000-episode runs.

The test asks the WHOLE box four things, all measured on the depth image:

* **bimodal** -- a second surface stands at least ``min_gap_m`` (or ``gap_fraction`` of the near range)
  behind the one the percentile chose;
* **the far surface spans the box** -- ONE connected surface over at least ``span_fraction`` of the
  box's height and of its width, which is what a detector's box outlines when it outlines a flat
  object. The backrest of a couch, the tank of a toilet, the headboard of a bed span the width and not
  the height, and are not "the far surface" in this sense;
* **it covers more of the box** than the near surface does;
* **the near surface leaves the top of the box to it** -- its highest pixel lies at least
  ``top_fraction`` of the box's height below the box's top edge. A floor-standing object in front of a
  wall (a chair, a plant) reaches the top of its own box, and the wall behind it is never read as the
  object. A rail, a sofa back, a counter in front of a television does not.

When all four hold the near surface stands in front of the detected object: every pixel of the box at
its depth is returned, for the caller to measure around. Depth only -- no class, no colour, no learned
model. The mask pipeline (``RPTSettings.segmentation``) answers the same question from the object's own
pixels and does not need this.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def depth_band_m(depth_m):
    """The depth tolerance the box pipeline treats as one surface: 12 cm, or 6 % of the range."""
    return max(0.12, float(depth_m) * 0.06)


def _largest_component(mask):
    """The largest 4-connected component of a boolean mask, or None when the mask is empty."""
    labels, count = ndimage.label(mask)
    if not count:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def occluder_in_front(depth_img, box, depth_m, min_depth_m, max_depth_m, min_gap_m=0.5, gap_fraction=0.3,
                      span_fraction=0.8, top_fraction=0.25, min_valid_px=20):
    """The pixels of a surface standing in front of the detected object inside ``box``, or None.

    Args:
        depth_img: ``(H, W)`` metric depth in metres, optical ``z``; NaN/inf are no reading.
        box: The detection's ``(x1, y1, x2, y2)`` in image pixels.
        depth_m: The range the box pipeline chose for the box (``robust_bbox_depth``).
        min_depth_m: Depths at or below it are no reading.
        max_depth_m: Depths at or above it are no reading.
        min_gap_m: The least depth gap between the near surface and the far one.
        gap_fraction: ... or this fraction of the near range, whichever is larger.
        span_fraction: The share of the box's height AND width the far surface must span.
        top_fraction: The share of the box's height, from the top, the near surface must leave to the far one.
        min_valid_px: Fewer valid pixels in the box is too little depth to tell.

    Returns:
        A boolean ``(H, W)`` mask of the near surface's pixels inside the box when the four
        conditions hold; None when there is no occluder in this sense, or too little depth to tell.

    Raises:
        ValueError: On a depth image that is not two-dimensional.
    """
    depth = np.asarray(depth_img)
    if depth.ndim != 2:
        raise ValueError("depth_img must be (H, W), got shape %r" % (depth.shape,))
    height, width = depth.shape
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    patch = depth[y1:y2, x1:x2]
    valid = np.isfinite(patch) & (patch > min_depth_m) & (patch < max_depth_m)
    if int(valid.sum()) < min_valid_px:
        return None
    band = depth_band_m(depth_m)
    near_band = valid & (np.abs(patch - depth_m) <= band)
    near = _largest_component(near_band)
    if near is None:
        return None
    gap = max(float(min_gap_m), float(gap_fraction) * float(depth_m))
    far = _largest_component(valid & (patch > depth_m + gap))
    if far is None or int(far.sum()) <= int(near.sum()):
        return None
    rows, cols = np.nonzero(far)
    if ((rows.max() - rows.min() + 1) < span_fraction * (y2 - y1)
            or (cols.max() - cols.min() + 1) < span_fraction * (x2 - x1)):
        return None
    if int(np.nonzero(near)[0].min()) < top_fraction * (y2 - y1):
        return None
    out = np.zeros(depth.shape, dtype=bool)
    out[y1:y2, x1:x2] = near_band
    return out

