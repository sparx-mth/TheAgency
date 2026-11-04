"""What a detection box's extent in the frame says: a sliver cut by an edge, and the footprint it implies.

Two image-plane readings every box-driven consumer shares -- the target
closing's takeover gate, the legacy target evidence and the perception
cycle's landmark association -- so that a box one of them refuses the
others cannot act on. Pure pinhole arithmetic on the box and the
intrinsics; no depth image, no class knowledge.
"""
from __future__ import annotations


def clipped_box(box, intrinsics, margin_px, min_fraction=0.3, floor_cut_min_fraction=0.15, floor_cut_min_width_fraction=0.08):
    """Whether ``box`` is a SLIVER at an image edge: touching it within ``margin_px`` AND narrow.

    The one test behind both target gates -- the takeover's start and the
    legacy target evidence -- so a sliver the one refuses the other cannot
    pursue. A clipped box's depth centroid is the centroid of what is in
    frame, not of the object, and the clipped end of a bed reads as a sofa
    (the Ranchester bed: 84 x 210 px on the left edge of a 640 x 480 frame).
    A box that touches an edge but spans at least ``min_fraction`` of the
    frame in both dimensions is the opposite case -- a big object close up,
    filling the view (the Ranchester couch from 0.7 m: 392 x 379 px) -- and
    is not clipped in this sense: refusing it meant the takeover never
    started from beside the couch.

    **The bottom edge alone is the camera's floor cutoff, not a sliver**
    (since 2026-10-07). A level camera 0.88 m up sees the floor from about
    1.4 m out, so a LOW object -- a toilet, a chair seat -- at 1-2 m is cut
    by the bottom edge on every frame while standing whole in the frame
    laterally; its depth centroid and near edge are on the object. The
    Allensville toilet run refused twenty-three such frames of a 0.97
    toilet at 1.2-1.7 m, never counted two in a row and released the
    candidate twice. A box that touches ONLY the bottom edge, whose top lies
    in the lower half of the frame and which spans at least
    ``floor_cut_min_fraction`` of the frame's height (the toilet: 187 of
    480 px; a 60 px sliver is still a sliver) and ``floor_cut_min_width_fraction``
    of its width, is not clipped; a box also touching a side edge, or the
    top, is judged as before. **The width bar is low** (8 %, since
    2026-10-14): a box cut only by the bottom edge stands whole laterally,
    so its width is the object's -- a toilet's tank at 1.3 m is 65-83 px of
    640, and the 15 % bar refused four 0.91-0.95 frames of it in
    Collierville/000023, where the takeover never started.
    """
    m = int(margin_px)
    x1, y1, x2, y2 = box
    lateral = bool(x1 < m or x2 > intrinsics.width - m)
    top = bool(y1 < m)
    bottom = bool(y2 > intrinsics.height - m)
    if not (lateral or top or bottom):
        return False
    narrow = (x2 - x1) < min_fraction * intrinsics.width or (y2 - y1) < min_fraction * intrinsics.height
    if not narrow:
        return False
    floor_cut = (bottom and not lateral and not top and y1 >= intrinsics.cy
                 and (y2 - y1) >= floor_cut_min_fraction * intrinsics.height
                 and (x2 - x1) >= floor_cut_min_width_fraction * intrinsics.width)
    return not floor_cut


def footprint_radius_m(box, depth_m, fx, floor_m=0.15, ceiling_m=1.5):
    """Half the box's width at its depth, as the object's footprint half-extent.

    A disc, not a box: the detector's box is an image-plane extent and the
    agent sees the object from one side, so the plan-view shape is unknown;
    the half-width at the measured depth is the one footprint figure the
    frame supports. Clamped so a sliver and a wall-to-wall box both stay
    inside the range of indoor furniture.
    """
    x1, _, x2, _ = box
    radius = 0.5 * max(0.0, float(x2) - float(x1)) * float(depth_m) / float(fx)
    return float(min(ceiling_m, max(floor_m, radius)))

