"""Unit tests for pure bbox geometry in ``core.common.math.bbox``.

Covers the stateless image-plane box arithmetic (centre/size/area/area-fraction,
signed normalised centre offset, clipping, IoU, xyxy<->cxcywh round-trip, and the
robust ``bounds_rect``) plus the :class:`Track2D` convenience properties that
mirror the same math.
"""
import numpy as np
import pytest

from sparx_agency.core.common.math.bbox import (
    area_frac,
    bounds_rect,
    center_offset_norm,
    clip_xyxy,
    cxcywh_to_xyxy,
    iou,
    xyxy_area,
    xyxy_center,
    xyxy_size,
    xyxy_to_cxcywh,
)
from sparx_agency.core.common.types.perception import Track2D


# --------------------------------------------------------------------------- #
# xyxy_center / xyxy_size / xyxy_area / area_frac
# --------------------------------------------------------------------------- #
def test_xyxy_center():
    assert xyxy_center((10.0, 20.0, 30.0, 50.0)) == (20.0, 35.0)


def test_xyxy_size():
    assert xyxy_size((10.0, 20.0, 30.0, 50.0)) == (20.0, 30.0)


def test_xyxy_size_clamped_non_negative():
    # x2 < x1 and y2 < y1 -> both dimensions clamped to 0.
    assert xyxy_size((30.0, 50.0, 10.0, 20.0)) == (0.0, 0.0)


def test_xyxy_area():
    assert xyxy_area((10.0, 20.0, 30.0, 50.0)) == pytest.approx(600.0)


def test_xyxy_area_degenerate_is_zero():
    assert xyxy_area((5.0, 5.0, 5.0, 5.0)) == 0.0


def test_area_frac():
    # Box area 600 over a 100x200 image -> 0.03.
    assert area_frac((10.0, 20.0, 30.0, 50.0), 100, 200) == pytest.approx(0.03)


def test_area_frac_denominator_floored():
    # Zero-sized frame is floored to a 1x1 denominator (no div-by-zero).
    assert area_frac((0.0, 0.0, 2.0, 3.0), 0, 0) == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# center_offset_norm  (SIGN semantics + clamping)
# --------------------------------------------------------------------------- #
def test_center_offset_centred_is_zero():
    ox, oy = center_offset_norm((45.0, 45.0, 55.0, 55.0), 100, 100)
    assert ox == pytest.approx(0.0)
    assert oy == pytest.approx(0.0)


def test_center_offset_box_right_is_positive_x():
    # Box centre at (75, 50) in a 100x100 image -> +x, oy ~ 0.
    ox, oy = center_offset_norm((70.0, 45.0, 80.0, 55.0), 100, 100)
    assert ox == pytest.approx(0.5)
    assert oy == pytest.approx(0.0)


def test_center_offset_box_below_is_positive_y():
    # Box centre at (50, 75) in a 100x100 image -> +y, ox ~ 0.
    ox, oy = center_offset_norm((45.0, 70.0, 55.0, 80.0), 100, 100)
    assert ox == pytest.approx(0.0)
    assert oy == pytest.approx(0.5)


def test_center_offset_box_left_is_negative_x():
    ox, _ = center_offset_norm((20.0, 45.0, 30.0, 55.0), 100, 100)
    assert ox == pytest.approx(-0.5)


def test_center_offset_box_above_is_negative_y():
    _, oy = center_offset_norm((45.0, 20.0, 55.0, 30.0), 100, 100)
    assert oy == pytest.approx(-0.5)


def test_center_offset_clamped_to_unit_range():
    # Centre far to the right/below the image -> saturates at +1.
    ox, oy = center_offset_norm((190.0, 190.0, 210.0, 210.0), 100, 100)
    assert ox == pytest.approx(1.0)
    assert oy == pytest.approx(1.0)
    # Centre far to the left/above -> saturates at -1.
    ox, oy = center_offset_norm((-210.0, -210.0, -190.0, -190.0), 100, 100)
    assert ox == pytest.approx(-1.0)
    assert oy == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# clip_xyxy
# --------------------------------------------------------------------------- #
def test_clip_inside_unchanged():
    box = (10.0, 20.0, 30.0, 40.0)
    assert clip_xyxy(box, 100, 100) == box


def test_clip_outside_is_bounded():
    clipped = clip_xyxy((-5.0, -10.0, 150.0, 250.0), 100, 200)
    assert clipped == (0.0, 0.0, 100.0, 200.0)


def test_clip_keeps_min_max_ordering():
    # A box wholly to the right of the frame collapses onto the right edge,
    # and the result stays ordered (x1<=x2, y1<=y2).
    x1, y1, x2, y2 = clip_xyxy((150.0, 50.0, 130.0, 70.0), 100, 100)
    assert x1 <= x2 and y1 <= y2
    assert (x1, x2) == (100.0, 100.0)


# --------------------------------------------------------------------------- #
# iou
# --------------------------------------------------------------------------- #
def test_iou_partial_overlap():
    # (0,0,2,2) & (1,1,3,3): inter=1, union=7 -> 1/7.
    assert iou((0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0)) == pytest.approx(1.0 / 7.0)


def test_iou_disjoint_is_zero():
    assert iou((0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)) == 0.0


def test_iou_identical_is_one():
    box = (4.0, 6.0, 14.0, 26.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_degenerate_boxes_zero_union():
    # Two zero-area boxes -> union 0 -> defined as 0 (no div-by-zero).
    assert iou((1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0)) == 0.0


# --------------------------------------------------------------------------- #
# xyxy <-> cxcywh round-trip
# --------------------------------------------------------------------------- #
def test_xyxy_to_cxcywh():
    assert xyxy_to_cxcywh((10.0, 20.0, 30.0, 50.0)) == (20.0, 35.0, 20.0, 30.0)


def test_cxcywh_to_xyxy():
    assert cxcywh_to_xyxy((20.0, 35.0, 20.0, 30.0)) == (10.0, 20.0, 30.0, 50.0)


def test_xyxy_cxcywh_round_trip():
    box = (12.5, 7.25, 88.0, 63.5)
    assert cxcywh_to_xyxy(xyxy_to_cxcywh(box)) == pytest.approx(box)


# --------------------------------------------------------------------------- #
# bounds_rect
# --------------------------------------------------------------------------- #
def _grid_cloud():
    """5x5 grid over [0,10] x [0,20] (clean, no outliers)."""
    xs = np.linspace(0.0, 10.0, 5)
    ys = np.linspace(0.0, 20.0, 5)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def test_bounds_rect_exact_min_max():
    pts = np.array([[1.0, 2.0], [5.0, -3.0], [-4.0, 8.0], [0.0, 0.0]])
    assert bounds_rect(pts, k_mad=0.0) == (-4.0, -3.0, 5.0, 8.0)


def test_bounds_rect_clean_cloud_not_shrunk():
    # For a clean cloud, robust rejection must not shrink the box relative to
    # the exact min/max rectangle.
    pts = _grid_cloud()
    exact = bounds_rect(pts, k_mad=0.0)
    robust = bounds_rect(pts, k_mad=3.0)
    assert robust == exact == (0.0, 0.0, 10.0, 20.0)


def test_bounds_rect_rejects_single_outlier():
    # One far point jumps onto the background; k_mad>0 drops it and the box
    # collapses back to the clean-cloud bounds (not the outlier-inflated ones).
    pts = np.vstack([_grid_cloud(), np.array([[100.0, 100.0]])])
    inflated = bounds_rect(pts, k_mad=0.0)
    robust = bounds_rect(pts, k_mad=3.0)
    assert inflated == (0.0, 0.0, 100.0, 100.0)
    assert robust == (0.0, 0.0, 10.0, 20.0)


def test_bounds_rect_accepts_list_input():
    # Non-ndarray sequence is coerced via np.asarray.
    assert bounds_rect([[0.0, 0.0], [3.0, 4.0]], k_mad=0.0) == (0.0, 0.0, 3.0, 4.0)


def test_bounds_rect_empty_raises_value_error():
    with pytest.raises(ValueError):
        bounds_rect(np.empty((0, 2), dtype=np.float64))


# --------------------------------------------------------------------------- #
# Track2D convenience properties
# --------------------------------------------------------------------------- #
def _track():
    return Track2D(
        label="target",
        bbox_xyxy=(10.0, 20.0, 30.0, 50.0),
        frame_w=100,
        frame_h=200,
    )


def test_track2d_center_props():
    t = _track()
    assert t.cx == pytest.approx(20.0)
    assert t.cy == pytest.approx(35.0)


def test_track2d_size_props():
    t = _track()
    assert t.w == pytest.approx(20.0)
    assert t.h == pytest.approx(30.0)


def test_track2d_size_props_clamped_non_negative():
    # Inverted box -> width/height clamped to 0.
    t = Track2D(label="x", bbox_xyxy=(30.0, 50.0, 10.0, 20.0), frame_w=100, frame_h=100)
    assert t.w == 0.0
    assert t.h == 0.0
    assert t.area == 0.0


def test_track2d_area_and_frac():
    t = _track()
    assert t.area == pytest.approx(600.0)
    # 600 / (100*200) = 0.03.
    assert t.area_frac == pytest.approx(0.03)


def test_track2d_area_frac_matches_free_function():
    t = _track()
    assert t.area_frac == pytest.approx(area_frac(t.bbox_xyxy, t.frame_w, t.frame_h))
