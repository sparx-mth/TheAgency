"""Box arithmetic and the points-to-pixels conversion."""
from __future__ import annotations

import pytest

from sparx_agency.tools.pdf_parser.geometry import POINTS_PER_INCH, BBox, union_all


def test_inverted_box_is_rejected():
    with pytest.raises(ValueError):
        BBox(100.0, 0.0, 50.0, 10.0)


def test_dimensions():
    box = BBox(10.0, 20.0, 40.0, 60.0)
    assert box.width == 30.0
    assert box.height == 40.0
    assert box.center_x == 25.0
    assert box.center_y == 40.0


def test_union_covers_both():
    merged = BBox(0.0, 0.0, 10.0, 10.0).union(BBox(20.0, 5.0, 30.0, 25.0))
    assert merged == BBox(0.0, 0.0, 30.0, 25.0)


def test_union_all_returns_none_when_empty():
    assert union_all([]) is None


def test_x_overlap_ratio_uses_the_narrower_box():
    """A one-column figure under a full-width caption must still read as aligned."""
    narrow = BBox(100.0, 0.0, 200.0, 10.0)
    wide = BBox(50.0, 0.0, 500.0, 10.0)
    assert narrow.x_overlap_ratio(wide) == pytest.approx(1.0)


def test_disjoint_boxes_do_not_overlap():
    assert BBox(0.0, 0.0, 10.0, 10.0).x_overlap(BBox(20.0, 0.0, 30.0, 10.0)) == 0.0


def test_vertical_gap_is_zero_when_boxes_overlap():
    assert BBox(0.0, 0.0, 10.0, 20.0).vertical_gap(BBox(0.0, 10.0, 10.0, 30.0)) == 0.0
    assert BBox(0.0, 0.0, 10.0, 10.0).vertical_gap(BBox(0.0, 25.0, 10.0, 30.0)) == 15.0


def test_padding_is_clipped_to_the_page():
    page = BBox(0.0, 0.0, 612.0, 792.0)
    padded = BBox(2.0, 2.0, 100.0, 100.0).padded(10.0, page)
    assert padded.x_min == 0.0
    assert padded.y_min == 0.0
    assert padded.x_max == 110.0


def test_to_pixels_scales_by_dpi():
    """One inch of page at 150 DPI is 150 pixels."""
    box = BBox(0.0, 0.0, POINTS_PER_INCH, POINTS_PER_INCH)
    assert box.to_pixels(150.0) == (0, 0, 150, 150)


def test_to_pixels_never_returns_zero_extent():
    _, _, width, height = BBox(10.0, 10.0, 10.05, 10.05).to_pixels(72.0)
    assert width >= 1 and height >= 1
