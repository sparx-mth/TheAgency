"""The map backdrop: does a world point land on the pixel it should?

These are the tests that would have caught the two defects the first hospital
render actually had -- a whole-map view zoomed out until the building was a
stripe, and a backdrop mirrored in y against the overlay drawn on it. Neither
looks broken in a video; both put the drone in the wrong room.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2", reason="OpenCV needed for rendering")
import cv2  # noqa: E402

from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import (  # noqa: E402
    FREE_BGR,
    OCCUPIED_BGR,
    OccupancyMapImage,
    load_map_backdrop,
)
from sparx_agency.tasks.planning.sjtu_internvla_n1.top_down import _to_px  # noqa: E402


def _map_with_marked_bottom_left():
    """A 10 x 20 cell map at 0.5 m, origin (-2, -5), occupied only near min y."""
    grid = np.full((20, 10), 254, dtype=np.uint8)  # row 0 = min y, by contract
    grid[0:2, 0:2] = 0                             # world x[-2,-1] y[-5,-4]
    return OccupancyMapImage(grid, resolution=0.5, origin_x=-2.0, origin_y=-5.0)


def test_map_extent_follows_origin_and_resolution():
    m = _map_with_marked_bottom_left()
    assert m.width == 10 and m.height == 20
    assert m.max_x == pytest.approx(3.0)   # -2 + 10 * 0.5
    assert m.max_y == pytest.approx(5.0)   # -5 + 20 * 0.5


def test_window_is_not_mirrored_in_y():
    """The occupied corner is at MINIMUM y, so it must render near the BOTTOM.

    A y-flip here is the failure that does not look like one: the building is
    still a building, the route is still a route, and they disagree.
    """
    m = _map_with_marked_bottom_left()
    win = m.whole((100, 200))
    occupied = np.all(win.image == np.array(OCCUPIED_BGR, dtype=np.uint8), axis=-1)
    rows = np.nonzero(occupied.any(axis=1))[0]
    assert rows.size, "the occupied corner vanished from the rendered window"
    # Row 0 is the top of a picture; minimum y must therefore be in the lower half.
    assert rows.min() > win.image.shape[0] // 2


def test_a_world_point_lands_on_its_own_pixel():
    """The extent a window reports must be the one the overlay draws through."""
    m = _map_with_marked_bottom_left()
    win = m.whole((100, 200))
    # The middle of the occupied corner, in world metres.
    px, py = _to_px(win.image, win.extent, -1.5, -4.5)
    assert np.array_equal(win.image[py, px], np.array(OCCUPIED_BGR, dtype=np.uint8))
    # ...and a point in the middle of the free area is free.
    px, py = _to_px(win.image, win.extent, 0.5, 0.0)
    assert np.array_equal(win.image[py, px], np.array(FREE_BGR, dtype=np.uint8))


def test_whole_fills_the_panel_on_the_limiting_axis():
    """A tall map must fill a panel's height, not be shrunk to fit its width."""
    m = _map_with_marked_bottom_left()          # 5 m wide, 10 m tall
    win = m.whole((100, 200))                   # panel of the same 1:2 aspect
    assert win.max_y - win.min_y == pytest.approx(10.0, abs=0.05)
    assert win.max_x - win.min_x == pytest.approx(5.0, abs=0.05)


def test_window_span_is_the_short_side():
    m = _map_with_marked_bottom_left()
    win = m.window(0.0, 0.0, span_m=4.0, size=(100, 200))
    assert min(win.max_x - win.min_x, win.max_y - win.min_y) == pytest.approx(4.0)
    assert win.image.shape[:2] == (200, 100)


def test_window_rejects_a_degenerate_size():
    m = _map_with_marked_bottom_left()
    with pytest.raises(ValueError):
        m.window(0.0, 0.0, span_m=4.0, size=(0, 10))


def test_missing_map_is_no_backdrop_not_a_crash(tmp_path):
    assert load_map_backdrop("") is None
    assert load_map_backdrop(str(tmp_path / "nope.yaml")) is None


def test_a_configured_but_broken_map_raises(tmp_path):
    """Silently ignoring a map that IS configured would fly a lie."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("image: missing.pgm\n")
    with pytest.raises(ValueError):
        load_map_backdrop(str(bad))


def test_round_trip_through_a_real_nav2_yaml(tmp_path):
    """Load the same map back through the nav2 convention: row 0 is MAXIMUM y."""
    grid = np.full((20, 10), 254, dtype=np.uint8)
    grid[-2:, 0:2] = 0                    # last rows of a nav2 pgm = minimum y
    cv2.imwrite(str(tmp_path / "m.pgm"), grid)
    (tmp_path / "m.yaml").write_text(
        "image: m.pgm\nresolution: 0.5\norigin: [-2.0, -5.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\nmode: trinary\n")
    m = OccupancyMapImage.from_yaml(str(tmp_path / "m.yaml"))
    # After the load's flip, the occupied cells are at row 0 -- minimum y.
    assert (m.grid[0:2, 0:2] == 0).all()
    assert (m.grid[-2:, :] == 254).all()
