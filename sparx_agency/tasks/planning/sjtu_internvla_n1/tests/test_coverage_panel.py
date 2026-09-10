"""Drawing what has been seen: does the wash land on the floor it measured?

The coverage number and the coverage wash come from the same mask, so the one
way they can lie to a viewer is by disagreeing about *where* -- a resampled layer
half a pixel off the backdrop it is painted on, or a wash that erases a wall and
opens a doorway the building does not have.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2", reason="OpenCV needed for rendering")

from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import (  # noqa: E402
    FREE_BGR,
    OCCUPIED_BGR,
    OccupancyMapImage,
)
from sparx_agency.tasks.planning.sjtu_internvla_n1.top_down import (  # noqa: E402
    SEEN_BGR,
    CoverageOverlay,
    TopDownRenderer,
    _coverage_banner,
    _shade_seen,
    _to_px,
)


def _map():
    """A 20 x 30 cell map at 0.5 m, origin (-3, -4), with a wall across the middle."""
    grid = np.full((30, 20), 254, dtype=np.uint8)   # row 0 = min y
    grid[14, :] = 0
    return OccupancyMapImage(grid, resolution=0.5, origin_x=-3.0, origin_y=-4.0)


def _overlay(seen, fraction=0.25):
    return CoverageOverlay(seen=seen, fraction=fraction,
                           area_seen_m2=fraction * 400.0, area_total_m2=400.0)


def test_a_resampled_layer_lands_where_the_overlay_transform_says():
    """The one thing that would make the wash a lie: half a pixel of drift."""
    m = _map()
    win = m.whole((120, 180))
    layer = np.zeros(m.grid.shape, dtype=np.uint8)
    layer[3, 5] = 1                                  # world x[-0.5,0) y[-2.5,-2)
    out = win.resample(layer)
    assert out is not None
    px, py = _to_px(win.image, win.extent, -0.25, -2.25)   # that cell's centre
    assert out[py, px] > 0, "the layer did not land on its own world point"


def test_a_hand_made_window_carries_no_transform_and_resamples_to_none():
    from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import MapWindow
    win = MapWindow(image=np.zeros((4, 4, 3), np.uint8),
                    min_x=0.0, min_y=0.0, max_x=1.0, max_y=1.0)
    assert win.resample(np.ones((4, 4), np.uint8)) is None


def test_the_wash_paints_free_space():
    m = _map()
    win = m.whole((120, 180))
    panel = win.image.copy()
    seen = np.zeros(m.grid.shape, dtype=bool)
    seen[2:10, 2:10] = True
    _shade_seen(panel, win, _overlay(seen))
    washed = np.all(panel == np.array(SEEN_BGR, dtype=np.uint8), axis=-1)
    assert washed.any(), "nothing was washed"
    px, py = _to_px(panel, win.extent, *m_centre_of(m, 5, 5))
    assert washed[py, px]


def test_the_wash_never_erases_a_wall():
    """A seen cell next to a wall must not open a doorway in the drawing.

    The backdrop dilates its walls so a one-cell partition survives a downscale
    while the seen layer is point-sampled at its true width, so without the
    free-pixel test the two disagree by exactly the dilation -- and what a
    viewer reads off the picture is a gap in a wall that is solid.
    """
    m = _map()
    win = m.whole((120, 180))
    panel = win.image.copy()
    walls_before = np.all(panel == np.array(OCCUPIED_BGR, dtype=np.uint8), axis=-1)
    seen = np.ones(m.grid.shape, dtype=bool)          # claim EVERYTHING is seen
    _shade_seen(panel, win, _overlay(seen, 1.0))
    walls_after = np.all(panel == np.array(OCCUPIED_BGR, dtype=np.uint8), axis=-1)
    assert np.array_equal(walls_before, walls_after)
    assert walls_after.any(), "the fixture has no wall to protect"


def test_no_coverage_leaves_the_panel_alone():
    m = _map()
    win = m.whole((120, 180))
    panel = win.image.copy()
    _shade_seen(panel, win, None)
    assert np.array_equal(panel, win.image)


def test_the_banner_carries_the_percentage_and_the_area():
    text = _coverage_banner(_overlay(np.zeros((4, 4), bool), 0.234))
    assert "23.4%" in text and "94 of 400 m2" in text
    assert _coverage_banner(None) == "HOSPITAL"


def test_the_composed_panel_is_exactly_the_width_it_was_asked_for():
    """cv2 drops a wrong-sized frame silently, so this is worth asserting.

    A panel one pixel wide of the VideoWriter's geometry produces a run that
    counts every frame it never wrote and reports a successful recording of a
    258-byte file.
    """
    m = _map()
    renderer = TopDownRenderer(size=(640, 480), backdrop=m,
                               local_span_m=6.0, overview_fraction=0.42)
    seen = np.zeros(m.grid.shape, dtype=bool)
    seen[2:10, 2:10] = True
    panel = renderer.render((0.0, 0.0, 0.0), None, None, None, _overlay(seen))
    assert panel.shape == (480, 640, 3)


def test_rendering_without_a_map_ignores_coverage_rather_than_raising():
    """Graph paper has no building to measure against; it must still draw."""
    renderer = TopDownRenderer(size=(320, 240), backdrop=None)
    renderer.add_pose(0.0, 0.0)
    panel = renderer.render((0.0, 0.0, 0.0), None, None, None,
                            _overlay(np.zeros((4, 4), bool)))
    assert panel.shape == (240, 320, 3)


def m_centre_of(m, gy, gx):
    """World centre of cell ``(gy, gx)``."""
    return (m.origin_x + (gx + 0.5) * m.resolution,
            m.origin_y + (gy + 0.5) * m.resolution)
