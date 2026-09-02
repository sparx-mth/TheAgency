"""Colouring and the world->pixel mapping of the BEV display image."""
import numpy as np

from sparx_agency.tasks.planning.nav_debug.bev_image import occupancy_to_bgr, render_bev
from sparx_agency.tasks.planning.nav_debug.frame import BevMap


def test_occupancy_colours_by_class():
    grid = np.array([[-1, 0, 100]], dtype=np.int8)
    img = occupancy_to_bgr(grid)
    assert img.shape == (1, 3, 3)
    unknown, free, occ = img[0, 0], img[0, 1], img[0, 2]
    # unknown darkest, occupied brightest.
    assert unknown.mean() < free.mean() < occ.mean()


def test_confidence_tints_only_occupied():
    grid = np.array([[100, 100]], dtype=np.int8)
    conf = np.array([[0, 100]], dtype=np.int8)      # low vs high confidence
    img = occupancy_to_bgr(grid, conf)
    # High-confidence occupied is whiter (higher blue+green+red) than low.
    assert img[0, 1].mean() > img[0, 0].mean()


def test_render_bev_flips_y_up_and_maps_origin():
    # 4 rows x 6 cols at 0.5 m, origin at (-1, -2).
    grid = np.zeros((4, 6), np.int8)
    bev = BevMap(grid=grid, resolution=0.5, origin_x=-1.0, origin_y=-2.0)
    img, to_px = render_bev(bev, target_px=60)
    scale = max(1, 60 // 6)                 # integer upscale from the longer edge
    assert img.shape[0] == 4 * scale and img.shape[1] == 6 * scale
    # The origin corner (min x, min y) sits at the image's BOTTOM-left (y is up).
    px, py = to_px(-1.0, -2.0)
    assert px == 0 and py == img.shape[0] - 1
    # Moving +x is +px; moving +y is -py (up the image).
    assert to_px(-0.5, -2.0)[0] > px
    assert to_px(-1.0, -1.5)[1] < py


# ── the follow window ────────────────────────────────────────────────────────
# The Sphera map is ~105 m across, so on a 900 px pane a 30 cm tracking error is
# about two pixels. The follow view crops to a few metres about the aircraft;
# what matters is that the aircraft stays centred even at a map edge, and that
# world points still land where they belong afterwards.

def _grid_bev(h=40, w=60, res=0.1, ox=-2.0, oy=-1.0):
    from sparx_agency.tasks.planning.nav_debug.frame import BevMap
    g = np.full((h, w), -1, np.int8)
    g[10:12, 5:50] = 100          # a thin wall
    return BevMap(grid=g, resolution=res, origin_x=ox, origin_y=oy)


def test_window_is_centred_on_the_requested_point():
    from sparx_agency.tasks.planning.nav_debug import bev_image
    bev = _grid_bev()
    centre = (0.0, 0.5)
    grid, _conf, ox, oy = bev_image.window(bev, None, centre, 1.0)
    assert grid.shape[0] == grid.shape[1]            # square window
    mid = grid.shape[0] // 2
    # the centre cell of the window must be the requested world point
    assert abs((ox + mid * bev.resolution) - centre[0]) < bev.resolution
    assert abs((oy + mid * bev.resolution) - centre[1]) < bev.resolution


def test_window_pads_with_unknown_off_the_map_edge():
    """At an edge the aircraft must stay centred, not slide into a corner."""
    from sparx_agency.tasks.planning.nav_debug import bev_image
    bev = _grid_bev()
    grid, _conf, _ox, _oy = bev_image.window(bev, None, (-2.0, -1.0), 1.0)  # the corner
    mid = grid.shape[0] // 2
    assert grid.shape == (2 * 10 + 1, 2 * 10 + 1)
    assert (grid[:mid, :mid] == -1).all()            # off-map quadrant is unknown
    assert grid.shape[0] == grid.shape[1]


def test_follow_view_maps_world_points_correctly():
    """to_px must still be right after cropping, or every overlay is wrong."""
    from sparx_agency.tasks.planning.nav_debug.bev_image import render_bev
    bev = _grid_bev()
    centre = (0.4, 0.6)
    img, to_px = render_bev(bev, None, 400, center=centre, radius_m=1.0)
    cx, cy = to_px(*centre)
    # the centre of the window lands at the centre of the image (within a cell)
    assert abs(cx - img.shape[1] / 2.0) <= img.shape[1] / 20.0
    assert abs(cy - img.shape[0] / 2.0) <= img.shape[0] / 20.0
    # and a point one metre east is to the right, one metre north is higher up
    assert to_px(centre[0] + 1.0, centre[1])[0] > cx
    assert to_px(centre[0], centre[1] + 1.0)[1] < cy


def test_follow_view_magnifies_versus_the_whole_map():
    """The whole point: a metre must occupy far more pixels in the follow view."""
    from sparx_agency.tasks.planning.nav_debug.bev_image import render_bev
    bev = _grid_bev(h=600, w=1000)                   # a big, Sphera-sized grid
    _full, full_px = render_bev(bev, None, 600)
    _foll, foll_px = render_bev(bev, None, 600, center=(0.0, 0.0), radius_m=5.0)
    metre_full = abs(full_px(1.0, 0.0)[0] - full_px(0.0, 0.0)[0])
    metre_foll = abs(foll_px(1.0, 0.0)[0] - foll_px(0.0, 0.0)[0])
    assert metre_foll > metre_full * 4
