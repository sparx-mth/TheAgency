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
