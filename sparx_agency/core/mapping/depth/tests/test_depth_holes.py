"""Enclosed depth holes are filled from the surface around them; open or giant voids are left as no reading."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.mapping.depth.depth_holes import DepthHoleFill, fill_depth_holes


def wall(depth_m=2.0, shape=(48, 64)):
    return np.full(shape, depth_m, np.float32)


def test_a_screen_in_a_wall_takes_the_walls_depth():
    depth = wall(2.0)
    depth[10:30, 20:44] = 0.0                                # a glossy screen: no return
    holes = depth <= 0
    filled, stats = fill_depth_holes(depth, holes, DepthHoleFill())
    assert stats == {"holes": 1, "filled": 1, "pixels_filled": 20 * 24, "pixels_left": 0}
    assert np.all(filled[10:30, 20:44] == 2.0) and np.all(filled == 2.0)
    assert depth[15, 30] == 0.0, "the input is not modified"


def test_the_fill_is_the_nearest_valid_pixel_not_an_average():
    depth = wall(2.0)
    depth[:, 32:] = 3.0                                      # a step in the wall: 2 m on the left, 3 m on the right
    depth[16:32, 24:40] = 0.0                                # a hole across the step
    filled, _ = fill_depth_holes(depth, depth <= 0, DepthHoleFill())
    assert filled[20, 25] == 2.0 and filled[20, 38] == 3.0, "each side of the hole continues its own surface"


def test_a_void_larger_than_the_fraction_is_left_as_no_reading():
    depth = wall(2.0, (40, 40))
    depth[2:38, 2:38] = 0.0                                  # 81 % of the frame
    filled, stats = fill_depth_holes(depth, depth <= 0, DepthHoleFill(max_fraction=0.5))
    assert stats["filled"] == 0 and stats["pixels_left"] == 36 * 36 and np.all(filled[5, 5] == 0.0)
    filled, stats = fill_depth_holes(depth, depth <= 0, DepthHoleFill(max_fraction=0.9))
    assert stats["filled"] == 1 and np.all(filled == 2.0)


def test_a_hole_open_to_the_image_edge_is_not_enclosed():
    """A window at the top of the frame: wall on three sides, the image border on the fourth -- 70 % of
    its rim is still valid, so it is filled; the same hole spanning the width, open on two edges, is not."""
    depth = wall(2.0, (48, 64))
    depth[0:12, 20:44] = 0.0                                 # rim: 12 + 12 + 24 interior + 24 border pixels
    filled, stats = fill_depth_holes(depth, depth <= 0, DepthHoleFill(min_rim=0.5))
    assert stats["filled"] == 1
    depth = wall(2.0, (48, 64))
    depth[0:12, :] = 0.0                                     # open on the top, left and right edges
    _, stats = fill_depth_holes(depth, depth <= 0, DepthHoleFill(min_rim=0.5))
    assert stats["filled"] == 0, "only the bottom edge of the hole is surface: 64 of 64 + 24 + 64 + 24 rim pixels"
    _, stats = fill_depth_holes(depth, depth <= 0, DepthHoleFill(min_rim=0.3))
    assert stats["filled"] == 1


def test_disabled_rule_and_frames_without_valid_depth_pass_through():
    depth = wall(2.0)
    depth[10:20, 10:20] = 0.0
    same, stats = fill_depth_holes(depth, depth <= 0, DepthHoleFill(enabled=False))
    assert same is depth and stats["filled"] == 0 and stats["pixels_left"] == 100
    nothing = np.zeros((8, 8), np.float32)
    same, stats = fill_depth_holes(nothing, nothing <= 0, DepthHoleFill())
    assert same is nothing and stats["filled"] == 0


def test_the_rule_is_validated():
    with pytest.raises(ValueError):
        DepthHoleFill(max_fraction=1.5)
    with pytest.raises(ValueError):
        DepthHoleFill(min_rim=-0.1)
    with pytest.raises(ValueError):
        DepthHoleFill(enabled="yes")
