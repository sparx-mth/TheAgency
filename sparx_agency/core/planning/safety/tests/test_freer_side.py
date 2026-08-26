"""Which way to turn when the corridor ahead is shut."""
import numpy as np
import pytest

from sparx_agency.core.planning.safety.depth_proximity_brake import (
    DepthProximityBrakeConfig,
    freer_side,
)

CFG = DepthProximityBrakeConfig(fx=390.6, fy=390.6, cx=300.0, cy=300.0,
                                corridor_halfheight_m=0.35, min_valid_m=0.15,
                                stride=4)


def frame(left_m, right_m):
    d = np.empty((600, 600), np.float32)
    d[:, :300] = left_m
    d[:, 300:] = right_m
    return d


def test_it_turns_toward_the_open_half():
    assert freer_side(frame(6.0, 1.0), CFG) == 1.0      # image-left = body left
    assert freer_side(frame(1.0, 6.0), CFG) == -1.0


def test_a_flat_wall_breaks_left_deterministically():
    # The halves rarely hold the same number of samples, so a symmetric scene
    # differs in the last float32 ulp; without an explicit tie band the
    # "deterministic" answer follows the rounding.
    for wall in (0.3, 0.45, 1.0, 3.0):
        assert freer_side(np.full((600, 600), wall, np.float32), CFG) == 1.0


def test_a_hole_in_the_depth_map_is_open_space_not_a_wall():
    # A doorway is exactly where a depth camera returns nothing. Scoring a miss
    # as zero range would make the one way out read as the most blocked
    # direction in the frame.
    d = frame(1.0, 1.0)
    d[:, :300] = np.nan
    assert freer_side(d, CFG) == 1.0
    d = frame(1.0, 1.0)
    d[:, 300:] = np.inf
    assert freer_side(d, CFG) == -1.0


def test_returns_are_below_the_valid_floor_count_as_open():
    # Self-view and sensor noise, not a surface. Same argument as the miss.
    d = frame(0.01, 1.0)
    assert freer_side(d, CFG) == 1.0


def test_it_never_raises_on_a_degenerate_frame():
    assert freer_side(np.zeros((0, 0), np.float32), CFG) == 1.0
    assert freer_side(np.full((600, 600), np.nan, np.float32), CFG) == 1.0
