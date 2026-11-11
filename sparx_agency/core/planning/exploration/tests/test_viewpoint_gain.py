"""Expected useful area from a viewpoint: the MapEx scorer with this camera's sensor model (2026-11-11)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.viewpoint_gain import (
    ExpectedView, SensorModel, arc_uncovered, near_floor_mask, sensor_from_camera)

RES = 0.1
FREE, OCC, UNK = 0, 100, -1
VALUES = OccupancyValues(free=FREE, occupied=OCC, unknown=UNK)
HALF_FOV = math.atan(320.0 / 388.1)            # the benchmark's 79-degree camera


def world(data):
    return OccupancyGrid2D(data, OccupancyGrid2DParams(RES, -10.0, -10.0, "world"), values=VALUES)


def cells(x0, x1, y0, y1):
    return (slice(int(round((y0 + 10) / RES)), int(round((y1 + 10) / RES))),
            slice(int(round((x0 + 10) / RES)), int(round((x1 + 10) / RES))))


def closed_room(doorway=False):
    """A 6 x 6 m room about the origin, walls known; with ``doorway`` a 1 m gap in the east wall onto unknown."""
    g = np.full((200, 200), UNK, np.int8)
    g[cells(-3.0, 3.0, -3.0, 3.0)] = FREE
    g[cells(-3.1, 3.1, 3.0, 3.1)] = OCC
    g[cells(-3.1, 3.1, -3.1, -3.0)] = OCC
    g[cells(-3.1, -3.0, -3.1, 3.1)] = OCC
    g[cells(3.0, 3.1, -3.1, 3.1)] = OCC
    if doorway:
        g[cells(3.0, 3.1, -0.5, 0.5)] = FREE
    return world(g)


def sensor(detection_range_m=5.0, free_run=2.0):
    return sensor_from_camera(640, 388.1, detection_range_m, 0.5, free_run)


def test_the_sensor_model_checks_its_domain_and_derives_the_camera_cone():
    s = sensor()
    assert s.half_fov_rad == pytest.approx(HALF_FOV) and s.detection_range_m == 5.0 and s.blind_range_m == 0.5
    assert s.rotation_turns(math.radians(30)) == 10, "ten turns show the 79-degree camera the circle"
    for bad in (dict(half_fov_rad=0.0, detection_range_m=5.0), dict(half_fov_rad=1.0, detection_range_m=0.0),
                dict(half_fov_rad=1.0, detection_range_m=3.0, blind_range_m=3.0),
                dict(half_fov_rad=1.0, detection_range_m=3.0, unknown_free_run_m=0.0),
                dict(half_fov_rad=1.0, detection_range_m=3.0, unknown_free_run_m=True)):
        with pytest.raises(ValueError):
            SensorModel(**bad)
    with pytest.raises(ValueError, match="resolved"):
        ExpectedView(closed_room(), s, resolved=np.zeros((10, 10), bool))


def test_a_fully_known_room_promises_nothing_and_a_doorway_promises_only_through_itself():
    s = sensor()
    assert ExpectedView(closed_room(), s).circle_gain((0.0, 0.0)) == 0.0
    view = ExpectedView(closed_room(doorway=True), s)
    profile = view.profile((0.0, 0.0))
    assert profile.sum() > 0.5
    east, west = view.frame_gains(profile, [0.0, math.pi])
    assert east == pytest.approx(profile.sum()) and west == 0.0, "every ray that finds unknown passes the doorway"
    # Standing at the doorway the rays fan out into the unknown room: far more to see than from the middle.
    assert view.circle_gain((2.9, 0.0)) > 3 * view.circle_gain((0.0, 0.0))


def test_the_survival_prior_integrates_to_the_exponential_and_none_is_mapex_transparency():
    """On an all-unknown map a 30-degree sliver is the polar integral of ``exp(-(r - blind) / lambda) r`` over it."""
    data = np.full((200, 200), UNK, np.int8)
    view = ExpectedView(world(data), sensor(5.0, 2.0))
    profile = view.profile((0.0, 0.0))
    sliver = view.sector_gain((0.0, 0.0), 0.0, math.radians(15), profile)
    r = np.linspace(0.5, 5.0, 100001)
    analytic = np.trapz(np.exp(-(r - 0.5) / 2.0) * r, r) * math.pi / 6
    assert sliver == pytest.approx(analytic, rel=0.01)
    assert profile.sum() == pytest.approx(12 * analytic, rel=0.01)
    optimistic = ExpectedView(world(data), sensor(5.0, None)).circle_gain((0.0, 0.0))
    assert optimistic == pytest.approx(math.pi * (5.0 ** 2 - 0.5 ** 2), rel=0.02), "no prior: the whole disc"
    assert optimistic > profile.sum() * 3


def test_a_wall_ends_the_ray_and_the_detection_range_caps_it():
    """Unknown beyond a known wall is not seen; unknown beyond the detection range is not useful."""
    g = np.full((200, 200), UNK, np.int8)
    g[cells(-0.3, 2.0, -0.3, 0.3)] = FREE                # a known strip ahead
    g[cells(2.0, 2.1, -3.0, 3.0)] = OCC                  # a wall across it at 2 m
    walled = ExpectedView(world(g), sensor(5.0, None))
    ahead = walled.sector_gain((0.0, 0.0), 0.0, math.radians(5))
    assert ahead < 0.05, "the wall at 2 m stops every ray in the narrow sector ahead: %.3f" % ahead
    open_g = np.full((200, 200), UNK, np.int8)
    open_g[cells(-0.3, 2.0, -0.3, 0.3)] = FREE
    far = ExpectedView(world(open_g), sensor(5.0, None)).sector_gain((0.0, 0.0), 0.0, math.radians(5))
    near = ExpectedView(world(open_g), sensor(3.0, None)).sector_gain((0.0, 0.0), 0.0, math.radians(5))
    assert far > near > 0.0
    # The cap is the integral's upper limit: area between 2 and 3 m against 2 and 5 m in a 10-degree sector.
    assert near == pytest.approx(0.5 * (3.0 ** 2 - 2.0 ** 2) * math.radians(10), rel=0.05)
    assert far == pytest.approx(0.5 * (5.0 ** 2 - 2.0 ** 2) * math.radians(10), rel=0.05)


def test_resolved_cells_block_and_do_not_count_while_ignored_cells_neither_block_nor_count():
    g = np.full((200, 200), UNK, np.int8)
    g[cells(-0.3, 1.0, -0.3, 0.3)] = FREE
    s = sensor(5.0, None)
    plain = ExpectedView(world(g), s).sector_gain((0.0, 0.0), 0.0, math.radians(5))
    band = np.zeros((200, 200), bool)
    band[cells(1.0, 1.5, -3.0, 3.0)] = True             # the half metre of unknown right past the known strip
    settled = ExpectedView(world(g), s, resolved=band).sector_gain((0.0, 0.0), 0.0, math.radians(5))
    assert settled == 0.0, "a settled band is a wall to the rays"
    skipped = ExpectedView(world(g), s, ignore=band).sector_gain((0.0, 0.0), 0.0, math.radians(5))
    assert 0.0 < skipped < plain
    assert plain - skipped == pytest.approx(0.5 * (1.5 ** 2 - 1.0 ** 2) * math.radians(10), rel=0.1)


def test_sector_masks_wrap_the_circle_and_frame_gains_read_the_profile():
    view = ExpectedView(closed_room(), sensor())
    full = view.sector_mask(0.0, math.pi)
    assert full.all()
    behind = view.sector_mask(math.pi, math.radians(15))
    assert behind.sum() == pytest.approx(view.rays / 12, abs=2), "a 30-degree sector straddling +-pi"
    assert not view.sector_mask(0.0, 0.0).any()
    profile = np.ones(view.rays)
    gains = view.frame_gains(profile, [0.0, math.pi / 2])
    assert gains[0] == pytest.approx(gains[1]) and gains[0] == pytest.approx(view.rays * (2 * HALF_FOV) / (2 * math.pi), rel=0.02)


def test_arc_uncovered_measures_what_no_frame_from_the_spot_has_covered():
    half = HALF_FOV
    assert arc_uncovered([], 0.0, half) == pytest.approx(2 * half)
    assert arc_uncovered([(-half, half)], 0.0, half) == 0.0
    after_left_turn = arc_uncovered([(-half, half)], math.radians(30), half)
    assert after_left_turn == pytest.approx(math.radians(30), abs=1e-3), "one turn adds exactly one sliver"
    # The frame before the first turn and ten turns of 30 degrees -- eleven frames -- cover the circle (300 + 79
    # degrees); the first ten leave an 11-degree gap, which is why the rotation is ten turns, not nine.
    eleven = [(math.radians(30 * k) - half, math.radians(30 * k) + half) for k in range(11)]
    ten = eleven[:10]
    assert arc_uncovered(eleven, 0.0, math.pi) == pytest.approx(0.0, abs=1e-3)
    assert arc_uncovered(ten, 0.0, math.pi) == pytest.approx(math.radians(360 - 270 - 79), abs=1e-2)
    assert arc_uncovered([(0.0, 2 * math.pi)], 1.0, 0.3) == 0.0


def test_near_floor_mask_is_the_floor_inside_the_frame_cone_under_the_horizon_not_a_disc():
    w = world(np.full((200, 200), UNK, np.int8))
    mask = near_floor_mask(w, [(0.0, 0.0, 0.0)], 1.4, HALF_FOV)
    gx, gy = w.world_to_grid(0.0, 0.0)
    assert mask[gy, gx], "the agent's own cell"
    assert mask[gy, gx + 10] and not mask[gy, gx - 10], "a metre ahead is under the horizon, a metre behind was never looked at"
    assert not mask[gy, gx + 16], "beyond the horizon the floor is sampled"
    assert mask[gy + 5, gx + 8] and not mask[gy + 8, gx + 2], "inside the 79-degree cone, not beside it"
    assert not near_floor_mask(w, [], 1.4, HALF_FOV).any()
    assert not near_floor_mask(w, [(0.0, 0.0, 0.0)], 0.0, HALF_FOV).any()
