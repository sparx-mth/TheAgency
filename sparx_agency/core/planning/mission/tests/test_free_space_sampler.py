"""Sampled missions must be clear at both ends, reachable, and worth flying."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import occupancy_from_mask
from sparx_agency.core.planning.mission import (
    connected_regions, largest_region, sample_goal_from, sample_start_goal,
    snap_to_region, traversable_mask,
)

RES = 0.5


def _open_room(width_cells: int = 40, height_cells: int = 40):
    """An empty room with a one-cell wall all the way round."""
    occupied = np.zeros((height_cells, width_cells), dtype=bool)
    occupied[0, :] = occupied[-1, :] = True
    occupied[:, 0] = occupied[:, -1] = True
    return occupancy_from_mask(occupied, RES, 0.0, 0.0)


def _two_rooms():
    """Two open rooms separated by a solid wall -- no route between them."""
    occupied = np.zeros((40, 40), dtype=bool)
    occupied[0, :] = occupied[-1, :] = True
    occupied[:, 0] = occupied[:, -1] = True
    occupied[:, 20] = True
    return occupancy_from_mask(occupied, RES, 0.0, 0.0)


def test_traversable_mask_excludes_cells_too_close_to_walls():
    grid = _open_room()
    mask = traversable_mask(grid, clearance_m=1.0)

    assert not mask[1, 1]      # one cell (0.5 m) from two walls
    assert mask[20, 20]        # middle of the room
    # No traversable cell may be within the clearance of the wall at gy=0.
    assert not mask[:2, :].any()


def test_traversable_mask_treats_unknown_as_blocked():
    occupied = np.zeros((10, 10), dtype=bool)
    known = np.ones((10, 10), dtype=bool)
    known[5, 5] = False
    grid = occupancy_from_mask(occupied, RES, 0.0, 0.0, known=known)

    assert not traversable_mask(grid, clearance_m=0.0)[5, 5]


def test_connected_regions_separates_disconnected_rooms():
    regions = connected_regions(traversable_mask(_two_rooms(), clearance_m=0.6))
    assert len(regions) == 2
    assert not (regions[0] & regions[1]).any()


def test_connected_regions_sorted_largest_first():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:2, 0:2] = True      # 4 cells
    mask[5:9, 5:9] = True      # 16 cells
    regions = connected_regions(mask)
    assert [int(r.sum()) for r in regions] == [16, 4]


def test_largest_region_raises_when_nothing_is_clear_enough():
    with pytest.raises(ValueError, match="clearance"):
        largest_region(_open_room(), clearance_m=50.0)


def test_sampled_ends_are_clear_and_far_enough():
    grid = _open_room()
    rng = np.random.default_rng(0)
    region = largest_region(grid, clearance_m=1.0)

    for _ in range(40):
        mission = sample_start_goal(grid, rng, clearance_m=1.0,
                                    min_separation_m=5.0, region=region)
        for pose in (mission.start, mission.goal):
            gx, gy = grid.world_to_grid(pose.x, pose.y)
            assert region[gy, gx], "an end landed outside the traversable region"
        assert mission.separation_m >= 5.0


def test_sampled_ends_are_always_in_the_same_room():
    """The whole point of sampling from one component: no unreachable goals."""
    grid = _two_rooms()
    rng = np.random.default_rng(3)
    region = largest_region(grid, clearance_m=0.6)

    for _ in range(30):
        mission = sample_start_goal(grid, rng, clearance_m=0.6,
                                    min_separation_m=2.0, region=region)
        sx, _ = grid.world_to_grid(mission.start.x, mission.start.y)
        gx, _ = grid.world_to_grid(mission.goal.x, mission.goal.y)
        assert (sx < 20) == (gx < 20), "start and goal ended up either side of the wall"


def test_start_yaw_points_at_the_goal_by_default():
    mission = sample_start_goal(_open_room(), np.random.default_rng(7),
                                clearance_m=1.0, min_separation_m=5.0)
    expected = math.atan2(mission.goal.y - mission.start.y,
                          mission.goal.x - mission.start.x)
    assert mission.start.yaw == pytest.approx(expected)


def test_start_yaw_jitter_stays_within_its_bound():
    grid = _open_room()
    rng = np.random.default_rng(11)
    region = largest_region(grid, clearance_m=1.0)
    jitter = 0.4

    for _ in range(30):
        mission = sample_start_goal(grid, rng, clearance_m=1.0, min_separation_m=5.0,
                                    start_yaw_jitter_rad=jitter, region=region)
        bearing = math.atan2(mission.goal.y - mission.start.y,
                             mission.goal.x - mission.start.x)
        error = abs(math.atan2(math.sin(mission.start.yaw - bearing),
                               math.cos(mission.start.yaw - bearing)))
        assert error <= jitter + 1e-9


def test_max_separation_is_respected():
    grid = _open_room()
    rng = np.random.default_rng(5)
    region = largest_region(grid, clearance_m=1.0)

    for _ in range(30):
        mission = sample_start_goal(grid, rng, clearance_m=1.0, min_separation_m=3.0,
                                    max_separation_m=6.0, region=region)
        assert 3.0 <= mission.separation_m <= 6.0


def test_same_seed_gives_the_same_mission():
    grid = _open_room()
    a = sample_start_goal(grid, np.random.default_rng(42), min_separation_m=4.0)
    b = sample_start_goal(grid, np.random.default_rng(42), min_separation_m=4.0)
    assert (a.start.x, a.start.y, a.goal.x, a.goal.y) == (b.start.x, b.start.y,
                                                          b.goal.x, b.goal.y)


def test_unreachable_separation_raises_rather_than_looping_forever():
    grid = _open_room()
    with pytest.raises(ValueError, match="from the sampled start"):
        sample_start_goal(grid, np.random.default_rng(1), clearance_m=1.0,
                          min_separation_m=500.0)


def test_sample_goal_from_respects_distance_bounds():
    grid = _open_room()
    rng = np.random.default_rng(9)
    region = largest_region(grid, clearance_m=1.0)
    start = snap_to_region(grid, region, 5.0, 5.0)

    goal, separation = sample_goal_from(grid, rng, start, clearance_m=1.0,
                                        min_separation_m=4.0, max_separation_m=8.0,
                                        region=region)
    assert 4.0 <= separation <= 8.0
    assert separation == pytest.approx(math.hypot(goal.x - start.x, goal.y - start.y))
    gx, gy = grid.world_to_grid(goal.x, goal.y)
    assert region[gy, gx]


def test_snap_to_region_pulls_an_off_map_pose_back_in():
    grid = _open_room()
    region = largest_region(grid, clearance_m=1.0)

    snapped = snap_to_region(grid, region, -50.0, -50.0)

    gx, gy = grid.world_to_grid(snapped.x, snapped.y)
    assert region[gy, gx]
