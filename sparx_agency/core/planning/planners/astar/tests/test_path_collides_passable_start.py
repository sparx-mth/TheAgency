"""path_collides + the passable-start exemption (drone sitting in its own skirt)."""
import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.planners.astar import (
    WeightedAStarParams, WeightedAStarPlanner2D)

BEV = OccupancyValues(free=0, occupied=100, unknown=-1)


def _grid(data, res=0.1):
    return OccupancyGrid2D(
        data.astype(np.int16),
        OccupancyGrid2DParams(resolution=res, origin_x=0.0, origin_y=0.0,
                              frame_id="world"),
        values=BEV)


def test_clear_path_no_collision():
    g = _grid(np.zeros((30, 30), np.int16))
    planner = WeightedAStarPlanner2D(WeightedAStarParams(inflate_radius_m=0.0))
    assert not planner.path_collides(g, [Pose2D(0.5, 1.5), Pose2D(2.5, 1.5)])


def test_obstacle_on_path_detected():
    data = np.zeros((30, 30), np.int16)
    data[15, 20] = 100  # wall cell on the horizontal route at row 15
    g = _grid(data)
    planner = WeightedAStarPlanner2D(WeightedAStarParams(inflate_radius_m=0.0))
    path = [Pose2D(0.5, 1.55), Pose2D(2.9, 1.55)]  # y=1.55 -> row 15
    assert planner.path_collides(g, path)


def test_passable_start_exempts_only_the_drone_cell():
    """The drone sits in an inflated skirt: its own cell reads blocked, but the
    collision check must not fire on it -- while a real obstacle one step ahead
    is still detected."""
    data = np.zeros((30, 30), np.int16)
    data[15, 5] = 100
    g = _grid(data)
    planner = WeightedAStarPlanner2D(WeightedAStarParams(inflate_radius_m=0.3))
    _, occ = planner.cost_for(g)
    # Drone at a cell that is inflated (within 0.3m of the wall) but not the wall.
    drone = Pose2D(0.65, 1.55)  # ~2 cells right of the wall -> in the skirt
    dsx, dsy = g.world_to_grid(drone.x, drone.y)
    assert occ[dsy, dsx], "test setup: drone cell should be inflated"
    # A short path forward that stays clear of the wall's inflation ahead.
    path = [drone, Pose2D(2.5, 1.55)]
    assert planner.path_collides(g, path), "without exemption the skirt reads blocked"
    assert not planner.path_collides(g, path, passable_start=drone), \
        "passable_start must exempt the drone's own inflated cell"
    # The exemption is restored (single-cell, temporary): a second call still blocks.
    assert planner.path_collides(g, path)


def test_passable_start_does_not_mask_real_obstacle_ahead():
    data = np.zeros((30, 30), np.int16)
    data[15, 20] = 100  # wall well ahead of the drone
    g = _grid(data)
    planner = WeightedAStarPlanner2D(WeightedAStarParams(inflate_radius_m=0.0))
    drone = Pose2D(0.5, 1.55)
    path = [drone, Pose2D(2.9, 1.55)]  # runs into the wall at col 20
    assert planner.path_collides(g, path, passable_start=drone)
