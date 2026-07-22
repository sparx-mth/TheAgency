"""Planning from a blocked start (drone hugging / painted inside a wall).

On a noisy map the drone's own cell often reads OCCUPIED or inflated. The planner
must still return a route OUT -- clearing the drone's own cell and its inflation
skirt -- rather than failing and stranding the drone. But it must NEVER route the
flown path through a real wall: a genuinely walled-in start returns NO_PATH so the
node can STOP and hand off to the reactive local planner (NavDP).
"""
import numpy as np

from sparx_agency.core.common.types import PlanStatus, Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.planners.astar import (
    WeightedAStarParams, WeightedAStarPlanner2D)
from sparx_agency.core.planning.planners.common.grid_geometry_2d import (
    line_of_sight_clear)

BEV = OccupancyValues(free=0, occupied=100, unknown=-1)
RES = 0.1
N = 40


def _grid(data):
    return OccupancyGrid2D(
        data.astype(np.int16),
        OccupancyGrid2DParams(resolution=RES, origin_x=0.0, origin_y=0.0,
                              frame_id="world"),
        values=BEV)


def _plan(data, start, goal, planner=None, **params):
    planner = planner or WeightedAStarPlanner2D(WeightedAStarParams(**params))
    req = PlanRequest(start=start, goal=goal, frame_id="world")
    return planner.plan(req, _grid(data))


def _free():
    return np.zeros((N, N), np.int16)


def _crosses_true_wall(data, points, exempt_cell):
    """True if the world polyline crosses a truly-occupied cell (exempting the
    drone's own cell) -- an independent check of the emitted-path safety net."""
    g = _grid(data)
    occ = (data == 100).copy()
    ex, ey = exempt_cell
    occ[ey, ex] = False
    cells = [g.world_to_grid(p.x, p.y) for p in points]
    return any(not line_of_sight_clear(occ, x0, y0, x1, y1)
               for (x0, y0), (x1, y1) in zip(cells[:-1], cells[1:]))


# ─── the user's case: hugging a wall (start inside the inflation skirt) ───────
def test_start_in_inflation_skirt_still_plans():
    """Drone hugging a wall: its cell is inflated (blocked) so the OLD single-cell
    clear dead-ended. The footprint/skirt clear now reaches the free side and A*
    routes out -- staying well clear of the (real) wall."""
    data = _free()
    data[:, 10] = 100                                  # vertical wall at column 10
    start = Pose2D(1.25, 2.0)                           # gx=12 (in the skirt), gy=20
    goal = Pose2D(3.05, 2.0)                            # gx=30, same (right) side
    res = _plan(data, start, goal, inflate_radius_m=0.3)
    assert res.ok, res.message
    assert res.path.points[-1].x > 2.5
    dsx, dsy = _grid(data).world_to_grid(start.x, start.y)
    assert not _crosses_true_wall(data, res.path.points, (dsx, dsy))


def test_single_noise_cell_under_drone_still_plans():
    """A depth frame paints the drone's OWN single cell OCCUPIED. It is cleared (the
    drone is there), the skirt is cleared, and A* escapes to free space -- no snap,
    and the emitted path never crosses a real obstacle."""
    data = _free()
    data[20, 11] = 100                                 # one occupied cell = drone cell
    start = Pose2D(1.15, 2.0)                           # gx=11, gy=20 (on that cell)
    goal = Pose2D(3.05, 2.0)                            # gx=30, free
    res = _plan(data, start, goal, inflate_radius_m=0.3)
    assert res.ok, res.message
    dsx, dsy = _grid(data).world_to_grid(start.x, start.y)
    assert (dsx, dsy) == (11, 20)
    assert not _crosses_true_wall(data, res.path.points, (dsx, dsy))


# ─── safety: never route the flown path through a real wall ──────────────────
def test_no_route_through_a_real_wall_fails_safe():
    """Drone wedged on a REAL wall with the goal only reachable by crossing it: the
    planner must NOT thread the wall (the old snap fallback would have). It returns
    NO_PATH so the node STOPs and hands off to NavDP."""
    data = _free()
    data[:, 20:23] = 100                               # 3-cell-thick full-height wall
    start = Pose2D(2.15, 2.0)                           # gx=21, on the wall
    goal = Pose2D(0.5, 2.0)                             # gx=5, other side (no way round)
    res = _plan(data, start, goal, inflate_radius_m=0.3)
    assert not res.ok
    assert res.status == PlanStatus.NO_PATH


def test_enclosed_by_occupied_blob_fails_safe():
    """Drone deep inside an occupied blob larger than its footprint: the footprint
    clear cannot reach free space and there is no snap-through-walls fallback, so
    the planner returns NO_PATH (-> node STOP + NavDP) rather than a route out that
    would cross the blob."""
    data = _free()
    data[10:25, 10:25] = 100                            # 15x15 occupied blob
    start = Pose2D(1.75, 1.75)                          # gx=17, gy=17, deep inside
    goal = Pose2D(3.5, 1.75)                            # gx=35, free (right of blob)
    res = _plan(data, start, goal, inflate_radius_m=0.3)
    assert not res.ok
    assert res.status == PlanStatus.NO_PATH


def test_partial_wall_reroutes_around_not_through():
    """Drone wedged in the skirt on one side of a THICK partial wall (a gap at the
    bottom), goal on the far side: the escape route must go AROUND the wall's end,
    never through it. Crosses no real obstacle."""
    data = _free()
    data[0:30, 20:23] = 100                             # thick wall, gap at rows 30..39
    start = Pose2D(2.45, 2.0)                           # gx=24 (right skirt), gy=20
    goal = Pose2D(0.5, 2.0)                             # gx=5, left side (must go round)
    res = _plan(data, start, goal, inflate_radius_m=0.3)
    assert res.ok, res.message
    dsx, dsy = _grid(data).world_to_grid(start.x, start.y)
    assert not _crosses_true_wall(data, res.path.points, (dsx, dsy))
    assert res.path.points[-1].x < 1.0                 # reached the far (left) side


# ─── the common (free-start) case must be untouched ──────────────────────────
def test_free_start_is_unaffected():
    """A free start must take the fast path: no footprint relaxation at all, and a
    clean straight corridor route."""
    planner = WeightedAStarPlanner2D(WeightedAStarParams(inflate_radius_m=0.3))
    calls = {"n": 0}
    orig = planner._clear_start_footprint
    planner._clear_start_footprint = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                                      orig(*a, **k))[1]
    data = _free()
    start = Pose2D(0.5, 2.0)
    goal = Pose2D(3.5, 2.0)
    res = _plan(data, start, goal, planner=planner)
    assert res.ok
    assert calls["n"] == 0, "a free start must not relax the footprint"
    assert res.path.points[0].x <= 0.6                 # starts at the drone
    assert res.path.points[-1].x >= 3.4                # ends at the goal
    assert all(abs(p.y - 2.0) < 0.2 for p in res.path.points)   # clean corridor
