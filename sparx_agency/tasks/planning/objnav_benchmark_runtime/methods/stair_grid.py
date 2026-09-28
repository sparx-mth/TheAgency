"""Observed-only support rasterization, independent of stair scheduling."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, minimum_filter, maximum_filter

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.falcon.ordering import Deadline
from sparx_agency.core.planning.exploration.falcon.travel import GridRoutes


def build_support_grid(terrain, pose):
    """Construct the same bounded measured-height graph used by stair steering."""
    t, r = terrain, terrain.resolution
    half = int(math.ceil(t.params.terrain_radius_m / r))
    t.offset = (int(math.floor(pose.x / r)) - half, int(math.floor(pose.y / r)) - half)
    heights = _visible_support(t, pose, 2 * half + 1)
    planar = _borrow_floor(t, pose, heights)
    _close_holes(heights, t.params.max_step_m)
    _footprint(t, pose, heights, half)
    obstacles = _obstacles(t, pose, heights, planar)
    clearance = distance_transform_edt(~obstacles) * r if obstacles.any() else np.full_like(heights, np.inf)
    safe = np.isfinite(heights) & ~obstacles & (clearance > t.body_radius)
    safe[half, half] = True
    data = np.full(heights.shape, -1, np.int8)
    data[safe] = 0
    data[obstacles] = 100
    t.heights, t.safe = heights, safe
    t.world = OccupancyGrid2D(data, OccupancyGrid2DParams(r, t.offset[0] * r, t.offset[1] * r, "world"),
                              values=OccupancyValues(free=0, occupied=100, unknown=-1))
    t.routes = _routes(safe, heights, r, t.params.max_step_m)
    t.start = (half, half)
    t.distances = t.routes.distances(t.start, Deadline(1.0))


def _visible_support(t, pose, n):
    heights = np.full((n, n), np.nan)
    for (x, y, _), (z, _, horizontal) in t.samples.items():
        x, y = x - t.offset[0], y - t.offset[1]
        if horizontal and 0 <= x < n and 0 <= y < n and abs(z - pose.z) <= 1.5:
            if not np.isfinite(heights[y, x]) or z > heights[y, x]:
                heights[y, x] = z
    return heights


def _borrow_floor(t, pose, heights):
    """Only previously observed near-field cells, on their measured source floor."""
    obstacles = np.zeros(heights.shape, bool)
    world = t._floor_world
    if world is None or t._floor_height is None or abs(pose.z - t._floor_height) >= t.params.floor_match_m:
        return obstacles
    yy, xx = np.indices(heights.shape)
    wx, wy = (xx + t.offset[0] + 0.5) * t.resolution, (yy + t.offset[1] + 0.5) * t.resolution
    gx = np.floor((wx - world.origin_x) / world.resolution).astype(int)
    gy = np.floor((wy - world.origin_y) / world.resolution).astype(int)
    inside = (gx >= 0) & (gx < world.width) & (gy >= 0) & (gy < world.height)
    close = (wx - pose.x) ** 2 + (wy - pose.y) ** 2 < 2.0 ** 2
    iy, ix = np.nonzero(inside & close)
    values = world.grid[gy[iy, ix], gx[iy, ix]]
    free = (values == world.values.free) & ~np.isfinite(heights[iy, ix])
    heights[iy[free], ix[free]] = t._floor_height - 0.05
    occupied = values == world.values.occupied
    obstacles[iy[occupied], ix[occupied]] = True
    return obstacles


def _close_holes(heights, max_step):
    known = np.isfinite(heights)
    if not known.any():
        return
    low = minimum_filter(np.where(known, heights, np.inf), size=3)
    high = maximum_filter(np.where(known, heights, -np.inf), size=3)
    enclosed = np.zeros_like(known)
    enclosed[1:-1, 1:-1] = ((known[1:-1, :-2] & known[1:-1, 2:]) | (known[:-2, 1:-1] & known[2:, 1:-1]))
    fill = ~known & enclosed & (high - low < max_step)
    heights[fill] = (high[fill] + low[fill]) * 0.5


def _footprint(t, pose, heights, half):
    yy, xx = np.indices(heights.shape)
    footprint = (xx - half) ** 2 + (yy - half) ** 2 <= (t.body_radius / t.resolution) ** 2
    foot, observed = t._foot_height, heights[half, half]
    if np.isfinite(observed) and pose.z - 0.40 <= observed <= pose.z + 0.05:
        foot = float(observed)
    elif foot is None:
        foot = pose.z - 0.05
    t.support_height_m = foot
    # Preserve real adjacent treads when the measured body spans two heights.
    heights[footprint & ~np.isfinite(heights)] = foot
    heights[half, half] = foot


def _obstacles(t, pose, heights, planar):
    # The borrowed slab is anchored to its SOURCE floor, not the moving base.
    # Real measured walls still enter the body-height check below.
    obstacles = (planar & (np.abs(heights - t._floor_height) < 0.08)
                 if t._floor_height is not None else np.zeros_like(planar))
    h, w = heights.shape
    for (x, y, _), (z, _, _) in t.samples.items():
        x, y = x - t.offset[0], y - t.offset[1]
        if 0 <= x < w and 0 <= y < h and np.isfinite(heights[y, x]):
            if 0.28 < z - heights[y, x] < t.body_height + 0.10:
                obstacles[y, x] = True
    for x, y, height in t.obstacles:
        x, y = x - t.offset[0], y - t.offset[1]
        if 0 <= x < w and 0 <= y < h and abs(height * 0.1 - pose.z) < t.params.floor_match_m:
            obstacles[y, x] = True
    return obstacles


def _routes(safe, heights, resolution, max_step):
    routes = GridRoutes(np.where(safe, 1.0, np.inf), resolution)
    graph = routes.graph.tocoo()
    ay, ax = routes.y[graph.row], routes.x[graph.row]
    by, bx = routes.y[graph.col], routes.x[graph.col]
    valid = np.abs(heights[ay, ax] - heights[by, bx]) <= max_step
    valid &= np.abs(heights[ay, bx] - heights[ay, ax]) <= max_step
    valid &= np.abs(heights[by, ax] - heights[ay, ax]) <= max_step
    graph.data[~valid] = 0
    graph.eliminate_zeros()
    routes.graph = graph.tocsr()
    # Any-angle shortening would discard the tread-by-tread height constraints.
    routes.shorten = lambda cells, deadline: tuple(cells)
    return routes
