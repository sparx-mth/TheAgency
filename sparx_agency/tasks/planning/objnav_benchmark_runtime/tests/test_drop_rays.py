"""A drop is measured along the depth ray, not only where it lands (00835-q3zU7Yy5E5s/000006, 2026-10-31).

The observed map carves free space along rays toward obstacle points, over a
stairwell's treads that a level camera never sees; the guard marked drops only
where depth points landed. These frames are ray-marched through a height field
-- a storey at 0 m with a stairwell 1.5 m deep -- so the guard sees what the
benchmark camera would.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.object_search.multifloor.drop_rays import cells_crossed_below
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_spawn_floor_guard import _step_terrain

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
RES = 0.1
WALL_M = 3.0


def stairwell(x, y):
    """A room 8 x 6 m at 0 m with a stairwell 1.5 m deep at x 4.5-6.0, y 2.0-4.0; the room's walls are 3 m tall."""
    height = np.zeros(np.broadcast(x, y).shape)
    height[(x >= 4.5) & (x < 6.0) & (y >= 2.0) & (y <= 4.0)] = -1.5
    height[(x >= 8.0) | (x <= 0.0) | (y <= 0.5) | (y >= 5.5)] = WALL_M
    return height


def flat(x, y):
    """The same room with a floor everywhere (a slab over the stairwell, as seen from elsewhere)."""
    height = np.zeros(np.broadcast(x, y).shape)
    height[(x >= 8.0) | (x <= 0.0) | (y <= 0.5) | (y >= 5.5)] = WALL_M
    return height


def sunken(depth_m):
    """The room with a sunken floor ``depth_m`` down where the stairwell was (a step down, not a flight)."""
    def scene(x, y):
        height = flat(x, y)
        height[in_well(x, y) & (x < 6.0) & (y >= 2.0) & (y <= 4.0)] = -depth_m
        return height
    return scene


def low_storey(x, y):
    """The storey 1.2 m below, seen by an agent that walked down to it."""
    return np.where((x >= 8.0) | (x <= 0.0) | (y <= 0.5) | (y >= 5.5), WALL_M, -1.2)


def frame(episode, step, pose, scene, stride=8):
    """A depth frame of ``scene`` from ``pose``: every ``stride``-th pixel (the guard's) ray-marched, the rest NaN."""
    camera = episode.camera
    k = camera.intrinsics
    vs, us = np.mgrid[0:k.height:stride, 0:k.width:stride]
    transform = world_T_camera_optical(pose, camera)
    optical = np.stack(((us - k.cx) / k.fx, (vs - k.cy) / k.fy, np.ones(us.shape)), axis=-1)
    direction = optical @ transform[:3, :3].T
    s = np.arange(camera.min_depth_m, camera.max_depth_m, 0.005)
    points = transform[:3, 3] + s[None, None, :, None] * direction[:, :, None, :]
    hit = points[..., 2] <= scene(points[..., 0], points[..., 1])
    depth = np.full((k.height, k.width), np.nan, np.float32)
    depth[::stride, ::stride] = np.where(hit.any(axis=-1), s[hit.argmax(axis=-1)], np.nan)
    return replace(observation(episode, step), depth_m=depth, pose=pose)


def carved_room():
    """The room as the observed map leaves it: every cell inside the walls free, the stairwell included."""
    ys, xs = np.mgrid[0:60, 0:80]
    cx, cy = (xs + 0.5) * RES, (ys + 0.5) * RES
    grid = np.where(flat(cx, cy) > 1.0, VALUES.occupied, VALUES.free).astype(np.int8)
    return OccupancyGrid2D(grid, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def cells(world, predicate):
    ys, xs = np.mgrid[0:world.height, 0:world.width]
    return predicate((xs + 0.5) * RES, (ys + 0.5) * RES)


AT_THE_EDGE = AgentPose(3.0, 3.0, 0.0, 0.0)          # 1.5 m before the stairwell, facing it


def guarded(monkeypatch, walked_level=True):
    """A storey walked level (step reach = the atlas's 0.45 m departure) unless ``walked_level`` is False (a
    provisional spawn storey: the whole 0.8 m step band)."""
    policy, episode = setup_policy("bed")
    monkeypatch.setattr(type(policy.floor_guard), "provisional", property(lambda self: not walked_level))
    policy.mapping.atlas.update(AT_THE_EDGE)
    policy.mapping._anchor = 0.0
    return policy, episode, policy.floor_guard, carved_room()


def in_well(x, y):
    """The stairwell and its faces, one cell round (a face's points fall either side of its line). A level camera
    1.5 m before it sees no tread: only the lower part of the far and side faces."""
    return (x >= 4.5) & (x < 6.0 + RES) & (y >= 2.0 - RES) & (y <= 4.0 + RES)


def test_a_ray_is_a_drop_from_where_it_sinks_under_the_drop_line_to_where_it_lands():
    camera, point = (0.05, 0.05, 0.88), (3.05, 0.05, -1.0)
    xs, ys = cells_crossed_below(camera, [point], -0.30, (0.0, 0.0), RES, (10, 40))
    sinks = 0.05 + 3.0 * (0.88 + 0.30) / (0.88 + 1.0)                 # 1.93 m: the ray is 0.30 m under the plane there
    assert set(ys) == {0} and min(xs) == math.floor(sinks / RES) and max(xs) == 30
    assert set(xs) == set(range(min(xs), 31)), "no cell skipped between"
    empty = cells_crossed_below(camera, np.zeros((0, 3)), -0.30, (0.0, 0.0), RES, (10, 40))
    assert len(empty[0]) == len(empty[1]) == 0
    off_grid = cells_crossed_below(camera, [(6.05, 0.05, -1.0)], -0.30, (0.0, 0.0), RES, (10, 40))
    assert set(off_grid[0]) == {38, 39}, "cells beyond the grid are dropped, not wrapped"


def test_a_stairwell_the_map_carved_free_is_a_drop_where_the_rays_sank_into_it_and_routes_go_around(monkeypatch):
    policy, episode, guard, world = guarded(monkeypatch)
    confined = guard.observe(frame(episode, 0, AT_THE_EDGE, stairwell), world)
    landed, crossed = guard.drops[0], guard.crossed[0]
    well = cells(world, in_well)
    assert landed.any() and not (landed & ~well).any(), "points land in the stairwell only"
    assert guard.stats["drop_cells"] == int(landed.sum())
    new = crossed & ~landed & ~guard.surfaced[0]
    assert guard.stats["drop_cells_crossed"] == int(new.sum()) > 2 * int(landed.sum())
    assert not (new & ~well).any(), "a crossing never leaves the stairwell"
    assert (confined.grid[new] == VALUES.occupied).all()
    # The near edge -- between the plane and the drop line, and closer than the lowest ray reaches -- stays as the
    # map had it: a ray there passes over a shallow step as readily as over a hole.
    gx, gy = world.world_to_grid(4.6, 3.0)
    assert not crossed[gy, gx] and confined.grid[gy, gx] == VALUES.free
    for xy in ((4.0, 3.0), (7.0, 3.0), (5.0, 1.5)):                    # the floor before, beyond and beside it
        gx, gy = world.world_to_grid(*xy)
        assert confined.grid[gy, gx] == VALUES.free and guard.goal_allowed(frame(episode, 1, AT_THE_EDGE, stairwell),
                                                                          world, xy, "frontier")
    # On the endpoints alone every one of these cells was floor to the planner; a route beyond now goes around.
    before = world.grid.copy()
    before[landed] = VALUES.occupied
    assert (before[new] == VALUES.free).all()
    around = policy._plan_to(frame(episode, 1, AT_THE_EDGE, stairwell), confined, (7.0, 3.0))
    assert around is not None
    ends = [world.world_to_grid(p.x, p.y) for p in around.points]
    route = {cell for (x0, y0), (x1, y1) in zip(ends, ends[1:]) for cell in line_cells(x0, y0, x1, y1)}
    assert not any(new[y, x] for x, y in route)


@pytest.mark.parametrize("seen_first", [False, True])
def test_a_surface_seen_above_the_line_outvotes_a_crossing_in_either_order(monkeypatch, seen_first):
    """A ray that sank into a stairwell can run on under the slab beyond it; floor seen there is the storey's.
    Only the cells the floor landed in are released (the observed map's own rule for visible floor)."""
    policy, episode, guard, world = guarded(monkeypatch)
    looking_across = frame(episode, 0, AT_THE_EDGE, stairwell)
    slab_seen = frame(episode, 1, AgentPose(5.25, 0.8, 0.0, math.pi / 2), flat)   # from the side, over a slab
    for obs in ((slab_seen, looking_across) if seen_first else (looking_across, slab_seen)):
        confined = guard.observe(obs, world)
    crossed, surfaced, landed = guard.crossed[0], guard.surfaced[0], guard.drops[0]
    released = crossed & surfaced & ~landed
    still = crossed & ~surfaced & ~landed
    assert released.any() and still.any()
    assert (confined.grid[released] == VALUES.free).all()
    assert (confined.grid[still] == VALUES.occupied).all()
    assert (confined.grid[landed] == VALUES.occupied).all(), "a point that landed under the line is a drop regardless"
    assert guard.stats["drop_cells_crossed"] == int(still.sum())


@pytest.mark.parametrize("walked_level", [True, False])
def test_a_sunken_floor_inside_the_step_reach_is_never_crossed_below_it(monkeypatch, walked_level):
    """00878/000035 walks down to a sunken floor 0.40 m below the storey: its points land under the drop line (the
    step layer decides whether they are floor), but no ray crosses deeper than the step reach on the way to it."""
    policy, episode, guard, world = guarded(monkeypatch, walked_level)
    reach = guard.step_reach_m
    guard.observe(frame(episode, 0, AT_THE_EDGE, sunken(0.40)), world)
    assert guard.drops[0].any() and not guard.crossed[0].any() and guard.stats["drop_cells_crossed"] == 0
    # A step past the reach: crossed from where the rays sink under it, as a stairwell.
    policy, episode, guard, world = guarded(monkeypatch, walked_level)
    guard.observe(frame(episode, 0, AT_THE_EDGE, sunken(reach + 0.5)), world)
    assert guard.crossed[0].any() and guard.stats["drop_cells_crossed"] > 0


def test_no_crossing_is_drawn_from_a_camera_under_the_storey_floor(monkeypatch):
    """Down a flight by accident, the camera is under the storey's slab: its rays say nothing about the floor above."""
    policy, episode, guard, world = guarded(monkeypatch)
    below = AgentPose(3.0, 3.0, -1.2, 0.0)
    guard.observe(frame(episode, 0, below, low_storey), world)
    assert guard.drops[0].any(), "the lower storey's floor still lands below the line (the drop rule as before)"
    assert not guard.crossed[0].any() and guard.stats["drop_cells_crossed"] == 0


def test_the_cells_the_agent_stood_on_stay_free_inside_a_crossing(monkeypatch):
    policy, episode, guard, world = guarded(monkeypatch)
    guard.observe(frame(episode, 0, AT_THE_EDGE, stairwell), world)
    gx, gy = world.world_to_grid(5.5, 3.0)
    assert guard.crossed[0][gy, gx]
    inside = AgentPose(5.5, 3.0, 0.0, 0.0)                              # the body was there: nothing outvotes it
    confined = guard.observe(replace(observation(episode, 1), pose=inside), world)
    assert confined.grid[gy, gx] == VALUES.free and guard.stats["walked_exempt_cells"] > 0
    gx, gy = world.world_to_grid(5.9, 3.0)                              # past the body radius and a cell
    assert guard.crossed[0][gy, gx] and confined.grid[gy, gx] == VALUES.occupied


def test_a_re_measured_plane_takes_its_crossings_and_surfaces_with_its_drops(monkeypatch):
    policy, episode, guard, world = guarded(monkeypatch)
    _step_terrain(policy, monkeypatch, [], -0.35)
    guard.observe(frame(episode, 0, AT_THE_EDGE, stairwell), world)
    assert guard.crossed[0].any() and guard.surfaced[0].any()
    for step in range(1, 4):
        guard.observe(replace(observation(episode, step), pose=AgentPose(1.0 + 0.1 * step, 3.0, -0.35, math.pi)),
                      world)
    assert guard.stats["plane_remeasured"] == 1
    assert not guard.crossed[0].any() and not guard.surfaced[0].any(), "measured against the old plane"
