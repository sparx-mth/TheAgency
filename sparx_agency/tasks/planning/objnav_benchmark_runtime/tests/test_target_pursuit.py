"""A locked target released for want of a path is kept, and walked back to once the map holds a route to it."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.object_search.closing.target_pursuit import PursuitSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
AGENT, TARGET = (2.0, 2.0), (6.5, 2.0)


def room(door_open):
    """A 9 x 4 m room split by a wall at x = 4 m; the doorway at y = 2 m is open or shut."""
    g = np.zeros((40, 90), np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    g[:, 39:41] = 100
    if door_open:
        g[16:25, 39:41] = 0
    return OccupancyGrid2D(g, OccupancyGrid2DParams(0.1, 0.0, 0.0, "world"), values=VALUES)


def at(ep, step, xy=AGENT):
    return replace(observation(ep, step), pose=AgentPose(xy[0], xy[1], 0.0, 0.0))


def released(policy, ep, step=10):
    closing = SimpleNamespace(xyz=(TARGET[0], TARGET[1], 0.5), floor_id=policy.mapping.floor_id, label="bed")
    policy.closing.boxed_in.append((AGENT, policy.mapping.floor_id, step))
    policy.pursuit.remember(at(ep, step), closing)


def test_a_released_target_is_pursued_once_a_route_exists_and_its_boxed_in_spot_is_cleared():
    policy, ep = setup_policy("bed")
    released(policy, ep)
    pursuit = policy.pursuit
    assert pursuit.command(at(ep, 12), room(False)) is None, "too soon after the release"
    assert pursuit.command(at(ep, 20), room(False)) is None, "no route through the wall"
    assert pursuit.stats["queries"] == 1 and pursuit.active is None
    command = pursuit.command(at(ep, 30), room(True))
    assert command is not None and command.info["kind"] == "target_pursuit"
    assert command.waypoints and abs(command.waypoints[-1][0] - TARGET[0]) <= policy.closing.settings.terminal_distance_m + 0.2
    assert pursuit.active is not None and pursuit.stats["started"] == 1
    assert policy.closing.boxed_in == [], "the spot no longer refuses the takeover: a route exists"
    pursuit.notify_takeover(at(ep, 31))
    assert pursuit.active is None and pursuit.stats["reacquired"] == 1


def test_a_target_is_tried_a_bounded_number_of_times_and_not_after_it_is_stale():
    policy, ep = setup_policy("bed")
    policy.pursuit.settings = PursuitSettings(max_attempts=1, max_age_actions=100)
    released(policy, ep)
    assert policy.pursuit.command(at(ep, 30), room(True)) is not None
    policy.pursuit._end(at(ep, 31), "test")
    assert policy.pursuit.command(at(ep, 60), room(True)) is None, "its one attempt is spent"
    fresh, ep2 = setup_policy("bed")
    fresh.pursuit.settings = PursuitSettings(max_age_actions=100)
    released(fresh, ep2)
    assert fresh.pursuit.command(at(ep2, 200), room(True)) is None, "released too long ago"


def test_disabled_nothing_is_remembered_and_settings_are_checked():
    policy, ep = setup_policy("bed")
    policy.pursuit.settings = PursuitSettings(enabled=False)
    released(policy, ep)
    assert policy.pursuit.targets == [] and policy.pursuit.command(at(ep, 30), room(True)) is None
    with pytest.raises(ValueError, match="retry_actions"):
        PursuitSettings(retry_actions=0)
    assert "target_pursuit" in policy.episode_info()


def test_a_standoff_on_a_step_down_is_a_standoff_of_this_storey_but_not_one_on_another():
    """00835/000006 (2026-10-31): every ring point of a television 0.4 m down three treads was skipped at 0.2 m."""
    from sparx_agency.core.planning.object_search.closing.target_path import TargetApproachPath
    confined, _ = setup_policy("bed")
    assert confined.floor_guard.provisional, "a spawn storey never walked level: the whole step band"
    assert TargetApproachPath(confined)._height_tolerance() == pytest.approx(confined.settings.step_surface_band_m)
    assert 0.4 < confined.settings.step_surface_band_m < confined.settings.multifloor.min_floor_separation_m
    confined.building.storey_confirmed = True          # walked level: as far as the guard frees, the atlas's departure
    assert TargetApproachPath(confined)._height_tolerance() == pytest.approx(confined.settings.multifloor.departure_m)
    assert confined.settings.multifloor.departure_m > 0.4
    climbing, _ = setup_policy("bed", allow_stair_traversal=True)
    assert TargetApproachPath(climbing)._height_tolerance() == pytest.approx(0.2), "multi-storey: a step, as before"


def test_a_bearing_approach_stopped_by_a_wall_hands_its_target_over_beyond_the_depth_range():
    """00853/000065 (2026-10-31): three walks into a wall toward a television seen from beyond 5 m, each given up."""
    import math
    policy, ep = setup_policy("bed")
    obs = at(ep, 40)
    policy.pursuit.remember_bearing(obs, "bed", 0.0, policy.mapping.floor_id)
    entry = policy.pursuit.targets[-1]
    assert entry["source"] == "bearing"
    reach = obs.camera.max_depth_m + policy.pursuit.settings.bearing_reach_m
    assert math.dist(entry["xyz"][:2], (AGENT[0] + reach, AGENT[1])) < 1e-6


def test_a_query_that_finds_no_route_leaves_the_searchs_committed_route_alone():
    policy, ep = setup_policy("bed")
    released(policy, ep)
    world = room(False)
    obs = at(ep, 20)
    committed = policy._navigate(obs, world, (3.0, 3.0), "frontier")
    assert committed is not None
    before = (policy.route_memory.goal, policy.route_memory.path, policy._goal)
    assert policy.pursuit.command(obs, world) is None
    assert (policy.route_memory.goal, policy.route_memory.path, policy._goal) == before


def test_a_stop_on_the_memory_alone_needs_the_map_to_show_no_wall_between():
    """00877/000043 (2026-10-31): a memory STOP 1.0 m from a toilet behind the bathroom wall, 2.64 m from any view point."""
    policy, ep = setup_policy("bed")
    closing = policy.closing
    closing.xyz = (5.0, 2.0, 0.4)
    obs = at(ep, 50, (3.6, 2.0))
    policy.last_world = room(False)                       # the wall at x = 4 m stands between
    assert closing._memory_in_sight(obs) is False and closing.memory_sight_refusals == 1
    policy.last_world = room(True)                        # the doorway on the line is open
    assert closing._memory_in_sight(obs) is True
    closing.xyz = (3.9, 2.0, 0.4)                         # within the object's own margin: nothing to read
    policy.last_world = room(False)
    assert closing._memory_in_sight(at(ep, 51, (3.55, 2.0))) is True
