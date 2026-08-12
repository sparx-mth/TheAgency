"""A sampled episode must be a route the aircraft can actually fly.

The one property that matters: no leg of the emitted waypoint list may pass
through a cell the map calls occupied. Everything else here supports that.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import occupancy_from_mask
from sparx_agency.core.planning.mission import largest_region
from sparx_agency.tasks.planning.sim_flight_recording.episode_plan import (
    EpisodeSpec, make_planner, plan_between, sample_episode, waypoints_with_heading,
)

RES = 0.25


def _two_rooms_with_a_doorway():
    """Two 5x10 m rooms joined by a 1.5 m doorway, walls 0.25 m thick."""
    height, width = 40, 80                       # 10 m x 20 m at 0.25 m/cell
    occupied = np.zeros((height, width), dtype=bool)
    occupied[0, :] = occupied[-1, :] = True
    occupied[:, 0] = occupied[:, -1] = True
    occupied[:, 40] = True                       # the dividing wall
    occupied[16:22, 40] = False                  # the doorway, 1.5 m tall
    return occupancy_from_mask(occupied, RES, 0.0, 0.0)


def _spec(**overrides) -> EpisodeSpec:
    base = dict(altitude_m=1.5, clearance_m=0.5, inflate_radius_m=0.35,
                inflate_floor_m=0.3, min_separation_m=6.0, waypoint_spacing_m=1.0,
                start_yaw_jitter_rad=0.0)
    base.update(overrides)
    return EpisodeSpec(**base)


def _crosses_obstacle(grid, a, b, step_m=0.05) -> bool:
    """Sample the straight segment a->b and report whether any cell is occupied."""
    steps = max(int(math.dist(a, b) / step_m), 1)
    for i in range(steps + 1):
        t = i / steps
        gx, gy = grid.world_to_grid(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if grid.is_occupied(gx, gy) or grid.is_unknown(gx, gy):
            return True
    return False


def test_route_between_rooms_never_crosses_a_wall():
    grid = _two_rooms_with_a_doorway()
    spec = _spec()
    start = Pose2D(3.0, 5.0, 0.0)
    goal = Pose2D(16.0, 5.0, 0.0)

    plan = plan_between(grid, start, goal, spec)

    assert plan is not None, "the doorway is wide enough; a route must exist"
    previous = (start.x, start.y)
    for x, y, _z, _yaw in plan.waypoints:
        assert not _crosses_obstacle(grid, previous, (x, y)), \
            f"leg {previous} -> {(x, y)} passes through a wall"
        previous = (x, y)


def test_route_goes_through_the_doorway_not_around_it():
    grid = _two_rooms_with_a_doorway()
    plan = plan_between(grid, Pose2D(3.0, 5.0, 0.0), Pose2D(16.0, 5.0, 0.0), _spec())
    doorway_x = 40 * RES
    assert plan is not None
    crossings = [w for w in plan.waypoints if abs(w[0] - doorway_x) < 0.6]
    assert crossings, "no waypoint near the only opening between the rooms"
    for _x, y, _z, _yaw in crossings:
        assert 3.9 <= y <= 5.6, "crossed the dividing wall somewhere other than the door"


def test_no_route_when_the_doorway_is_sealed():
    height, width = 40, 80
    occupied = np.zeros((height, width), dtype=bool)
    occupied[:, 40] = True
    grid = occupancy_from_mask(occupied, RES, 0.0, 0.0)

    assert plan_between(grid, Pose2D(3.0, 5.0, 0.0), Pose2D(16.0, 5.0, 0.0),
                        _spec()) is None


def test_every_waypoint_is_at_the_cruise_altitude():
    grid = _two_rooms_with_a_doorway()
    plan = plan_between(grid, Pose2D(3.0, 5.0, 0.0), Pose2D(16.0, 5.0, 0.0),
                        _spec(altitude_m=2.25))
    assert plan is not None
    assert all(w[2] == pytest.approx(2.25) for w in plan.waypoints)


def test_the_last_waypoint_is_the_goal():
    grid = _two_rooms_with_a_doorway()
    goal = Pose2D(16.0, 5.0, 0.0)
    plan = plan_between(grid, Pose2D(3.0, 5.0, 0.0), goal, _spec())
    assert plan is not None
    assert plan.waypoints[-1][0] == pytest.approx(goal.x, abs=1e-3)
    assert plan.waypoints[-1][1] == pytest.approx(goal.y, abs=1e-3)


def test_waypoint_yaw_faces_along_the_leg():
    start = Pose2D(0.0, 0.0, 0.0)
    points = [Pose2D(1.0, 0.0, 0.0), Pose2D(1.0, 1.0, 0.0), Pose2D(0.0, 1.0, 0.0)]

    waypoints = waypoints_with_heading(start, points, altitude_m=1.5)

    assert [round(w[3], 6) for w in waypoints] == [
        0.0, round(math.pi / 2, 6), round(math.pi, 6)]


def test_waypoint_yaw_is_held_through_a_zero_length_leg():
    waypoints = waypoints_with_heading(
        Pose2D(0.0, 0.0, 1.0), [Pose2D(0.0, 0.0, 0.0)], altitude_m=1.0)
    assert waypoints[0][3] == pytest.approx(1.0), "a duplicate point must not snap the camera"


def test_sampled_episodes_are_flyable_and_long_enough():
    grid = _two_rooms_with_a_doorway()
    spec = _spec(min_separation_m=6.0, start_yaw_jitter_rad=math.pi)
    region = largest_region(grid, spec.clearance_m)
    rng = np.random.default_rng(0)

    for _ in range(8):
        plan = sample_episode(grid, rng, spec, region=region)
        assert plan.straight_line_m >= 6.0
        assert plan.path_length_m >= plan.straight_line_m - 1e-6
        previous = (plan.start.x, plan.start.y)
        for x, y, _z, _yaw in plan.waypoints:
            assert not _crosses_obstacle(grid, previous, (x, y))
            previous = (x, y)


def test_chained_episode_starts_where_the_last_one_landed():
    grid = _two_rooms_with_a_doorway()
    spec = _spec(min_separation_m=6.0)
    region = largest_region(grid, spec.clearance_m)
    rng = np.random.default_rng(2)

    first = sample_episode(grid, rng, spec, region=region)
    second = sample_episode(grid, rng, spec, region=region, start_from=first.goal)

    assert math.dist((second.start.x, second.start.y),
                     (first.goal.x, first.goal.y)) <= RES * 2


def test_a_chained_start_keeps_the_heading_it_was_given():
    # The heading has to survive the snap-onto-the-region step, or the planner
    # charges its turn cost against a direction the aircraft is not facing.
    # A campaign chains from ``plan.goal``, whose own yaw is a meaningless zero
    # (an arrival has no heading), so the caller must supply the real one --
    # and this asserts the plumbing carries it when it does.
    grid = _two_rooms_with_a_doorway()
    spec = _spec()
    region = largest_region(grid, spec.clearance_m)
    for yaw in (0.0, math.pi / 2, -2.0):
        plan = sample_episode(grid, np.random.default_rng(4), spec, region=region,
                              start_from=Pose2D(5.0, 5.0, yaw))
        assert plan.start.yaw == pytest.approx(yaw)


def test_an_arrival_has_no_heading_of_its_own():
    # Point 6: a flight ends on a radius, at whatever angle. plan.goal's yaw is
    # therefore a placeholder, and this pins that so nobody starts trusting it.
    grid = _two_rooms_with_a_doorway()
    plan = plan_between(grid, Pose2D(5.0, 5.0, 1.0), Pose2D(15.0, 5.0, 2.5), _spec())
    assert plan is not None
    assert plan.goal.yaw == 0.0


def test_impossible_spec_raises_instead_of_looping():
    grid = _two_rooms_with_a_doorway()
    with pytest.raises(RuntimeError, match="could not plan an episode"):
        sample_episode(grid, np.random.default_rng(0),
                       _spec(min_separation_m=500.0, max_attempts=3))


def test_same_seed_gives_the_same_episode():
    grid = _two_rooms_with_a_doorway()
    spec = _spec()
    a = sample_episode(grid, np.random.default_rng(7), spec)
    b = sample_episode(grid, np.random.default_rng(7), spec)
    assert a.waypoints == b.waypoints


def test_ends_are_drawn_from_the_goal_region_while_the_route_uses_all_of_it():
    """Fly over the furniture, but never take off or land on it."""
    grid = _two_rooms_with_a_doorway()
    spec = _spec(min_separation_m=6.0)
    region = largest_region(grid, spec.clearance_m)
    # Pretend only the left room has floor clear enough to land on.
    landable = region.copy()
    landable[:, 40:] = False
    rng = np.random.default_rng(4)

    for _ in range(10):
        plan = sample_episode(grid, rng, spec, region=region, goal_region=landable)
        for pose in (plan.start, plan.goal):
            gx, gy = grid.world_to_grid(pose.x, pose.y)
            assert landable[gy, gx], "an end landed outside the landable region"


def test_a_chained_episode_may_start_outside_the_goal_region():
    """After a failed flight the aircraft can be anywhere; it still needs a goal."""
    grid = _two_rooms_with_a_doorway()
    spec = _spec(min_separation_m=4.0)
    region = largest_region(grid, spec.clearance_m)
    landable = region.copy()
    landable[:, 40:] = False

    plan = sample_episode(grid, np.random.default_rng(1), spec, region=region,
                          goal_region=landable, start_from=Pose2D(16.0, 5.0, 0.0))

    gx, gy = grid.world_to_grid(plan.goal.x, plan.goal.y)
    assert landable[gy, gx]
    assert plan.start.x > 12.0, "the start should stay where the aircraft actually is"


def test_the_spec_hands_its_turn_cost_to_the_planner():
    # The wiring, not the search: a spec that says turning is expensive must
    # produce a planner that charges for it, over metres rather than one cell.
    spec = _spec(start_turn_cost_m_per_rad=2.5, start_turn_radius_m=4.0)
    params = make_planner(spec).params
    assert params.start_turn_cost_m_per_rad == 2.5
    assert params.start_turn_radius_m == 4.0
    assert params.heading_penalty_m == 0.0, "the reversal shape stays off here"


def test_a_route_with_only_one_way_out_is_flown_however_the_aircraft_faces():
    # The heading cost is a tie-break, never a veto: the doorway is the only way
    # between the rooms, so it must be taken even facing hard away from it.
    grid = _two_rooms_with_a_doorway()
    spec = _spec(start_turn_cost_m_per_rad=8.0, start_turn_radius_m=3.0)
    goal = Pose2D(15.0, 5.0, 0.0)
    for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
        plan = plan_between(grid, Pose2D(5.0, 5.0, yaw), goal, spec)
        assert plan is not None, f"no route when facing {math.degrees(yaw):.0f} deg"
        assert max(p[0] for p in plan.waypoints) > 12.0
