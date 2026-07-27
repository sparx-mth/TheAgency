"""Waypoint sequencing: pass through the route, but actually arrive at the goal."""
from __future__ import annotations

import pytest

from sparx_agency.tasks.planning.sim_flight_recording.waypoint_mission import WaypointMission

ROUTE = [(1.0, 0.0, 1.5, 0.0), (2.0, 0.0, 1.5, 0.0), (3.0, 0.0, 1.5, 0.0)]


def test_empty_mission_rejected():
    with pytest.raises(ValueError):
        WaypointMission([])


def test_intermediate_waypoints_are_passed_through_without_dwelling():
    mission = WaypointMission(ROUTE, arrival_radius_m=0.8, dwell_s=5.0)
    # 0.5 m short of the first waypoint, and moving: inside the pass-through
    # radius, so it should hand over immediately despite the long dwell.
    assert mission.update((0.5, 0.0, 1.5), 0.0)
    assert mission.index == 1


def test_the_final_waypoint_requires_the_tight_radius():
    mission = WaypointMission(ROUTE, arrival_radius_m=0.8, final_radius_m=0.3, dwell_s=0.5)
    mission.update((1.0, 0.0, 1.5), 0.0)
    mission.update((2.0, 0.0, 1.5), 1.0)
    assert mission.on_final

    assert not mission.update((2.5, 0.0, 1.5), 2.0), "0.5 m out is inside the pass-through " \
                                                     "radius but not the final one"
    assert not mission.update((3.0, 0.0, 1.5), 3.0), "arrived, but has not dwelled yet"
    assert mission.update((3.0, 0.0, 1.5), 3.6)
    assert mission.finished


def test_leaving_the_goal_restarts_the_dwell():
    mission = WaypointMission([(0.0, 0.0, 1.5, 0.0)], final_radius_m=0.3, dwell_s=0.5)
    mission.update((0.0, 0.0, 1.5), 0.0)          # inside, dwell starts at t=0
    mission.update((5.0, 0.0, 1.5), 0.2)          # pushed out
    assert not mission.update((0.0, 0.0, 1.5), 0.4), "the dwell must restart, not resume"
    assert mission.update((0.0, 0.0, 1.5), 1.0)


def test_an_unreachable_waypoint_is_skipped_rather_than_blocking():
    mission = WaypointMission(ROUTE, timeout_s=10.0)
    mission.update((0.0, 0.0, 1.5), 0.0)          # starts the clock, nowhere near
    assert not mission.update((0.0, 0.0, 1.5), 5.0)
    assert mission.update((0.0, 0.0, 1.5), 11.0)
    assert mission.index == 1
    assert mission.skipped == 1


def test_the_timeout_clock_starts_at_the_first_update_not_construction():
    """A mission is built before arming; the first waypoint must get its full budget."""
    mission = WaypointMission(ROUTE, timeout_s=10.0)
    # PX4 spent 100 s booting and arming before anything called update().
    assert not mission.update((0.0, 0.0, 1.5), 100.0)
    assert not mission.update((0.0, 0.0, 1.5), 105.0)
    assert mission.update((0.0, 0.0, 1.5), 111.0)


def test_finished_mission_is_inert():
    mission = WaypointMission([(0.0, 0.0, 0.0, 0.0)], final_radius_m=1.0, dwell_s=0.0)
    mission.update((0.0, 0.0, 0.0), 0.0)
    assert mission.finished
    assert mission.current() is None
    assert not mission.update((0.0, 0.0, 0.0), 1.0)


def test_altitude_counts_towards_arrival():
    """A drone 3 m above its waypoint has not reached it."""
    mission = WaypointMission([(0.0, 0.0, 1.5, 0.0)], final_radius_m=0.3, dwell_s=0.0)
    assert not mission.update((0.0, 0.0, 4.5), 0.0)
    assert mission.update((0.0, 0.0, 1.5), 1.0)
