"""Channels are read off the clearance field along a route: the doorway, its throat, its axis and its width."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from sparx_agency.core.planning.object_search.navigation.bottlenecks import (heading_error, nearest_channel,
                                                                              route_channels, route_position)
from sparx_agency.core.planning.object_search.navigation.tests.rig import DOOR_X, DOOR_Y, RES, doorway_policy, doorway_world


def channels_of(route, world, policy, doors=()):
    fields = policy.planner.fields_for(world)
    return route_channels(route, fields.clearance, 0.0, 0.0, RES, policy.planner_params.channel_half_width_m, doors=doors)


def test_a_route_through_the_doorway_has_one_doorway_channel_at_the_wall():
    policy, _ = doorway_policy()
    world = doorway_world(0.8)
    route = [(3.0, DOOR_Y), (9.0, DOOR_Y)]
    channels = channels_of(route, world, policy)
    assert len(channels) == 1
    door = channels[0]
    assert door.kind == "doorway" and door.door_id is None
    assert abs(door.throat_xy[0] - DOOR_X) <= 0.15 and abs(door.throat_xy[1] - DOOR_Y) <= 0.05
    assert door.width_m == pytest.approx(0.8, abs=0.15)
    assert abs(heading_error(door, 0.0)) < 1e-6 and heading_error(door, math.radians(40)) == pytest.approx(math.radians(40))
    assert door.entry_s_m < door.throat_s_m < door.exit_s_m and door.length_m < 1.0
    assert door.entry_s_m == pytest.approx(DOOR_X - 3.0 - (policy.planner_params.channel_half_width_m - 0.1), abs=0.25)


def test_a_route_that_stays_in_the_open_has_no_channel_and_a_landmark_names_a_doorway():
    policy, _ = doorway_policy()
    world = doorway_world(0.8)
    assert channels_of([(1.5, 1.5), (4.5, 4.5)], world, policy) == []
    door = SimpleNamespace(id=7, xy=(DOOR_X + 0.2, DOOR_Y - 0.1))
    channels = channels_of([(3.0, DOOR_Y), (9.0, DOOR_Y)], world, policy, doors=[door])
    assert channels[0].door_id == 7
    far = SimpleNamespace(id=8, xy=(DOOR_X + 2.0, DOOR_Y))
    assert channels_of([(3.0, DOOR_Y), (9.0, DOOR_Y)], world, policy, doors=[far])[0].door_id is None


def test_nearest_channel_and_route_position():
    policy, _ = doorway_policy()
    world = doorway_world(0.8)
    route = [(3.0, DOOR_Y), (9.0, DOOR_Y)]
    channels = channels_of(route, world, policy)
    assert nearest_channel(channels, (DOOR_X - 0.3, DOOR_Y), 0.75) is channels[0]
    assert nearest_channel(channels, (DOOR_X - 2.0, DOOR_Y), 0.75) is None
    assert route_position(route, (4.5, DOOR_Y + 0.3)) == pytest.approx(1.5, abs=0.1)
    assert route_position(route, (9.5, DOOR_Y)) == pytest.approx(6.0, abs=0.1)
