"""The camera pitches on the approach to a doorway, the entry is re-planned on the pitched frame, the view restores past it."""
from __future__ import annotations

import math

from sparx_agency.core.planning.object_search.navigation.tests.rig import (DOOR_X, DOOR_Y, doorway_policy,
                                                                              doorway_world, observe, planned_route)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand


def rig(**overrides):
    policy, ep = doorway_policy(**overrides)
    world = doorway_world()
    route = planned_route(policy, world, (DOOR_X - 3.0, DOOR_Y), (DOOR_X + 2.5, DOOR_Y))
    points = [(p.x, p.y) for p in route.points]
    return policy, ep, world, route, points


def test_the_pitch_is_requested_within_approach_m_of_the_entry_and_released_past_the_exit():
    policy, ep, world, route, points = rig()
    look = policy.look_down
    assert look.pitch_request(observe(ep, 1, DOOR_X - 3.0, DOOR_Y, 0.0), world, points) is None, "3 m out: too early"
    assert look.pitch_request(observe(ep, 2, DOOR_X - 1.6, DOOR_Y, 0.0), world, points) is None, \
        "1.6 m out: too early since 2026-11-05 (a third of the look-downs requested from here were released before the doorway)"
    assert look.pitch_request(observe(ep, 2, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) == 30.0, "within 0.9 m of the entry"
    assert look.active and look.stats["requests"] == 1
    assert look.pitch_request(observe(ep, 3, DOOR_X - 0.2, DOOR_Y, 0.0), world, points) == 30.0, "held through the doorway"
    assert look.pitch_request(observe(ep, 4, DOOR_X + 1.5, DOOR_Y, 0.0), world, points) is None, "half a metre past the exit"
    assert look.stats["released_past_exit"] == 1 and not look.active
    assert look.pitch_request(observe(ep, 5, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) is None, "one look per doorway"


def test_a_wide_channel_is_not_looked_down_at_and_a_throat_near_one_looked_at_is_the_same_doorway():
    """2026-11-05: of 214 recorded look-downs 49 were at channels a metre wide and 50 within a metre of a throat
    already looked at; the plan through the channel changed in 21 % of them whatever the width."""
    policy, ep = doorway_policy()
    world = doorway_world(width_m=1.0)                                   # a 1.0 m doorway: the body passes it blind
    route = planned_route(policy, world, (DOOR_X - 3.0, DOOR_Y), (DOOR_X + 2.5, DOOR_Y))
    points = [(p.x, p.y) for p in route.points]
    assert policy.look_down.pitch_request(observe(ep, 2, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) is None
    assert policy.look_down.stats["requests"] == 0
    narrow, ep2 = doorway_policy(doorway_look_down={"max_width_m": 1.5})
    assert narrow.look_down.pitch_request(observe(ep2, 2, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) == 30.0, \
        "the bound is the setting's"
    # A throat looked at: a channel named up to done_radius_m away is the same doorway.
    policy, ep, world, route, points = rig()
    look = policy.look_down
    assert look.pitch_request(observe(ep, 2, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) == 30.0
    throat = look._active["throat"]
    look.pitch_request(observe(ep, 3, DOOR_X + 1.5, DOOR_Y, 0.0), world, points)      # past the exit: released, done
    assert not look.active and look._done == [throat]
    look._done = [(throat[0], throat[1] + 0.8)]                                       # the doorway named 0.8 m over
    assert look.pitch_request(observe(ep, 4, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) is None, "looked at already"
    look._done = [(throat[0], throat[1] + 1.3)]
    assert look.pitch_request(observe(ep, 5, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) == 30.0, "another doorway"
    import pytest
    from sparx_agency.core.planning.object_search.navigation.settings import LookDownSettings
    for kwargs in ({"max_width_m": 0.0}, {"done_radius_m": -1.0}):
        with pytest.raises(ValueError):
            LookDownSettings(**kwargs)


def test_the_first_pitched_frame_refreshes_once_and_drops_the_follower_plan():
    policy, ep, world, route, points = rig()
    look = policy.look_down
    assert not look.refreshed(observe(ep, 1, DOOR_X - 1.0, DOOR_Y, 0.0))
    assert look.pitch_request(observe(ep, 1, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) == 30.0
    assert not look.refreshed(observe(ep, 2, DOOR_X - 1.0, DOOR_Y, 0.0, pitch=0.0)), "the LOOK has not executed yet"
    assert look.refreshed(observe(ep, 3, DOOR_X - 1.0, DOOR_Y, 0.0, pitch=math.radians(30))), "the first pitched frame"
    assert not look.refreshed(observe(ep, 4, DOOR_X - 0.75, DOOR_Y, 0.0, pitch=math.radians(30))), "once"
    assert look.stats["refreshes"] == 1
    # Through the policy: the frame after the LOOK drops the plan in force, and the next command plans afresh.
    obs = observe(ep, 10, DOOR_X - 1.6, DOOR_Y, 0.0)
    policy.route_memory.adopt(route, points[-1], "frontier", obs)
    command = policy.kinematic.command(obs, world, policy.planner, NavigationCommand.follow(route))
    assert command.actions
    policy.look_down._active = {"throat": (0.0, 0.0), "channel": {}, "since": 10, "gone_since": None, "pitched": False,
                                "refreshed": False}
    policy.look_down._last_pitch = 0.0
    policy.look_down.refreshed(observe(ep, 11, DOOR_X - 1.6, DOOR_Y, 0.0, pitch=math.radians(30)))
    assert policy.look_down.stats["refreshes"] == 2


def test_the_camera_controller_pitches_for_the_doorway_under_the_search_owner_only():
    policy, ep, world, route, points = rig()
    obs = observe(ep, 2, DOOR_X - 1.6, DOOR_Y, 0.0)
    policy.camera_control.bottleneck_pitch_deg = 30.0
    held = policy.camera_control.apply(obs, NavigationCommand.follow(route), "SEARCH")
    assert held.camera_pitch == math.radians(30.0) and held.info["camera"]["owner"] == "DOORWAY"
    closing = policy.camera_control.apply(obs, NavigationCommand.follow(route), "TARGET_CLOSING")
    assert closing.info["camera"]["owner"] != "DOORWAY" and closing.camera_pitch == 0.0
    policy.camera_control.bottleneck_pitch_deg = None
    pitched = observe(ep, 3, DOOR_X - 1.6, DOOR_Y, 0.0, pitch=math.radians(30))
    restored = policy.camera_control.apply(pitched, NavigationCommand.follow(route), "SEARCH")
    assert restored.camera_pitch == 0.0 and restored.info["camera"]["owner"] == "RESTORE"


def test_a_hold_that_never_arrives_is_released_and_an_interrupted_walk_releases_after_a_grace():
    policy, ep, world, route, points = rig(doorway_look_down={"max_hold_actions": 3})
    look = policy.look_down
    assert look.pitch_request(observe(ep, 1, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) == 30.0
    assert look.pitch_request(observe(ep, 2, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) == 30.0
    assert look.pitch_request(observe(ep, 4, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) is None
    assert look.stats["released_hold"] == 1 and look.pitch_request(observe(ep, 5, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) is None
    other, _ = doorway_policy()
    assert other.look_down.pitch_request(observe(ep, 1, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) == 30.0
    assert other.look_down.pitch_request(observe(ep, 2, DOOR_X - 1.0, DOOR_Y, 0.0), world, None) == 30.0, "a hold: held"
    assert other.look_down.pitch_request(observe(ep, 3, DOOR_X - 1.0, DOOR_Y, 0.0), world, None) is None, "two: released"
    assert other.look_down.stats["released_route_gone"] == 1
    assert other.look_down.pitch_request(observe(ep, 4, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) is None, \
        "a released channel is never asked for again: the pair of LOOKs is spent once"
    facing_away, _ = doorway_policy()
    assert facing_away.look_down.pitch_request(observe(ep, 1, DOOR_X - 1.0, DOOR_Y, math.pi), world, points) is None, \
        "facing away from the throat: the turn comes first"


def test_disabled_no_pitch_is_ever_requested():
    policy, ep, world, route, points = rig(doorway_look_down={"enabled": False})
    assert policy.look_down.pitch_request(observe(ep, 2, DOOR_X - 1.0, DOOR_Y, 0.0), world, points) is None
