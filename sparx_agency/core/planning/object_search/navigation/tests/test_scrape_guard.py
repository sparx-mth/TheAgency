"""A refused step in a doorway, aligned with it, stays off the map; off the axis or repeated, it is written."""
from __future__ import annotations

import math

from sparx_agency.core.planning.object_search.navigation.tests.rig import (DOOR_X, DOOR_Y, doorway_policy,
                                                                              doorway_world, observe, planned_route)


def rig(**overrides):
    policy, ep = doorway_policy(**overrides)
    world = doorway_world()
    route = planned_route(policy, world, (DOOR_X - 2.0, DOOR_Y), (DOOR_X + 2.5, DOOR_Y))
    policy.last_world = world
    return policy, ep, world, route


def written(policy):
    """How many refused steps the observed map was told about (each is a cell painted occupied)."""
    return len(policy.mapping._bumps.get(policy.mapping.floor_id, ()))


def test_an_aligned_refusal_at_the_doorway_is_a_scrape_and_writes_nothing():
    policy, ep, world, route = rig()
    obs = observe(ep, 5, DOOR_X - 0.4, DOOR_Y + 0.1, math.radians(25))      # one turn off the axis: a clipped jamb
    verdict = policy.scrape_guard.judge(obs, world, route)
    assert verdict is not None and verdict["kind"] == "doorway" and verdict["heading_error_deg"] == 25.0
    assert verdict["excused"] == 1 and abs(verdict["throat_xy"][0] - DOOR_X) <= 0.15
    before = written(policy)
    policy._route = route
    policy.notify_blocked(obs)
    assert written(policy) == before and policy.scrape_guard.stats["scrapes"] == 2


def test_a_refusal_across_the_doorway_or_far_from_one_is_written_as_before():
    policy, ep, world, route = rig()
    across = observe(ep, 5, DOOR_X - 0.4, DOOR_Y, math.radians(80))
    assert policy.scrape_guard.judge(across, world, route) is None and policy.scrape_guard.stats["misaligned"] == 1
    far = observe(ep, 6, DOOR_X - 1.8, DOOR_Y, 0.0)
    assert policy.scrape_guard.judge(far, world, route) is None and policy.scrape_guard.stats["no_channel"] == 1
    assert policy.scrape_guard.judge(far, world, None) is None and policy.scrape_guard.stats["no_route"] == 1
    before = written(policy)
    policy._route = route
    policy.notify_blocked(across)
    assert written(policy) == before + 1, "the cell ahead is painted, as before"


def test_the_excuse_runs_out_after_scrapes_before_mark_within_the_window():
    policy, ep, world, route = rig(doorway_scrape={"scrapes_before_mark": 2, "window_actions": 10})
    obs = observe(ep, 5, DOOR_X - 0.4, DOOR_Y, math.radians(20))
    assert policy.scrape_guard.judge(obs, world, route) is not None
    assert policy.scrape_guard.judge(observe(ep, 7, DOOR_X - 0.4, DOOR_Y, math.radians(-20)), world, route) is not None
    assert policy.scrape_guard.judge(observe(ep, 9, DOOR_X - 0.4, DOOR_Y, 0.0), world, route) is None
    assert policy.scrape_guard.stats["exhausted"] == 1 and policy.scrape_guard.stats["written"] == 1
    assert policy.scrape_guard.judge(observe(ep, 30, DOOR_X - 0.4, DOOR_Y, 0.0), world, route) is not None, \
        "outside the window the doorway is given the benefit of the doubt again"


def test_disabled_every_refusal_is_written():
    policy, ep, world, route = rig(doorway_scrape={"enabled": False})
    obs = observe(ep, 5, DOOR_X - 0.4, DOOR_Y, 0.0)
    assert policy.scrape_guard.judge(obs, world, route) is None
    assert policy.scrape_guard.diagnostics()["stats"]["refusals"] == 1
