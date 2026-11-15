"""A refused step in a doorway, aligned with it, stays off the map; off the axis or repeated, it is written."""
from __future__ import annotations

import math

from sparx_agency.core.planning.object_search.navigation.tests.rig import (DOOR_X, DOOR_Y, confirm_door,
                                                                              doorway_policy, doorway_world, observe,
                                                                              planned_route)


def rig(door=True, **overrides):
    """The doorway world with its route; ``door`` confirms a door landmark at the throat (the guard's default needs one)."""
    policy, ep = doorway_policy(**overrides)
    world = doorway_world()
    route = planned_route(policy, world, (DOOR_X - 2.0, DOOR_Y), (DOOR_X + 2.5, DOOR_Y))
    policy.last_world = world
    if door:
        confirm_door(policy)
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
    policy, ep, world, route = rig(doorway_scrape={"scrapes_before_mark": 2, "window_actions": 10,
                                                  "closed_memory_actions": 50})
    obs = observe(ep, 5, DOOR_X - 0.4, DOOR_Y, math.radians(20))
    assert policy.scrape_guard.judge(obs, world, route) is not None
    assert policy.scrape_guard.judge(observe(ep, 7, DOOR_X - 0.4, DOOR_Y, math.radians(-20)), world, route) is not None
    assert policy.scrape_guard.judge(observe(ep, 9, DOOR_X - 0.4, DOOR_Y, 0.0), world, route) is None
    assert policy.scrape_guard.stats["exhausted"] == 1 and policy.scrape_guard.stats["written"] == 1
    # The place is closed: outside the window but within the closed memory a refusal there is written at once.
    assert policy.scrape_guard.judge(observe(ep, 30, DOOR_X - 0.4, DOOR_Y, 0.0), world, route) is None
    assert policy.scrape_guard.stats["closed"] == 1 and policy.scrape_guard.stats["written"] == 2
    assert policy.scrape_guard.events[-1]["detail"] == "written: the place was closed at action 9"
    # Beyond the closed memory the doorway is given the benefit of the doubt again.
    assert policy.scrape_guard.judge(observe(ep, 60, DOOR_X - 0.4, DOOR_Y, 0.0), world, route) is not None
    assert policy.scrape_guard.stats["closed"] == 1
    # Another doorway is another place: its own excuses.
    policy2, ep2, world2, route2 = rig(doorway_scrape={"scrapes_before_mark": 2, "window_actions": 10})
    assert policy2.scrape_guard.judge(observe(ep2, 5, DOOR_X - 0.4, DOOR_Y, 0.0), world2, route2) is not None
    assert policy2.scrape_guard.judge(observe(ep2, 6, DOOR_X - 0.4, DOOR_Y, 0.0), world2, route2) is not None
    assert policy2.scrape_guard.judge(observe(ep2, 7, DOOR_X - 0.4, DOOR_Y, 0.0), world2, route2) is None
    assert policy2.scrape_guard.judge(observe(ep2, 8, DOOR_X - 0.4, DOOR_Y + 3.0, 0.0), world2, route2) is None, \
        "far from the channel: not a scrape at all (no_channel), and no closure applies either"
    assert policy2.scrape_guard.stats["no_channel"] == 1 and policy2.scrape_guard.stats["closed"] == 0


def test_scrapes_are_counted_by_place_across_the_routes_re_planned_after_each_refusal():
    """00843/000007 (``25a50599d098``, 2026-11-05), actions 117-132: refused at one spot at four headings 90 degrees
    apart -- a hole in the mesh -- and every refusal excused as the first scrape of a new channel, because each
    refusal clears the route and the re-planned route's throat fell into another half-metre bucket. The excuses
    are per place now; a second visit 80 actions later (211-216) found the place closed and wrote the map."""
    policy, ep, world, _ = rig(doorway_scrape={"scrapes_before_mark": 3, "window_actions": 40,
                                             "closed_memory_actions": 120})
    # Routes re-planned from the spot after each refusal: different starts, the same doorway, throats a few
    # centimetres apart; the refusals from one standing position at successive headings.
    routes = [planned_route(policy, world, (DOOR_X - 1.2 - 0.3 * i, DOOR_Y + 0.15 * (i - 1)), (DOOR_X + 2.5, DOOR_Y))
              for i in range(4)]
    headings = (math.radians(20), math.radians(-20), math.radians(35), math.radians(-35))
    verdicts = [policy.scrape_guard.judge(observe(ep, 117 + 4 * i, DOOR_X - 0.4, DOOR_Y, headings[i]), world, routes[i])
                for i in range(4)]
    assert [v is not None for v in verdicts] == [True, True, True, False], "three excuses at the place, then written"
    assert [v["excused"] for v in verdicts[:3]] == [1, 2, 3]
    assert policy.scrape_guard.stats["exhausted"] == 1 and policy.scrape_guard.stats["written"] == 1
    # The second visit, 80 actions on: the place is closed, the first refusal is written.
    later = planned_route(policy, world, (DOOR_X - 1.5, DOOR_Y - 0.2), (DOOR_X + 2.5, DOOR_Y))
    assert policy.scrape_guard.judge(observe(ep, 211, DOOR_X - 0.45, DOOR_Y + 0.05, math.radians(25)), world, later) is None
    assert policy.scrape_guard.stats["closed"] == 1 and policy.scrape_guard.stats["written"] == 2
    before = len(policy.mapping._bumps.get(policy.mapping.floor_id, ()))
    policy._route = later
    policy.notify_blocked(observe(ep, 213, DOOR_X - 0.45, DOOR_Y + 0.05, math.radians(-25)))
    assert len(policy.mapping._bumps.get(policy.mapping.floor_id, ())) == before + 1, "the cell ahead is painted"


def test_a_squeeze_no_door_names_is_written_at_once_unless_the_guard_is_told_otherwise():
    """Since 2026-11-10: the gap between furniture and a wall is a channel too, and excusing refusals there kept the
    map from learning the furniture (2azQ1b91cZZ/000058: six refusals from one spot, the lock released as stalled)."""
    policy, ep, world, route = rig(door=False)
    obs = observe(ep, 5, DOOR_X - 0.4, DOOR_Y + 0.1, math.radians(25))
    assert policy.scrape_guard.judge(obs, world, route) is None and policy.scrape_guard.stats["no_door"] == 1
    assert policy.scrape_guard.events[-1]["detail"] == "written: no door names the channel"
    before = written(policy)
    policy._route = route
    policy.notify_blocked(obs)
    assert written(policy) == before + 1, "the furniture the body pushed against is on the map at once"
    legacy, ep, world, route = rig(door=False, doorway_scrape={"doorways_only": False})
    assert legacy.scrape_guard.judge(obs, world, route) is not None, "the geometry-only channel, on request"
    assert legacy.scrape_guard.stats["no_door"] == 0


def test_disabled_every_refusal_is_written():
    policy, ep, world, route = rig(doorway_scrape={"enabled": False})
    obs = observe(ep, 5, DOOR_X - 0.4, DOOR_Y, 0.0)
    assert policy.scrape_guard.judge(obs, world, route) is None
    assert policy.scrape_guard.diagnostics()["stats"]["refusals"] == 1
