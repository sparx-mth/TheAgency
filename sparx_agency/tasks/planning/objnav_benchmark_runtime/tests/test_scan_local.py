"""The two room visits of the scan mode: vantage point and a full circle, or the nearest frontiers with no middle.

Built on the two-room world of ``test_room_search_loop``: room B (east) has a small unknown patch by the door
and an unexplored east end. ``RPTSettings.room_visit`` picks the visit; the loop's ``scan_local`` is what it maps to.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from sparx_agency.core.planning.exploration.object_search_supervisor import EXHAUSTED, SEARCH
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import ROOM_VISIT_LOOP, RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_explore_visit import resolve_around, walk
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_opening_nodes import refresh
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    IN_A, loop_policy, obs_at)


def scan_policy(scan_local, order=(1, 0), **overrides):
    policy, episode, world, rooms, reasoned = loop_policy(order=order, visit="scan", scan_local=scan_local, **overrides)
    policy.sight.settings = replace(policy.sight.settings, enabled=False)
    refresh(policy, world, IN_A)
    return policy, episode, world, rooms, reasoned


def enter_room_b(policy, episode, world, revisit=False):
    """Plan once toward room B, then stand on the entry the transit aimed at; return it."""
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    if revisit:
        policy.loop._revisits.add(1)
    entry = tuple(policy.route_memory.goal)
    resolve_around(world, entry, 1.2)
    refresh(policy, world, entry)
    policy.sight.observe(obs_at(episode, 1, entry), world)
    return entry


@pytest.mark.parametrize("name", sorted(ROOM_VISIT_LOOP))
def test_each_room_visit_maps_to_the_loop_settings(name):
    policy, _ = setup_policy(room_visit=name)
    for key, value in ROOM_VISIT_LOOP[name].items():
        assert getattr(policy.loop_settings, key) == value


def test_an_unknown_room_visit_is_refused():
    with pytest.raises(ValueError, match="room_visit"):
        RPTSettings(room_visit="spiral")


def test_a_nearest_frontier_scan_is_budgeted_like_an_explore_visit():
    settings = scan_policy("nearest_frontier")[0].loop.settings
    assert settings.nearest_frontier_scan and settings.visit_budget() == settings.explore_visit_steps
    assert not scan_policy("rotate")[0].loop.settings.nearest_frontier_scan


def test_the_rotate_visit_turns_a_full_circle_at_the_vantage_point():
    policy, episode, world, rooms, _ = scan_policy("rotate")
    entry = enter_room_b(policy, episode, world)
    assert entry == policy.loop._vantage(world, policy.navigation_cost(world), rooms[1])[0]
    first = policy.loop.plan(obs_at(episode, 1, entry), world)
    assert policy.supervisor.state == SEARCH and first.info["kind"] == "room_scan"
    policy.loop.charge()
    commands, _, _ = walk(policy, episode, world, 2, entry)
    assert all(c.info.get("kind") == "room_scan" for c in commands[:-1])
    assert policy.scans.records[-1]["source"] == "room_scan"


def test_the_nearest_frontier_visit_enters_at_the_nearest_frontier_not_the_middle():
    policy, episode, world, rooms, _ = scan_policy("nearest_frontier")
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    cost = policy.navigation_cost(world)
    nearest = policy.sweep.nearest(obs_at(episode, 0, IN_A), world, cost, rooms[1].mask,
                                   labels=policy.graph.labels, label=rooms[1].id + 1)
    assert tuple(policy.route_memory.goal) == tuple(nearest.xy)
    assert policy.loop._entries[1][1] is True and policy.loop._entries[1][0] != policy.loop._vantage(world, cost, rooms[1])[0]


def test_the_nearest_frontier_visit_walks_frontiers_instead_of_turning_and_then_finishes_the_room():
    policy, episode, world, _, _ = scan_policy("nearest_frontier")
    entry = enter_room_b(policy, episode, world)
    first = policy.loop.plan(obs_at(episode, 1, entry), world)
    assert policy.supervisor.state == SEARCH and first.info["kind"] == "frontier" and first.waypoints
    assert any(e["event"] == "scan_begin" and e["phase"] == "frontier" for e in policy.loop.events)
    policy.loop.charge()
    commands, _, _ = walk(policy, episode, world, 2, entry)
    assert {c.info.get("kind") for c in commands[:-1]} <= {"frontier"}
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)
    assert policy.scans.records[-1]["source"] == "room_frontier_walk"
    assert policy.loop.stats["scans_completed"] == 1


def test_a_revisit_under_the_nearest_frontier_visit_still_turns_a_circle():
    policy, episode, world, rooms, _ = scan_policy("nearest_frontier")
    enter_room_b(policy, episode, world, revisit=True)
    vantage = policy.loop._vantage(world, policy.navigation_cost(world), rooms[1], revisit=True)[0]
    command = policy.loop.plan(obs_at(episode, 1, vantage), world)
    assert policy.supervisor.state == SEARCH and command.info["kind"] == "room_scan"
