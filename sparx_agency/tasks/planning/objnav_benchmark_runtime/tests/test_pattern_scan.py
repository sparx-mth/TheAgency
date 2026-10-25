"""The gain-ranked look-around (``scan_kind="pattern"``): it turns only toward unknown floor and stops early."""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.exploration.object_search_supervisor import EXHAUSTED, SEARCH
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.pattern_scan import PatternScanSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_explore_visit import walk
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    IN_A, loop_policy, obs_at)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_scan_local import enter_room_b, scan_policy


def reveal_ahead(policy, world, obs):
    """What the camera does on a turn: the unknown cells in its cone become free floor."""
    view = policy.pattern_scan._view(world)
    world.grid[view.unknown_ahead((obs.pose.x, obs.pose.y), obs.pose.yaw)] = world.values.free


def test_a_known_room_needs_no_look_around():
    policy, episode, world, _, _ = loop_policy()
    world.grid[world.grid == world.values.unknown] = world.values.free
    policy.pattern_scan.begin()
    assert policy.pattern_scan.step(obs_at(episode, 0, IN_A), world) is None


def test_the_first_turn_heads_toward_the_unknown_west_end_of_the_room():
    policy, episode, world, _, _ = loop_policy()
    policy.pattern_scan.begin()
    command = policy.pattern_scan.step(obs_at(episode, 0, IN_A), world)
    assert command.waypoints == () and command.info["kind"] == "pattern_scan"
    assert abs(abs(command.info["target_deg"]) - 180.0) <= 60.0


def test_the_look_stops_before_a_full_circle_once_the_unknown_is_revealed():
    policy, episode, world, _, _ = loop_policy()
    policy.pattern_scan.begin()
    yaw, turns = 0.0, 0
    for step in range(policy.rotation_turns() + 2):
        obs = obs_at(episode, step, IN_A, yaw=yaw)
        reveal_ahead(policy, world, obs)
        command = policy.pattern_scan.step(obs, world)
        if command is None:
            break
        turns += 1
        yaw = command.final_yaw
    assert command is None and 0 < turns < policy.rotation_turns()


def test_the_look_never_exceeds_a_full_circles_turns():
    policy, episode, world, _, _ = loop_policy()
    policy.pattern_scan.begin()
    turns = 0
    while policy.pattern_scan.step(obs_at(episode, turns, IN_A), world) is not None:
        turns += 1
        assert turns <= policy.rotation_turns()
    assert turns == policy.rotation_turns()


def test_the_scan_kind_and_its_settings_are_validated():
    with pytest.raises(ValueError, match="scan_kind"):
        RPTSettings(scan_kind="spiral")
    with pytest.raises(ValueError, match="min_gain_m2"):
        PatternScanSettings(min_gain_m2=-1.0)
    assert RPTSettings(pattern_scan={"min_gain_m2": 2.5}).pattern_scan.min_gain_m2 == 2.5


def test_the_warmup_under_the_pattern_scan_ends_with_the_ledger_record():
    p, episode = setup_policy("bed", discovery=True, scan_kind="pattern", doorway_peek={"enabled": False})
    budget = p.warmup_turns
    for step in range(budget + 1):
        obs = observation(episode, step, depth=3)
        command = p.plan(obs)
        if command.info.get("kind") != "warmup":
            break
        assert "pattern_turn" in command.info and command.waypoints == ()
        p.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert p.warmup_actions == budget and p.scans.records[0]["source"] == "warmup"


def test_a_room_scan_under_the_pattern_scan_records_a_pattern_scan_and_finishes_the_room():
    policy, episode, world, _, _ = scan_policy("rotate", settings={"scan_kind": "pattern"})
    entry = enter_room_b(policy, episode, world)
    first = policy.loop.plan(obs_at(episode, 1, entry), world)
    assert policy.supervisor.state == SEARCH and first.info["kind"] == "room_scan" and "pattern_turn" in first.info
    policy.loop.charge()
    commands, _, _ = walk(policy, episode, world, 2, entry)
    assert all(c.info.get("kind") == "room_scan" for c in commands[:-1])
    assert policy.supervisor.history[-1][:2] == (1, EXHAUSTED)
    assert policy.scans.records[-1]["source"] == "pattern_scan"
    assert policy.loop.stats["scans_completed"] == 1
