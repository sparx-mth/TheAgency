"""Informative glances along a route (2026-10-05): where one is scheduled, how it is performed, when not.

Built on a corridor with a doorway in its north wall and an unknown room
beyond: the route runs east along the corridor past the door. The policy is
a stand-in carrying exactly what the scheduler reads -- the route in force,
the action spec, the warm-up state, the scan ledger -- so every decision can
be checked against the drawing.
"""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.discovery import SUSPENDED_PHASES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.path_glances import (
    FULL, LEFT, RIGHT, GlanceScheduler, GlanceSettings)

RES = 0.1
FREE, OCC, UNK = 0, 100, -1
VALUES = OccupancyValues(free=FREE, occupied=OCC, unknown=UNK)


def corridor_world(door=True):
    g = np.full((100, 120), UNK, np.int8)
    g[20:30, 0:120] = FREE                 # the corridor, y 2.0-3.0 m
    g[30, 0:120] = OCC
    if door:
        g[30, 50:58] = FREE                # the doorway at x 5.0-5.8 m
    g[19, 0:120] = OCC
    g[0:19, :] = OCC
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


class FakeRoute:
    def __init__(self, path):
        self.path = path
        self.pauses = []

    def resume_after_pause(self, actions):
        self.pauses.append(actions)


def fake_policy(path, scans=()):
    camera = PROTOCOL.camera()
    actions = PROTOCOL.actions()
    policy = SimpleNamespace(
        route_memory=FakeRoute(path), episode=SimpleNamespace(action_spec=actions, camera=camera),
        settings=SimpleNamespace(warmup_steps=12), warmup_actions=12, _action_owner="search",
        closing=SimpleNamespace(active=False), _target_xy=None, building=None,
        mapping=SimpleNamespace(floor_id=0),
        scans=SimpleNamespace(on_floor=lambda floor_id=None: [{"xy": xy} for xy in scans]))
    return policy, camera, actions


def obs_at(camera, step, xy, yaw=0.0):
    k = camera.intrinsics
    return ObjNavObservation(np.zeros((k.height, k.width, 3), np.uint8), np.full((k.height, k.width), 3.0, np.float32),
                             AgentPose(xy[0], xy[1], 0.0, yaw), camera, "toilet", step)


def route_east(x0=1.0, x1=10.0, y=2.5, spacing=0.25):
    n = int(round((x1 - x0) / spacing)) + 1
    return [(x0 + i * spacing, y) for i in range(n)]


def follow(route):
    return NavigationCommand.follow(route, info={"kind": "transit/1"})


def test_settings_reject_nonsense():
    for kwargs in (dict(horizon_m=0.0), dict(cooldown_actions=0), dict(max_range_m=0.3), dict(side_rad=4.0), dict(enabled="yes")):
        with pytest.raises(ValueError):
            GlanceSettings(**kwargs)
    assert GlanceSettings().enabled


def test_a_left_glance_is_scheduled_abreast_of_the_doorway_not_where_the_agent_stands():
    route = route_east()
    policy, camera, actions = fake_policy(route)
    scheduler = GlanceScheduler(policy)
    world = corridor_world()
    command = follow(route)
    out = scheduler.apply(obs_at(camera, 0, (1.0, 2.5)), world, command)
    assert out is command, "the glance point is ahead: walk on"
    plan = scheduler.plan_in_force
    assert plan is not None and plan.kind == LEFT
    assert 4.3 <= plan.candidate.xy[0] <= 5.6, "where the room beyond the door shows best, not at x = 1"
    assert plan.actions == 6 and plan.gain_m2 >= GlanceSettings().min_gain_m2
    assert plan.candidate.right_m2 == 0.0, "the south wall is solid"
    scheduled = [e for e in scheduler.events if e["event"] == "glance_scheduled"]
    assert len(scheduled) == 1 and scheduled[0]["kind"] == LEFT
    assert scheduler.stats["evaluations"] == 1
    # Walking on without reaching the point changes nothing until the re-score cadence.
    again = scheduler.apply(obs_at(camera, 1, (1.25, 2.5)), world, command)
    assert again is command and scheduler.stats["evaluations"] == 1
    scheduler.apply(obs_at(camera, 3, (1.75, 2.5)), world, command)
    assert scheduler.stats["evaluations"] == 2, "re-scored every revalue_actions"


def test_no_glance_where_there_is_nothing_to_see():
    route = route_east()
    policy, camera, actions = fake_policy(route)
    scheduler = GlanceScheduler(policy)
    command = follow(route)
    assert scheduler.apply(obs_at(camera, 0, (1.0, 2.5)), corridor_world(door=False), command) is command
    assert scheduler.plan_in_force is None and scheduler.events == []


def test_the_glance_is_performed_at_the_point_one_turn_per_action_then_the_route_resumes():
    route = route_east()
    policy, camera, actions = fake_policy(route)
    scheduler = GlanceScheduler(policy)
    world = corridor_world()
    command = follow(route)
    scheduler.apply(obs_at(camera, 0, (1.0, 2.5)), world, command)
    point = scheduler.plan_in_force.candidate.xy
    turn = actions.turn_angle_rad
    # Arrive at the point: the first turn is emitted and the policy's action owner is the glance.
    first = scheduler.apply(obs_at(camera, 10, point, yaw=0.0), world, command)
    assert first is not command and first.waypoints == () and first.info["kind"] == "glance"
    assert first.final_yaw == pytest.approx(turn) and policy._action_owner == "glance"
    assert scheduler.active is not None and scheduler.active["kind"] == LEFT
    assert "glance" in SUSPENDED_PHASES, "the loop is not charged and the clocks pause while it turns"
    started = [e for e in scheduler.events if e["event"] == "glance_started"]
    assert len(started) == 1 and started[0]["kind"] == LEFT
    yaw = float(first.final_yaw)
    second = scheduler.continue_look(obs_at(camera, 11, point, yaw=yaw))
    assert second.final_yaw == pytest.approx(2 * turn)
    yaw = float(second.final_yaw)
    third = scheduler.continue_look(obs_at(camera, 12, point, yaw=yaw))
    assert third.final_yaw == pytest.approx(math.pi / 2, abs=1e-6)
    yaw = float(third.final_yaw)
    done = scheduler.continue_look(obs_at(camera, 13, point, yaw=yaw))
    assert done is None, "facing the side: the look is complete and the search resumes this action"
    assert scheduler.active is None and scheduler.stats["completed"] == 1 and scheduler.stats["turns"] == 3
    assert policy.route_memory.pauses == [3], "the committed route's watchdogs are told how long the pause was"
    complete = [e for e in scheduler.events if e["event"] == "glance_complete"][0]
    assert complete["turns"] == 3 and complete["swept_deg"] == pytest.approx(90.0, abs=1.0)
    # The cooldown holds the next glance off for a while, then the route is scored afresh.
    assert scheduler.apply(obs_at(camera, 14, point, yaw=yaw), world, command) is command
    assert scheduler.plan_in_force is None


def test_a_side_glance_beats_a_full_circle_unless_the_rear_holds_something_too():
    """Two doorways opposite each other: a full circle (12 actions) sees both rooms, a left glance (6)
    sees the bigger one at twice the rate. The other side is a second glance's business later."""
    g = corridor_world().grid.copy()
    g[19, 50:58] = FREE                                     # a second doorway, in the south wall, opposite the first
    g[0:19, 30:80] = UNK                                    # ... with a big unknown room beyond it
    world = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    route = route_east()
    policy, camera, actions = fake_policy(route)
    scheduler = GlanceScheduler(policy)
    scheduler.apply(obs_at(camera, 0, (1.0, 2.5)), world, follow(route))
    plan = scheduler.plan_in_force
    assert plan is not None and plan.kind in (LEFT, RIGHT) and plan.actions == 6
    assert plan.candidate.left_m2 > 5 and plan.candidate.right_m2 > 3 and plan.candidate.full_m2 > plan.gain_m2
    assert plan.rate > plan.candidate.full_m2 / 12, "the full circle's gain per action is lower"


def test_a_full_glance_turns_all_the_way_round_in_the_richer_direction():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.path_glances import GlanceCandidate, GlancePlan
    route = route_east()
    policy, camera, actions = fake_policy(route)
    scheduler = GlanceScheduler(policy)
    point = (5.4, 2.5)
    candidate = GlanceCandidate(9, point, 0.0, 4.4, left_m2=12.0, right_m2=3.0, full_m2=20.0)
    turn = actions.turn_angle_rad
    command = scheduler._start(obs_at(camera, 10, point, yaw=0.0), GlancePlan(candidate, FULL, 20.0, 12))
    assert command.info["glance"] == FULL and scheduler.active["direction"] == 1.0, "the left is the richer side: turn left"
    assert command.final_yaw == pytest.approx(turn)
    yaw, turns, step = float(command.final_yaw), 1, 10
    while scheduler.active is not None and turns < 20:
        step += 1
        nxt = scheduler.continue_look(obs_at(camera, step, point, yaw=yaw))
        if nxt is None:
            break
        assert nxt.final_yaw == pytest.approx(math.atan2(math.sin(yaw + turn), math.cos(yaw + turn)), abs=1e-6)
        yaw = float(nxt.final_yaw)
        turns += 1
    assert scheduler.active is None and 12 <= turns <= 14
    assert policy.route_memory.pauses == [turns]
    complete = [e for e in scheduler.events if e["event"] == "glance_complete"][0]
    assert complete["kind"] == FULL and complete["swept_deg"] >= 345.0


def test_no_glance_during_the_warmup_a_peek_a_takeover_or_within_a_scanned_spot():
    route = route_east()
    world = corridor_world()
    command = follow(route)
    policy, camera, actions = fake_policy(route)
    policy.warmup_actions = 3
    assert GlanceScheduler(policy).apply(obs_at(camera, 0, (1.0, 2.5)), world, command) is command
    policy, camera, actions = fake_policy(route)
    policy._action_owner = "doorway_peek"
    scheduler = GlanceScheduler(policy)
    assert scheduler.apply(obs_at(camera, 0, (1.0, 2.5)), world, command) is command and scheduler.plan_in_force is None
    policy, camera, actions = fake_policy(route)
    policy._target_xy = (7.0, 2.5)
    assert GlanceScheduler(policy).apply(obs_at(camera, 0, (1.0, 2.5)), world, command) is command
    policy, camera, actions = fake_policy(route)
    policy.closing = SimpleNamespace(active=True)
    assert GlanceScheduler(policy).apply(obs_at(camera, 0, (1.0, 2.5)), world, command) is command
    # A full rotation stood abreast of the door already: nothing more to see from there.
    policy, camera, actions = fake_policy(route, scans=[(5.4, 2.5)])
    scheduler = GlanceScheduler(policy)
    scheduler.apply(obs_at(camera, 0, (1.0, 2.5)), world, command)
    plan = scheduler.plan_in_force
    assert plan is None or math.dist(plan.candidate.xy, (5.4, 2.5)) > GlanceSettings().scanned_m
    # Disabled: never anything but the command.
    policy, camera, actions = fake_policy(route)
    scheduler = GlanceScheduler(policy, GlanceSettings(enabled=False))
    assert scheduler.apply(obs_at(camera, 0, (1.0, 2.5)), world, command) is command and scheduler.plan_in_force is None
    # A hold (a scan's rotation, a peek's look) is never wrapped.
    policy, camera, actions = fake_policy(route)
    hold = NavigationCommand.hold(final_yaw=0.5, info={"kind": "room_scan"})
    assert GlanceScheduler(policy).apply(obs_at(camera, 0, (1.0, 2.5)), world, hold) is hold


def test_a_target_sighting_aborts_the_glance_and_resumes_the_route_clocks():
    route = route_east()
    policy, camera, actions = fake_policy(route)
    scheduler = GlanceScheduler(policy)
    world = corridor_world()
    command = follow(route)
    scheduler.apply(obs_at(camera, 0, (1.0, 2.5)), world, command)
    point = scheduler.plan_in_force.candidate.xy
    scheduler.apply(obs_at(camera, 10, point, yaw=0.0), world, command)
    scheduler.continue_look(obs_at(camera, 11, point, yaw=actions.turn_angle_rad))
    assert scheduler.active is not None and scheduler.active["turns"] == 2
    scheduler.abort(obs_at(camera, 12, point), "target_priority")
    assert scheduler.active is None and scheduler.plan_in_force is None
    assert scheduler.stats["aborted"] == 1 and policy.route_memory.pauses == [2]
    assert scheduler.events[-1]["event"] == "glance_aborted" and scheduler.events[-1]["reason"] == "target_priority"
    state = scheduler.state()
    assert state == {"active": None, "planned": None}
    diagnostics = scheduler.diagnostics()
    assert diagnostics["stats"]["started"] == 1 and len(diagnostics["events"]) >= 3


def test_a_glance_point_walked_past_is_taken_from_where_the_agent_is():
    route = route_east()
    policy, camera, actions = fake_policy(route)
    scheduler = GlanceScheduler(policy, replace(GlanceSettings(), revalue_actions=1000))
    world = corridor_world()
    command = follow(route)
    scheduler.apply(obs_at(camera, 0, (1.0, 2.5)), world, command)
    point = scheduler.plan_in_force.candidate.xy
    beyond = (point[0] + 1.0, 2.5)                          # the converter overshot the point by a metre
    out = scheduler.apply(obs_at(camera, 10, beyond, yaw=0.0), world, command)
    assert out.info["kind"] == "glance", "passed without arriving: look from here rather than never"


def test_the_policy_wraps_the_loops_transit_in_a_glance_and_resumes_the_search_when_it_ends():
    """End to end on the two-room fixture: the transit east leaves the unexplored west end behind the
    agent, a full glance is worth its twelve actions, the loop is not charged while it turns, and the
    action the look completes on goes to the transit again."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
        IN_A, loop_policy, obs_at as loop_obs)
    policy, episode, world, rooms, reasoned = loop_policy(order=(1, 0), visit="scan")
    policy.glance_settings = GlanceSettings(min_gain_m2=1.0, min_gain_per_action_m2=0.1)
    policy.glances = GlanceScheduler(policy, policy.glance_settings)
    obs = loop_obs(episode, 0, IN_A)
    command = policy._search(obs, world, False)
    assert command.info["kind"] == "glance" and command.waypoints == (), "the west end behind the agent is worth a look"
    assert policy._action_owner == "glance" and policy.glances.active["kind"] == FULL
    assert policy.supervisor.state == "transit" and policy.route_memory.path is not None, "the transit was planned and kept"
    charged = policy.loop.local_steps
    policy.notify_action(obs, __import__("sparx_agency.core.planning.objnav.types.actions", fromlist=["DiscreteAction"]).DiscreteAction.TURN_LEFT)
    assert policy.loop.local_steps == charged, "a suspended phase: the loop is not charged"
    yaw = float(command.final_yaw)
    turns, step = 1, 1
    while policy.glances.active is not None and turns < 20:
        nxt = policy._search(loop_obs(episode, step, IN_A, yaw), world, False)
        if nxt.info.get("kind") != "glance":
            break
        yaw, turns, step = float(nxt.final_yaw), turns + 1, step + 1
    assert policy.glances.active is None and 12 <= turns <= 14
    assert nxt.waypoints and nxt.info["kind"] == "transit/1", "the look is over: this very action is the transit's"
    assert policy.glances.stats["completed"] == 1 and policy.route_memory.stats["kept"] >= 1
