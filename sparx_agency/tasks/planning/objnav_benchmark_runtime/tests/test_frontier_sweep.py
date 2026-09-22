"""Regressions for the bounded frontier sweep and the inspection gate.

Each test is one wasted-action pattern read off the Ranchester recording
``e7c4f2ad5402``; the supervisor is scripted so a room can be exhausted,
released and re-selected in microseconds, and the planner is stubbed so the
sweep's decisions are tested rather than A*.
"""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    EXHAUSTED, SEARCH, SELECT, TRANSIT, UNREACHABLE, FlyTo, ObjectSearchState, Release)
from sparx_agency.core.planning.exploration.room_search_policy import Hold
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods import frontier_sweep
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.frontier_sweep import SweepSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.route_memory import CommittedRoute
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy, RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import (
    FakeDetector, FakeLLM, observation, setup_policy)


class ScriptedSupervisor:
    """Returns the scripted states in order (the last one repeats) and records every call."""

    def __init__(self, *states):
        self.states = list(states)
        self.calls = []
        self.state = states[0].state

    def update(self, *args, **kwargs):
        self.calls.append(kwargs)
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        self.state = state.state
        return state


def searching(room_id=0):
    return ObjectSearchState(state=SEARCH, action=Hold("mapping"), room_id=room_id, goal_xy=(0.0, 0.0))


def released(room_id, verdict):
    return ObjectSearchState(state=SELECT, action=Release(room_id, verdict), completed=(room_id, verdict))


def flying_to(room_id, xy):
    return ObjectSearchState(state=TRANSIT, action=FlyTo(room_id, xy), room_id=room_id, goal_xy=xy, changed=True)


def swept_policy(monkeypatch):
    """A policy after one real observation, with no frontier anywhere and one room."""
    policy, episode = setup_policy()
    policy.plan(observation(episode, 0, depth=3.0))
    world = policy.last_world
    monkeypatch.setattr(frontier_sweep, "ranked_frontier_goals", lambda *a, **k: [])
    policy.graph.registry.rooms = {0: SimpleNamespace(mask=np.ones(world.grid.shape, bool))}
    return policy, episode, world


def test_a_swept_room_gets_one_rotation_then_leaves_on_the_same_action(monkeypatch):
    policy, episode, world = swept_policy(monkeypatch)
    sup = ScriptedSupervisor(searching(0))
    policy.supervisor = sup
    turns = int(math.ceil(2 * math.pi / episode.action_spec.turn_angle_rad))
    commands = [policy.sweep.plan(observation(episode, step, depth=3.0), world) for step in range(1, turns + 1)]
    assert all(c.info.get("kind") == "room_scan" and c.final_yaw is not None for c in commands)
    assert [c.info["scan_turn"] for c in commands] == list(range(1, turns + 1))
    assert not any(call.get("frontier_exhausted") for call in sup.calls)

    sup.states = [searching(0), released(0, EXHAUSTED), ObjectSearchState(state=SELECT, action=Hold("no room"))]
    after = policy.sweep.plan(observation(episode, turns + 1, depth=3.0), world)
    assert after.info.get("kind") != "room_scan"
    assert sup.calls[-2].get("frontier_exhausted") is True, "the exhausted flag released the room"
    assert sup.calls[-1].get("frontier_exhausted", False) is False, "the reselect call carries no stale flag"
    assert policy.sweep.stats["rooms_released"] == 1
    assert policy.sweep.stats["scan_turns"] == turns


def test_room_scan_budget_zero_leaves_at_once(monkeypatch):
    policy, episode, world = swept_policy(monkeypatch)
    policy.sweep.settings = replace(policy.sweep.settings, room_scan_turns=0)
    sup = ScriptedSupervisor(searching(0), released(0, EXHAUSTED),
                             ObjectSearchState(state=SELECT, action=Hold("no room")))
    policy.supervisor = sup
    command = policy.sweep.plan(observation(episode, 1, depth=3.0), world)
    assert command.info.get("kind") != "room_scan"
    assert sup.calls[1]["frontier_exhausted"] is True


def test_a_released_room_is_replaced_by_a_transit_on_the_same_action(monkeypatch):
    """The recording planned a throwaway 7-9 m floor route on every release."""
    policy, episode, world = swept_policy(monkeypatch)
    navigated = []

    def navigate(obs, world_, goal, kind, final_yaw=None):
        navigated.append((goal, kind))
        return NavigationCommand.follow([(0.0, 0.0), goal], info={"kind": kind})

    monkeypatch.setattr(policy, "_navigate", navigate)
    sup = ScriptedSupervisor(released(0, EXHAUSTED), flying_to(1, (2.0, 0.0)))
    policy.supervisor = sup
    command = policy.sweep.plan(observation(episode, 1, depth=3.0), world)
    assert len(sup.calls) == 2
    assert navigated == [((2.0, 0.0), "transit/1")]
    assert command.info["kind"] == "transit/1"
    assert policy.route_memory.path is None, "the released room's route was dropped before the transit"


def test_a_refused_transit_route_ends_the_room_instead_of_idling(monkeypatch):
    policy, episode, world = swept_policy(monkeypatch)
    answers = iter([None, NavigationCommand.follow([(0.0, 0.0), (4.0, 0.0)], info={"kind": "transit/2"})])
    monkeypatch.setattr(policy, "_navigate", lambda *a, **k: next(answers))
    sup = ScriptedSupervisor(flying_to(1, (2.0, 0.0)), released(1, UNREACHABLE), flying_to(2, (4.0, 0.0)))
    policy.supervisor = sup
    command = policy.sweep.plan(observation(episode, 1, depth=3.0), world)
    assert sup.calls[1]["route_failed"] is True
    assert command.info["kind"] == "transit/2"
    assert policy.sweep.stats["plan_failures"] == 1


def test_a_committed_goal_survives_a_moved_room_mask_but_not_a_resolved_boundary():
    policy, episode = setup_policy()
    policy.plan(observation(episode, 0, depth=3.0))
    world = policy.last_world
    gx, gy = world.world_to_grid(1.5, 0.0)
    grid = world.grid
    reach = int(math.ceil(policy.sweep.settings.informative_radius_m / world.resolution)) + 2
    grid[gy - reach:gy + reach + 1, gx - reach:gx + reach + 1] = world.values.free
    boundary = (slice(gy - 2, gy + 3), gx + 3)                   # five unknown cells beyond the goal
    grid[boundary] = world.values.unknown
    assert np.count_nonzero(grid[boundary] == world.values.unknown) >= policy.sweep.settings.informative_cells
    cost = np.ones(grid.shape)
    policy._goal, policy.route_memory.kind = (1.5, 0.0), "frontier"
    policy.route_memory.path = SimpleNamespace(points=[Pose2D(0, 0), Pose2D(1.5, 0)])
    policy.route_memory.progress.adopt(((0.0, 0.0), (1.5, 0.0)))
    obs = observation(episode, 1)
    arrival = policy.converter_params.goal_tolerance_m + world.resolution
    elsewhere = np.zeros(grid.shape, bool)
    elsewhere[gy + 5, gx] = True                                 # mask moved, still within the slack
    assert policy.sweep._current_goal(obs, world, cost, elsewhere, arrival) == (1.5, 0.0)
    grid[boundary] = world.values.free                           # the boundary got observed
    assert policy.sweep._current_goal(obs, world, cost, elsewhere, arrival) is None
    assert (1.5, 0.0) in policy._visited_frontiers and policy.route_memory.path is None
    assert policy.sweep.stats["goals_resolved"] == 1


def test_a_committed_goal_far_outside_the_room_is_dropped():
    policy, episode = setup_policy()
    policy.plan(observation(episode, 0, depth=3.0))
    world = policy.last_world
    cost = np.ones(world.grid.shape)
    policy._goal, policy.route_memory.kind = (1.5, 0.0), "frontier"
    policy.route_memory.path = SimpleNamespace(points=[Pose2D(0, 0), Pose2D(1.5, 0)])
    policy.route_memory.progress.adopt(((0.0, 0.0), (1.5, 0.0)))
    arrival = policy.converter_params.goal_tolerance_m + world.resolution
    far_room = np.zeros(world.grid.shape, bool)
    gx, gy = world.world_to_grid(1.5, 5.0)
    far_room[gy, gx] = True
    assert policy.sweep._current_goal(observation(episode, 1), world, cost, far_room, arrival) is None


def test_route_memory_records_why_the_previous_route_was_replaced():
    route = CommittedRoute(PROTOCOL.actions(), ActionConverterParams())
    obs = SimpleNamespace(step=0, pose=AgentPose(0, 0, 0, 0))
    route.adopt([Pose2D(0, 0), Pose2D(3, 0)], (3, 0), "frontier", obs)
    assert route.replaced == "unplanned"
    route.clear("route_obstructed")
    route.adopt([Pose2D(0, 0), Pose2D(3, 1)], (3, 1), "frontier", obs)
    assert route.replaced == "route_obstructed"
    assert route.stats["cleared:route_obstructed"] == 1 and route.stats["invalidations"] == 1


def test_no_progress_is_a_one_shot_verdict_that_is_still_reported_on_the_next_adoption():
    """A latched no_progress would refuse every later goal without planning."""
    route = CommittedRoute(PROTOCOL.actions(), ActionConverterParams())
    planner = SimpleNamespace(path_collides=lambda *a, **k: False)
    still = AgentPose(0, 0, 0, 0)
    start = SimpleNamespace(step=0, pose=still)
    route.adopt([Pose2D(0, 0), Pose2D(3, 0)], (3, 0), "frontier", start)
    assert route.reusable(start, None, planner, (3, 0), "frontier")
    stuck = SimpleNamespace(step=route.settings.progress_timeout_steps + 1, pose=still)
    assert not route.reusable(stuck, None, planner, (3, 0), "frontier")
    assert route.reason == "no_progress"
    later = SimpleNamespace(step=stuck.step + 1, pose=still)
    assert not route.reusable(later, None, planner, (5, 0), "frontier")
    assert route.reason == "unplanned", "the next goal is answered afresh"
    route.adopt([Pose2D(0, 0), Pose2D(5, 0)], (5, 0), "frontier", later)
    assert route.replaced == "no_progress"


def test_navigate_surfaces_the_replacement_reason_only_on_adoption(monkeypatch):
    policy, episode = setup_policy()
    monkeypatch.setattr(policy, "_plan_to", lambda obs, world, goal: SimpleNamespace(points=[Pose2D(0, 0), Pose2D(*goal)]))
    world = SimpleNamespace(resolution=0.1)
    monkeypatch.setattr(policy.planner, "path_collides", lambda *a, **k: False)
    policy.route_memory.clear("goal_changed")
    first = policy._navigate(observation(episode, 1), world, (2.0, 0.0), "frontier")
    assert first.info["route"] == "new_goal_or_invalid_route" and first.info["route_replaced"] == "goal_changed"
    second = policy._navigate(observation(episode, 2), world, (2.0, 0.0), "frontier")
    assert second.info["route"] == "committed_safe_route" and "route_replaced" not in second.info


@pytest.mark.parametrize("kwargs", [{"plan_attempts": 0}, {"room_scan_turns": -2}, {"informative_radius_m": 0},
                                    {"visited_memory": 0}, {"ranking": {"gain_exponent": -1}}])
def test_sweep_settings_reject_nonsense(kwargs):
    with pytest.raises(ValueError):
        SweepSettings(**kwargs)


def test_sweep_settings_accept_ranking_overrides_as_a_mapping():
    assert SweepSettings(ranking={"heading_weight": 0.5}).ranking.heading_weight == 0.5


# -- the inspection gate ----------------------------------------------------
def tilt_policy():
    episode = ObjNavEpisode("synthetic/0", "synthetic", "gibson", "development", "chair",
                            MULTIFLOOR_PROTOCOL.camera(), MULTIFLOOR_PROTOCOL.actions(), 500)
    policy = RPTSearchPolicy(FakeDetector(), FakeLLM(), RPTSettings(map_size_m=20.0))
    policy.reset(episode, gibson_label_mapper().target_labels("chair"))
    return policy, episode


def at_step(episode, step, x=0.0):
    return replace(observation(episode, step), pose=AgentPose(x, 0, 0, 0))


def test_inspection_waits_for_a_pause_names_its_trigger_and_keeps_a_long_cadence():
    policy, episode = tilt_policy()
    building, camera = policy.building, policy.camera_control
    building.entered_step = -building.params.floor_search_actions      # allowance spent from the start
    policy.route_memory.path = SimpleNamespace(points=())               # a route is committed
    assert building._inspect(at_step(episode, 10), exhausted=False) is None
    assert camera.inspection is None and not building.events

    policy.route_memory.path = None                                     # the natural pause
    command = building._inspect(at_step(episode, 10), exhausted=False)
    assert command is not None and command.info["kind"] == "stair_inspection"
    assert building.events[-1] == {"action": 10, "event": "inspection_started", "reason": "periodic", "floor_id": 0}
    camera.inspection = None                                            # sweep finished

    # A stair detection while a route is committed is remembered, not acted on.
    policy.route_memory.path = SimpleNamespace(points=())
    building.semantic_stair_hint = True
    assert building._inspect(at_step(episode, 40, x=3.0), exhausted=False) is None
    assert building.stair_hint_pending
    building.semantic_stair_hint = False
    policy.route_memory.path = None
    command = building._inspect(at_step(episode, 41, x=3.0), exhausted=False)
    assert command is not None and building.events[-1]["reason"] == "stair_detection"
    assert not building.stair_hint_pending
    camera.inspection = None

    # No unprompted sweep inside the long cadence, even when idle in a new place.
    assert building._inspect(at_step(episode, 70, x=6.0), exhausted=False) is None
    assert len(building.events) == 2
    command = building._inspect(at_step(episode, 41 + camera.settings.periodic_inspection_actions, x=9.0),
                                exhausted=False)
    assert command is not None and building.events[-1]["reason"] == "periodic"
    camera.inspection = None

    # An exhausted floor may inspect even mid-route: there is nothing else to do.
    policy.route_memory.path = SimpleNamespace(points=())
    command = building._inspect(at_step(episode, 200, x=12.0), exhausted=True)
    assert command is not None and building.events[-1]["reason"] == "floor_exhausted"


def test_camera_settings_refuse_a_periodic_cadence_shorter_than_the_cooldown():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.camera_control import CameraControlSettings
    with pytest.raises(ValueError):
        CameraControlSettings(inspection_cooldown_actions=24, periodic_inspection_actions=10)
    assert CameraControlSettings().periodic_inspection_actions >= CameraControlSettings().inspection_cooldown_actions




