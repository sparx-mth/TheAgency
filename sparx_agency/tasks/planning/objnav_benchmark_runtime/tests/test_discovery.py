"""CPU-only discovery integration: real geometry/planning, scripted perception."""
from dataclasses import replace
import math
import pytest

from sparx_agency.core.planning.exploration.object_search_supervisor import SEARCH, TRANSIT
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_candidates import PeekSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doorway_peek import DoorwayPeek
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_search_loop import LoopSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import IN_A, loop_policy, obs_at, see
from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import search_snapshot


@pytest.mark.parametrize("explorer", ["frontier", "falcon"])
def test_first_ten_emitted_actions_are_discovery_not_llm_decisions(explorer):
    p, episode = setup_policy("bed", discovery=True, local_exploration=explorer, doorway_peek={"enabled": False})
    for step in range(10):
        obs = observation(episode, step, depth=3)
        command = p.plan(obs)
        assert command.info["kind"] == "warmup"
        assert p.graph.queries == 0 and p.solver.calls == 0
        assert p.plan(obs) is command  # repeated reads neither infer nor spend actions
        p.notify_action(obs, DiscreteAction.TURN_LEFT)
        p.notify_action(obs, DiscreteAction.TURN_LEFT)
        assert p.warmup_actions == step + 1 and p.loop.local_steps == 0
    p.plan(observation(episode, 10, depth=3))
    assert p.warmup_actions == 10
    if explorer == "frontier":
        assert p.graph.queries == 1 and p.solver.calls == 1
    else:
        assert p.hierarchy.machine.actions == 10
        assert p.hierarchy.machine.burst.actions == 0


def test_confirmed_target_preempts_warmup_without_waiting_ten_steps():
    from sparx_agency.core.planning.objnav.types.pose import AgentPose
    p, episode = setup_policy(discovery=True)
    p.plan(observation(episode, 0))
    p.plan(observation(episode, 1))
    distinct = replace(observation(episode, 2), pose=AgentPose(0, 0.25, 0, 0))
    assert p.plan(distinct).stop
    assert p.warmup_actions == 0


def peek_rig(order=(0, 1)):
    p, ep, world, rooms, reasoned = loop_policy(order=order)
    p.settings = replace(p.settings, doorway_peek=PeekSettings())
    p.peek = DoorwayPeek(p)
    p.loop.settings = LoopSettings(entry_frontier=False)
    p.loop.plan(obs_at(ep, 0, IN_A), world)
    cost = p.navigation_cost(world)
    p.graph.refresh_accessibility(world, cost, IN_A)
    p.peek._remember_start(obs_at(ep, 0, IN_A), world)
    return p, ep, world, rooms, reasoned


def refresh(p, ep, world, step, xy, yaw=0.0):
    obs = obs_at(ep, step, xy, yaw)
    p.planner.invalidate_cache()  # this fixture edits one grid; production mapping returns a new snapshot
    cost = p.navigation_cost(world)
    p.graph.refresh_accessibility(world, cost, xy, yaw)
    return obs, cost


def complete_peek(p, ep, world, start=1):
    obs, cost = refresh(p, ep, world, start, (5.0, 3.0))
    command = p._discover(obs, world, cost)
    assert command.info["kind"] == "doorway_peek_approach"
    assert p.peek.plan(obs, world) is command
    p.notify_action(obs, DiscreteAction.MOVE_FORWARD)
    active = p.peek.active
    xy, heading = active["xy"], active["heading"]
    step = start + 1
    # Each observation is a new frame at the measured pose. A full inward
    # semicircle, not repeated planning at one yaw, must precede completion.
    n = int(math.ceil(math.pi / ep.action_spec.turn_angle_rad))
    for i in range(n + 1):
        obs, cost = refresh(p, ep, world, step, xy, heading - math.pi / 2 + i * ep.action_spec.turn_angle_rad)
        command = p._discover(obs, world, cost)
        if i < n:
            assert command.info["kind"] == "doorway_peek_scan"
            p.notify_action(obs, DiscreteAction.TURN_LEFT)
        step += 1
    return obs, command


def test_peek_scans_once_and_resumes_the_same_room_without_refilling_its_budget():
    p, ep, world, rooms, reasoned = peek_rig()
    assert p.supervisor.state == SEARCH and p.supervisor.room_id == 0
    p.loop.local_steps = 4
    saved = p.route_memory
    deadline = p.supervisor.inner._search_end_s
    obs, _ = complete_peek(p, ep, world)
    assert p.peek.active is None and not p.peek.pending_reason
    assert p.route_memory is saved
    assert p.loop.local_steps == 4 and p.supervisor.room_id == 0
    assert p.supervisor.inner._search_end_s > deadline
    assert len(reasoned) == 2
    assert len([e for e in p.peek.events if e["event"] == "peek_completed"]) == 1
    assert p.peek.records[-1]["done"]
    # Walking past the same room again does not buy another scan.
    again, _ = refresh(p, ep, world, obs.step + 1, (5.0, 3.0))
    assert p.peek.plan(again, world) is None


def test_peek_can_switch_to_the_new_room_and_start_a_normal_local_turn():
    p, ep, world, rooms, reasoned = peek_rig()
    p.loop.local_steps = 4
    p.supervisor.inner._solver = lambda candidates, instance: [1, 0]
    obs, _ = complete_peek(p, ep, world)
    assert p.supervisor.state == "select" and p.loop.room_id is None
    p.loop.plan(obs, world)
    assert p.supervisor.room_id == 1 and p.supervisor.state == SEARCH
    assert p.loop.local_steps == 0


def test_target_approach_cancels_peek_before_any_more_scan_action(monkeypatch):
    p, ep, world, _, _ = peek_rig()
    obs, cost = refresh(p, ep, world, 1, (5.0, 3.0))
    p._discover(obs, world, cost)
    assert p.peek.active is not None
    p._target_xy, p._target_step, p._target_id = (7.0, 3.0), 2, 42
    approach = NavigationCommand.follow([(5.0, 3.0), (6.0, 3.0)], info={"kind": "target"})
    monkeypatch.setattr(p, "_approach", lambda *args: approach)
    assert p._search(obs_at(ep, 2, (5.0, 3.0)), world, False) is approach
    assert p.peek.active is None
    assert p.peek.events[-1]["reason"] == "target_priority"


def test_peek_disabled_and_unreachable_doorways_do_not_interrupt():
    p, ep, world, _, _ = peek_rig()
    p.peek.settings = PeekSettings(enabled=False)
    obs, _ = refresh(p, ep, world, 1, (5.0, 3.0))
    assert p.peek.plan(obs, world) is None
    p.peek.settings = PeekSettings()
    world.grid[25:36, 55:66] = 100  # close the only doorway
    obs, _ = refresh(p, ep, world, 2, (5.0, 3.0))
    assert p.peek.plan(obs, world) is None
    assert p.peek.events == []


def test_inaccessible_frontiers_are_absent_from_counts_and_display():
    p, ep, world, _, _ = peek_rig()
    world.grid[25:36, 55:66] = 100
    obs, cost = refresh(p, ep, world, 1, IN_A)
    inventory = p.graph.frontier_inventory
    assert p.graph.facts[0].frontier_clusters > 0
    assert p.graph.facts[1].frontier_clusters == 0
    assert 2 not in inventory.by_room
    assert all(goal.xy[0] < 5.5 for goal in inventory.goals)
    snapshot = search_snapshot(p)
    assert snapshot["accessible_frontiers"] == [list(g.xy) for g in inventory.goals]
    world.grid[25:36, 55:66] = 0
    refresh(p, ep, world, 2, IN_A)
    assert p.graph.facts[1].frontier_clusters > 0  # inaccessible is not permanently blacklisted


def test_non_destination_semantic_clue_triggers_reasoning_even_without_peeks(monkeypatch):
    p, ep, world, _, reasoned = loop_policy(order=(1, 0))
    obs = obs_at(ep, 0, IN_A)
    p.loop.plan(obs, world)
    p._discover(obs, world, p.navigation_cost(world))
    see(p, {0: ["sink"], 1: []}, 1)
    p._discover(obs_at(ep, 1, IN_A), world, p.navigation_cost(world))
    assert reasoned == [0, 1]
    assert p.supervisor.room_id == 1 and p.supervisor.state == TRANSIT


def test_peek_without_yaw_progress_is_bounded_and_does_not_claim_a_scan():
    p, ep, world, _, _ = peek_rig()
    obs, cost = refresh(p, ep, world, 1, (5.0, 3.0))
    p._discover(obs, world, cost)
    xy, heading = p.peek.active["xy"], p.peek.active["heading"]
    for step in range(2, 30):
        obs, cost = refresh(p, ep, world, step, xy, heading - math.pi / 2)
        command = p._discover(obs, world, cost)
        if p.peek.active is None:
            break
        p.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert p.peek.active is None
    assert p.peek.records[-1]["reason"] == "scan_budget"
    assert not p.peek.records[-1]["done"]


@pytest.mark.parametrize("kwargs", [{"warmup_steps": 0}, {"warmup_steps": True},
                                     {"doorway_peek": {"enabled": "false"}},
                                     {"doorway_peek": {"approach_actions": 0}}])
def test_discovery_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        RPTSettings(**kwargs)
