"""Real policy/converter orchestration with controlled observed geometry."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.planning.exploration.falcon.bursts import DONE, EXPLORE, RECOVER, VERIFY
from sparx_agency.core.planning.exploration.falcon.params import FalconParams
from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.falcon_regions import ObservedRegions
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_falcon import open_fixture
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy


def rig(monkeypatch, **settings):
    old, episode = setup_policy("bed")
    policy = RPTSearchPolicy(old.detector, old.llm_client,
                             replace(old.settings, local_exploration="falcon", falcon=FalconParams(**settings)))
    agent = HeadlessObjNavAgent(policy, gibson_label_mapper())
    agent.reset(episode)
    world, cost, scope, request = open_fixture()
    monkeypatch.setattr(policy.mapping, "update", lambda obs, **kwargs: world)
    monkeypatch.setattr(policy, "_perceive", lambda obs: False)
    make = lambda step: replace(observation(episode, step, depth=3), pose=request.pose)
    return policy, agent, world, make


def test_startup_is_falcon_without_opening_panorama_or_llm(monkeypatch):
    p, agent, _, make = rig(monkeypatch)
    first = agent.act(make(0))
    assert not first.info["policy"].get("stop", False)
    assert first.action != DiscreteAction.STOP
    assert p.graph.queries == 0
    assert p.hierarchy.explorer.records[0]["status"] == "planned"
    assert p.hierarchy.machine.burst.actions == 1


def test_small_burst_expires_before_more_local_actions(monkeypatch):
    p, agent, _, make = rig(monkeypatch, burst_actions=3)
    for step in range(4):
        agent.act(make(step))
    m = p.hierarchy.machine
    assert m.history[0].actions == 3
    assert m.history[0].status == "budget_exhausted"
    assert p.graph.queries == 1
    assert m.actions == sum(p.telemetry.actions.values()) == 4


def test_target_interrupt_rejection_resumes_same_committed_route(monkeypatch):
    p, agent, _, make = rig(monkeypatch, target_actions=2)
    agent.act(make(0))
    saved = p.route_memory
    p._target_id, p._target_xy, p._target_step = 42, (4.0, 4.0), 1
    monkeypatch.setattr(p, "_approach", lambda obs, world: NavigationCommand.hold(final_yaw=1.0))
    agent.act(make(1))
    assert p.hierarchy.machine.phase == VERIFY
    agent.act(make(2))
    agent.act(make(3))
    assert p._target_xy is None
    assert p.route_memory is saved
    assert p.hierarchy.machine.phase == EXPLORE
    assert len(p.hierarchy.machine.history) == 1
    assert p.hierarchy.machine.burst.actions == 4
    assert p.hierarchy.machine.actions == 4


def test_low_gain_is_partial_not_room_fully_searched(monkeypatch):
    p, agent, _, make = rig(monkeypatch, low_gain_actions=2)
    agent.act(make(0))
    agent.act(make(1))
    burst = p.hierarchy.machine.history[0]
    assert burst.reason == "negligible_gain" and burst.status == "partial_ended"


def test_discrete_forward_sweep_cannot_enter_unknown_or_cut_scope(monkeypatch):
    p, agent, world, make = rig(monkeypatch)
    obs = make(0)
    agent.act(obs)
    path = Path2D((Pose2D(3.05, 3.05), Pose2D(4.05, 3.05)))
    p.route_memory.adopt(path, (4.05, 3.05), "falcon", obs)
    p.hierarchy.cost[30, 32] = np.inf
    action = p.hierarchy.filter_action(obs, DiscreteAction.MOVE_FORWARD)
    assert action == DiscreteAction.TURN_LEFT
    assert p.hierarchy.machine.phase == RECOVER


def test_room_entry_goal_is_reachable_and_not_centroid_or_boundary():
    world, cost, _, obs = open_fixture()
    mask = np.zeros(cost.shape, bool)
    mask[20:40, 32:44] = True
    room = SimpleNamespace(mask=mask, centroid=(5.9, 5.9))  # obstructed/unobserved centroid
    regions = ObservedRegions(FalconParams())
    goals, _ = regions.room_goals(world, cost, {7: room}, obs.pose, 0)
    x, y = world.world_to_grid(*goals[7])
    assert mask[y, x] and np.isfinite(cost[y, x])
    assert x >= 35 and x <= 40 and 23 <= y <= 36


def test_unavailable_llm_ends_with_visible_error_not_fallback(monkeypatch):
    p, agent, _, make = rig(monkeypatch, burst_actions=1)
    agent.act(make(0))
    def unavailable(*args, **kwargs):
        raise TimeoutError("test LLM request deadline")
    monkeypatch.setattr(p.graph, "reason", unavailable)
    decision = agent.act(make(1))
    assert decision.action == DiscreteAction.STOP
    assert p.hierarchy.machine.phase == DONE
    assert "deadline" in p.hierarchy.errors[-1]["error"]
    assert p.hierarchy.diagnostics()["fallback"] is None


def test_observed_floor_reset_retains_global_ledger(monkeypatch):
    p, agent, _, make = rig(monkeypatch)
    agent.act(make(0))
    p.mapping.floor_revision += 1
    agent.act(make(1))
    assert p.hierarchy.machine.actions == 2
    assert p.hierarchy.machine.history[0].reason == "floor_change"


def test_larger_split_child_cannot_steal_the_burst_anchor():
    world, _, _, obs = open_fixture()
    regions = ObservedRegions(FalconParams())
    whole = world.grid == 0
    regions.start(world, obs.pose, {0: SimpleNamespace(mask=whole)}, 0)
    small, large = whole.copy(), whole.copy()
    small[:29, :] = False
    small[33:, :] = False
    large[29:33, :] = False
    scope = regions.scope(world, {0: SimpleNamespace(mask=small), 1: SimpleNamespace(mask=large)})
    assert regions.room_id == 0 and scope[30, 30]
    assert len(regions.visits) == 1


def test_route_bank_preserves_cursor_without_budget_refill(monkeypatch):
    p, agent, _, make = rig(monkeypatch)
    agent.act(make(0))
    h = p.hierarchy
    original = p.route_memory.path
    cursor = p.route_memory.progress
    h.routes.suspend(h.regions.mask, h.current, p, 1)
    p.route_memory.clear("transit")
    restored = h.routes.restore(h.regions.mask, p, 20)
    assert restored is not None and p.route_memory.path == original
    assert p.route_memory.progress is not cursor  # independent snapshot
    assert p.route_memory.last_progress_step >= 19
    assert h.machine.actions == 1 and h.machine.burst.actions == 1


def test_final_guard_and_accounting_survive_recording_probe(monkeypatch):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import PolicyProbe
    for recorded in (False, True):
        p, _, episode_make_world, make = rig(monkeypatch)
        agent = HeadlessObjNavAgent(PolicyProbe(p) if recorded else p, gibson_label_mapper())
        episode = p.episode
        agent.reset(episode)
        monkeypatch.setattr(p.mapping, "update", lambda obs: episode_make_world)
        monkeypatch.setattr(p, "_perceive", lambda obs: False)
        monkeypatch.setattr(p, "plan", lambda obs: NavigationCommand.follow(((3.05, 3.05), (4.05, 3.05))))
        monkeypatch.setattr(p, "filter_action", lambda obs, action: DiscreteAction.TURN_LEFT if action == DiscreteAction.MOVE_FORWARD else action)
        decision = agent.act(make(0))
        assert decision.action == DiscreteAction.TURN_LEFT
        assert not agent._converter.forward_blocked(make(1).pose)
        assert p.hierarchy.machine.actions == 1
        assert p.telemetry.actions["TURN_LEFT"] == 1


def test_capacity_and_extra_action_rejection_do_not_mutate_ledgers():
    import pytest
    from sparx_agency.core.planning.exploration.falcon.bursts import BurstMachine
    from sparx_agency.core.planning.exploration.falcon.ordering import PlanningCapacity
    from sparx_agency.core.planning.exploration.falcon.travel import GridRoutes
    with pytest.raises(PlanningCapacity):
        GridRoutes(np.ones((3, 3)), 0.1, max_nodes=2)
    machine = BurstMachine(FalconParams(burst_actions=1), 500)
    machine.begin("room")
    machine.charge(0)
    before = machine.snapshot()
    with pytest.raises(RuntimeError, match="Local burst"):
        machine.charge(1)
    assert machine.snapshot() == before

