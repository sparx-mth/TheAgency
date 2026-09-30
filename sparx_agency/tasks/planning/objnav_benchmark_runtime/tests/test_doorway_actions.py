"""Doorway inspection through the real discrete action converter, without a GPU."""
from dataclasses import replace
import math
import numpy as np
import pytest

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_discovery import peek_rig, refresh, complete_peek
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import IN_A, obs_at


def moved(obs, action, spec):
    pose = obs.pose
    x, y, yaw = pose.x, pose.y, pose.yaw
    if action == DiscreteAction.MOVE_FORWARD:
        x += spec.forward_step_m * math.cos(yaw)
        y += spec.forward_step_m * math.sin(yaw)
    elif action == DiscreteAction.TURN_LEFT:
        yaw = normalize_angle(yaw + spec.turn_angle_rad)
    elif action == DiscreteAction.TURN_RIGHT:
        yaw = normalize_angle(yaw - spec.turn_angle_rad)
    return replace(pose, x=x, y=y, yaw=yaw)


@pytest.mark.parametrize("yaw", [0.0, 3.0, -3.0])
def test_real_converter_enters_threshold_scans_then_returns_with_budget_intact(yaw):
    p, ep, world, rooms, reasoned = peek_rig()
    p.loop.local_steps = 4
    converter = DiscreteActionConverter(ep.action_spec, p.converter_params)
    pose = replace(obs_at(ep, 1, (5.0, 3.0), yaw).pose)
    owners = []
    for step in range(1, 100):
        obs, cost = refresh(p, ep, world, step, (pose.x, pose.y), pose.yaw)
        p._action_owner = "search"
        command = p._discover(obs, world, cost)
        if command is None:
            break
        owners.append(p._action_owner)
        result = converter.step(obs.pose, command)
        action = DiscreteAction.TURN_LEFT if result.idle else result.action
        p.notify_action(obs, action)
        pose = moved(obs, action, ep.action_spec)
        gx, gy = world.world_to_grid(pose.x, pose.y)
        assert world.in_bounds(gx, gy) and world.grid[gy, gx] == 0
        assert p.loop.local_steps == 4
    else:
        pytest.fail("peek or return did not terminate")
    completed = [event for event in p.peek.events if event["event"] == "peek_completed"]
    assert len(completed) == 1 and completed[0]["swept_degrees"] >= 180 - 1e-6
    assert "doorway_peek" in owners and "room_return" in owners
    assert p.graph.room_at(world, (pose.x, pose.y)) == 0
    assert p.supervisor.room_id == 0 and len(reasoned) == 2
    assert not [event for event in p.peek.events if event["event"] == "peek_cancelled"]


def test_yaw_wrap_is_progress_but_oscillation_is_not_a_semicircle():
    p, ep, world, _, _ = peek_rig()
    obs, cost = refresh(p, ep, world, 1, (5.0, 3.0))
    p._discover(obs, world, cost)
    xy = p.peek.active["xy"]
    p.peek.active.update(phase="scan", last_yaw=3.0, swept=0.0)
    for step, yaw in enumerate([-3.0, 3.0] * 8, 2):
        obs, _ = refresh(p, ep, world, step, xy, yaw)
        assert p.peek.plan(obs, world) is not None
        p.peek.charge()
        assert p.peek.active["swept"] < 0.3
    assert not p.peek.records[-1]["done"]


def test_target_and_floor_preemptions_do_not_resume_an_old_floor_route():
    p, ep, world, _, _ = peek_rig()
    obs, cost = refresh(p, ep, world, 1, (5.0, 3.0))
    original = p.route_memory
    p._discover(obs, world, cost)
    p.peek.cancel(obs_at(ep, 2, (5.0, 3.0)), "target_priority")
    assert p.route_memory is original and p.peek.active is None
    assert p.peek.records[-1]["attempts"] == 0
    p.peek.records[-1]["retry"] = 0
    obs, cost = refresh(p, ep, world, 3, (5.0, 3.0))
    p._discover(obs, world, cost)
    assert p.peek.active is not None
    p.mapping.floor_id = 1
    p.peek.cancel(obs_at(ep, 4, (5.0, 3.0)), "floor_changed", restore=False)
    assert p.peek.active is None and p.route_memory is not original
    assert p.peek.events[-1]["floor"] == 0


def test_room_identity_change_does_not_repeat_a_completed_peek():
    p, ep, world, rooms, _ = peek_rig()
    complete_peek(p, ep, world)
    p.graph.registry.rooms[7] = replace(p.graph.registry.rooms.pop(1), id=7)
    p.graph.labels[p.graph.registry.rooms[7].mask] = 8
    obs, _ = refresh(p, ep, world, 20, (5.0, 3.0))
    assert p.peek.plan(obs, world) is None


def test_a_room_entered_without_a_scan_still_needs_inspection():
    p, ep, world, _, _ = peek_rig()
    obs, _ = refresh(p, ep, world, 1, (9.0, 3.0))
    p.peek._remember_start(obs, world)
    assert not p.peek.records[-1]["done"] and not p.peek.floor_ready()
    obs, _ = refresh(p, ep, world, 2, (5.0, 3.0))
    assert p.peek.plan(obs, world) is not None
    assert p.peek.active["room"] == 1


def test_peek_actions_during_startup_count_once_towards_warmup():
    p, ep, world, _, _ = peek_rig()
    p.warmup_actions = 9
    obs, cost = refresh(p, ep, world, 1, (5.0, 3.0))
    command = p._discover(obs, world, cost)
    assert command.info["kind"] == "doorway_peek_approach"
    p.notify_action(obs, DiscreteAction.TURN_LEFT)
    p.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert p.warmup_actions == 10 and p.peek.active["actions"] == 1


def test_frontier_origin_snap_cannot_teleport_through_a_wall():
    from sparx_agency.core.planning.exploration.frontier_ranking import accessible_frontiers
    from sparx_agency.core.planning.environment import OccupancyGrid2DParams, OccupancyValues
    grid = np.full((20, 20), -1, dtype=np.int8)
    grid[2:18, 2:8] = 0
    grid[:, 8:10] = 100
    grid[2:18, 10:18] = 0
    world = OccupancyGrid2D(grid, OccupancyGrid2DParams(0.1, 0.0, 0.0), values=OccupancyValues(free=0, occupied=100, unknown=-1))
    cost = np.where(grid == 0, 1.0, np.inf)
    cost[:, :8] = np.inf  # no body-clearance-safe cell on the robot's side
    inventory = accessible_frontiers(world, cost, np.where(grid == 0, 1, 0), world.grid_to_world(7, 8), 0)
    assert not inventory.goals and not inventory.cells.any()


def test_llm_failure_after_peek_uses_bounded_backoff_without_repeating_the_scan():
    p, ep, world, _, _ = peek_rig()
    p.graph.reason = lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("fake outage"))
    obs, command = complete_peek(p, ep, world)
    assert command is not None and p.peek.active is None and p.peek.pending_reason
    calls = p.fallback.stats["room_llm_failures"]
    p._discover(replace(obs, step=obs.step + 1), world, p.navigation_cost(world))
    assert p.fallback.stats["room_llm_failures"] == calls == 1
    assert len([e for e in p.peek.events if e["event"] == "peek_completed"]) == 1


def test_peek_evidence_flows_through_classifier_oracle_and_real_rpt_star():
    from types import MethodType
    from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_labels import RevisableRoomLabels
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.oracle_retry import RepairingNodeOracle
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.scene_graph import ObservedSceneGraph

    class BathroomLLM:
        def __init__(self):
            self.calls = []

        def chat_json(self, system, user, **kwargs):
            self.calls.append((user, kwargs))
            if "Room observed objects" in user:
                return {"label": "bathroom" if "sink" in user else "living_room", "confidence": 0.9}
            assert "type=bathroom?" in user and "seen: sink" in user
            assert kwargs.get("reasoning") is True
            return {"nodes": [{"id": 0, "p": 2, "why": "living room"},
                              {"id": 1, "p": 95, "why": "bathroom clue"}]}

    p, ep, world, _, _ = peek_rig()
    client = BathroomLLM()
    p.target = gibson_label_mapper().target_labels("toilet")
    p.graph.label_tracker = RevisableRoomLabels(client)
    p.graph.oracle = RepairingNodeOracle(client)
    p.graph._objects = {0: ["sofa"], 1: ["sink"]}
    p.graph.reason = MethodType(ObservedSceneGraph.reason, p.graph)
    p.supervisor.inner._solver = p.solver
    complete_peek(p, ep, world)
    assert p.graph.label_info(1)["label"] == "bathroom"
    assert len(client.calls) == 3  # two room labels, then one synchronous oracle
    assert p.solver.last.source == "rpt_star" and p.solver.last.guarantee == "optimal"
    assert p.solver.last.order[0] == 1
    assert p.graph.probs[1] == pytest.approx(0.95)


def test_falcon_external_actions_keep_global_ledger_but_not_local_budget():
    from sparx_agency.core.planning.exploration.falcon.bursts import BurstMachine
    from sparx_agency.core.planning.exploration.falcon.params import FalconParams
    machine = BurstMachine(FalconParams(burst_actions=10), 500)
    machine.begin("original_room")
    machine.charge(0)
    for step in range(1, 8):
        machine.charge(step, external_phase="doorway_peek")
    assert machine.actions == 8 and machine.burst.actions == machine.phase_actions == 1
    machine.charge(8)
    assert machine.burst.actions == 2


def test_falcon_active_room_is_not_removed_by_its_own_cooldown():
    from sparx_agency.core.planning.exploration.falcon.params import FalconParams
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.falcon_regions import ObservedRegions
    p, ep, world, rooms, _ = peek_rig()
    regions = ObservedRegions(FalconParams())
    obs = obs_at(ep, 1, IN_A)
    regions.start(world, obs.pose, rooms, 0)
    normal, _ = regions.room_goals(world, p.navigation_cost(world), rooms, obs.pose, 1)
    retained, _ = regions.room_goals(world, p.navigation_cost(world), rooms, obs.pose, 1, retain_room_id=0)
    assert 0 not in normal and 0 in retained


def test_accessible_frontier_count_is_not_limited_to_thirty_metres():
    from sparx_agency.core.planning.exploration.frontier_ranking import accessible_frontiers
    from sparx_agency.core.planning.environment import OccupancyGrid2DParams, OccupancyValues
    grid = np.full((12, 450), 100, dtype=np.int8)
    grid[2:10, 1:420] = 0
    grid[2:10, 420:449] = -1
    world = OccupancyGrid2D(grid, OccupancyGrid2DParams(0.1, 0.0, 0.0),
                            values=OccupancyValues(free=0, occupied=100, unknown=-1))
    cost = np.where(grid == 0, 1.0, np.inf)
    inventory = accessible_frontiers(world, cost, np.where(grid == 0, 1, 0), world.grid_to_world(3, 5), 0)
    assert len(inventory.by_room[1]) == 1
    assert inventory.goals[0].geodesic_m > 40.0
