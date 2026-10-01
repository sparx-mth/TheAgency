"""Synthetic control/geometry regressions, not Habitat navigation scores."""
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.camera_geometry import project_to_image
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_closing import TargetClosingSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation


def forbidden(*args, **kwargs):
    raise AssertionError("Global exploration was called after target takeover")


def freeze_global(monkeypatch, policy):
    for owner, names in ((policy, ("_decide", "_refresh_graph")),
                         (policy.loop, ("plan", "charge")), (policy.fallback, ("plan",)),
                         (policy.graph, ("reason",)), (policy.llm_client, ("chat_json",)),
                         (policy.building, ("plan", "prepare_observation", "observe", "filter_action"))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)


@pytest.mark.parametrize("explorer", ["frontier", "falcon"])
def test_two_frames_take_over_both_explorers_without_global_calls(monkeypatch, explorer):
    p, ep = setup_policy(duplicate=True, local_exploration=explorer)
    freeze_global(monkeypatch, p)
    first = p.plan(observation(ep, 0))
    assert p.closing.active and not p.closing.locked and not first.stop
    assert p.plan(observation(ep, 0)) is first
    assert p.closing.count == 1
    command = p.plan(observation(ep, 1))
    assert command.stop and p.closing.locked
    assert p.filter_action(observation(ep, 1), DiscreteAction.STOP) == DiscreteAction.STOP
    p.notify_action(observation(ep, 1), DiscreteAction.STOP)
    assert p.graph.queries == p.solver.calls == 0
    assert p.episode_info()["target_closing"]["locked"]
    p.reset(ep, p.target)
    assert not p.closing.active


def test_confidence_threshold_does_not_enter_legacy_pursuit():
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("chair", .49, (280, 200, 360, 280))]
    assert not p.plan(observation(ep, 0, depth=3)).stop
    assert not p.closing.active and p._target_xy is None


@pytest.mark.parametrize("depth", [float("nan"), float("inf"), 0.0])
def test_invalid_depth_never_confirms_or_moves_forward(monkeypatch, depth):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    for step in range(3):
        command = p.plan(observation(ep, step, depth=depth))
        assert not command.stop and not command.waypoints and not p.closing.locked


def test_nonconsecutive_and_different_objects_do_not_confirm(monkeypatch):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=4))
    assert not p.closing.locked
    p.plan(observation(ep, 2, depth=2))
    assert p.closing.count == 1 and not p.closing.locked
    p.plan(observation(ep, 3, depth=2))
    assert p.closing.locked


def test_occlusion_retains_goal_and_same_path_beyond_old_loss_timeout(monkeypatch):
    p, ep = setup_policy(target_closing={"max_reacquire_steps": 2})
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=2))
    initial = p.plan(observation(ep, 1, depth=2))
    goal, xyz = p.closing.path.goal_xyz, p.closing.xyz
    plans = p._plan_calls
    p.detector.detect = lambda rgb: []
    for step in range(2, 12):
        command = p.plan(observation(ep, step, depth=2))
        assert not command.stop and command.waypoints == initial.waypoints and command.waypoints
        assert p.closing.xyz == xyz and p.closing.path.goal_xyz == goal
        assert p.closing.locked and p.closing.phase == "CLOSE_OCCLUDED"
    assert p._plan_calls == plans


def test_terminal_loss_does_not_move_or_stop_on_stale_evidence(monkeypatch):
    p, ep = setup_policy(target_closing={"max_reacquire_steps": 2})
    freeze_global(monkeypatch, p)
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=2))
    p.detector.detect = lambda rgb: []
    near = replace(observation(ep, 2), pose=AgentPose(1.1, 0, 0, 0))
    for step in (2, 3):
        command = p.plan(replace(near, step=step))
        assert not command.stop and not command.waypoints
    with pytest.raises(RuntimeError, match="terminal visual confirmation"):
        p.plan(replace(near, step=4))
    assert p.closing.locked and p.closing.phase == "FAILED"


def test_reobservation_refines_goal_but_missing_frames_do_not(monkeypatch):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=3))
    p.plan(observation(ep, 1, depth=3))
    goal = p.closing.path.goal_xyz
    p.plan(observation(ep, 2, depth=3.1))  # small update, keep stable endpoint
    assert p.closing.path.goal_xyz == goal
    p.plan(observation(ep, 3, depth=3.4))  # still associated, meaningful refinement
    assert p.closing.path.goal_xyz != goal and p.closing.path.refinements == 1


def test_expected_closing_failure_is_recordable_not_infrastructure():
    from sparx_agency.tasks.planning.objnav_benchmark.agent_contract import decide
    p, ep = setup_policy()
    agent = SimpleNamespace(act=lambda obs: p.closing.fail("terminal visual confirmation unavailable"))
    action, error = decide(agent, observation(ep, 0), ep.action_spec, "record")
    assert action == DiscreteAction.STOP and "terminal visual confirmation" in error
    assert p.closing.phase == "FAILED"  # forced STOP is not target-confirmed success


def test_planner_exception_does_not_reach_exploration_fallback(monkeypatch):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    p.plan(observation(ep, 0, depth=3))
    monkeypatch.setitem(p.__dict__, "_plan_to", lambda *a: (_ for _ in ()).throw(RuntimeError("planner broken")))
    with pytest.raises(RuntimeError, match="planner broken"):
        p.plan(observation(ep, 1, depth=3))
    assert p.closing.active and p.closing.locked


def test_unreachable_navmesh_never_uses_unsnapped_goal(monkeypatch):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    p.target_projector = lambda point: None
    monkeypatch.setitem(p.__dict__, "_plan_to", forbidden)
    p.plan(observation(ep, 0, depth=3))
    command = p.plan(observation(ep, 1, depth=3))
    assert not command.stop and not command.waypoints and p.closing.locked


@pytest.mark.parametrize("column, expected", [(100, DiscreteAction.TURN_LEFT), (540, DiscreteAction.TURN_RIGHT)])
def test_bbox_yaw_sign_and_offcenter_stop_rejection(column, expected):
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (column-30, 200, column+30, 280))]
    p.plan(observation(ep, 0, depth=.6))
    command = p.plan(observation(ep, 1, depth=.6))
    assert not command.stop
    assert DiscreteActionConverter(ep.action_spec).step(AgentPose(0, 0, 0, 0), command).action == expected


@pytest.mark.parametrize("occlusion", [False, True])
def test_actual_converter_closes_low_target_looks_down_and_stops(monkeypatch, occlusion):
    p, original = setup_policy(target_closing={"max_reacquire_steps": 2})
    ep = replace(original, target_category="toilet", action_spec=MULTIFLOOR_PROTOCOL.actions())
    p.reset(ep, gibson_label_mapper().target_labels("toilet"))
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    target = np.array([[2.0, 0.0, .45]])
    converter, pose, actions = DiscreteActionConverter(ep.action_spec), AgentPose(0, 0, 0, 0), []
    projections = []
    p.target_projector = lambda point: projections.append(point) or point
    for step in range(40):
        pixels, depths = project_to_image(target, ep.camera, pose)
        u, v = pixels[0]
        visible = depths[0] > 0 and 22 < u < 618 and 22 < v < 458
        boxes = [DetectionWire("toilet", .9, (u-20, v-20, u+20, v+20))] if visible and not (occlusion and 2 <= step <= 5) else []
        p.detector.detect = lambda rgb: boxes
        obs = replace(observation(ep, step, depth=float(depths[0])), pose=pose, target_category="toilet")
        command = p.plan(obs)
        action = converter.step(pose, command).action
        assert action is not None
        actions.append(action)
        p.notify_action(obs, action)
        if action == DiscreteAction.STOP:
            assert math.dist((pose.x, pose.y), target[0, :2]) <= p.settings.target_closing.terminal_distance_m
            assert visible and p.closing.locked
            break
        pose = apply_action(pose, action, ep.action_spec)
    assert actions[-1] == DiscreteAction.STOP
    assert DiscreteAction.MOVE_FORWARD in actions and DiscreteAction.LOOK_DOWN in actions
    assert projections and p.graph.queries == 0
    if occlusion:
        assert p.closing.occluded_path_steps >= 4


def test_bbox_servo_does_not_interrupt_astar_corner_heading(monkeypatch):
    from sparx_agency.core.planning.objnav.types.command import NavigationCommand
    p, ep = setup_policy()
    p.plan(observation(ep, 0, depth=3))
    p.plan(observation(ep, 1, depth=3))
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (60, 200, 140, 280))]
    # Even if the live box asks left, a safe planned detour may need right.
    route = NavigationCommand.follow([(0, 0), (0, -1), (2, -1), (2, 0)])
    monkeypatch.setitem(p.closing.path.__dict__, "command", lambda *args: route)
    obs = replace(observation(ep, 2, depth=2.6), pose=AgentPose(0, 0, 0, -math.pi / 6))
    command = p.plan(obs)
    assert p.closing.last_seen == 2
    assert command.waypoints == route.waypoints and command.final_yaw is None
    assert DiscreteActionConverter(ep.action_spec).step(obs.pose, command).action == DiscreteAction.TURN_RIGHT


@pytest.mark.parametrize("snap", [(float("nan"), 0, 0), (-2, 4, -1), (-20, 3, -1)])
def test_navmesh_rejects_invalid_remote_or_other_floor_snaps(snap):
    sim = HabitatRGBDSimulator(None, None, .18, True, height_m=0.88)
    sim._sim = SimpleNamespace(pathfinder=SimpleNamespace(snap_point=lambda point: snap))
    assert sim.project_target((1, 2, 3)) is None


def test_navmesh_round_trip_enu_native_coordinates():
    received = []
    sim = HabitatRGBDSimulator(None, None, .18, True, height_m=0.88)
    sim._sim = SimpleNamespace(pathfinder=SimpleNamespace(snap_point=lambda point: received.append(point) or point))
    assert sim.project_target((1, 2, 3)) == (1, 2, 3)
    np.testing.assert_allclose(received[0], [-2, 3, -1])


@pytest.mark.parametrize("kwargs", [{"confirmation_frames": 1}, {"confirmation_frames": True},
                                    {"confidence": float("nan")}, {"confidence": 0}, {"max_verify_steps": 0}])
def test_invalid_closing_settings(kwargs):
    with pytest.raises(ValueError):
        TargetClosingSettings(**kwargs)
