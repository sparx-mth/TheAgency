"""The FALCON + YOLO + closing baseline: explores without an LLM, closes exactly as the full method, stays on its storey."""
from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.exploration.falcon.planner import FalconPlan
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.camera_geometry import project_to_image
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import FloorPanels
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.progress import lean_episode_info
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.core.planning.object_search.baselines.falcon_yolo import (
    BaselineSettings, FalconYoloBaseline)
from sparx_agency.core.planning.object_search.world_model.peek_stairs import stair_peek_mask
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import BODY_ACTIONS_PROTOCOL, FakeDetector
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    FLIGHT, IN_A, STRUCTURE, two_room_world)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import FRAME_SIZE, method_snapshot, render_dashboard


class NoDetector:
    def detect(self, rgb):
        return []


def baseline(detector=None, label="chair", metadata=None, actions=None, baseline_settings=None, **overrides):
    """The baseline on a 20 m synthetic map; ``overrides`` are RPTSettings fields."""
    closing = dict(overrides.pop("target_closing", None) or {})
    if not any(key.startswith("stop_overshoot") for key in closing):
        closing.update(stop_overshoot_steps=0, stop_overshoot_max_steps=0)
    overrides["target_closing"] = closing
    episode = ObjNavEpisode("synthetic/0", "synthetic", "gibson", "val", label, PROTOCOL.camera(),
                            actions or BODY_ACTIONS_PROTOCOL.actions(), 500, metadata=metadata or {})
    if "stair_connectors" in (metadata or {}) and "multifloor" not in overrides:
        overrides["multifloor"] = {"stair_source": "ground_truth"}      # a scene structure handed in is the GT ablation
    policy = FalconYoloBaseline(detector or NoDetector(), RPTSettings(**dict({"map_size_m": 20.0}, **overrides)),
                                baseline=baseline_settings)
    policy.reset(episode, gibson_label_mapper().target_labels(label))
    return policy, episode


def observation(episode, step, depth=3.0, pose=None, label="chair"):
    k = episode.camera.intrinsics
    return ObjNavObservation(np.zeros((k.height, k.width, 3), np.uint8), np.full((k.height, k.width), depth, np.float32),
                             pose or AgentPose(0, 0, 0, 0), episode.camera, label, step)


def forbidden(*args, **kwargs):
    raise AssertionError("the exploration was consulted while the closing owned the episode")


# -- what the baseline is -------------------------------------------------------------------------
def test_the_baseline_has_no_llm_and_declares_the_full_methods_closing_and_confinement():
    policy, _ = baseline()
    configuration = policy.configuration()
    assert configuration["method"] == "sparx-falcon-yolo-baseline"
    assert configuration["llm"] is None and configuration["room_llm"] == configuration["rpt_star"] == "none"
    assert configuration["local_exploration"] == "falcon" and configuration["falcon_running"]
    assert configuration["allow_stair_traversal"] is False and configuration["ground_truth_stairs"] is False
    assert configuration["stair_source"] == "observed", "the baseline reads no navmesh stairs either (2026-10-21)"
    assert baseline(multifloor={"stair_source": "ground_truth"})[0].configuration()["ground_truth_stairs"] is True
    assert configuration["target_closing"] == configuration["adaptation"]["target_closing"]
    assert policy.llm_client is None and not hasattr(policy, "graph") and not hasattr(policy, "loop")
    assert policy.settings.local_exploration == "falcon", "a frontier setting is overridden: the baseline is FALCON"
    assert FalconYoloBaseline(NoDetector(), RPTSettings(local_exploration="frontier")).settings.local_exploration == "falcon"
    json.dumps(configuration, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"bootstrap_turns": -1}, {"exhausted_turns": 0}, {"relocation_retire_m": 0.0},
                                    {"stop_when_exhausted": "no"}, {"relocation_candidates": 0}])
def test_invalid_baseline_settings_are_refused(kwargs):
    with pytest.raises(ValueError, match="baseline"):
        BaselineSettings(**kwargs)


def test_the_baseline_explores_with_falcon_from_the_first_frame_and_keeps_its_route():
    policy, episode = baseline()
    first = policy.plan(observation(episode, 0))
    assert not first.stop and first.info["kind"] == "falcon" and first.info["route"] == "planned"
    assert first.info["explorer"] == "falcon" and len(first.waypoints) >= 2
    assert policy.plan(observation(episode, 0)) is first, "one decision per action"
    policy.notify_action(observation(episode, 0), DiscreteAction.MOVE_FORWARD)
    second = policy.plan(observation(episode, 1, pose=AgentPose(0.1, 0.0, 0.0, 0.0)))
    assert second.info["kind"] == "falcon" and second.info["route"] == "committed"
    assert policy.explorer.current is not None and policy.route_memory.kind == "falcon"
    stats = policy.episode_info()["exploration"]["stats"]
    assert stats["falcon_plans"] == 1 and stats["falcon:planned"] == 1
    assert policy.filter_action(observation(episode, 1), DiscreteAction.MOVE_FORWARD) in (
        DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_LEFT)


def test_bootstrap_turns_come_first_and_count_against_nothing_but_themselves():
    policy, episode = baseline(baseline_settings=BaselineSettings(bootstrap_turns=2))
    for step in range(2):
        command = policy.plan(observation(episode, step))
        assert command.info["kind"] == "bootstrap_scan" and command.final_yaw is not None and not command.waypoints
        policy.notify_action(observation(episode, step), DiscreteAction.TURN_LEFT)
    assert policy.plan(observation(episode, 2)).info["kind"] == "falcon"
    assert policy.episode_info()["exploration"]["stats"]["bootstrap_turns"] == 2


# -- the closing is the full method's ---------------------------------------------------------------
def test_two_consecutive_frames_take_over_and_stop_without_consulting_the_exploration(monkeypatch):
    policy, episode = baseline(FakeDetector("chair", duplicate=True))
    first = policy.plan(observation(episode, 0, depth=0.7))
    assert policy.closing.active and not policy.closing.locked and not first.stop
    assert first.info["phase"].startswith("target_")
    monkeypatch.setattr(policy.explorer, "plan", forbidden)
    monkeypatch.setattr(policy.explorer, "prepare", forbidden)
    command = policy.plan(observation(episode, 1, depth=0.7))
    assert command.stop and policy.closing.locked
    assert policy.filter_action(observation(episode, 1), DiscreteAction.STOP) == DiscreteAction.STOP
    policy.notify_action(observation(episode, 1), DiscreteAction.STOP)
    info = policy.episode_info()
    assert info["target_closing"]["locked"] and info["perception"]["counts"]["raw_frames"] == 2
    policy.reset(episode, policy.target)
    assert not policy.closing.active and policy.explorer.current is None


def test_a_weak_or_depthless_box_starts_nothing_and_falcon_goes_on():
    policy, episode = baseline()
    policy.detector.detect = lambda rgb: [DetectionWire("chair", .49, (280, 200, 360, 280))]
    assert policy.plan(observation(episode, 0, depth=3.0)).info["kind"] == "falcon"
    assert not policy.closing.active and policy._target_xy is None
    policy.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    assert policy.plan(observation(episode, 1, depth=0.0)).info["kind"] == "falcon"
    assert not policy.closing.active


def test_the_actual_converter_closes_a_toilet_from_two_metres_looks_down_and_stops(monkeypatch):
    policy, episode = baseline(label="toilet", actions=MULTIFLOOR_PROTOCOL.actions(),
                               target_closing={"max_reacquire_steps": 2})
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(policy.mapping.__dict__, "update", lambda *a, **kw: world)
    target = np.array([[2.0, 0.0, .45]])
    converter, pose, actions = DiscreteActionConverter(episode.action_spec), AgentPose(0, 0, 0, 0), []
    projections = []
    policy.target_projector = lambda point: projections.append(point) or point
    for step in range(40):
        pixels, depths = project_to_image(target, episode.camera, pose)
        u, v = pixels[0]
        visible = depths[0] > 0 and 22 < u < 618 and 22 < v < 458
        boxes = [DetectionWire("toilet", .9, (u - 20, v - 20, u + 20, v + 20))] if visible else []
        policy.detector.detect = lambda rgb, boxes=boxes: boxes
        obs = observation(episode, step, depth=float(depths[0]), pose=pose, label="toilet")
        command = policy.plan(obs)
        action = converter.step(pose, command).action
        assert action is not None
        guarded = policy.filter_action(obs, action)
        assert guarded == action, "the closing's route is never vetoed by the exploration's mask"
        actions.append(action)
        policy.notify_action(obs, action)
        if action == DiscreteAction.STOP:
            assert math.dist((pose.x, pose.y), target[0, :2]) <= policy.settings.target_closing.terminal_distance_m
            assert visible and policy.closing.locked
            break
        pose = apply_action(pose, action, episode.action_spec)
    assert actions[-1] == DiscreteAction.STOP
    assert DiscreteAction.MOVE_FORWARD in actions and DiscreteAction.LOOK_DOWN in actions
    assert projections, "the standoff ring is projected onto the navmesh exactly as in the full method"
    assert policy.episode_info()["target_closing"]["locked"]


def test_a_blocked_step_rejects_the_falcon_viewpoint_and_a_blocked_step_under_the_closing_is_the_closings():
    policy, episode = baseline()
    policy.plan(observation(episode, 0))
    assert policy.explorer.current is not None
    policy.notify_blocked(observation(episode, 0))
    assert policy.explorer.current is None and policy.route_memory.path is None
    assert policy.episode_info()["exploration"]["stats"]["viewpoints_rejected"] == 1
    assert policy.mapping.worlds, "the bump was written to the observed map"
    locked, locked_episode = baseline(FakeDetector("chair", duplicate=True))
    locked.plan(observation(locked_episode, 0, depth=0.7))
    assert locked.closing.active
    locked.notify_blocked(observation(locked_episode, 0, depth=0.7))
    assert locked.closing.inspection_blocked or locked._blocked == 1


# -- exploration: relocation and exhaustion --------------------------------------------------------
def free_world(size=200, resolution=0.1, origin=-10.0):
    return OccupancyGrid2D(np.zeros((size, size), np.int8), OccupancyGrid2DParams(resolution, origin, origin))


def falcon_says(status, *args):
    """A FALCON planner stub answering ``status`` with the record the real planner attaches."""
    return lambda *a, **kw: FalconPlan(status, *args, diagnostics={"status": status, "planning_ms": 0.0})


def test_an_exhausted_disc_relocates_to_the_nearest_frontier_of_the_storey(monkeypatch):
    policy, episode = baseline()
    world = free_world()
    monkeypatch.setitem(policy.mapping.__dict__, "update", lambda *a, **kw: world)
    explorer = policy.explorer
    monkeypatch.setattr(explorer.explorer, "plan", falcon_says("no_useful_reachable_frontiers"))
    boundary = np.zeros(world.grid.shape, dtype=bool)
    far = world.world_to_grid(7.0, 0.0)              # 7 m east: outside the 4 m disc about the agent
    nearer = world.world_to_grid(-6.0, 0.0)          # 6 m west: nearer, and the one chosen
    for gx, gy in (far, nearer):
        boundary[gy - 2:gy + 3, gx] = True
    monkeypatch.setattr(explorer.explorer, "observe", lambda world_: None)
    explorer.explorer.frontiers.boundary = boundary
    command = policy.plan(observation(episode, 0))
    assert command.info["kind"] == "relocate" and not command.stop
    assert explorer.relocation is not None and explorer.relocation["goal"][0] == pytest.approx(-6.0, abs=0.2)
    assert math.dist(command.waypoints[-1], (-6.0, 0.0)) < 0.5
    assert policy.route_memory.kind == "relocate"
    info = policy.episode_info()["exploration"]
    assert info["stats"]["relocations"] == 1 and info["events"][-1]["event"] == "relocation_started"
    # The frontier the relocation walks to disappears (the walk showed it): the relocation ends early
    # and the next decision is FALCON's again.
    explorer.explorer.frontiers.boundary = np.zeros(world.grid.shape, dtype=bool)
    monkeypatch.setattr(explorer.explorer, "plan", falcon_says("planned", (), ()))
    policy.notify_action(observation(episode, 0), DiscreteAction.MOVE_FORWARD)
    with pytest.raises(IndexError):          # the stub plan has no viewpoint: FALCON was asked, which is the point
        policy._search(observation(episode, 1, pose=AgentPose(-0.25, 0, 0, math.pi)), world, False)
    assert explorer.relocation is None and info["stats"]["relocations"] == 1


def test_with_nothing_left_anywhere_the_agent_looks_all_round_and_then_stops(monkeypatch):
    policy, episode = baseline(baseline_settings=BaselineSettings(exhausted_turns=3))
    world = free_world()
    monkeypatch.setitem(policy.mapping.__dict__, "update", lambda *a, **kw: world)
    explorer = policy.explorer
    monkeypatch.setattr(explorer.explorer, "plan", falcon_says("frontiers_exhausted"))
    monkeypatch.setattr(explorer.explorer, "observe", lambda world_: None)
    explorer.explorer.frontiers.boundary = np.zeros(world.grid.shape, dtype=bool)
    yaw = 0.0
    for step in range(2):
        command = policy.plan(observation(episode, step, pose=AgentPose(0, 0, 0, yaw)))
        assert command.info["kind"] == "exploration_exhausted_turn" and not command.stop
        assert command.final_yaw == pytest.approx(yaw + episode.action_spec.turn_angle_rad)
        policy.notify_action(observation(episode, step), DiscreteAction.TURN_LEFT)
        yaw += episode.action_spec.turn_angle_rad
    final = policy.plan(observation(episode, 2, pose=AgentPose(0, 0, 0, yaw)))
    assert final.stop and final.info["kind"] == "exploration_exhausted"
    stats = policy.episode_info()["exploration"]["stats"]
    assert stats["exhausted_decisions"] == 3 and stats["exhausted_stops"] == 1
    # The same with STOP disabled keeps turning.
    patient, episode = baseline(baseline_settings=BaselineSettings(exhausted_turns=1, stop_when_exhausted=False))
    monkeypatch.setitem(patient.mapping.__dict__, "update", lambda *a, **kw: world)
    monkeypatch.setattr(patient.explorer.explorer, "plan", falcon_says("frontiers_exhausted"))
    monkeypatch.setattr(patient.explorer.explorer, "observe", lambda world_: None)
    patient.explorer.explorer.frontiers.boundary = np.zeros(world.grid.shape, dtype=bool)
    assert not patient.plan(observation(episode, 0)).stop
    assert not patient.plan(observation(episode, 1)).stop


def test_a_decision_failure_is_recorded_and_answered_with_a_turn_not_a_dead_episode(monkeypatch):
    policy, episode = baseline()
    monkeypatch.setattr(policy.explorer, "plan", lambda *a, **kw: 1 / 0)
    command = policy.plan(observation(episode, 0))
    assert command.info["kind"] == "decision_failure_turn" and not command.stop
    fallback = policy.episode_info()["exploration_fallback"]
    assert fallback["stats"]["failures"] == 1 and fallback["failures"][0]["kind"] == "decision"


# -- spawn-floor confinement: the full method's ---------------------------------------------------
def confined_baseline(monkeypatch):
    """The baseline on the two-room world with the one ground-truth staircase SEEN, traversal forbidden."""
    policy, episode = baseline(metadata=STRUCTURE)
    world, _ = two_room_world()
    monkeypatch.setitem(policy.mapping.__dict__, "update", lambda *a, **kw: world)
    obs = observation(episode, 0, pose=AgentPose(IN_A[0], IN_A[1], 0.0, 0.0))
    policy.mapping.atlas.update(obs.pose)
    policy.mapping._anchor = 0.0
    for connector in policy.building.ground_truth.connectors:
        policy.building.sightings.mark_seen(connector.id, 0, connector.polyline)
    policy.building.observe(obs)
    assert len(policy.building.portals) == 1, "the seen staircase is placed: knowledge, not a destination"
    return policy, episode, world, obs


def test_the_seen_staircase_is_impassable_on_the_world_falcon_plans_on_and_no_goal_on_it_is_accepted(monkeypatch):
    policy, episode, world, obs = confined_baseline(monkeypatch)
    assert policy.building.allowed is False and policy.floor_guard.enabled
    foot = world.world_to_grid(*FLIGHT[0][:2])
    assert world.is_free(*foot)
    command = policy.plan(obs)
    confined = policy.last_world
    mask = stair_peek_mask(policy, world)
    assert mask.any() and confined is not world
    assert (confined.grid[mask] == world.values.occupied).all(), "the flight is written occupied for every plan"
    assert policy.mapping.worlds[policy.mapping.floor_id] is confined, "the recorder and the display read the same map"
    assert not command.stop and command.info["kind"] in ("falcon", "relocate")
    # FALCON's viewpoint and its route stand clear of the flight ...
    goal = policy.route_memory.goal
    gx, gy = world.world_to_grid(*goal)
    assert not mask[gy, gx]
    for waypoint in command.waypoints:
        wx, wy = world.world_to_grid(*waypoint)
        assert not mask[wy, wx]
    # ... a FALCON step into the masked flight is vetoed by the swept-motion check that reads the confined
    # map (the route in force is FALCON's), and a goal on the flight is refused by the guard the full method uses.
    before_the_flight = AgentPose(FLIGHT[0][0], FLIGHT[0][1] + 0.35, 0.0, -math.pi / 2)
    at_the_foot = observation(episode, 1, pose=before_the_flight)
    policy.explorer.prepare(at_the_foot, confined)
    assert policy.route_memory.kind == "falcon"
    assert policy.filter_action(at_the_foot, DiscreteAction.MOVE_FORWARD) == DiscreteAction.TURN_LEFT
    assert policy.floor_guard.goal_allowed(obs, confined, FLIGHT[1][:2], "falcon") is False
    assert policy._navigate(obs, confined, FLIGHT[1][:2], "relocate") is None
    assert policy.floor_guard.stats["goals_refused"] >= 2
    assert policy.episode_info()["spawn_floor_guard"]["stats"]["stair_cells"] > 0


def test_an_unplanned_height_departure_starts_no_traversal_and_the_building_never_plans(monkeypatch):
    policy, episode, world, obs = confined_baseline(monkeypatch)
    policy.plan(obs)
    raised = observation(episode, 1, pose=AgentPose(IN_A[0], IN_A[1], 0.6, 0.0))
    policy.building.prepare_observation(raised)
    assert policy.building.transition is None and policy.building.phase == "SEARCH"
    assert any(e["event"] == "height_departure_ignored" for e in policy.building.events)
    assert policy.building.plan(raised, world) is None
    building = policy.episode_info()["building"]
    assert building["allow_stair_traversal"] is False and building["portals"][0]["stair_source"] == "ground_truth"
    assert building["decides_floor_changes"].startswith("never")


# -- the record ---------------------------------------------------------------------------------------
def test_the_recorder_snapshot_the_floor_panels_and_the_lean_record_work_without_rooms():
    policy, episode = baseline()
    obs = observation(episode, 0)
    policy.plan(obs)
    snapshot = method_snapshot(policy)
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["state"] == "FALCON_FALCON" and snapshot["exploration"]["mode"] == "falcon"
    assert snapshot["rooms"] == 0 and snapshot["search"] == {} and snapshot["explorer"] == "falcon"
    assert snapshot["target_closing"]["phase"] == "SEARCH"
    panels = FloorPanels(1, policy.settings, obs.pose)
    panels.capture(policy, obs)
    assert panels.slots[0]["floor_id"] == 0 and panels.slots[0]["rooms"] == 0
    assert panels.render(0, snapshot["planned_path"], size=(640, 720)).shape == (720, 640, 3)
    agent = HeadlessObjNavAgent(policy, gibson_label_mapper(), name=policy.name)
    agent.reset(episode)
    agent.act(observation(episode, 0))
    full = agent.episode_info()
    lean = lean_episode_info(full)
    assert lean["lean"] and lean["policy"]["method"] == "sparx-falcon-yolo-baseline"
    assert "stats" in lean["policy"]["exploration"] and "records" not in lean["policy"]["exploration"]
    assert "target_closing" in lean["policy"] and "spawn_floor_guard" in lean["policy"]
    json.dumps(full, allow_nan=False)


def test_the_dashboard_draws_the_baseline_column_in_place_of_the_empty_search_panel():
    policy, episode = baseline(FakeDetector("chair", duplicate=True))
    obs = observation(episode, 0, depth=0.7)
    command = policy.plan(obs)
    snapshot = method_snapshot(policy)
    assert snapshot["search"] == {} and snapshot["exploration"] is not None
    decision = {"action": "MOVE_FORWARD", "info": dict(command.info)}
    frame = render_dashboard(policy, obs, [(0.0, 0.0)], decision, episode.episode_id, snapshot)
    assert frame.shape == (FRAME_SIZE[1], FRAME_SIZE[0], 3)
    column = frame[40:760, 1280:]
    assert (column > 150).any(), "the baseline column is written, not the blank search panel"
    final = render_dashboard(policy, obs, [(0.0, 0.0)], {}, episode.episode_id, snapshot, final=True)
    assert final.shape == frame.shape


def corridor_world(size=200, resolution=0.1, origin=-10.0):
    """Unknown everywhere but a 2 m wide free corridor from x=-1 to x=8 m: frontiers along its edges."""
    grid = np.full((size, size), -1, np.int8)
    x0, y0 = int((-1.0 - origin) / resolution), int((-1.0 - origin) / resolution)
    x1, y1 = int((8.0 - origin) / resolution), int((1.0 - origin) / resolution)
    grid[y0:y1 + 1, x0:x1 + 1] = 0
    return OccupancyGrid2D(grid, OccupancyGrid2DParams(resolution, origin, origin))


def test_the_legacy_pursuit_drops_the_falcon_viewpoint_without_rejecting_it_and_is_not_held_to_the_disc(monkeypatch):
    policy, episode = baseline()
    world = corridor_world()
    monkeypatch.setitem(policy.mapping.__dict__, "update", lambda *a, **kw: world)
    first = policy.plan(observation(episode, 0))
    assert first.info["kind"] == "falcon" and policy.explorer.current is not None
    # A confirmed target landmark 6 m away (outside the 4 m disc), as the perception cycle would leave it.
    policy._target_xy, policy._target_id, policy._target_step = (6.0, 0.0), 1, 1
    policy.target_evidence.verification_started = lambda landmark_id: False
    command = policy.plan(observation(episode, 1, pose=AgentPose(0.0, 0.0, 0.0, 0.0)))
    assert command.info["kind"] == "target" and policy.route_memory.kind == "target"
    assert policy.explorer.current is None, "the viewpoint is dropped ..."
    stats = policy.episode_info()["exploration"]["stats"]
    assert stats.get("viewpoints_rejected", 0) == 0 and stats["viewpoints_suspended"] == 1, "... not rejected"
    far = observation(episode, 1, pose=AgentPose(4.5, 0.0, 0.0, 0.0))     # on the target route, beyond the disc
    policy.explorer.prepare(far, world)
    assert policy.filter_action(far, DiscreteAction.MOVE_FORWARD) == DiscreteAction.MOVE_FORWARD


def test_an_astar_route_is_not_rejudged_by_the_swept_motion_check_but_a_falcon_route_is(monkeypatch):
    policy, episode = baseline()
    world = corridor_world()
    monkeypatch.setitem(policy.mapping.__dict__, "update", lambda *a, **kw: world)
    obs = observation(episode, 0)
    assert policy.plan(obs).info["kind"] == "falcon" and policy.route_memory.kind == "falcon"
    monkeypatch.setattr(policy.explorer.motion, "permits", lambda *a, **kw: False)
    assert policy.filter_action(obs, DiscreteAction.MOVE_FORWARD) == DiscreteAction.TURN_LEFT
    assert policy.explorer.current is None and policy.episode_info()["exploration"]["stats"]["safety_vetoes"] == 1
    # The same refusal under a relocation route (A*, qualified at the body radius): the step stands.
    monkeypatch.setattr(policy.explorer.explorer, "plan", falcon_says("no_useful_reachable_frontiers"))
    boundary = np.zeros(world.grid.shape, dtype=bool)
    gx, gy = world.world_to_grid(7.0, 0.0)
    boundary[gy - 2:gy + 3, gx] = True
    monkeypatch.setattr(policy.explorer.explorer, "observe", lambda world_: None)
    policy.explorer.explorer.frontiers.boundary = boundary
    later = observation(episode, 1)
    command = policy.plan(later)
    assert command.info["kind"] == "relocate" and policy.route_memory.kind == "relocate"
    assert policy.filter_action(later, DiscreteAction.MOVE_FORWARD) == DiscreteAction.MOVE_FORWARD
    assert policy.episode_info()["exploration"]["stats"]["safety_vetoes"] == 1
