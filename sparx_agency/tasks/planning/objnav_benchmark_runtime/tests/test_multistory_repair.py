"""Failure-driven multi-story tests; synthetic passes are not navigation scores."""
from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest

from sparx_agency.core.planning.exploration.floor_atlas import FloorAtlas, ObservedFloor
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import FloorPanels, UNKNOWN_GRAY
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import habitat_pose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.camera_control import CameraController
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import observed_objects
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_traversal import StairTraversal
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import method_snapshot


def test_inspection_is_one_bounded_transaction_not_pitch_or_yaw_keyed():
    actions = MULTIFLOOR_PROTOCOL.actions()
    camera, converter = CameraController(actions), DiscreteActionConverter(actions)
    pose = AgentPose(0, 0, 0, 0)
    emitted = []
    for step in range(6, 70):
        obs = SimpleNamespace(pose=pose, step=step)
        camera.begin_inspection(obs, 0)
        command = camera.inspection_command(obs) or NavigationCommand.hold()
        command = camera.apply(obs, command)
        action = converter.step(pose, command).action
        emitted.append(action)
        if action == DiscreteAction.LOOK_DOWN:
            pose = replace(pose, camera_pitch=pose.camera_pitch + actions.tilt_angle_rad)
        elif action == DiscreteAction.LOOK_UP:
            pose = replace(pose, camera_pitch=pose.camera_pitch - actions.tilt_angle_rad)
        elif action == DiscreteAction.TURN_LEFT:
            pose = replace(pose, yaw=pose.yaw + actions.turn_angle_rad)
        elif action == DiscreteAction.TURN_RIGHT:
            pose = replace(pose, yaw=pose.yaw - actions.turn_angle_rad)
    assert emitted.count(DiscreteAction.LOOK_DOWN) == 2
    assert emitted.count(DiscreteAction.LOOK_UP) == 2
    assert camera.inspection is None and len(camera.inspected) == 1
    assert pose.camera_pitch == pytest.approx(0)


def test_camera_does_not_restore_or_oscillate_during_committed_transition():
    camera = CameraController(MULTIFLOOR_PROTOCOL.actions())
    obs = SimpleNamespace(pose=AgentPose(0, 0, 0.2, 0, camera_pitch=math.pi / 3), step=10)
    for phase in ("TRAVERSE", "CONFIRM_DESTINATION", "RETREAT", "TRAVERSE"):
        command = camera.apply(obs, NavigationCommand.hold(), phase, 1, close_support=phase == "TRAVERSE")
        assert command.camera_pitch == pytest.approx(math.pi / 3)
        assert command.info["camera"]["owner"] == phase
    assert camera.apply(obs, NavigationCommand.hold(), "SEARCH").camera_pitch == 0


def test_no_look_commands_on_no_tilt_protocol():
    _, episode = setup_policy()
    camera = CameraController(episode.action_spec)
    obs = observation(episode, 8)
    assert not camera.begin_inspection(obs, 0)
    assert camera.apply(obs, NavigationCommand.hold(), "TRAVERSE", -1).camera_pitch is None


@pytest.mark.parametrize("owner", ["SEARCH", "SAFE_HALT"])
def test_stop_never_carries_nonexecutable_camera_pitch(owner):
    camera = CameraController(MULTIFLOOR_PROTOCOL.actions())
    obs = SimpleNamespace(pose=AgentPose(0, 0, 1, 0, camera_pitch=math.pi / 3), step=30)
    command = camera.apply(obs, NavigationCommand.stop_here(), owner, -1)
    assert command.stop and command.camera_pitch is None
    assert command.info["camera"]["observed_pitch_rad"] == pytest.approx(math.pi / 3)


def test_raw_perception_continues_on_stairs_without_fusing_or_stopping(monkeypatch):
    policy, episode = setup_policy()
    obs = observation(episode, 0)
    policy.plan(obs)
    before = len(policy.landmarks), policy.graph.queries
    building = policy.building
    building.active = building._portal(obs, 1, [(0, 0, 0), (1, 0, 0.6)])
    building._start(obs)
    monkeypatch.setattr(building.transition, "plan", lambda obs: NavigationCommand.hold(info={"kind": "TRAVERSE"}))
    for step in range(1, 5):
        command = policy.plan(replace(observation(episode, step), pose=AgentPose(step * 0.1, 0, 0.2, 0)))
        assert not command.stop
        snapshot = method_snapshot(policy)
        assert snapshot["perception"]["observation_step"] == step
        assert snapshot["detections"] and snapshot["perception"]["projections"][0]["status"] == "transition_in_progress"
    assert (len(policy.landmarks), policy.graph.queries) == before
    assert policy.perception.counts["raw_frames"] == 5
    assert building.active is not None


def test_committed_approach_preempts_target_reasoning(monkeypatch):
    policy, episode = setup_policy()
    obs = observation(episode, 0)
    policy.plan(obs)
    building = policy.building
    building.active = building._portal(obs, -1, [(3, 0, 0), (4, 0, -0.6)])
    monkeypatch.setattr(policy, "_navigate", lambda *args: NavigationCommand.hold(info={"kind": "portal/0"}))
    command = policy.plan(observation(episode, 1))
    assert command.info["phase"] == "APPROACH_STAIRS" and not command.stop


def test_projection_uses_actual_supported_pixels_not_box_center_ray():
    _, episode = setup_policy()
    obs = observation(episode, 0, depth=5)
    depth = obs.depth_m.copy()
    depth[200:280, 270:310] = 1
    box = DetectionWire("bed", 0.9, (240, 160, 400, 320))
    diagnostics = []
    points = list(observed_objects(replace(obs, depth_m=depth), [box], 0.35, diagnostics))
    assert len(points) == 1
    assert points[0][1][1] > 0.02  # left of optical centre, not the centre ray
    assert diagnostics[0]["status"] == "projected"


def test_fragmented_depth_is_quarantined():
    _, episode = setup_policy()
    obs = observation(episode, 0, depth=2)
    depth = obs.depth_m.copy()
    yy, xx = np.indices((80, 80))
    depth[200:280, 280:360] = 1 + ((xx + yy) % 2) * 3
    diagnostics = []
    boxes = [DetectionWire("sofa", 0.95, (240, 160, 400, 320))]
    assert not list(observed_objects(replace(obs, depth_m=depth), boxes, 0.35, diagnostics))
    assert diagnostics[0]["status"] == "mixed_depth"


def test_stairs_are_context_only_and_do_not_create_portals_from_rgb(monkeypatch):
    mapper = gibson_label_mapper()
    assert {"stairs", "staircase"} <= set(mapper.vocabulary())
    assert all(not mapper.target_labels(c).accepts("stairs") for c in ("chair", "couch", "bed", "toilet", "tv"))
    policy, episode = setup_policy("stairs")
    policy.plan(observation(episode, 0, depth=3))
    assert not policy.building.portals and not policy.landmarks.all_landmarks()


def test_wrong_floor_support_rejected_but_real_bed_not_blacklisted():
    policy, episode = setup_policy("bed")
    policy.plan(observation(episode, 0, depth=3))
    terrain = policy.building.terrain
    terrain.samples = {(30, i, -20): (-2.0, 0, True) for i in range(-3, 4)}
    assert policy.perception._floor_support((3, 0, 0.5)) == "unsupported_floor_association"
    terrain.samples = {(30, i, 0): (0.0, 0, True) for i in range(-3, 4)}
    assert policy.perception._floor_support((3, 0, 0.5)) is None


def test_three_dimensional_association_cannot_walk_a_landmark(monkeypatch):
    policy, episode = setup_policy()
    policy.plan(observation(episode, 0))
    original = policy.landmarks.all_landmarks()[0]
    policy._object_geometry[original.id] = (original.xy[0], original.xy[1], -1.0)
    count = original.count
    assert not policy._perceive(observation(episode, 1))
    assert original.count == count
    assert policy.perception.projections[0]["status"] == "inconsistent_3d_association"


def transition_fixture():
    policy, episode = setup_policy("bed")
    obs = observation(episode, 0)
    policy.plan(obs)
    building = policy.building
    building.active = building._portal(obs, 1, [(0, 0, 0), (1, 0, 0.6)])
    building._start(obs)
    return policy, episode, building.transition


def test_near_source_height_is_not_retreat_completion(monkeypatch):
    policy, episode, transition = transition_fixture()
    transition.retreat_path = [(1, 0, 0.2), (0, 0, 0)]
    transition.retreat_step = 0
    transition.phase = "RETREAT"
    monkeypatch.setattr(transition.terrain, "plateau_area", lambda pose: 9)
    monkeypatch.setattr(transition.terrain, "stair_distance", lambda pose: 2)
    for step, x in enumerate((1, 0.75, 0.5, 0.25), 1):
        transition.observe(replace(observation(episode, step), pose=AgentPose(x, 0, 0.206, 0)))
    assert not transition.arrival_allowed
    with pytest.raises(RuntimeError, match="Cannot abandon"):
        policy.building._abandon(observation(episode, 5), "not actually returned")


def test_landing_area_and_exit_are_both_required(monkeypatch):
    _, episode, transition = transition_fixture()
    monkeypatch.setattr(transition.terrain, "plateau_area", lambda pose: 1)
    monkeypatch.setattr(transition.terrain, "stair_distance", lambda pose: 2)
    for step, x in enumerate((1, 1.25, 1.5, 1.75, 2), 1):
        transition.observe(replace(observation(episode, step), pose=AgentPose(x, 0, 1.6, 0)))
    assert not transition.arrival_allowed
    monkeypatch.setattr(transition.terrain, "plateau_area", lambda pose: 9)
    monkeypatch.setattr(transition.terrain, "stair_distance", lambda pose: 0.1)
    for step, x in enumerate((2.25, 2.5, 2.75), 6):
        transition.observe(replace(observation(episode, step), pose=AgentPose(x, 0, 2.7, 0)))
    assert transition.phase == "CONFIRM_DESTINATION" and not transition.arrival_allowed
    monkeypatch.setattr(transition.terrain, "stair_distance", lambda pose: 2)
    transition.observe(replace(observation(episode, 9), pose=AgentPose(3.25, 0, 2.7, 0)))
    assert transition.arrival_allowed and transition.completion_reason == "stable_platform_and_stair_exit"


def test_atlas_defers_new_and_revisited_floor_until_exit():
    atlas = FloorAtlas()
    atlas.update(AgentPose(0, 0, 0, 0))
    atlas.floors[1] = ObservedFloor(1, 2.7)
    atlas.begin_transition(AgentPose(0, 0, 0, 0))
    for x in (1, 1.25, 1.5, 1.75):
        atlas.update(AgentPose(x, 0, 2.7, 0), 9, arrival_allowed=False)
    assert atlas.active_id == 0 and atlas.in_transition and not atlas.connections
    atlas.update(AgentPose(2, 0, 2.7, 0), 9, arrival_allowed=True)
    assert atlas.active_id == 1 and len(atlas.connections) == 1


def test_continuation_uses_support_height_not_smoothed_base():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_multifloor import terrain_with_surface
    terrain = terrain_with_surface(lambda x: 0.0)
    assert terrain.continuation(AgentPose(0, 0, 0.3, 0), 1, 0)
    assert terrain.params.max_step_m == 0.24


def test_footprint_does_not_flatten_a_measured_next_tread():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_multifloor import terrain_with_surface
    terrain = terrain_with_surface(lambda x: 0.18 if x >= 1 else 0.0)
    x, y = terrain.start
    assert terrain.heights[y, x + 1] == pytest.approx(0.18)


def test_retreat_includes_observed_platform_approach():
    policy, episode = setup_policy("bed")
    obs = observation(episode, 0)
    policy.plan(obs)
    building = policy.building
    building.source_trail = [(-1.5, 0, 0), (-1, 0, 0), (-0.5, 0, 0), (0, 0, 0)]
    building.active = building._portal(obs, 1, [(0, 0, 0), (1, 0, 0.6)])
    building._start(obs)
    assert building.transition.source_xyz == (-1.5, 0, 0)
    assert building.transition.trace == building.source_trail


def test_flight_guidance_uses_farthest_connected_support(monkeypatch):
    _, episode, transition = transition_fixture()
    transition.portal["path"] = [(0, 0, 0), (0.6, 0, 0.2), (1.2, 0, 0.6)]
    terrain = transition.terrain
    for point in transition.portal["path"]:
        x, y = terrain.world.world_to_grid(*point[:2])
        terrain.heights[y, x] = point[2]
    monkeypatch.setattr(terrain, "path", lambda cell: [(0, 0, 0), (*terrain.world.grid_to_world(*cell), float(terrain.heights[cell[1], cell[0]]))])
    path = transition._surface_route(observation(episode, 1))
    assert path[-1][0] > 1.0 and path[-1][2] == pytest.approx(0.6)


def test_retreat_cursor_skips_loops_and_cannot_regress(monkeypatch):
    _, episode, transition = transition_fixture()
    transition.retreat_path = [(0, 0, 0), (0.5, 0, 0), (0, 0, 0), (-0.5, 0, 0), (-1, 0, 0)]
    terrain = transition.terrain
    monkeypatch.setattr(terrain, "path", lambda cell: [(0, 0, 0), (*terrain.world.grid_to_world(*cell), 0)])
    path = transition._retreat_route(observation(episode, 1))
    assert path[-1][0] < 0 and transition.retreat_index == 2
    transition._retreat_route(replace(observation(episode, 2), pose=AgentPose(-0.5, 0, 0, 0)))
    cursor = transition.retreat_index
    transition._retreat_route(observation(episode, 3))
    assert transition.retreat_index >= cursor


def test_unknown_panels_are_independent_gray_and_do_not_teach_policy(tmp_path):
    policy, episode = setup_policy("bed")
    obs = observation(episode, 0, depth=3)
    panels = FloorPanels(3, policy.settings, obs.pose)
    assert all(np.all(s["grid"] == -1) for s in panels.slots)
    assert not np.shares_memory(panels.slots[0]["grid"], panels.slots[1]["grid"])
    policy.plan(obs)
    panels.capture(policy, obs)
    first = panels.slots[0]["grid"].copy()
    assert len(policy.mapping.maps) == 1 and len(policy.mapping.atlas.floors) == 1
    image = panels.render(0)
    assert np.all(image[60:280, 600:800] == UNKNOWN_GRAY)
    assert np.all(panels.slots[1]["grid"] == -1)
    policy.floors.activate(1)
    panels.capture(policy, obs)
    np.testing.assert_array_equal(panels.slots[0]["grid"], first)
    policy.floors.activate(0)
    panels.capture(policy, obs)
    assert panels.bindings == {0: 0}
    panels.save(tmp_path)
    saved = np.load(tmp_path / "floor_maps.npz", allow_pickle=False)
    assert len(saved.files) == 3 and np.all(saved["slot_2"] == -1)


@pytest.mark.parametrize("pitch", [-30, 0, 30, 60])
def test_habitat_pitch_and_enu_alignment(pitch):
    # Native LOOK_DOWN is negative rotation about Habitat camera +X.
    a = math.radians(-pitch)
    camera = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    pose = habitat_pose((2, 3, -4), np.eye(3), camera)
    assert (pose.x, pose.y, pose.z) == (4, -2, 3)
    assert pose.camera_pitch == pytest.approx(math.radians(pitch))


def test_selected_proposal_refines_geometry_but_measured_edge_does_not():
    policy, episode = setup_policy("bed")
    policy.plan(observation(episode, 0, depth=3))
    building = policy.building
    original = [(0, 0, 0), (1, 0, 0.6)]
    portal = building._portal(observation(episode, 1), 1, original)
    building.active = portal
    portal["destination"] = 1
    refined = [(0.1, 0, 0), (0.4, 0, 0.18), (1.1, 0, 0.7)]
    building.terrain.candidates = [{"path": refined, "direction": 1}]
    building._discover(observation(episode, 2))
    assert building.active is portal and portal["path"] == refined
    assert portal["direction"] == 1 and portal["destination"] == 1
    portal["edge_id"] = 42
    building.terrain.candidates = [{"path": original, "direction": 1}]
    building._discover(observation(episode, 3))
    assert portal["path"] == refined


def test_source_floor_slab_does_not_rise_with_base_and_reblock_treads():
    from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
    from sparx_agency.core.planning.exploration.floor_atlas import MultiFloorParams
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_terrain import StairTerrain
    terrain = StairTerrain(MultiFloorParams(), 0.1, 0.18, 0.88)
    for x in range(-10, 30):
        for y in range(-10, 11):
            z = 0.15 + max(0, x // 3) * 0.15
            terrain.samples[(x, y, round(z / 0.1))] = (z, 0, True)
    terrain._floor_world = OccupancyGrid2D(np.full((120, 120), 100, np.int8),
        OccupancyGrid2DParams(0.1, -6, -6), values=OccupancyValues(free=0, occupied=100, unknown=-1))
    terrain._floor_height = 0.0
    pose = AgentPose(0, 0, 0.27, 0)
    terrain._rasterize(pose)
    assert terrain.permits(pose, 0.25)
    assert terrain.params.max_step_m == 0.24
    terrain.samples[(2, 0, 6)] = (0.65, 0, False)
    terrain._rasterize(pose)
    assert not terrain.permits(pose, 0.25)


