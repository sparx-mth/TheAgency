"""Multi-story regressions: stable identity, isolated evidence, stairs and scoring."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest

from sparx_agency.core.planning.exploration.floor_atlas import FloorAtlas, MultiFloorParams
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_distance import MultiFloorDistance
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_metrics import ExplorationMetrics
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.observed_map import ObservedMap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_terrain import StairTerrain
from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import EpisodeRecorder
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation


def ascend(atlas, height=2.7, start_x=0.0):
    for i, z in enumerate(np.arange(0.18, height, 0.18), 1):
        atlas.update(AgentPose(start_x + i * 0.25, 0, float(z), 0))
        assert len(atlas.floors) == 1
    for x in (4.0, 4.25, 4.5):
        atlas.update(AgentPose(start_x + x, 0, height, 0))


def test_treads_and_stationary_turns_never_create_floors():
    atlas = FloorAtlas()
    atlas.update(AgentPose(0, 0, 0, 0))
    for i in range(20):
        atlas.update(AgentPose(0.25, 0, 1.8, i * 0.1))
    assert len(atlas.floors) == 1 and atlas.in_transition
    assert not atlas.connections


def test_floor_hysteresis_and_revisit_make_one_bidirectional_edge():
    atlas = FloorAtlas()
    atlas.update(AgentPose(0, 0, 0, 0))
    ascend(atlas)
    assert atlas.active_id == 1 and len(atlas.floors) == 2
    assert len(atlas.connections) == 1
    edge = atlas.connections[0]
    assert edge.source == 0 and edge.destination == 1 and edge.length_m > 2.7
    assert atlas.first_hop(0) is edge
    for i, z in enumerate(np.arange(2.52, 0, -0.18), 1):
        atlas.update(AgentPose(4.5 - i * 0.25, 0, float(z), math.pi))
    for x in (0.5, 0.25, 0.0):
        atlas.update(AgentPose(x, 0, 0, math.pi))
    assert atlas.active_id == 0 and len(atlas.floors) == 2
    assert len(atlas.connections) == 1 and edge.traversals == 2
    assert atlas.floors[0].visits == 2


@pytest.mark.parametrize("kwargs", [{"stable_samples": 1}, {"enabled": "yes"},
                                   {"floor_match_m": 0.9}, {"max_step_m": float("nan")},
                                   {"floor_search_actions": 0}])
def test_multifloor_settings_reject_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        MultiFloorParams(**kwargs)


def test_per_floor_grids_are_retained_and_not_fused_on_stairs():
    _, episode = setup_policy()
    mapping = ObservedMap(size_m=20)
    original = mapping.update(observation(episode, 0, depth=3))
    grid = mapping.grid
    before = grid.get_grid()[1].copy()
    step = 1
    for x, z in [(0.25 * i, float(z)) for i, z in enumerate(np.arange(0.18, 2.7, 0.18), 1)] + [(4, 2.7), (4.25, 2.7), (4.5, 2.7)]:
        obs = replace(observation(episode, step, depth=2), pose=AgentPose(x, 0, z, 0))
        mapping.update(obs, integrate=False)
        step += 1
    assert len(mapping.maps) == 2 and mapping.floor_id == 1
    assert mapping.maps[0] is grid and mapping.worlds[0] is original
    np.testing.assert_array_equal(grid.get_grid()[1], before)
    assert mapping.maps[1] is not grid


def test_floor_contexts_keep_target_rejections_and_room_history_isolated():
    policy, episode = setup_policy()
    policy.plan(observation(episode, 0))
    first_graph, first_landmarks = policy.graph, policy.landmarks
    policy.target_evidence.reject(0, 0)
    first_evidence = policy.target_evidence
    policy._floor_time = 45
    policy._visited_frontiers.append((1, 1))
    policy.floors.activate(1)
    assert policy.graph is not first_graph and len(policy.landmarks) == 0
    assert policy._floor_time == 0 and policy._target_xy is None
    policy._floor_time = 12
    policy.floors.activate(0)
    assert policy.graph is first_graph and policy.landmarks is first_landmarks
    assert policy.target_evidence is first_evidence
    assert policy._floor_time == 45 and policy._visited_frontiers == [(1, 1)]
    assert policy._target_xy is None and policy.route_memory.path is None


def test_coverage_return_does_not_count_known_floor_twice():
    _, episode = setup_policy()
    mapping = ObservedMap(size_m=20)
    world = mapping.update(observation(episode, 0, depth=3))
    metrics = ExplorationMetrics()
    metrics.observe(observation(episode, 0), world, 0)
    area = metrics.curve[-1]["observed_m2"]
    metrics.observe(observation(episode, 1), world, 1)
    assert metrics.curve[-1]["observed_m2"] == pytest.approx(area * 2)
    metrics.observe(observation(episode, 2), world, 0)
    assert metrics.curve[-1]["new_m2"] == 0
    assert metrics.curve[-1]["observed_m2"] == pytest.approx(area * 2)


def terrain_with_surface(height_fn):
    terrain = StairTerrain(MultiFloorParams(), 0.1, 0.18, 0.88)
    for x in range(-8, 30):
        for y in range(-8, 9):
            z = float(height_fn(x))
            terrain.samples[(x, y, int(round(z / 0.1)))] = (z, 0, True)
    terrain._rasterize(AgentPose(0, 0, 0, 0))
    return terrain


def test_measured_treads_are_connected_but_a_cliff_is_not():
    terrain = terrain_with_surface(lambda x: max(0, x // 4) * 0.18 - 0.05)
    assert any(c["direction"] == 1 for c in terrain.candidates)
    assert terrain.permits(AgentPose(0, 0, 0, 0), 0.25)
    cliff = terrain_with_surface(lambda x: 1.0 if x > 1 else -0.05)
    assert not cliff.candidates
    assert not cliff.permits(AgentPose(0, 0, 0, 0), 0.25)


def test_depth_wall_is_not_mistaken_for_stairs():
    _, episode = setup_policy()
    terrain = StairTerrain(MultiFloorParams(), 0.1, 0.18, 0.88)
    points, horizontal = terrain._surfaces(observation(episode, 0, depth=3))
    assert len(points) > 100 and not horizontal.any()
    terrain.update(observation(episode, 0, depth=3))
    assert not terrain.candidates


def test_upstairs_xy_overlap_cannot_succeed_downstairs():
    semantic = np.zeros((7, 80, 80), np.uint8)
    semantic[0] = 1
    semantic[1, 20, 20] = 1
    def find_path(path):
        path.geodesic_distance = 8.0
        return True
    field = MultiFloorDistance(SimpleNamespace(find_path=find_path), [[1, 0, 1]], semantic,
                               (0, 0), 0, 0, path_factory=SimpleNamespace)
    assert field.distance((1, 0, 1)) == 0
    assert field.distance((1, 3, 1)) == 8
    assert not field.success((1, 3, 1))
    assert not field.start_uses_sentinel((1, 3, 1))


def test_protocol_is_separate_and_permits_bounded_camera_inspection():
    assert MULTIFLOOR_PROTOCOL.path_length_dimension == "3d"
    assert MULTIFLOOR_PROTOCOL.max_steps == 500
    assert MULTIFLOOR_PROTOCOL.split != "val"
    assert MULTIFLOOR_PROTOCOL.actions().allows(DiscreteAction.LOOK_DOWN)
    assert "incomplete" in MULTIFLOOR_PROTOCOL.goal_definition


def test_unselected_recording_does_not_touch_frames_or_policy(tmp_path):
    policy, episode = setup_policy()
    recorder = EpisodeRecorder(tmp_path, policy, selected_episode_ids={"another/episode"})
    recorder.begin(episode, observation(episode, 0), {})
    recorder.decision(None, None, None)
    recorder.after_step(None, None, True)
    recorder.complete(None)
    assert not list(tmp_path.iterdir())


def test_near_field_support_uses_observed_free_space_on_matching_floor_only():
    from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
    terrain = StairTerrain(MultiFloorParams(), 0.1, 0.18, 0.88)
    for x in range(8, 30):
        for y in range(-6, 7):
            z = max(0, (x - 8) // 4) * 0.18 - 0.05
            terrain.samples[(x, y, int(round(z / 0.1)))] = (z, 0, True)
    terrain._rasterize(AgentPose(0, 0, 0, 0))
    assert not terrain.candidates  # unobserved blind gap is not carved free
    grid = np.full((80, 80), -1, np.int8)
    grid[32:48, 38:50] = 0
    terrain._floor_world = OccupancyGrid2D(grid, OccupancyGrid2DParams(0.1, -4, -4),
                                         values=OccupancyValues(free=0, occupied=100, unknown=-1))
    terrain._floor_height = 0.0
    terrain._rasterize(AgentPose(0, 0, 0, 0))
    assert any(c["direction"] == 1 for c in terrain.candidates)
    terrain._rasterize(AgentPose(0, 0, 3.0, 0))
    assert not terrain.candidates  # same XY, other floor: cannot reuse support


def test_stair_steering_uses_safe_discrete_heading_and_converter_moves():
    from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
    from sparx_agency.core.planning.objnav.types.command import NavigationCommand
    terrain = terrain_with_surface(lambda x: max(0, x // 4) * 0.18 - 0.05)
    pose = AgentPose(0, 0, 0, 0)
    actions = MULTIFLOOR_PROTOCOL.actions()
    path = terrain.steering_path(pose, [(0, 0, 0), (2.0, 0, 0.85)], actions)
    assert len(path) == 2 and math.dist(path[0][:2], path[1][:2]) == pytest.approx(actions.forward_step_m + terrain.resolution)
    result = DiscreteActionConverter(actions).step(pose, NavigationCommand.follow(path))
    assert result.action == DiscreteAction.MOVE_FORWARD
    assert terrain.permits(pose, actions.forward_step_m)


def test_campaign_jobs_select_one_explorer_without_doubling_episodes():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distinct_buildings import jobs_for
    assert jobs_for(["A", "B"], ["frontier"]) == [("frontier", "A"), ("frontier", "B")]
    with pytest.raises(ValueError):
        jobs_for(["A"], ["frontier", "frontier"])


def test_development_runner_import_and_first_recording_cli():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run_development import parser
    args = parser().parse_args(["--manifest", "episodes.json", "--scene", "A", "--output", "results",
                               "--explorer", "frontier", "--detector-url", "http://localhost:18095",
                               "--detector-backend", "yolo_world", "--record-first", "--limit", "1"])
    assert args.record_first and args.limit == 1


def test_observed_underfoot_height_handles_smoothed_navmesh_offset(monkeypatch):
    terrain = StairTerrain(MultiFloorParams(), 0.1, 0.18, 0.88)
    points = np.array([(x, y, -0.30) for x in np.arange(-0.3, 1.5, 0.03)
                       for y in np.arange(-0.5, 0.5, 0.03)])
    monkeypatch.setattr(terrain, "_surfaces", lambda obs: (points, np.ones(len(points), bool)))
    pose = AgentPose(0, 0, 0, 0)
    terrain.update(SimpleNamespace(pose=pose, step=0))
    assert terrain._foot_height == pytest.approx(-0.30)
    assert terrain.permits(pose, 0.25)
    assert terrain.params.max_step_m == 0.24


def test_small_stair_landing_is_not_promoted_to_a_floor():
    atlas = FloorAtlas()
    atlas.update(AgentPose(0, 0, 0, 0), plateau_area_m2=8.0)
    for i in range(8):
        atlas.update(AgentPose(i * 0.25, 0, 1.6, 0), plateau_area_m2=1.0)
    assert atlas.in_transition and len(atlas.floors) == 1
    for i in range(3):
        atlas.update(AgentPose(2.0 + i * 0.25, 0, 2.7, 0), plateau_area_m2=8.0)
    assert atlas.active_id == 1 and len(atlas.floors) == 2
    assert atlas.elevation_m == pytest.approx(2.7)


@pytest.mark.parametrize("plane_z,pitch,is_support", [(0.0, math.pi / 6, True), (1.8, -math.pi / 6, False)])
def test_surface_orientation_rejects_ceiling_undersides(plane_z, pitch, is_support):
    from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
    _, episode = setup_policy()
    pose = AgentPose(0, 0, 0, 0, camera_pitch=pitch)
    k = episode.camera.intrinsics
    yy, xx = np.indices((k.height, k.width))
    rays = np.stack(((xx - k.cx) / k.fx, (yy - k.cy) / k.fy, np.ones_like(xx)), axis=-1)
    transform = world_T_camera_optical(pose, episode.camera)
    rays = rays @ transform[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = ((plane_z - transform[2, 3]) / rays[..., 2]).astype(np.float32)
    depth[depth <= 0] = np.nan
    obs = replace(observation(episode, 0), pose=pose, depth_m=depth)
    _, horizontal = StairTerrain(MultiFloorParams(), 0.1, 0.18, 0.88)._surfaces(obs)
    assert bool(horizontal.any()) == is_support


def test_tread_above_old_floor_samples_is_not_projected_underneath():
    terrain = terrain_with_surface(lambda x: max(0, x // 4) * 0.18 - 0.05)
    for x in range(4, 25):
        for y in range(-6, 7):
            terrain.samples[(x, y, -1)] = (-0.10, 0, True)
    terrain._rasterize(AgentPose(0, 0, 0, 0))
    x, y = terrain.world.world_to_grid(1.25, 0.05)
    assert terrain.heights[y, x] > 0.4
    assert any(c["direction"] == 1 for c in terrain.candidates)


def test_vertical_navmesh_correction_is_bounded_and_not_a_val_protocol_change():
    from sparx_agency.core.planning.objnav.errors import EnvContractError
    from sparx_agency.tasks.planning.objnav_benchmark.kinematics import check_motion
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
    before = AgentPose(0, 0, 0, 0)
    corrected = AgentPose(0.25, 0, 0.269, 0)
    check_motion(DiscreteAction.MOVE_FORWARD, before, corrected, MULTIFLOOR_PROTOCOL.actions(), MULTIFLOOR_PROTOCOL.kinematics())
    for dz in (0.305, 0.52):
        check_motion(DiscreteAction.MOVE_FORWARD, before, AgentPose(0.25, 0, dz, 0), MULTIFLOOR_PROTOCOL.actions(), MULTIFLOOR_PROTOCOL.kinematics())
    assert MULTIFLOOR_PROTOCOL.max_native_vertical_m == 0.60
    with pytest.raises(EnvContractError):
        check_motion(DiscreteAction.MOVE_FORWARD, before, corrected, PROTOCOL.actions(), PROTOCOL.kinematics())
    with pytest.raises(EnvContractError):
        check_motion(DiscreteAction.MOVE_FORWARD, before, AgentPose(0.25, 0, 1.0, 0), MULTIFLOOR_PROTOCOL.actions(), MULTIFLOOR_PROTOCOL.kinematics())


