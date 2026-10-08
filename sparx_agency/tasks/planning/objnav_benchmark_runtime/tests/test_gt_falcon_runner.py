"""Ground-truth POV scoring and HM3D asset layout for the standalone runner."""
from types import SimpleNamespace
import csv

import pytest
import numpy as np

from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.run_gt_falcon import (
    GoalView, resolve_scene_assets, target_view_error, target_view_match,
    final_video_labels, target_view_outcome, write_occupancy_map_png,
    write_trajectory_csv,
)


def test_target_view_scoring_checks_position_and_heading():
    pose = SimpleNamespace(position=lambda: (1.0, 2.0, 0.0), yaw=0.0)
    views = (GoalView(1.2, 2.0, 0.0, 0.1, (0, 0, 0), (1, 0, 0, 0)),
             GoalView(4.0, 2.0, 0.0, 2.0, (0, 0, 0), (1, 0, 0, 0)))
    distance, heading = target_view_error(pose, views)
    assert distance == pytest.approx(0.2)
    assert heading == pytest.approx(0.1)


def test_target_pov_accepts_any_view_meeting_both_thresholds():
    pose = SimpleNamespace(position=lambda: (1.0, 2.0, 0.0), yaw=0.0)
    views = (GoalView(1.1, 2.0, 0.0, 2.0, (0, 0, 0), (1, 0, 0, 0)),
             GoalView(1.4, 2.0, 0.0, 0.2, (0, 0, 0), (1, 0, 0, 0)))
    matched, (distance, heading) = target_view_match(pose, views, 0.5, 0.3)
    assert matched
    assert distance == pytest.approx(0.4)
    assert heading == pytest.approx(0.2)


def test_target_pov_hit_after_action_is_successful_termination():
    pose = SimpleNamespace(position=lambda: (1.4, 2.0, 0.0), yaw=0.2)
    success, termination, error = target_view_outcome(
        pose, (GoalView(1.0, 2.0, 0.0, 0.0, (0, 0, 0), (1, 0, 0, 0)),), 0.5, 0.3)
    assert success and termination == "gt_viewpoint"
    assert error[0] == pytest.approx(0.4)


def test_success_video_frame_is_explicitly_labeled():
    labels = final_video_labels(True, "plant", 0.2, 12.0)
    assert labels[0] == "SUCCESS: TARGET POV REACHED"
    assert "plant" in labels[1] and "0.20 m" in labels[1]


def test_scene_assets_resolve_v2_release_alias(tmp_path):
    episode = SimpleNamespace(scene_relative_path=(
        "hm3d_v0.2/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"))
    scene = tmp_path / "hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"mesh")
    scene.with_name("TEEsavR23oF.basis.navmesh").write_bytes(b"navmesh")
    resolved_scene, resolved_navmesh = resolve_scene_assets(episode, tmp_path, "hm3d")
    assert resolved_scene == scene
    assert resolved_navmesh.is_file()


def test_trajectory_csv_contains_each_action_pose(tmp_path):
    path = tmp_path / "trajectory.csv"
    rows = [{"step": 0, "action": "START", "x": 1.0, "y": 2.0, "z": 0.0,
             "yaw": 0.1, "target_distance_m": 3.0, "target_yaw_error_deg": 4.0,
             "planner_status": "not_started"}]
    write_trajectory_csv(path, rows)
    with path.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[0]["action"] == "START"
    assert float(saved[0]["x"]) == pytest.approx(1.0)


def test_occupancy_map_render_includes_route(tmp_path):
    grid = np.full((20, 20), -1, dtype=np.int8)
    grid[8:13, 8:13] = 0
    grid[8:10, 10] = 100
    world = OccupancyGrid2D(
        grid, OccupancyGrid2DParams(0.2, 0.0, 0.0, "world"),
        values=OccupancyValues(free=0, occupied=100, unknown=-1))
    trace = [{"x": 1.0, "y": 1.0}, {"x": 1.2, "y": 1.0}]
    routes = [{"route_xy_world": [[1.0, 1.0], [1.2, 1.0]]}]
    plans = [{"connectivity_groups_world": [
        {"id": 0, "unknown": False, "center_xy": [1.0, 1.0]},
        {"id": 1, "unknown": True, "center_xy": [1.6, 1.0]}],
        "connectivity_edges_world": [{"from": 0, "to": 1, "cost_m": 0.6}],
        "frontier_clusters_world": [{"boundary_samples_xy": [[1.4, 1.2]]}],
        "viewpoint_candidates_world": [{"xy": [1.2, 1.2]}]}]
    path = tmp_path / "occupancy_map.png"
    write_occupancy_map_png(path, world, trace, routes, plans)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")