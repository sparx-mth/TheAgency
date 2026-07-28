"""The interactive voxel viewer, exercised without opening a window.

Everything here builds geometry or manipulates the environment; nothing calls
``create_window``, so the suite runs headless. Open3D is an optional dependency
(it is not installed in the Isaac container), so the parts that need it skip
rather than fail.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.core.planning.environment import VoxelGrid3D
from sparx_agency.core.planning.environment.voxel_grid_3d import FREE, OCCUPIED
from sparx_agency.tasks.planning.sim_flight_recording import view_voxel_map

o3d = pytest.importorskip("open3d", reason="open3d is an optional viewer dependency")


def _grid() -> VoxelGrid3D:
    voxels = np.full((20, 10, 10), FREE, dtype=np.int8)
    voxels[0, :, :] = OCCUPIED        # floor at z in [0.0, 0.1)
    voxels[19, :, :] = OCCUPIED       # ceiling at z in [1.9, 2.0)
    voxels[0:8, 4, 4] = OCCUPIED      # a column standing on the floor
    return VoxelGrid3D(voxels, 0.1, (0.0, 0.0, 0.0))


def _write_recording(root, frames: int = 5, columns: int = 15):
    """A minimal recording on disk, enough for the viewer to draw a track."""
    (root / "depth").mkdir(parents=True, exist_ok=True)
    poses = np.zeros((frames, columns), np.float32)
    poses[:, 0] = np.arange(frames) * 0.1          # t
    poses[:, 1] = np.arange(frames) * 0.5          # x
    poses[:, 2] = 1.0                              # y
    if columns > 4:
        poses[:, 4] = 1.5                          # z
    np.save(root / "poses.npy", poses)
    (root / "intrinsics.json").write_text(json.dumps(
        {"width": 4, "height": 4, "fx": 2.0, "fy": 2.0, "cx": 2.0, "cy": 2.0}))
    (root / "meta.json").write_text(json.dumps({"rate_hz": 10.0, "frames": frames}))
    return root


def test_points_are_the_default_because_a_million_cubes_will_not_orbit():
    points = _grid().occupied_points()
    geometry = view_voxel_map.voxel_geometry(points, 0.1, cubes=False)
    assert isinstance(geometry, o3d.geometry.PointCloud)
    assert len(geometry.points) == len(points)
    assert len(geometry.colors) == len(points)


def test_cubes_produce_a_real_voxel_grid():
    geometry = view_voxel_map.voxel_geometry(_grid().occupied_points(), 0.1, cubes=True)
    assert isinstance(geometry, o3d.geometry.VoxelGrid)
    assert geometry.voxel_size == pytest.approx(0.1)


def test_colours_are_normalised_for_open3d():
    """Open3D wants 0-1 floats; height_colours produces 0-255 bytes."""
    geometry = view_voxel_map.voxel_geometry(_grid().occupied_points(), 0.1, cubes=False)
    colours = np.asarray(geometry.colors)
    assert colours.min() >= 0.0 and colours.max() <= 1.0
    assert colours.max() > 0.1, "an all-black cloud means the scaling is wrong"


def test_a_flight_becomes_a_line_and_two_end_markers(tmp_path):
    _write_recording(tmp_path / "flight_a")
    geometries = view_voxel_map.flight_geometries(tmp_path)

    assert len(geometries) == 3
    lines = [g for g in geometries if isinstance(g, o3d.geometry.LineSet)]
    assert len(lines) == 1
    assert len(lines[0].points) == 5
    assert len(lines[0].lines) == 4


def test_several_flights_are_all_drawn(tmp_path):
    for name in ("a", "b", "c"):
        _write_recording(tmp_path / f"flight_{name}")
    assert len(view_voxel_map.flight_geometries(tmp_path)) == 9


def test_a_legacy_four_column_recording_is_skipped_not_crashed_on(tmp_path):
    """Rosbag extractions have no altitude column; there is no 3D track to draw."""
    _write_recording(tmp_path / "legacy", columns=4)
    assert view_voxel_map.flight_geometries(tmp_path) == []


def test_the_drawn_track_can_be_lifted_clear_of_the_floor(tmp_path):
    _write_recording(tmp_path / "flight")
    plain = view_voxel_map.flight_geometries(tmp_path, altitude_offset=0.0)
    lifted = view_voxel_map.flight_geometries(tmp_path, altitude_offset=0.5)

    low = np.asarray([g for g in plain if isinstance(g, o3d.geometry.LineSet)][0].points)
    high = np.asarray([g for g in lifted if isinstance(g, o3d.geometry.LineSet)][0].points)
    np.testing.assert_allclose(high[:, 2] - low[:, 2], 0.5)


def test_the_viewer_starts_with_the_requested_cut():
    viewer = view_voxel_map.VoxelMapViewer(_grid(), max_z=1.0)
    assert viewer.max_z == 1.0
    assert len(viewer.points) == int(_grid().occupied.sum())


class _FakeControl:
    """Just enough view control for _rebuild's save-and-restore of the camera."""

    def __init__(self):
        self.restored = 0

    def convert_to_pinhole_camera_parameters(self):
        return "camera"

    def convert_from_pinhole_camera_parameters(self, camera, allow_arbitrary=False):
        assert camera == "camera", "the camera must survive the geometry swap"
        self.restored += 1


class _FakeVis:
    """A visualizer that records geometry swaps instead of drawing them."""

    def __init__(self):
        self.control = _FakeControl()
        self.geometries = []

    def get_view_control(self):
        return self.control

    def add_geometry(self, geometry, reset_bounding_box=True):
        self.geometries.append(geometry)

    def remove_geometry(self, geometry, reset_bounding_box=True):
        self.geometries.remove(geometry)


def test_lowering_the_cut_hides_the_ceiling():
    viewer = view_voxel_map.VoxelMapViewer(_grid())
    vis = _FakeVis()
    viewer._rebuild(vis)
    with_ceiling = len(vis.geometries[0].points)

    viewer._move_cut(-view_voxel_map.CLIP_STEP_M)(vis)
    viewer._move_cut(-view_voxel_map.CLIP_STEP_M)(vis)

    assert viewer.max_z == pytest.approx(1.6)
    assert len(vis.geometries) == 1, "the old geometry must be removed, not stacked"
    assert len(vis.geometries[0].points) < with_ceiling


def test_the_cut_cannot_be_pushed_through_the_floor():
    grid = _grid()
    viewer = view_voxel_map.VoxelMapViewer(grid)
    vis = _FakeVis()
    for _ in range(50):
        viewer._move_cut(-view_voxel_map.CLIP_STEP_M)(vis)
    assert viewer.max_z == pytest.approx(grid.origin_z)


def test_the_cut_cannot_be_raised_above_the_map():
    grid = _grid()
    viewer = view_voxel_map.VoxelMapViewer(grid, max_z=1.0)
    vis = _FakeVis()
    for _ in range(50):
        viewer._move_cut(+view_voxel_map.CLIP_STEP_M)(vis)
    assert viewer.max_z == pytest.approx(grid.origin_z + grid.depth * grid.resolution)


def test_an_empty_cut_still_draws_something_rather_than_crashing():
    """Cutting below every voxel must not hand Open3D a zero-point cloud."""
    viewer = view_voxel_map.VoxelMapViewer(_grid(), max_z=-5.0)
    vis = _FakeVis()
    viewer._rebuild(vis)
    assert len(vis.geometries[0].points) == 1


def test_the_camera_is_restored_around_every_rebuild():
    viewer = view_voxel_map.VoxelMapViewer(_grid())
    vis = _FakeVis()
    viewer._toggle_cubes(vis)
    assert viewer.cubes is True
    assert vis.control.restored == 1
    assert isinstance(vis.geometries[0], o3d.geometry.VoxelGrid)


def test_x11_is_preferred_only_when_there_is_an_x_display(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":1")
    assert view_voxel_map.prefer_x11() is True
    assert "WAYLAND_DISPLAY" not in __import__("os").environ


def test_a_wayland_session_with_no_x_display_is_left_alone(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert view_voxel_map.prefer_x11() is False
    assert __import__("os").environ["WAYLAND_DISPLAY"] == "wayland-0"


def test_an_x11_session_needs_no_change(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    assert view_voxel_map.prefer_x11() is False
