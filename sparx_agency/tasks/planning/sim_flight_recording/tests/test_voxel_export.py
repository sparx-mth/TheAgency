"""Exporting a voxel map to something a human can open.

The PLY must be readable by Open3D on a machine that has it, written from one
that does not -- so the format is checked byte by byte rather than by loading it.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.environment import VoxelGrid3D
from sparx_agency.core.planning.environment.voxel_grid_3d import FREE, OCCUPIED
from sparx_agency.tasks.planning.sim_flight_recording.voxel_export import (
    clip_height, export_voxel_grid, height_colours, render_isometric, write_ply,
)

HEADER_FIELDS = [b"ply", b"format binary_little_endian 1.0", b"property float x",
                 b"property uchar red", b"end_header"]


def _grid() -> VoxelGrid3D:
    voxels = np.full((20, 10, 10), FREE, dtype=np.int8)
    voxels[0, :, :] = OCCUPIED       # floor
    voxels[19, :, :] = OCCUPIED      # ceiling
    voxels[0:8, 4, 4] = OCCUPIED     # something standing on the floor
    return VoxelGrid3D(voxels, 0.1, (0.0, 0.0, 0.0))


def test_the_ply_header_is_what_open3d_reads(tmp_path):
    path = write_ply(tmp_path / "c.ply", np.zeros((3, 3), np.float32))
    head = path.read_bytes()[:200]
    for field in HEADER_FIELDS:
        assert field in head, f"missing {field!r}"
    assert b"element vertex 3" in head


def test_the_payload_is_fifteen_bytes_a_point(tmp_path):
    """float x/y/z + uchar rgb, no padding -- anything else and Open3D misreads it."""
    points = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    path = write_ply(tmp_path / "c.ply", points)

    raw = path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    assert len(raw) - header_end == 500 * 15


def test_the_points_round_trip_exactly(tmp_path):
    points = np.array([[1.5, -2.25, 3.0], [0.0, 0.0, 0.0]], np.float32)
    colours = np.array([[10, 20, 30], [200, 100, 50]], np.uint8)
    path = write_ply(tmp_path / "c.ply", points, colours)

    raw = path.read_bytes()
    payload = raw[raw.index(b"end_header\n") + len(b"end_header\n"):]
    record = np.frombuffer(payload, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    np.testing.assert_array_equal(
        np.stack([record["x"], record["y"], record["z"]], axis=1), points)
    np.testing.assert_array_equal(
        np.stack([record["red"], record["green"], record["blue"]], axis=1), colours)


def test_mismatched_colours_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        write_ply(tmp_path / "c.ply", np.zeros((3, 3)), np.zeros((2, 3), np.uint8))


def test_an_empty_cloud_still_writes_a_valid_file(tmp_path):
    path = write_ply(tmp_path / "c.ply", np.zeros((0, 3), np.float32))
    assert b"element vertex 0" in path.read_bytes()


def test_colours_run_low_to_high_with_height():
    points = np.array([[0, 0, 0.0], [0, 0, 5.0]], np.float32)
    low, high = height_colours(points)
    assert not np.array_equal(low, high)
    assert high[0] > low[0], "high should be the warm end"


def test_clipping_the_ceiling_is_what_makes_an_indoor_map_legible():
    """Without it the largest surface in the building is between you and the rest."""
    grid = _grid()
    everything = grid.occupied_points()
    clipped = clip_height(everything, max_z=1.5)

    assert len(clipped) < len(everything)
    assert clipped[:, 2].max() <= 1.5
    assert len(clipped) > 0, "the floor and the furniture must survive"


def test_clipping_the_floor_away_works_too():
    clipped = clip_height(_grid().occupied_points(), min_z=1.0)
    assert clipped[:, 2].min() >= 1.0


def test_clipping_nothing_keeps_everything():
    points = _grid().occupied_points()
    np.testing.assert_array_equal(clip_height(points), points)


def test_exporting_a_grid_honours_the_height_band(tmp_path):
    path = export_voxel_grid(_grid(), tmp_path / "c.ply", max_z=1.0)
    raw = path.read_bytes()
    full = export_voxel_grid(_grid(), tmp_path / "full.ply").read_bytes()
    assert len(raw) < len(full)


def test_the_isometric_render_produces_a_picture():
    image = render_isometric(_grid(), width=200, max_z=1.5)
    assert image.ndim == 3 and image.shape[1] == 200
    assert image.std() > 0, "a blank canvas means nothing was drawn"


def test_the_isometric_render_survives_an_empty_grid():
    empty = VoxelGrid3D(np.full((4, 4, 4), FREE, np.int8), 0.1, (0, 0, 0))
    assert render_isometric(empty, width=100).shape[1] == 100
