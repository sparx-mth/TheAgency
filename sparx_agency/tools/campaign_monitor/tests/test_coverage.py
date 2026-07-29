"""Coverage is the stopping criterion, so it has to mean what it claims.

Two ways it could quietly lie: counting free space no flight can reach, which
puts a ceiling below 100 % that reads as a plateau; and counting a corridor
flown ten times the same way as well covered, when the camera has seen it once.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.tools.campaign_monitor import coverage


def _map(tmp_path, grid, resolution=0.1, origin=(0.0, 0.0)):
    path = tmp_path / "scene_alt0150cm.npz"
    np.savez_compressed(path, grid=grid.astype(np.int16),
                        resolution=np.array(resolution),
                        origin=np.asarray(origin, dtype=np.float64))
    return path


def _flight(directory, scene, xs, ys, yaws):
    directory.mkdir(parents=True, exist_ok=True)
    n = len(xs)
    poses = np.zeros((n, 21), dtype=np.float32)
    poses[:, 0] = np.arange(n) * 0.1
    poses[:, 1] = xs
    poses[:, 2] = ys
    poses[:, 3] = yaws
    np.save(directory / "poses.npy", poses)
    (directory / "meta.json").write_text(json.dumps({"scene": scene}))


def test_unreachable_free_space_is_not_counted(tmp_path):
    """A sealed room is free but no flight can get there.

    ``free_space_sampler`` draws goals from the largest connected component
    only, so counting the rest would cap coverage below 100 % forever and look
    exactly like the campaign having saturated.
    """
    grid = np.ones((40, 40), np.int16)
    grid[2:18, 2:18] = 0          # the reachable hall
    grid[25:35, 25:35] = 0        # a sealed room, not connected to it
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    mask, _, _ = coverage.reachable_mask(_map(maps_dir, grid))
    assert mask.sum() == 16 * 16, "only the largest connected component counts"


def test_flying_the_same_line_both_ways_doubles_heading_coverage(tmp_path):
    """The number that says whether re-flying a corridor was worth anything."""
    grid = np.zeros((40, 40), np.int16)
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    _map(maps_dir, grid)
    scenes = {"scene": maps_dir / "scene_alt0150cm.npz"}

    root = tmp_path / "campaign"
    xs = np.linspace(0.5, 3.5, 40)
    ys = np.full(40, 1.5)
    _flight(root / "a", "scene", xs, ys, np.zeros(40))
    one_way = coverage.measure(root, scenes, pose_stride=1)[0]

    _flight(root / "b", "scene", xs[::-1], ys, np.full(40, np.pi))
    both_ways = coverage.measure(root, scenes, pose_stride=1)[0]

    assert both_ways.cells_seen == one_way.cells_seen, "same cells, other direction"
    assert both_ways.mean_headings > one_way.mean_headings
    assert both_ways.flights == 2


def test_coverage_is_a_fraction_of_reachable_not_of_the_map(tmp_path):
    grid = np.ones((40, 40), np.int16)
    grid[0:20, 0:20] = 0
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    _map(maps_dir, grid)
    scenes = {"scene": maps_dir / "scene_alt0150cm.npz"}

    root = tmp_path / "campaign"
    _flight(root / "a", "scene", np.full(20, 0.5), np.linspace(0.1, 1.9, 20),
            np.zeros(20))
    result = coverage.measure(root, scenes, pose_stride=1)[0]
    assert 0.0 < result.fraction < 1.0
    assert result.cells_reachable == 4      # a 2x2 m hall at one-metre cells


def test_a_scene_with_no_recordings_is_omitted(tmp_path):
    grid = np.zeros((20, 20), np.int16)
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    _map(maps_dir, grid)
    root = tmp_path / "campaign"
    root.mkdir()
    assert coverage.measure(root, {"scene": maps_dir / "scene_alt0150cm.npz"}) == []


def test_a_recording_naming_an_unsurveyed_scene_is_ignored(tmp_path):
    """A campaign may hold buildings this map directory does not have."""
    grid = np.zeros((20, 20), np.int16)
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    _map(maps_dir, grid)
    root = tmp_path / "campaign"
    _flight(root / "a", "somewhere_else", np.zeros(5), np.zeros(5), np.zeros(5))
    assert coverage.measure(root, {"scene": maps_dir / "scene_alt0150cm.npz"}) == []


def test_default_scene_maps_finds_the_surveyed_buildings():
    """Reads the directory, so a newly surveyed scene needs no code change."""
    found = coverage.default_scene_maps()
    assert "office" in found
    assert all(path.is_file() for path in found.values())
