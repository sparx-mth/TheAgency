"""End-to-end tests: a hand-made world in, three map artifacts out."""
from __future__ import annotations

import textwrap

import numpy as np
import pytest
import yaml

from sparx_agency.core.planning.environment.occupancy_io import load_occupancy_grid
from sparx_agency.tasks.mapping.gazebo_world_occupancy import nav2_map
from sparx_agency.tasks.mapping.gazebo_world_occupancy.build_map import main

RESOLUTION = 0.1
Z_MIN = 0.3
Z_MAX = 2.0
MARGIN = 1.0

# Both boxes stand from the floor to 3 m, so they cross the whole band and
# neither lid lands inside it -- a lid at exactly z_max would fill the
# footprint solid and hide whether the walls were drawn at all.
BOX_A = dict(name="box_a", x=0.0, y=0.0, z=1.5, sx=1.0, sy=1.0, sz=3.0)
BOX_B = dict(name="box_b", x=5.0, y=3.0, z=1.5, sx=2.0, sy=1.0, sz=3.0)


def _box_model(spec):
    """One world-level <model> holding a single box collision."""
    return textwrap.dedent(
        """
        <model name="%(name)s">
          <pose>%(x)s %(y)s %(z)s 0 0 0</pose>
          <link name="link">
            <collision name="collision">
              <geometry><box><size>%(sx)s %(sy)s %(sz)s</size></box></geometry>
            </collision>
          </link>
        </model>
        """
        % spec
    )


def _write_world(tmp_path, models, name="two_boxes"):
    """Write a world file made of the given <model> blocks."""
    worlds = tmp_path / "worlds"
    worlds.mkdir(parents=True, exist_ok=True)
    path = worlds / (name + ".world")
    path.write_text(
        '<sdf version="1.6">\n  <world name="w">\n%s\n  </world>\n</sdf>\n'
        % ("\n".join(models),)
    )
    return path


def _build(tmp_path, world, extra_args=()):
    """Run the CLI into ``tmp_path/out`` and return the output directory."""
    out = tmp_path / "out"
    argv = [
        "--world", str(world),
        "--output-dir", str(out),
        "--resolution", str(RESOLUTION),
        "--z-min", str(Z_MIN),
        "--z-max", str(Z_MAX),
        "--margin", str(MARGIN),
    ]
    assert main(argv + list(extra_args)) == 0
    return out


def _occupied_world_bbox(grid):
    """World-space bounding box of the occupied cells, measured at cell centres.

    Centres rather than edges, so a wall lying exactly on a cell boundary is
    half a cell away from the answer either way instead of a whole cell away
    on one side.
    """
    rows, cols = np.nonzero(grid.grid == grid.values.occupied)
    res = grid.resolution
    return (
        grid.origin_x + (cols.min() + 0.5) * res,
        grid.origin_x + (cols.max() + 0.5) * res,
        grid.origin_y + (rows.min() + 0.5) * res,
        grid.origin_y + (rows.max() + 0.5) * res,
    )


def test_two_box_world_puts_occupied_cells_where_the_boxes_are(tmp_path):
    world = _write_world(tmp_path, [_box_model(BOX_A), _box_model(BOX_B)])
    out = _build(tmp_path, world)

    grid, metadata, _layers = load_occupancy_grid(out / "two_boxes.npz")

    # The union footprint is x [-0.5, 6.0], y [-0.5, 3.5].
    min_x, max_x, min_y, max_y = _occupied_world_bbox(grid)
    assert min_x == pytest.approx(-0.5, abs=RESOLUTION)
    assert max_x == pytest.approx(6.0, abs=RESOLUTION)
    assert min_y == pytest.approx(-0.5, abs=RESOLUTION)
    assert max_y == pytest.approx(3.5, abs=RESOLUTION)

    # A point on box A's south wall is occupied ...
    wall_x, wall_y = grid.world_to_grid(0.0, -0.5)
    assert grid.grid[wall_y, wall_x] == grid.values.occupied
    # ... the empty floor between the two boxes is free ...
    gap_x, gap_y = grid.world_to_grid(2.5, 1.5)
    assert grid.grid[gap_y, gap_x] == grid.values.free
    # ... and nothing is unknown, because this is ground truth.
    assert not np.any(grid.grid == grid.values.unknown)
    assert metadata["unknown_cells"] is False


def test_metadata_records_the_provenance(tmp_path):
    world = _write_world(tmp_path, [_box_model(BOX_A), _box_model(BOX_B)])
    out = _build(tmp_path, world)
    _grid, metadata, _layers = load_occupancy_grid(out / "two_boxes.npz")

    assert metadata["world_path"] == str(world.resolve())
    assert len(metadata["world_sha256"]) == 64
    assert metadata["resolution"] == pytest.approx(RESOLUTION)
    assert metadata["z_min"] == pytest.approx(Z_MIN)
    assert metadata["z_max"] == pytest.approx(Z_MAX)
    assert metadata["instance_count"] == 2
    assert metadata["triangle_count"] == 24
    assert metadata["frame_id"] == "map"


def test_a_closed_box_leaves_a_hollow_watertight_footprint(tmp_path):
    """A box is a *surface*: its inside is enclosed, not filled."""
    world = _write_world(tmp_path, [_box_model(BOX_B)], name="one_box")
    out = _build(tmp_path, world)
    grid, _metadata, _layers = load_occupancy_grid(out / "one_box.npz")

    centre_x, centre_y = grid.world_to_grid(5.0, 3.0)
    assert grid.grid[centre_y, centre_x] == grid.values.free

    # Every straight line out of the centre must cross an occupied cell, so
    # the interior is unreachable even though it reads as free.
    row = grid.grid[centre_y, :]
    column = grid.grid[:, centre_x]
    assert (row[:centre_x] == grid.values.occupied).any()
    assert (row[centre_x:] == grid.values.occupied).any()
    assert (column[:centre_y] == grid.values.occupied).any()
    assert (column[centre_y:] == grid.values.occupied).any()


def test_geometry_outside_the_height_band_is_excluded(tmp_path):
    """A 0.2 m kerb is not an obstacle to something flying at 1.2 m."""
    kerb = dict(BOX_A, name="kerb", z=0.1, sz=0.2)
    world = _write_world(tmp_path, [_box_model(kerb), _box_model(BOX_B)], name="kerb")
    out = _build(tmp_path, world)
    grid, metadata, _layers = load_occupancy_grid(out / "kerb.npz")

    assert metadata["instance_count"] == 1
    min_x, _max_x, min_y, _min_y = _occupied_world_bbox(grid)
    assert min_x == pytest.approx(4.0, abs=RESOLUTION)
    assert min_y == pytest.approx(2.5, abs=RESOLUTION)


def test_nav2_pgm_is_written_top_down_and_matches_the_grid(tmp_path):
    """PGM row 0 is maximum y; the npz grid's row 0 is minimum y."""
    world = _write_world(tmp_path, [_box_model(BOX_A), _box_model(BOX_B)])
    out = _build(tmp_path, world)

    grid, _metadata, _layers = load_occupancy_grid(out / "two_boxes.npz")
    image = nav2_map.read_pgm(out / "two_boxes.pgm")
    assert image.shape == grid.grid.shape

    occupied = grid.grid == grid.values.occupied
    np.testing.assert_array_equal(np.flipud(image) == nav2_map.OCCUPIED_PIXEL, occupied)
    assert set(np.unique(image)) == {nav2_map.OCCUPIED_PIXEL, nav2_map.FREE_PIXEL}

    # Box B's north wall is the highest occupied y, so it is the first PGM row.
    first_row = int(np.argmax((image == nav2_map.OCCUPIED_PIXEL).any(axis=1)))
    top_y = grid.origin_y + (grid.height - first_row - 0.5) * grid.resolution
    assert top_y == pytest.approx(3.5, abs=grid.resolution)


def test_nav2_yaml_has_the_map_server_fields(tmp_path):
    world = _write_world(tmp_path, [_box_model(BOX_A), _box_model(BOX_B)])
    out = _build(tmp_path, world)
    document = yaml.safe_load((out / "two_boxes.yaml").read_text())
    grid, _metadata, _layers = load_occupancy_grid(out / "two_boxes.npz")

    assert document["image"] == "two_boxes.pgm"
    assert document["resolution"] == pytest.approx(RESOLUTION)
    assert document["origin"] == pytest.approx([grid.origin_x, grid.origin_y, 0.0])
    assert document["negate"] == 0
    assert document["occupied_thresh"] == pytest.approx(0.65)
    assert document["mode"] == "trinary"

    # nav2 decodes a pixel as occ = 1 - pixel/255 and calls it free below
    # free_thresh. The unknown grey has to stay on the unknown side of that,
    # or a map_server hands back open space wherever nothing was surveyed.
    assert document["free_thresh"] <= 1.0 - nav2_map.UNKNOWN_PIXEL / 255.0
    assert document["free_thresh"] > 1.0 - nav2_map.FREE_PIXEL / 255.0


def test_the_robot_is_kept_out_of_its_own_map(tmp_path):
    drone = dict(BOX_A, name="sjtu_drone")
    world = _write_world(tmp_path, [_box_model(drone), _box_model(BOX_B)], name="w")
    out = _build(tmp_path, world)
    _grid, metadata, _layers = load_occupancy_grid(out / "w.npz")
    assert metadata["skipped_models"] == ["sjtu_drone"]
    assert metadata["instance_count"] == 1


def test_a_bare_world_name_resolves_against_a_search_path(tmp_path):
    world = _write_world(tmp_path, [_box_model(BOX_A)], name="named")
    out = tmp_path / "out"
    assert main([
        "--world", "named",
        "--search-path", str(world.parent),
        "--output-dir", str(out),
        "--resolution", str(RESOLUTION),
    ]) == 0
    assert (out / "named.pgm").exists()


def test_an_empty_band_fails_loudly(tmp_path):
    """Better a crash on the ground than a blank map that looks flyable."""
    kerb = dict(BOX_A, name="kerb", z=0.1, sz=0.2)
    world = _write_world(tmp_path, [_box_model(kerb)], name="empty")
    with pytest.raises(ValueError):
        _build(tmp_path, world)


def test_strict_mode_rejects_an_unresolved_model(tmp_path):
    include = "<include><uri>model://Ward</uri></include>"
    world = _write_world(tmp_path, [_box_model(BOX_A), include], name="missing")
    with pytest.raises(FileNotFoundError):
        _build(tmp_path, world, extra_args=["--strict"])


def test_strict_mode_passes_over_gazebo_builtins(tmp_path):
    """Every world includes sun and ground_plane, and neither resolves here.

    If they trip it, --strict cannot be used on any real world at all, which
    is how a genuinely missing model went on reading as routine.
    """
    builtins = (
        "<include><uri>model://sun</uri></include>"
        "<include><uri>model://ground_plane</uri></include>"
    )
    world = _write_world(tmp_path, [_box_model(BOX_A), builtins], name="builtins")
    out = _build(tmp_path, world, extra_args=["--strict"])
    assert (out / "builtins.pgm").exists()


def test_a_missing_mesh_file_is_fatal_even_without_strict(tmp_path):
    """A wall whose mesh is gone must never come back as a warning."""
    models = tmp_path / "models"
    sdf = models / "Ward" / "model.sdf"
    sdf.parent.mkdir(parents=True, exist_ok=True)
    sdf.write_text(
        '<sdf version="1.6"><model name="Ward"><link name="link">'
        '<collision name="collision"><geometry><mesh>'
        "<uri>model://Ward/meshes/gone.dae</uri>"
        "</mesh></geometry></collision></link></model></sdf>\n"
    )
    include = "<include><uri>model://Ward</uri></include>"
    world = _write_world(tmp_path, [_box_model(BOX_A), include], name="gone")
    with pytest.raises(FileNotFoundError):
        _build(tmp_path, world, extra_args=["--search-path", str(models)])


def test_the_far_wall_is_inside_the_map_when_there_is_no_margin(tmp_path):
    """--margin 0 lands the east and north walls exactly on the extent.

    With the grid sized by ceil() they fell one cell outside it and were
    dropped in silence -- a hole in the map exactly where its edge wall is.
    """
    world = _write_world(tmp_path, [_box_model(BOX_A)], name="tight")
    out = tmp_path / "tight_out"
    assert main([
        "--world", str(world),
        "--output-dir", str(out),
        "--resolution", str(RESOLUTION),
        "--z-min", str(Z_MIN),
        "--z-max", str(Z_MAX),
        "--margin", "0",
    ]) == 0

    grid, _metadata, _layers = load_occupancy_grid(out / "tight.npz")
    occupied = grid.grid == grid.values.occupied
    assert occupied[:, -1].any(), "the east wall is off the right edge"
    assert occupied[-1, :].any(), "the north wall is off the top edge"
