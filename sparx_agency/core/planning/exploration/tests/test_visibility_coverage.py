"""What the camera can and cannot be said to have seen."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment.occupancy_io import occupancy_from_mask
from sparx_agency.core.planning.exploration.visibility_coverage import (
    SensorCone,
    VisibilityCoverage,
    cone_from_intrinsics,
)

RES = 0.1


def _room(width_cells=60, height_cells=60):
    """A rectangular room with a one-cell wall all the way round.

    Origin is at (0, 0), so world x runs 0..width_cells*RES and the interior is
    everything from one cell in.
    """
    occupied = np.zeros((height_cells, width_cells), dtype=bool)
    occupied[0, :] = occupied[-1, :] = True
    occupied[:, 0] = occupied[:, -1] = True
    return occupied


def _grid(occupied):
    return occupancy_from_mask(occupied, RES, 0.0, 0.0)


def _cone(half_fov_deg=37.5, max_range_m=8.0, forward_offset_m=0.0):
    return SensorCone(half_fov_rad=math.radians(half_fov_deg),
                      max_range_m=max_range_m,
                      forward_offset_m=forward_offset_m)


def test_the_denominator_is_the_room_and_excludes_the_wall():
    occupied = _room(20, 20)
    cov = VisibilityCoverage(_grid(occupied), _cone())
    assert cov.cells_total == 18 * 18
    assert cov.area_total_m2 == pytest.approx(18 * 18 * RES * RES)
    assert cov.cells_seen == 0
    assert cov.fraction_seen == 0.0


def test_a_full_spin_in_an_empty_room_sees_all_of_it():
    cov = VisibilityCoverage(_grid(_room(40, 40)), _cone())
    centre = 20 * RES
    for degrees in range(0, 360, 5):
        cov.observe(centre, centre, math.radians(degrees))
    assert cov.fraction_seen == pytest.approx(1.0, abs=0.02)


def test_a_wall_hides_what_is_behind_it():
    #  A room split by a partition with no opening. Standing on one side, the
    #  far side must stay at zero however long you look at it. The denominator
    #  is given explicitly as the whole floor, because the default would notice
    #  the two halves are separate rooms and measure only the one we are in --
    #  which is right, and would hide the thing this test is about.
    occupied = _room(60, 40)
    occupied[:, 30] = True
    grid = _grid(occupied)
    cov = VisibilityCoverage(grid, _cone(), countable=~occupied)
    for degrees in range(0, 360, 5):
        cov.observe(15 * RES, 20 * RES, math.radians(degrees))

    seen = cov.seen_mask
    assert seen[:, 1:30].any(), "the near side should be covered"
    assert not seen[:, 31:].any(), "nothing behind the partition may be seen"
    # Near side 29 columns, far side 28, so a perfect near side is just over half.
    assert 0.45 < cov.fraction_seen < 0.55


def test_a_doorway_lets_a_sliver_of_the_next_room_through():
    #  The measure has to reward looking INTO a room from its doorway, which is
    #  most of what an exploration flight actually does.
    occupied = _room(60, 40)
    occupied[:, 30] = True
    occupied[18:23, 30] = False              # a 0.5 m opening
    cov = VisibilityCoverage(_grid(occupied), _cone())
    cov.observe(25 * RES, 20 * RES, 0.0)     # 0.5 m back from the doorway
    seen = cov.seen_mask
    assert seen[20, 40], "straight through the opening is seen"
    assert not seen[12, 40], "and the far room in the doorway's shadow is not"
    beyond = seen[:, 31:].sum()
    assert 0 < beyond < int((~occupied)[:, 31:].sum()), \
        "a doorway shows part of the next room, never all of it"


def test_the_cone_is_a_wedge_and_not_a_disc():
    cov = VisibilityCoverage(_grid(_room(60, 60)), _cone(half_fov_deg=20.0))
    centre = 30 * RES
    cov.observe(centre, centre, 0.0)          # looking towards +x
    seen = cov.seen_mask
    assert seen[30, 45], "straight ahead is seen"
    assert not seen[30, 15], "directly behind is not"
    assert not seen[50, 30], "90 degrees to the left is not"


def test_range_is_a_hard_limit():
    cov = VisibilityCoverage(_grid(_room(120, 20)), _cone(max_range_m=2.0))
    cov.observe(1.0, 10 * RES, 0.0)           # x = 1.0 m, looking towards +x
    seen = cov.seen_mask
    assert seen[10, 25], "0.5 m ahead is within range"
    assert not seen[10, 45], "3.5 m ahead is past the 2 m clip"


def test_the_forward_offset_moves_the_eye():
    #  The camera sits ahead of the body, so a body origin pressed against the
    #  left wall still has its eye inside the room and can see.
    occupied = _room(40, 40)
    grid = _grid(occupied)
    at_body = VisibilityCoverage(grid, _cone(forward_offset_m=0.0))
    offset = VisibilityCoverage(grid, _cone(forward_offset_m=0.5))
    pose = (0.05, 20 * RES, 0.0)              # inside the wall cell itself
    assert at_body.observe(*pose) == 0        # eye in a wall sees nothing
    assert offset.observe(*pose) > 0


def test_looking_again_at_the_same_place_adds_nothing():
    cov = VisibilityCoverage(_grid(_room(40, 40)), _cone())
    first = cov.observe(20 * RES, 20 * RES, 0.0)
    assert first > 0
    assert cov.observe(20 * RES, 20 * RES, 0.0) == 0
    assert cov.cells_seen == first


def test_a_pose_that_is_not_a_pose_is_ignored():
    cov = VisibilityCoverage(_grid(_room(20, 20)), _cone())
    assert cov.observe(float("nan"), 1.0, 0.0) == 0
    assert cov.observe(1.0, float("inf"), 0.0) == 0
    assert cov.observe(1.0, 1.0, float("nan")) == 0
    assert cov.cells_seen == 0


def test_a_pose_off_the_map_sees_nothing():
    cov = VisibilityCoverage(_grid(_room(20, 20)), _cone())
    assert cov.observe(-50.0, -50.0, 0.0) == 0


def test_contains_separates_the_floor_from_everywhere_else():
    cov = VisibilityCoverage(_grid(_room(20, 20)), _cone())
    assert cov.contains(1.0, 1.0)
    assert not cov.contains(0.05, 1.0), "inside the wall"
    assert not cov.contains(-1.0, 1.0), "off the map"


def test_a_sealed_cavity_never_counts_against_coverage():
    #  A closed cupboard reads as free on a map computed from geometry, and
    #  nothing can ever see into it. Counting it would cap coverage below 100%.
    occupied = _room(40, 40)
    occupied[18:23, 18:23] = True
    occupied[19:22, 19:22] = False            # a 3x3 sealed void
    cov = VisibilityCoverage(_grid(occupied), _cone())
    assert cov.cells_total == 38 * 38 - 5 * 5
    assert not cov.countable_mask[20, 20]


def test_an_explicit_denominator_is_honoured():
    occupied = _room(20, 20)
    only = np.zeros(occupied.shape, dtype=bool)
    only[5:10, 5:10] = True
    cov = VisibilityCoverage(_grid(occupied), _cone(), countable=only)
    assert cov.cells_total == 25


def test_a_denominator_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError):
        VisibilityCoverage(_grid(_room(20, 20)), _cone(),
                           countable=np.ones((5, 5), dtype=bool))


def test_a_map_with_no_enclosed_region_is_refused():
    with pytest.raises(ValueError):
        VisibilityCoverage(_grid(np.zeros((10, 10), dtype=bool)), _cone())


def test_cone_from_intrinsics_reproduces_the_sjtu_front_camera():
    cone = cone_from_intrinsics(width=600, fx=390.642735, max_range_m=10.0,
                                forward_offset_m=0.2)
    assert math.degrees(2.0 * cone.half_fov_rad) == pytest.approx(75.046, abs=1e-3)
    assert cone.max_range_m == 10.0
    assert cone.forward_offset_m == 0.2


def test_summary_reads_as_a_percentage_and_an_area():
    cov = VisibilityCoverage(_grid(_room(20, 20)), _cone())
    assert "0.0% seen" in cov.summary()
    assert "m2" in cov.summary()
