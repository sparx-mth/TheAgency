"""The memory bill, and the volume rule that quietly writes it."""
from __future__ import annotations

import pytest

from sparx_agency.tasks.planning.falcon_pegasus.mapsize.area import Box, ExplorationArea
from sparx_agency.tasks.planning.falcon_pegasus.mapsize.memory import (
    TOTAL_BYTES_PER_VOXEL,
    grid_cost,
    implicit_resolution,
)

# The whole-office area, as the six run files describe it today. Repeated rather
# than imported from the sibling test: the tests directory is not a package.
OFFICE = ExplorationArea(
    building=(-23.0, -27.2, 5.1, 38.7),
    flight_band=(1.0, 2.2),
    vertical_extent=(-0.2, 2.4),
    resolution=0.10,
    margin=(2.0, 8.0, 2.0, 2.0),
)


def test_bytes_per_voxel_is_the_six_full_sized_arrays():
    """4 occupancy + 16 TSDF + 8 ESDF + two 8-byte scratch buffers + one bit."""
    assert TOTAL_BYTES_PER_VOXEL == pytest.approx(44.125)


def test_the_esdf_scratch_buffers_are_a_third_of_the_bill():
    """Both are only ever used over the local update region."""
    cost = grid_cost(Box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0), 0.2)
    scratch = sum(
        total for name, total, _ in cost.breakdown() if name.startswith("esdf_scratch")
    )
    assert scratch / cost.total_bytes == pytest.approx(16.0 / 44.125)


def test_office_run_costs_what_it_costs_today():
    """The whole-office grid at 10 cm, which is what the runs allocate now."""
    cost = grid_cost(OFFICE.map, OFFICE.resolution)
    assert cost.shape == (321, 759, 26)
    assert cost.megabytes == pytest.approx(266.4, rel=1e-2)


def test_doubling_the_resolution_costs_an_eighth():
    fine = grid_cost(OFFICE.map, 0.1)
    coarse = grid_cost(OFFICE.map, 0.2)
    assert coarse.total_bytes / fine.total_bytes == pytest.approx(0.125, rel=0.02)


def test_breakdown_sums_to_the_total():
    cost = grid_cost(OFFICE.map, 0.2)
    assert sum(total for _, total, _ in cost.breakdown()) == pytest.approx(
        cost.total_bytes
    )


def test_implicit_rule_picks_fine_below_the_threshold():
    assert implicit_resolution(3999.0) == 0.1
    assert implicit_resolution(4001.0) == 0.2


def test_a_smaller_exploration_box_can_cost_eight_times_more():
    """The trap: crossing the threshold downwards refines the whole grid."""
    big, small = 4500.0, 3500.0
    assert implicit_resolution(big) == 0.2
    assert implicit_resolution(small) == 0.1
    grid = Box(0.0, 0.0, 0.0, 60.0, 40.0, 12.0)
    cheap = grid_cost(grid, implicit_resolution(big))
    dear = grid_cost(grid, implicit_resolution(small))
    assert dear.total_bytes > 7.5 * cheap.total_bytes
