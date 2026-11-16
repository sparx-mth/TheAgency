import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import (
    UNASSIGNED, assign_cells, compass_yaws, spread_points)


def objects(*rows):
    return [{"region": region, "center": (x, 1.0, z)} for region, x, z in rows]


def test_a_cell_takes_the_region_of_the_objects_around_it():
    cells = np.array([[0.2, 0.0, 0.0], [10.2, 0.0, 0.0]])
    found = assign_cells(cells, objects((1, 0, 0), (1, 0.5, 0), (1, 0, 0.5), (2, 10, 0), (2, 10.5, 0), (2, 10, 0.5)), max_gap_m=2.0)
    assert list(found) == [1, 2]


def test_a_cell_far_from_every_object_belongs_to_no_region():
    found = assign_cells(np.array([[50.0, 0.0, 50.0]]), objects((1, 0, 0), (1, 1, 0), (1, 0, 1)), max_gap_m=2.0)
    assert list(found) == [UNASSIGNED]


def test_objects_on_another_floor_do_not_count():
    upstairs = [{"region": 3, "center": (0.0, 4.5, 0.0)}] * 3
    assert list(assign_cells(np.array([[0.0, 0.0, 0.0]]), upstairs, max_gap_m=2.0)) == [UNASSIGNED]


def test_the_majority_of_the_nearest_objects_decides():
    near = objects((1, 0.5, 0), (2, 0.6, 0), (2, 0.7, 0), (1, 5, 0))
    assert list(assign_cells(np.array([[0.0, 0.0, 0.0]]), near, max_gap_m=2.0)) == [2]


def test_spread_points_start_in_the_middle_and_then_go_far_apart():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    chosen = spread_points(points, 3)
    assert chosen[0] == 2 and set(chosen[1:]) == {0, 4}


def test_spread_points_never_returns_more_than_there_are():
    assert len(spread_points(np.array([[0.0, 0.0], [1.0, 0.0]]), 5)) == 2


@pytest.mark.parametrize("count", [4, 8])
def test_the_headings_are_evenly_spaced_around_the_circle(count):
    yaws = compass_yaws(count)
    assert len(yaws) == count and np.allclose(np.diff(yaws), 2 * np.pi / count)
