import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import random_pick, select


def candidate(seen, near=0.0, objects=(), cells=6):
    return {"cells": np.array([i in seen for i in range(cells)]), "near": near, "objects": [{"cls": c, "point": p} for c, p in objects]}


def test_a_view_filling_the_image_with_a_near_surface_is_never_selected():
    views = [candidate({0, 1, 2, 3}, near=0.9), candidate({0, 1}, near=0.1)]
    assert select(views, count=2, with_objects=False) == [1]


def test_the_objects_break_a_tie_in_floor_coverage_only_when_they_are_asked_for():
    wall = candidate({0, 1, 2}, objects=[])
    bed = candidate({3, 4, 5}, objects=[("bed", (1.0, 1.0))])
    wall_first = [wall, bed, candidate({0, 1, 2})]
    assert select(wall_first, count=1, with_objects=True) == [1]
    assert select(wall_first, count=1, with_objects=False) == [0]


def test_selection_without_any_usable_view_is_empty():
    assert select([candidate({0}, near=0.9)], count=3, with_objects=True) == []


def test_random_picks_are_reproducible_distinct_and_limited_to_the_count():
    views = [candidate({0}) for _ in range(10)]
    first, second = random_pick(views, 4, np.random.default_rng(1), filtered=False), random_pick(views, 4, np.random.default_rng(1), filtered=False)
    assert first == second and len(set(first)) == 4


def test_filtered_random_picks_leave_out_the_near_views():
    views = [candidate({0}, near=0.9)] * 5 + [candidate({0}, near=0.1)] * 5
    assert all(i >= 5 for i in random_pick(views, 5, np.random.default_rng(0), filtered=True))
