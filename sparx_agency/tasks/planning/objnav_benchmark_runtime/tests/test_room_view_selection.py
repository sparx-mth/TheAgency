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


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import select_walls


def walled(hit, near=0.0, size=6):
    return {"walls": np.array([i in hit for i in range(size)]), "near": near}


def test_wall_selection_takes_the_view_that_covers_the_most_perimeter_and_skips_close_ups():
    views = [walled({0, 1, 2, 3}, near=0.9), walled({0, 1}), walled({2, 3, 4})]
    assert select_walls(views, count=2) == [2, 1]


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import sample_names


def test_the_sample_is_a_sorted_reproducible_subset_of_the_asked_size():
    names = ["r%02d" % i for i in range(42)]
    first = sample_names(names, 40, seed=17)
    assert first == sample_names(names, 40, seed=17) == sorted(first) and len(set(first)) == 40 and set(first) <= set(names)
    assert first != sample_names(names, 40, seed=18)
    assert sample_names(names, 99, seed=17) == names
