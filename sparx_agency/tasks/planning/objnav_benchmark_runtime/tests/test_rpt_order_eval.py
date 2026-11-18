import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.rpt_order_eval import (
    adjusted_probability, efficiency, nearest_first, rpt_order, walk_cost)

MATRIX = [[0, 1, 3], [1, 0, 3], [3, 3, 0]]


def test_a_never_entered_room_is_lifted_and_floored_like_the_planner_does():
    assert adjusted_probability(0) == 0.15
    assert abs(adjusted_probability(10) - 0.15) < 1e-9
    assert abs(adjusted_probability(50) - 0.75) < 1e-9
    assert adjusted_probability(100) < 1.0


def test_the_walk_adds_legs_until_the_first_room_that_holds_the_target():
    assert walk_cost([1, 0], MATRIX, {0}) == (6, 2)
    assert walk_cost([0, 1], MATRIX, {1}) == (4, 2)
    assert walk_cost([0], MATRIX, {0}) == (1, 1)


def test_an_order_that_never_reaches_a_holder_has_no_cost():
    assert walk_cost([0], MATRIX, {1}) == (None, None)


def test_efficiency_is_the_shortest_trip_to_a_holder_over_the_walk():
    assert efficiency(4, MATRIX, {1}) == 3 / 4
    assert efficiency(1, MATRIX, {0}) == 1.0
    assert efficiency(1, MATRIX, {0, 1}) == 1.0


def test_with_equal_beliefs_the_order_goes_to_the_nearer_room_first():
    assert rpt_order([0.5, 0.5], MATRIX) == [0, 1]


def test_a_much_likelier_room_is_visited_first_even_if_farther():
    assert rpt_order([0.01, 0.99], MATRIX) == [1, 0]


def test_nearest_first_ignores_beliefs():
    assert nearest_first(MATRIX) == [0, 1]
    assert nearest_first([[0, 5, 2], [5, 0, 4], [2, 4, 0]]) == [1, 0]


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.eval_rpt_order import UNREACHABLE, area_percent, cluster_bootstrap, episode_matrix


def test_the_episode_matrix_puts_the_start_first_and_routes_missing_pairs_through_the_others():
    distances = [[0.0, 2.0, None], [2.0, 0.0, 5.0], [None, 5.0, 0.0]]
    assert episode_matrix(distances, 0, [1, 2]) == [[0.0, 2.0, 7.0], [2.0, 0.0, 5.0], [7.0, 5.0, 0.0]]
    assert episode_matrix(distances, 1, [2]) == [[0.0, 5.0], [5.0, 0.0]]


def test_a_pair_with_no_route_at_all_stays_unreachable():
    distances = [[0.0, None], [None, 0.0]]
    assert episode_matrix(distances, 0, [1])[0][1] == UNREACHABLE


def test_the_episode_matrix_satisfies_the_triangle_inequality_even_when_the_distances_do_not():
    distances = [[0.0, 1.0, 1.02], [1.0, 0.0, 3.0], [1.02, 3.0, 0.0]]
    matrix = episode_matrix(distances, 0, [1, 2])
    assert matrix[1][2] <= matrix[1][0] + matrix[0][2] + 1e-9


def test_area_becomes_a_percent_within_the_house():
    assert area_percent([4.0, 14.0, 9.0]) == [0.0, 100.0, 50.0]
    assert area_percent([7.0, 7.0]) == [50.0, 50.0]


def test_the_scene_bootstrap_interval_covers_the_mean_and_collapses_for_equal_scenes():
    rng = np.random.default_rng(0)
    mean, low, high = cluster_bootstrap([[1.0, 1.0], [1.0]], rng)
    assert mean == low == high == 1.0
    mean, low, high = cluster_bootstrap([[0.0], [1.0], [0.5, 0.5]], rng)
    assert low <= mean <= high
