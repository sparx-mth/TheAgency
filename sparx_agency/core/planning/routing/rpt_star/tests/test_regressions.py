"""Defects that were once real. Each of these shipped, and each is pinned here.

Found by an adversarial audit of the first working version. They shared a
shape worth naming: **none of them crashed.** Every one produced a plausible
result that was quietly wrong -- a missing route, a route stamped optimal that
was twenty percent expensive, a bound the true optimum sat below. That is the
failure mode this package is organised against, so the tests that catch them
belong together where the pattern is visible.
"""
from __future__ import annotations

import random

import pytest

from sparx_agency.core.planning.routing.rpt_star import (
    GUARANTEE_NONE,
    GUARANTEE_OPTIMAL,
    STATUS_BUDGET_EXCEEDED,
    DisconnectedGraphError,
    InvalidCostError,
    RouteProblem,
    RouteVertex,
    RptStarParams,
    brute_force_order,
    costs_from_pairs,
    costs_from_points,
    dense_costs,
    solve,
)
from sparx_agency.core.planning.routing.rpt_star.result import (
    ROUTE_FROM_FALLBACK,
    ROUTE_FROM_SEARCH,
)


def normalised(n, seed, spread=50.0):
    """A problem shaped like a real one: probabilities summing to one.

    This is the regime a calibrated oracle produces, and the one the exact
    search struggles in -- mean probability ``1/n``, which keeps the survival
    term near one and makes the problem behave like a travelling-salesman
    instance. Several of the defects below only appear here.
    """
    rng = random.Random(seed)
    points = [(rng.uniform(0.0, spread), rng.uniform(0.0, spread))
              for _ in range(n)]
    raw = [rng.random() for _ in range(n - 1)]
    total = sum(raw)
    rooms = [RouteVertex(id="r%d" % i, prob=raw[i] / total)
             for i in range(n - 1)]
    return RouteProblem.with_external_start(rooms), costs_from_points(points)


# -- the budget used to return nothing at all -----------------------------

def test_a_hard_instance_under_default_params_still_yields_somewhere_to_fly():
    """Once returned ``found=False`` and ``next_id=None`` at 21 places.

    The incumbent was only ever set from a *complete* ordering the search had
    reached, and on a hard instance the budget expires long before any
    ordering is complete. So the headline configuration handed a mission node
    nothing, at exactly the size and belief shape this package is meant for.
    """
    problem, matrix = normalised(21, seed=3)
    solution = solve(problem, matrix, RptStarParams(max_expansions=400,
                                                   time_budget_s=None))
    assert solution.found
    assert solution.next_id is not None
    assert solution.status == STATUS_BUDGET_EXCEEDED
    assert solution.guarantee == GUARANTEE_NONE
    assert solution.route_source == ROUTE_FROM_FALLBACK


def test_the_fallback_route_is_a_real_route_and_priced_correctly():
    """It has to be flyable, not merely non-empty.

    Every place exactly once, starting where the robot is, and a cost that
    matches the ordering -- the reconstruction tripwire runs on it too, so a
    fallback whose cost disagreed with its route would raise rather than fly.
    """
    problem, matrix = normalised(24, seed=11)
    solution = solve(problem, matrix, RptStarParams(max_expansions=400,
                                                   time_budget_s=None))
    assert solution.route_source == ROUTE_FROM_FALLBACK
    assert solution.order_indices[0] == problem.start
    assert sorted(solution.order_indices) == list(range(problem.n))
    assert solution.expected_cost < float("inf")
    assert solution.lower_bound <= solution.expected_cost + 1e-9


def test_an_easy_instance_still_comes_from_the_search():
    """The fallback must not quietly take over work the search can do."""
    problem, matrix = normalised(8, seed=5)
    solution = solve(problem, matrix)
    assert solution.route_source == ROUTE_FROM_SEARCH
    assert solution.guarantee == GUARANTEE_OPTIMAL


# -- a waived check used to still claim optimality ------------------------

def test_waiving_the_triangle_check_withdraws_the_guarantee():
    """Once returned ``guarantee='optimal'`` on a route 20.8% too expensive.

    Without the triangle inequality, Lemma 6 fails and the dominance rule can
    discard the optimal completion -- and the search cannot tell that it did.
    The instance below is the one the audit found: exact search, check waived,
    a confident wrong answer.
    """
    rows = [[0, 29, 3, 13, 3], [20, 0, 34, 33, 31], [29, 34, 0, 7, 27],
            [26, 5, 7, 0, 20], [26, 16, 23, 6, 0]]
    probs = [0.0, 0.0, 0.05, 0.0, 0.05]
    vertices = [RouteVertex(id=i, prob=probs[i]) for i in range(5)]
    problem = RouteProblem(vertices, start_id=0)
    matrix = dense_costs(rows)

    solution = solve(problem, matrix,
                     RptStarParams(epsilon=None, time_budget_s=None,
                                   require_triangle_inequality=False))
    _, true_best = brute_force_order(problem, matrix)

    assert solution.expected_cost > true_best + 1e-9, (
        "the fixture stopped being a case where pruning loses the optimum")
    assert not solution.is_optimal
    assert solution.guarantee == GUARANTEE_NONE
    assert any("waived" in w for w in solution.warnings)


def test_a_waived_check_publishes_no_bound_the_optimum_could_sit_below():
    """The bound used to be the returned cost, which was above the optimum.

    A result that certifies itself against the wrong number is worse than one
    that certifies nothing, so the only honest bound here is the trivial one.
    """
    rows = [[0, 29, 3, 13, 3], [20, 0, 34, 33, 31], [29, 34, 0, 7, 27],
            [26, 5, 7, 0, 20], [26, 16, 23, 6, 0]]
    vertices = [RouteVertex(id=i, prob=[0.0, 0.0, 0.05, 0.0, 0.05][i])
                for i in range(5)]
    problem = RouteProblem(vertices, start_id=0)
    matrix = dense_costs(rows)
    solution = solve(problem, matrix,
                     RptStarParams(epsilon=None, time_budget_s=None,
                                   require_triangle_inequality=False))
    _, true_best = brute_force_order(problem, matrix)
    assert solution.lower_bound <= true_best + 1e-9


def test_a_budget_stop_keeps_its_bound_because_pruning_was_still_sound():
    """Withdrawing the guarantee must not withdraw the bound as well.

    A timed-out search discarded nothing it should not have, so the smallest
    ``f`` left in its queue is still a genuine lower bound -- and it is the
    only thing such a caller gets. Losing it to an over-eager fix would be a
    regression of its own.
    """
    problem, matrix = normalised(20, seed=7)
    solution = solve(problem, matrix, RptStarParams(max_expansions=200,
                                                   time_budget_s=None))
    _, _ = solution.status, solution.guarantee
    assert solution.lower_bound > 0.0
    assert solution.lower_bound <= solution.expected_cost + 1e-9


# -- validation gaps ------------------------------------------------------

def test_a_raw_matrix_with_negative_costs_is_refused_by_solve():
    """``dense_costs`` refused these; ``solve`` on a raw list-of-lists did not.

    A potential matrix -- ``c(u,w) = phi(w) - phi(u)`` -- satisfies the
    triangle inequality with *equality* everywhere, so it sails past that
    check while containing negative edges that break the same lemma.
    """
    potential = [0, 1, 5, 2]
    raw = [[potential[w] - potential[u] for w in range(4)] for u in range(4)]
    vertices = [RouteVertex(id="v%d" % i, prob=[0.0, 0.3, 0.2, 0.1][i])
                for i in range(4)]
    problem = RouteProblem(vertices, start_id="v0")
    with pytest.raises(InvalidCostError, match="negative"):
        solve(problem, raw, RptStarParams(epsilon=None, time_budget_s=None))


def test_the_disconnected_error_names_places_the_way_the_caller_does():
    """It used to carry dense indices, which no caller can act on.

    Its sibling, the triangle-inequality error, has always translated. The
    documented attribute says ``(from_id, to_id)``, and a hybrid vertex set
    of ``("room", 7)`` and ``("frontier", 3)`` is exactly where an integer
    would be useless.
    """
    rooms = [RouteVertex(id=("room", 7), prob=0.4),
             RouteVertex(id=("frontier", 3), prob=0.6)]
    problem = RouteProblem.with_external_start(rooms)
    matrix = costs_from_pairs(3, {(0, 1): 1.0, (1, 0): 1.0,
                                  (0, 2): 1.0, (2, 0): 1.0})
    with pytest.raises(DisconnectedGraphError) as caught:
        solve(problem, matrix)
    assert caught.value.pairs
    for source, target in caught.value.pairs:
        assert isinstance(source, tuple) and isinstance(target, tuple)


# -- reporting ------------------------------------------------------------

def test_peak_frontier_is_a_high_water_mark_not_a_final_reading():
    """It was read off the frontiers after the loop, so it under-reported.

    Admitting one route can evict several, so a frontier shrinks. The number
    exists to size a memory budget, and a final reading is not that.
    """
    rng = random.Random(34188)
    points = [(rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0))
              for _ in range(9)]
    vertices = [RouteVertex(id=i, prob=rng.uniform(0.0, 0.3))
                for i in range(9)]
    problem = RouteProblem(vertices, start_id=0)
    matrix = costs_from_points(points)
    solution = solve(problem, matrix,
                     RptStarParams(epsilon=1.0, time_budget_s=None))
    assert solution.stats.peak_frontier >= 1
    assert solution.stats.peak_frontier <= solution.stats.generated + 1


def test_the_dominance_rule_has_exactly_one_definition():
    """The frontier used to inline its own copy, so the named rule was dead.

    Two copies of a pruning rule is the drift this repository has been bitten
    by before, and it hides in plain sight: both copies agreed, the tests
    passed, and mutating the canonical one changed nothing at all -- which is
    how you discover that nothing calls it.
    """
    import inspect

    from sparx_agency.core.planning.routing.rpt_star import dominance

    for method in (dominance.VertexFrontier.is_pruned,
                   dominance.VertexFrontier.filter_and_add):
        body = inspect.getsource(method)
        assert "dominates(" in body, (
            "%s no longer calls dominates() -- the rule has been inlined again"
            % method.__name__)


# -- input hygiene: found by an end-to-end review -------------------------

def test_mixing_2d_and_3d_points_is_refused():
    """Once measured only the shared prefix, and called the answer optimal.

    Zipping a three-component robot pose against two-component room centroids
    silently drops the third axis. The result is a perfectly good metric over
    the projection, so validation passes and the search returns a provably
    optimal route -- to a problem nobody posed. Assembling that list by hand
    from poses and map centroids is exactly how a caller reaches this.
    """
    from sparx_agency.core.planning.routing.rpt_star import costs_from_points

    with pytest.raises(InvalidCostError, match="shared prefix"):
        costs_from_points([(0.0, 0.0, 0.0), (1.0, 0.0), (9.0, 0.0)])


def test_uniform_points_of_any_one_dimension_are_fine():
    """The rule is consistency, not two dimensions."""
    from sparx_agency.core.planning.routing.rpt_star import costs_from_points

    flat = costs_from_points([(0.0, 0.0), (3.0, 4.0)])
    solid = costs_from_points([(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)])
    assert flat[0][1] == pytest.approx(5.0)
    assert solid[0][1] == pytest.approx(5.0)


def test_coordinates_too_large_to_square_are_refused_clearly():
    """They used to overflow to an all-infinite matrix, blamed on connectivity.

    The caller then got a DisconnectedGraphError telling them to drop an
    unreachable place -- a diagnosis pointing at entirely the wrong cause.
    """
    from sparx_agency.core.planning.routing.rpt_star import costs_from_points

    with pytest.raises(InvalidCostError, match="overflowed"):
        costs_from_points([(0.0, 0.0), (1e200, 1e200)])


def test_a_negative_epsilon_is_refused():
    """It used to ship ``guarantee='bounded'`` with a bound below one.

    The route was never wrong -- a band narrower than a point admits nothing,
    so the search quietly ran exact -- but the result promised a ratio no route
    can keep, and a caller checking it against the lower bound would read a
    good answer as a failure.
    """
    with pytest.raises(ValueError, match=r"\[0, inf\)"):
        RptStarParams(epsilon=-0.5)


def test_a_nan_epsilon_is_refused():
    with pytest.raises(ValueError, match="NaN"):
        RptStarParams(epsilon=float("nan"))


def test_zero_and_positive_epsilon_are_accepted():
    assert RptStarParams(epsilon=0.0).epsilon == 0.0
    assert RptStarParams(epsilon=2.5).epsilon == 2.5
    assert RptStarParams(epsilon=None).epsilon is None


def test_both_variants_report_comparable_pruning_counts():
    """The focal queue drops dead states itself, so its count read zero.

    That made the two variants' statistics silently incomparable, and invited
    someone to delete the pop-time dominance check as apparent dead weight --
    which costs well over double the expansions.
    """
    problem, matrix = normalised(11, seed=2)
    exact = solve(problem, matrix, RptStarParams(epsilon=None,
                                                 time_budget_s=None))
    focal = solve(problem, matrix, RptStarParams(epsilon=0.0,
                                                 time_budget_s=None))
    assert exact.stats.pruned_on_pop > 0
    assert focal.stats.pruned_on_pop > 0, (
        "the focal variant reported no pop-time pruning at all, which is the "
        "regression: it filters internally and the count was never collected")
