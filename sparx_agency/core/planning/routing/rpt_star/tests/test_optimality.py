"""The claim the whole package rests on: RPT* returns the cheapest ordering.

Theorem 2 (p.7) is a statement about an algorithm. Whether *this* is that
algorithm is a different question, and the only honest way to answer it is to
compute the optimum a second way and compare. So these tests enumerate every
ordering and demand the search agree -- on the cost, exactly, and on hundreds
of random instances rather than a handful of hand-picked ones.

They are the tests that would have caught the paper's own bug. With the initial
state as Algorithm 1 prints it, this file does not fail gracefully: the
heuristic indexes one row past the end of its table and the search raises
before it returns anything.

Three groups:

* **optimality** -- the search against exhaustive enumeration, over random
  instances in several probability regimes, because the regime turns out to
  matter more than the size;
* **the bound** -- that F-RPT* never exceeds ``(1 + epsilon)``, and that an
  epsilon of zero recovers the optimum exactly;
* **the pruning** -- that dominance changes only the effort, never the answer.
"""
from __future__ import annotations

import random

import pytest

from sparx_agency.core.planning.routing.rpt_star import (
    GUARANTEE_OPTIMAL,
    RouteProblem,
    RouteVertex,
    RptStarParams,
    brute_force_order,
    costs_from_points,
    dense_costs,
    expected_cost,
    metric_closure,
    solve,
)

EXACT = RptStarParams(epsilon=None, time_budget_s=None)


def instance(rng, n, max_prob, spread=100.0):
    """A random problem: ``n`` places in a square, with random probabilities.

    Euclidean costs, so the triangle inequality holds by construction and the
    instance is always legal.
    """
    points = [(rng.uniform(0.0, spread), rng.uniform(0.0, spread))
              for _ in range(n)]
    vertices = [RouteVertex(id=index, prob=rng.uniform(0.0, max_prob),
                            label="v%d" % index)
                for index in range(n)]
    problem = RouteProblem(vertices, start_id=rng.randrange(n))
    return problem, costs_from_points(points)


# -- optimality -----------------------------------------------------------

@pytest.mark.parametrize("max_prob", [0.9, 0.5, 0.05])
def test_matches_exhaustive_enumeration(max_prob):
    """Over every probability regime, the search finds the true optimum.

    The regimes are not decoration. Large probabilities make the survival term
    decay fast, which flattens the cost differences and makes the search easy;
    small ones keep it near one and the problem degenerates towards a plain
    travelling-salesman instance, which is where an incorrect pruning rule
    would actually show up.
    """
    rng = random.Random(20260903 + int(max_prob * 1000))
    for _ in range(60):
        problem, matrix = instance(rng, rng.randint(2, 8), max_prob)
        best_order, best_cost = brute_force_order(problem, matrix)
        solution = solve(problem, matrix, EXACT)
        assert solution.found
        assert solution.guarantee == GUARANTEE_OPTIMAL
        assert solution.expected_cost == pytest.approx(best_cost, abs=1e-9)
        # The order itself may differ from brute force's when two orderings
        # tie, which is common; the cost may not.
        assert sorted(solution.order_indices) == sorted(best_order)


def test_optimal_on_directed_graphs_where_the_cost_depends_on_direction():
    """The paper's graph is directed, so ``c(u,w)`` need not equal ``c(w,u)``.

    Easy to claim and easy to get wrong: a single transposed index in the
    successor rule or the heuristic table is invisible on symmetric costs,
    which is what every convenient test fixture produces. So this builds a
    genuine asymmetry -- a wind that makes travel cheaper in one direction --
    and takes the metric closure so the result is still a legal directed
    metric.
    """
    rng = random.Random(77)
    for _ in range(60):
        n = rng.randint(2, 7)
        points = [(rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0))
                  for _ in range(n)]
        rows = []
        for u in range(n):
            row = []
            for w in range(n):
                if u == w:
                    row.append(0.0)
                    continue
                dx = points[w][0] - points[u][0]
                dy = points[w][1] - points[u][1]
                row.append((dx * dx + dy * dy) ** 0.5 + 0.3 * dx)
            rows.append(row)
        matrix = metric_closure(dense_costs(rows))
        assert any(abs(matrix[u][w] - matrix[w][u]) > 1e-9
                   for u in range(n) for w in range(n) if u != w), \
            "fixture stopped being asymmetric"
        vertices = [RouteVertex(id=i, prob=rng.uniform(0.0, 0.5))
                    for i in range(n)]
        problem = RouteProblem(vertices, start_id=rng.randrange(n))
        _, best_cost = brute_force_order(problem, matrix)
        solution = solve(problem, matrix, EXACT)
        assert solution.expected_cost == pytest.approx(best_cost, abs=1e-9)


def test_returned_order_is_a_permutation_starting_at_the_start():
    """Every place exactly once, beginning where the robot is."""
    rng = random.Random(7)
    for _ in range(40):
        problem, matrix = instance(rng, rng.randint(2, 9), 0.4)
        solution = solve(problem, matrix, EXACT)
        assert solution.order_indices[0] == problem.start
        assert sorted(solution.order_indices) == list(range(problem.n))


def test_reported_cost_is_the_cost_of_the_reported_order():
    """The route and its price tag describe the same route.

    Scored independently of the search, which is the point -- a g-value that
    has drifted from the objective is invisible without this.
    """
    rng = random.Random(11)
    for _ in range(40):
        problem, matrix = instance(rng, rng.randint(2, 8), 0.6)
        solution = solve(problem, matrix, EXACT)
        rescored = expected_cost(solution.order_indices, problem.probs, matrix)
        assert solution.expected_cost == pytest.approx(rescored, abs=1e-12)


# -- the bound ------------------------------------------------------------

@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1, 0.5])
def test_focal_search_honours_its_bound(epsilon):
    """F-RPT* never returns worse than ``(1 + epsilon)`` times optimal."""
    rng = random.Random(4242 + int(epsilon * 100))
    params = RptStarParams(epsilon=epsilon, time_budget_s=None)
    for _ in range(50):
        problem, matrix = instance(rng, rng.randint(2, 8), 0.4)
        _, best_cost = brute_force_order(problem, matrix)
        solution = solve(problem, matrix, params)
        assert solution.found
        if best_cost > 0.0:
            assert solution.expected_cost <= (1.0 + epsilon) * best_cost + 1e-9


def test_zero_epsilon_recovers_the_optimum():
    """A band of zero width is exact, and says so.

    Worth pinning separately: it is the one setting where the focal machinery
    must produce exactly what the exact search does, so it is the cheapest
    check that the two code paths have not diverged.
    """
    rng = random.Random(99)
    params = RptStarParams(epsilon=0.0, time_budget_s=None)
    for _ in range(40):
        problem, matrix = instance(rng, rng.randint(2, 8), 0.5)
        _, best_cost = brute_force_order(problem, matrix)
        solution = solve(problem, matrix, params)
        assert solution.expected_cost == pytest.approx(best_cost, abs=1e-9)
        assert solution.guarantee == GUARANTEE_OPTIMAL


def test_bound_ratio_reports_the_slack_actually_used():
    """The realised sub-optimality, which is normally far inside the bound."""
    rng = random.Random(5)
    problem, matrix = instance(rng, 8, 0.3)
    solution = solve(problem, matrix, RptStarParams(epsilon=0.2,
                                                    time_budget_s=None))
    assert 1.0 <= solution.bound_ratio <= 1.2 + 1e-9
    assert solution.lower_bound <= solution.expected_cost + 1e-12


# -- the pruning ----------------------------------------------------------

def test_dominance_changes_the_effort_not_the_answer():
    """Turning the search loose without pruning must reach the same cost.

    Simulated by comparing against exhaustive enumeration rather than by
    disabling the pruning, since the pruning is what makes the search finish.
    The assertion that matters is that pruning never costs optimality; the
    statistics confirm it is doing something.
    """
    rng = random.Random(31337)
    pruned_total = 0
    for _ in range(30):
        problem, matrix = instance(rng, 8, 0.05)
        _, best_cost = brute_force_order(problem, matrix)
        solution = solve(problem, matrix, EXACT)
        assert solution.expected_cost == pytest.approx(best_cost, abs=1e-9)
        pruned_total += (solution.stats.pruned_on_generation
                         + solution.stats.pruned_on_pop)
    assert pruned_total > 0, "dominance pruned nothing at all -- suspicious"
