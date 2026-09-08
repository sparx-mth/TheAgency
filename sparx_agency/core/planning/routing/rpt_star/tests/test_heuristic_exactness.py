"""The heuristic must be exactly Eq. 5 and Eq. 6, not merely admissible.

Admissibility is a one-sided property, so a test that only checks
``h <= h*`` passes for *any* weaker heuristic -- including ``h = 0``, which
deletes the heuristic entirely. That is not hypothetical: replacing
``GammaTable.estimate`` with ``return 0.0`` leaves every other test in this
package green, while costing 75% more expansions and slackening the lower bound
that a budget-exceeded solve reports, which is the only guarantee such a caller
gets.

So these tests pin the values themselves, against an enumeration that shares no
code with :mod:`.heuristic`.

**The independent oracle.** Unrolling Eq. 5 shows that ``gamma(v, k)`` is the
cost of the cheapest ``k``-step walk out of ``v`` in which no step returns
immediately to where it just came from -- and "cost" there is the ordinary
expected cost of Lemma 1 applied to that walk. So the oracle is: enumerate
every such walk, score it with :func:`expected_cost`, take the minimum. Cheap
enough for five or six places, and it re-derives the recurrence rather than
restating it.
"""
from __future__ import annotations

import itertools
import random

import pytest

from sparx_agency.core.planning.routing.rpt_star import (
    GammaTable,
    RouteProblem,
    RouteVertex,
    costs_from_points,
    expected_cost,
)


def instance(rng, n, max_prob=0.6):
    """A random Euclidean problem."""
    points = [(rng.uniform(0.0, 50.0), rng.uniform(0.0, 50.0))
              for _ in range(n)]
    vertices = [RouteVertex(id=i, prob=rng.uniform(0.0, max_prob))
                for i in range(n)]
    return RouteProblem(vertices, start_id=0), costs_from_points(points)


def walks(start, n, steps):
    """Every ``steps``-step walk from ``start`` that never doubles straight back.

    The relaxation Eq. 5 encodes: a place may be revisited later, but a step
    may not return immediately to the place it just left.
    """
    if steps == 0:
        yield (start,)
        return
    for tail in itertools.product(range(n), repeat=steps):
        walk = (start,) + tail
        if all(walk[i] != walk[i + 1] for i in range(len(walk) - 1)):
            yield walk


def cheapest_walk(start, steps, probs, matrix, n):
    """The oracle: the cheapest such walk, scored by Lemma 1."""
    return min(expected_cost(walk, probs, matrix)
               for walk in walks(start, n, steps))


def test_gamma_equals_the_cheapest_walk_of_that_many_steps():
    """Every table entry, against an enumeration that shares no code with it.

    This is the test that fails if the ``(1 - p(v))`` factor moves, if the
    minimisation ranges over the wrong set, or if a table row is off by one.
    """
    rng = random.Random(101)
    for _ in range(12):
        n = rng.randint(2, 5)
        problem, matrix = instance(rng, n)
        gamma = GammaTable(problem.probs, matrix)
        for vertex in range(n):
            for steps in range(n):
                assert gamma.steps(vertex, steps) == pytest.approx(
                    cheapest_walk(vertex, steps, problem.probs, matrix, n),
                    abs=1e-9), (
                    "gamma(%d, %d) disagrees with the cheapest %d-step walk"
                    % (vertex, steps, steps))


def test_estimate_is_exactly_equation_six():
    """``h(s) = q(s) / (1 - p(v(s))) * gamma(v(s), |V| - |A(s)|)``.

    Recomputed here from the table rather than trusted, so that dropping the
    leading factor -- the mutation that halves the heuristic and still passes
    an admissibility test -- fails.
    """
    rng = random.Random(202)
    for _ in range(20):
        n = rng.randint(2, 7)
        problem, matrix = instance(rng, n)
        probs = problem.probs
        gamma = GammaTable(probs, matrix)
        for vertex in range(n):
            for visited_count in range(1, n + 1):
                survival = rng.uniform(0.05, 1.0) * (1.0 - probs[vertex])
                got = gamma.estimate(vertex, survival, visited_count)
                remaining = n - visited_count
                if remaining <= 0:
                    assert got == 0.0
                    continue
                expected = (survival / (1.0 - probs[vertex])
                            * gamma.steps(vertex, remaining))
                assert got == pytest.approx(expected, rel=1e-12)


def test_the_heuristic_is_not_trivially_zero():
    """Guard against the mutation the admissibility test cannot see.

    ``h = 0`` is perfectly admissible, so nothing else in this package would
    notice it. On any instance with real distances the estimate must be
    strictly positive while places remain.
    """
    rng = random.Random(303)
    problem, matrix = instance(rng, 6, max_prob=0.3)
    gamma = GammaTable(problem.probs, matrix)
    positive = 0
    for vertex in range(6):
        for visited_count in range(1, 6):
            if gamma.estimate(vertex, 0.5, visited_count) > 0.0:
                positive += 1
    assert positive == 6 * 5, "some estimate collapsed to zero"


def test_the_heuristic_actually_saves_expansions():
    """It is not decoration: without it the search does measurably more work.

    Compared against the same search with the estimate forced to zero, which
    is the uninformed baseline the paper calls RPT*_noh and reports as 50-70%
    slower (Fig. 9a, p.12). The direction is what is pinned here, not the
    ratio -- that depends on the instance.
    """
    from sparx_agency.core.planning.routing.rpt_star.params import (
        RptStarParams,
    )
    from sparx_agency.core.planning.routing.rpt_star.solver import solve

    class Uninformed(GammaTable):
        """The same table, but refusing to say anything about the future."""

        def estimate(self, vertex, survival, visited_count):
            return 0.0

    import sparx_agency.core.planning.routing.rpt_star.solver as solver_module

    rng = random.Random(404)
    problem, matrix = instance(rng, 9, max_prob=0.05)
    params = RptStarParams(epsilon=None, time_budget_s=None)
    informed = solve(problem, matrix, params).stats.expansions

    original = solver_module.GammaTable
    solver_module.GammaTable = Uninformed
    try:
        uninformed = solve(problem, matrix, params).stats.expansions
    finally:
        solver_module.GammaTable = original

    assert uninformed > informed, (
        "the heuristic saved nothing: %d expansions with, %d without"
        % (informed, uninformed))
