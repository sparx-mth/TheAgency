"""The answer computed the stupid way, so the clever way can be checked.

Theorem 2 says RPT* returns the optimum. That is a claim about an
implementation as much as about an algorithm, and the only way to hold an
implementation to it is to compute the optimum some other way and compare.
Enumerating every ordering is that other way: it shares no code with the search,
so no bug can be common to both.

It is shipped with the package rather than hidden in the tests because the
obligation belongs to the package. A reader who wants to know whether this is
really optimal should be able to check on their own data, not just trust that
somebody once ran a test.

Factorial, obviously. It refuses anything above :data:`MAX_VERTICES`, where the
enumeration is already ten million orderings.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple

from sparx_agency.core.planning.routing.rpt_star.objective import expected_cost

#: Above this the enumeration stops being a test and becomes a hang.
MAX_VERTICES = 11


def brute_force_order(problem, matrix):
    # type: (object, Sequence[Sequence[float]]) -> Tuple[Tuple[int, ...], float]
    """The cheapest ordering, found by trying all of them.

    Args:
        problem: A
            :class:`~sparx_agency.core.planning.routing.rpt_star.problem.RouteProblem`.
        matrix: The cost matrix.

    Returns:
        The best order as indices, and its expected cost.

    Raises:
        ValueError: If the problem has more than :data:`MAX_VERTICES` vertices.
    """
    n = problem.n
    if n > MAX_VERTICES:
        raise ValueError(
            "brute force refuses %d vertices: that is %d orderings. It exists "
            "to check the search on small instances, not to replace it."
            % (n, _factorial(n - 1)))
    start = problem.start
    others = [v for v in range(n) if v != start]
    best_order = (start,)                       # type: Tuple[int, ...]
    best_cost = 0.0 if n == 1 else float("inf")
    for permutation in itertools.permutations(others):
        order = (start,) + permutation
        cost = expected_cost(order, problem.probs, matrix)
        if cost < best_cost:
            best_cost = cost
            best_order = order
    return best_order, best_cost


def _factorial(k):
    # type: (int) -> int
    """``k!``, for the error message."""
    out = 1
    for value in range(2, k + 1):
        out *= value
    return out
