"""The planners the paper compares, each behind one signature.

Sections VII-B and VII-C name five things to run, and all five are here:

============  =========================================================
name          what it is
============  =========================================================
``RPT*``      The exact search of Sec. IV-B. Optimal under the belief
              it is given (Thm. 2).
``RPT*_noh``  The same search with ``h(s) = 0``, the paper's uninformed
              ablation (Sec. VII-B-1), reported to cost 50-70% more
              runtime. Still exact.
``F-RPT*``    The focal variant of Sec. IV-D at a given ``epsilon``.
              The paper sweeps ``{0.001, 0.01, 0.1, 0.2}`` (Sec.
              VII-B-3).
``Greedy``    "always move to a vertex with the largest probability
              value among the unvisited" (Sec. VII-C-1). Distance never
              enters it.
``LKH``       Ignore the belief, minimise raw distance (Sec. VII-C-1).
              See
              :func:`~sparx_agency.core.planning.routing.rpt_star.baselines.lkh_style_order`
              for what stands in for LKH itself and why it has to be
              strong.
============  =========================================================

Two more are included that the paper does not run. ``NN`` is plain
nearest-neighbour, which shows how much of ``LKH``'s result is the local search
rather than the idea of ignoring the belief. ``Optimal`` enumerates every
ordering, and exists so that "RPT* is optimal" is checked on this data rather
than assumed from Theorem 2.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

from sparx_agency.core.planning.routing.rpt_star import (
    RouteProblem,
    RouteVertex,
    RptStarParams,
    brute_force_order,
    greedy_probability_order,
    lkh_style_order,
    nearest_neighbour_order,
    solve,
)
from sparx_agency.core.planning.routing.rpt_star.result import (
    GUARANTEE_BOUNDED,
    GUARANTEE_OPTIMAL,
    ROUTE_FROM_SEARCH,
    STATUS_SOLVED,
)

#: The paper's per-instance runtime limit (Sec. VII-B-1, p.12).
PAPER_TIME_LIMIT_S = 60.0

#: The epsilons swept in Sec. VII-B-3 (p.12).
PAPER_EPSILONS = (0.001, 0.01, 0.1, 0.2)

#: Above this, enumerating every ordering stops being feasible.
BRUTE_FORCE_MAX = 11


class PlanResult(object):
    """What a planner returned, plus how it went.

    Attributes:
        order: The visiting order, as vertex indices.
        planning_s: Wall-clock seconds spent deciding it.
        expansions: States expanded, or zero for a non-search planner.
        status: The solver's status, or ``"n/a"``.
        guarantee: The solver's guarantee, or ``"n/a"``.
        solved: Whether the planner stands behind the route. For the search
            planners this means the search terminated on a goal state within
            its budget rather than falling back -- the paper's success-rate
            criterion. For the baselines it is always True: they always return
            something, they just make no optimality claim.
    """

    __slots__ = ("order", "planning_s", "expansions", "status", "guarantee",
                 "solved")

    def __init__(self, order, planning_s, expansions=0, status="n/a",
                 guarantee="n/a", solved=True):
        # type: (Tuple[int, ...], float, int, str, str, bool) -> None
        self.order = order
        self.planning_s = planning_s
        self.expansions = expansions
        self.status = status
        self.guarantee = guarantee
        self.solved = solved


def build_problem(belief, start_index=0):
    # type: (Sequence[float], int) -> RouteProblem
    """Wrap a belief vector in the solver's problem type.

    Vertex ids are their own indices, so a route reads the same in every
    module here and in the raw JSON.

    Args:
        belief: ``p(v)`` by index.
        start_index: Which vertex the robot starts at. Its own probability is
            *not* zeroed: the paper's Def. 2 treats the start as an ordinary
            vertex that happens to be first, and zeroing it would quietly
            change the instance.

    Returns:
        The problem.
    """
    vertices = [RouteVertex(id=i, prob=float(p), label=str(i))
                for i, p in enumerate(belief)]
    return RouteProblem(vertices, start_index)


def plan_rpt_star(problem, matrix, epsilon=None, use_heuristic=True,
                  time_limit_s=PAPER_TIME_LIMIT_S):
    # type: (RouteProblem, Sequence[Sequence[float]], Optional[float], bool, float) -> PlanResult
    """Run the solver, exact or focal, guided or not.

    Args:
        problem: The problem.
        matrix: The cost matrix.
        epsilon: ``None`` for exact RPT*, a number for F-RPT*.
        use_heuristic: ``False`` for the ``RPT*_noh`` ablation.
        time_limit_s: The per-instance budget.

    Returns:
        The result. ``solved`` is True only when the search itself produced the
        route with a guarantee attached; a budget-exhausted run returns its
        fallback route with ``solved`` False, which is what the paper's success
        rate counts.
    """
    params = RptStarParams(
        epsilon=epsilon,
        use_heuristic=use_heuristic,
        time_budget_s=time_limit_s,
    )
    started = time.monotonic()
    solution = solve(problem, matrix, params)
    elapsed = time.monotonic() - started

    solved = (solution.status == STATUS_SOLVED
              and solution.route_source == ROUTE_FROM_SEARCH
              and solution.guarantee in (GUARANTEE_OPTIMAL, GUARANTEE_BOUNDED))
    return PlanResult(
        order=tuple(solution.order_indices),
        planning_s=elapsed,
        expansions=solution.stats.expansions,
        status=solution.status,
        guarantee=solution.guarantee,
        solved=solved,
    )


def plan_greedy(problem, matrix):
    # type: (RouteProblem, Sequence[Sequence[float]]) -> PlanResult
    """The paper's ``Greedy``: most likely place next, distance ignored."""
    started = time.monotonic()
    order = greedy_probability_order(problem, matrix)
    return PlanResult(order, time.monotonic() - started)


def plan_lkh(problem, matrix):
    # type: (RouteProblem, Sequence[Sequence[float]]) -> PlanResult
    """The paper's ``LKH``: shortest route, belief ignored."""
    started = time.monotonic()
    order = lkh_style_order(problem, matrix)
    return PlanResult(order, time.monotonic() - started)


def plan_nearest_neighbour(problem, matrix):
    # type: (RouteProblem, Sequence[Sequence[float]]) -> PlanResult
    """Nearest unvisited place next. Not in the paper; the weak-tour control."""
    started = time.monotonic()
    order = nearest_neighbour_order(problem, matrix)
    return PlanResult(order, time.monotonic() - started)


def plan_optimal(problem, matrix):
    # type: (RouteProblem, Sequence[Sequence[float]]) -> PlanResult
    """Every ordering, scored. The ground truth RPT* is checked against.

    Returns a result with an empty order when the instance is too large to
    enumerate, so a caller can skip it without special-casing the size.
    """
    if problem.n > BRUTE_FORCE_MAX:
        return PlanResult((), 0.0, status="too-large", solved=False)
    started = time.monotonic()
    order, _cost = brute_force_order(problem, matrix)
    return PlanResult(order, time.monotonic() - started,
                      status=STATUS_SOLVED, guarantee=GUARANTEE_OPTIMAL)


def default_planners(include_optimal=False, epsilons=(0.01,)):
    # type: (bool, Sequence[float]) -> List[Tuple[str, Callable[[RouteProblem, Sequence[Sequence[float]]], PlanResult]]]
    """The planner set used by most experiments, in report order.

    Args:
        include_optimal: Whether to add the brute-force enumerator. Only
            sensible when every instance is small.
        epsilons: Which F-RPT* variants to include.

    Returns:
        ``(name, callable)`` pairs.
    """
    planners = [
        ("RPT*", lambda p, m: plan_rpt_star(p, m)),
    ]
    for epsilon in epsilons:
        planners.append((
            "F-RPT*(%g)" % epsilon,
            lambda p, m, e=epsilon: plan_rpt_star(p, m, epsilon=e),
        ))
    planners.extend([
        ("Greedy", plan_greedy),
        ("LKH", plan_lkh),
        ("NN", plan_nearest_neighbour),
    ])
    if include_optimal:
        planners.append(("Optimal", plan_optimal))
    return planners

