"""The one entry point: validate, build the table, search, dress the answer.

There is a single :func:`solve` rather than an ``rpt_star()`` and an
``f_rpt_star()``. The two variants differ in one line of Algorithm 1 and share
their validation, their heuristic, their pruning and their result type; two
front doors would invite those to drift, and this repository has a documented
history of one algorithm acquiring three spellings and only one of them getting
fixed.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

from sparx_agency.core.planning.routing.rpt_star.errors import (
    RoutingInternalError,
)
from sparx_agency.core.planning.routing.rpt_star.heuristic import GammaTable
from sparx_agency.core.planning.routing.rpt_star.objective import (
    decompose,
    expected_cost,
)
from sparx_agency.core.planning.routing.rpt_star.params import (
    RECONSTRUCTION_TOLERANCE,
    RptStarParams,
)
from sparx_agency.core.planning.routing.rpt_star.result import (
    GUARANTEE_BOUNDED,
    GUARANTEE_NONE,
    GUARANTEE_OPTIMAL,
    STATUS_SOLVED,
    RouteSolution,
    SearchStats,
)
from sparx_agency.core.planning.routing.rpt_star.search import search
from sparx_agency.core.planning.routing.rpt_star.validation import validate


def solve(problem, matrix, params=None):
    # type: (object, Sequence[Sequence[float]], Optional[RptStarParams]) -> RouteSolution
    """Find the visiting order that minimises expected cost.

    Args:
        problem: A
            :class:`~sparx_agency.core.planning.routing.rpt_star.problem.RouteProblem`
            -- the places, their probabilities, and where the robot starts.
        matrix: The cost of flying between every ordered pair, from
            :mod:`~sparx_agency.core.planning.routing.rpt_star.costs`.
        params: Tuning. The default runs the exact search under a five-second
            ceiling, so the answer is provably optimal whenever the search
            finishes. When it does not, the result says so and carries a
            fallback route rather than nothing -- see
            :class:`~sparx_agency.core.planning.routing.rpt_star.result.RouteSolution`.

    Returns:
        The ordering, what it is expected to cost, and what that is worth. The
        caller normally flies :attr:`RouteSolution.next_id` and re-solves.

    Raises:
        InvalidCostError: The matrix is malformed.
        DisconnectedGraphError: Some place cannot be reached from another.
        TriangleInequalityError: A detour beats going direct, which would make
            the pruning unsound.
        RoutingInternalError: The returned route does not cost what the search
            said it costs. A bug here, not in the input.
    """
    if params is None:
        params = RptStarParams()

    warnings = validate(problem, matrix, params.require_triangle_inequality)

    build_started = time.monotonic()
    gamma = GammaTable(problem.probs, matrix)
    build_seconds = time.monotonic() - build_started

    outcome = search(problem, matrix, gamma, params)
    stats = _with_build_time(outcome.stats, build_seconds)

    if not outcome.order:
        return RouteSolution(
            status=outcome.status,
            guarantee=GUARANTEE_NONE,
            route_source=outcome.source,
            epsilon=params.epsilon,
            lower_bound=_lower_bound(outcome.lower_bound, float("inf"),
                                     params.require_triangle_inequality),
            warnings=warnings,
            stats=stats,
        )

    cost = expected_cost(outcome.order, problem.probs, matrix)
    if params.verify_reconstruction:
        _check_reconstruction(cost, outcome.cost, outcome.order)

    guarantee = _guarantee(outcome.status, params)
    bound = _lower_bound(outcome.lower_bound, cost,
                         params.require_triangle_inequality)
    return RouteSolution(
        order=problem.ids_of(outcome.order),
        order_indices=outcome.order,
        next_id=(problem.id_of(outcome.order[1])
                 if len(outcome.order) > 1 else None),
        expected_cost=cost,
        lower_bound=bound,
        bound_ratio=_bound_ratio(cost, bound),
        status=outcome.status,
        guarantee=guarantee,
        route_source=outcome.source,
        epsilon=params.epsilon,
        legs=decompose(outcome.order, problem.probs, matrix),
        warnings=warnings,
        stats=stats,
    )


def _guarantee(status, params):
    # type: (str, RptStarParams) -> str
    """What the answer is worth, given how the search ended and what it ran.

    Three ways to owe nothing, and the last one is easy to forget:

    * a search stopped by its budget never looked at the orderings it did not
      reach, so no claim survives about them;
    * a caller who waived the triangle-inequality check waived the premise of
      Lemma 6, which is what makes the dominance pruning sound. The search may
      then have discarded the optimum, and it cannot tell that it did. This is
      the case :mod:`.result`'s own docstring names as the reason ``status``
      and ``guarantee`` are separate fields, so it has to be implemented and
      not merely described;
    * otherwise a clean exact run is optimal (Theorem 2) and a clean focal run
      is within its epsilon (Theorem 3) -- except that an epsilon of exactly
      zero is a band of zero width, and therefore optimal after all.
    """
    if status != STATUS_SOLVED:
        return GUARANTEE_NONE
    if not params.require_triangle_inequality:
        return GUARANTEE_NONE
    if params.epsilon is None or params.epsilon == 0.0:
        return GUARANTEE_OPTIMAL
    return GUARANTEE_BOUNDED


def _lower_bound(searched, cost, pruning_was_sound):
    # type: (float, float, bool) -> float
    """The tightest bound that is actually true.

    The search's bound is the smallest ``f`` still queued, and that bounds the
    optimum only if nothing optimal was pruned away. A budget that expired does
    not threaten it -- the optimum is then either the route returned or still
    sitting in the queue at no less than ``f_min`` -- so a timed-out search
    still certifies itself, which is the whole reason the field exists.

    A waived triangle-inequality check *does* threaten it: without Lemma 6 the
    dominance rule can discard the optimal completion, and then the "bound" is
    a number the true optimum sits below. A result that certifies the wrong
    thing is worse than one that certifies nothing, so the only honest answer
    there is the trivial bound.
    """
    if not pruning_was_sound:
        return 0.0
    return min(searched, cost)


def _bound_ratio(cost, bound):
    # type: (float, float) -> float
    """How far above the bound the answer is, as a factor.

    A zero bound is not a licence to claim a ratio of one: it means nothing is
    known, and infinity says so.
    """
    if bound > 0.0:
        return cost / bound
    return 1.0 if cost == 0.0 else float("inf")


def _check_reconstruction(rescored, searched, order):
    # type: (float, float, tuple) -> None
    """Confirm the route costs what the search believed it did.

    Cheap -- linear against an exponential search -- and it closes the whole
    family of bugs in which the state machine and the objective drift apart:
    a mis-set survival probability, a parent pointer recycled between states, a
    route reconstructed backwards. Any of those produce a plausible ordering
    with a wrong cost, which nothing downstream would ever question.

    Raises:
        RoutingInternalError: If they disagree.
    """
    scale = max(1.0, abs(rescored), abs(searched))
    if abs(rescored - searched) > RECONSTRUCTION_TOLERANCE * scale:
        raise RoutingInternalError(
            "the reconstructed route costs %.12g but the search terminated on "
            "a g-value of %.12g. The state machine and the objective disagree, "
            "so this route must not be flown. Order: %r"
            % (rescored, searched, order))


def _with_build_time(stats, seconds):
    # type: (SearchStats, float) -> SearchStats
    """Fold the heuristic build time into the stats the search produced."""
    return SearchStats(
        expansions=stats.expansions,
        generated=stats.generated,
        pruned_on_generation=stats.pruned_on_generation,
        pruned_on_pop=stats.pruned_on_pop,
        peak_frontier=stats.peak_frontier,
        heuristic_build_s=seconds,
        search_s=stats.search_s,
    )
