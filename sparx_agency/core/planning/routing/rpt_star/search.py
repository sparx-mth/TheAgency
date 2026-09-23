"""Algorithm 1: the search itself, once, with the queue handed to it.

RPT* and F-RPT* differ in exactly one thing -- which state gets expanded next
-- and the paper writes them as one listing with the focal additions in
parentheses (Algorithm 1, p.5). This module does the same: the queue is
injected, and the rest of the loop does not know or care which variant is
running. Two copies of this loop would be two places for the pruning rules to
drift apart, and a drifted pruning rule does not crash, it just quietly returns
a route that is not the one it promised.

The loop, in the paper's own order:

1. Pop the most promising state.
2. Discard it if something better has reached its vertex since it was queued
   (``IsPruned``).
3. Otherwise admit it to that vertex's frontier, evicting anything it beats
   (``FilterAndAddFront``).
4. If it has visited everywhere, it is the answer -- return it.
5. Otherwise expand it, discarding successors that are already dominated.

Step 4 is why the goal test lives on the *pop* and not on generation: a
complete route only becomes trustworthy when nothing cheaper is left in the
queue, which is precisely what popping it means.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

from sparx_agency.core.planning.routing.rpt_star.baselines import (
    nearest_neighbour_order,
)
from sparx_agency.core.planning.routing.rpt_star.dominance import VertexFrontier
from sparx_agency.core.planning.routing.rpt_star.focal_list import FocalList
from sparx_agency.core.planning.routing.rpt_star.heuristic import GammaTable
from sparx_agency.core.planning.routing.rpt_star.objective import expected_cost
from sparx_agency.core.planning.routing.rpt_star.open_list import OpenList
from sparx_agency.core.planning.routing.rpt_star.params import (
    BUDGET_CHECK_INTERVAL,
    RptStarParams,
)
from sparx_agency.core.planning.routing.rpt_star.result import (
    ROUTE_FROM_FALLBACK,
    ROUTE_FROM_SEARCH,
    STATUS_BUDGET_EXCEEDED,
    STATUS_NO_ROUTE,
    STATUS_SOLVED,
    SearchStats,
)
from sparx_agency.core.planning.routing.rpt_star.state import (
    SearchState,
    initial_state,
    reconstruct,
    successors,
)


class SearchOutcome(object):
    """The raw result of one search, before it is dressed up for a caller.

    Attributes:
        order: The visiting order as indices, or ``()`` if none was found.
        cost: Its expected cost, or infinity.
        lower_bound: A proven lower bound on the optimal cost.
        status: One of the ``STATUS_*`` constants.
        source: One of the ``ROUTE_FROM_*`` constants -- whether the search
            produced this route or a fallback did.
        stats: What the search did.
    """

    __slots__ = ("order", "cost", "lower_bound", "status", "source", "stats")

    def __init__(self, order, cost, lower_bound, status, source, stats):
        # type: (Tuple[int, ...], float, float, str, str, SearchStats) -> None
        self.order = order
        self.cost = cost
        self.lower_bound = lower_bound
        self.status = status
        self.source = source
        self.stats = stats


def search(problem, matrix, gamma, params):
    # type: (object, Sequence[Sequence[float]], GammaTable, RptStarParams) -> SearchOutcome
    """Run Algorithm 1 over a validated problem.

    Args:
        problem: A
            :class:`~sparx_agency.core.planning.routing.rpt_star.problem.RouteProblem`.
        matrix: The cost matrix.
        gamma: The prebuilt heuristic table.
        params: Tuning, including which variant to run.

    Returns:
        The outcome. Never raises on a hard instance -- a search that runs out
        of budget returns what it has, says so in its status, and reports the
        lower bound it proved on the way.
    """
    n = problem.n
    probs = problem.probs
    goal_mask = (1 << n) - 1

    frontiers = [VertexFrontier() for _ in range(n)]
    queue = _make_queue(params, frontiers)

    start = initial_state(problem.start, probs)
    queue.push(start, gamma.estimate(start.vertex, start.survival,
                                     start.visited_count))

    expansions = 0
    generated = 0
    pruned_on_generation = 0
    pruned_on_pop = 0
    #: The best complete route seen so far. Recording it costs one integer
    #: comparison per generated state and gives a timed-out search something
    #: to return, without changing which states the search expands.
    incumbent = None                            # type: Optional[SearchState]
    started = time.monotonic()
    deadline = (None if params.time_budget_s is None
                else started + float(params.time_budget_s))

    # A one-vertex problem is already solved: the robot is standing on the only
    # place there is, and the goal test would otherwise never be reached.
    if n == 1:
        return SearchOutcome((problem.start,), 0.0, 0.0, STATUS_SOLVED,
                             ROUTE_FROM_SEARCH,
                             SearchStats(search_s=time.monotonic() - started))

    peak_frontier = 0
    status = STATUS_NO_ROUTE
    while queue:
        if _out_of_budget(params, expansions, deadline):
            status = STATUS_BUDGET_EXCEEDED
            break

        state = queue.pop()
        frontier = frontiers[state.vertex]
        if frontier.is_pruned(state.cost, state.visited):
            pruned_on_pop += 1
            continue
        frontier.filter_and_add(state.cost, state.visited)
        # Sampled here rather than read off the frontiers at the end: an
        # admission can evict several entries and append one, so a frontier
        # shrinks, and a final reading would under-report the memory high-water
        # mark that this number exists to describe.
        if len(frontier) > peak_frontier:
            peak_frontier = len(frontier)

        if state.visited == goal_mask:
            status = STATUS_SOLVED
            incumbent = state
            break

        expansions += 1
        for child in successors(state, probs, matrix, n):
            generated += 1
            if frontiers[child.vertex].is_pruned(child.cost, child.visited):
                pruned_on_generation += 1
                continue
            if child.visited == goal_mask and (
                    incumbent is None or child.cost < incumbent.cost):
                incumbent = child
            queue.push(child, gamma.estimate(child.vertex, child.survival,
                                             child.visited_count))

    stats = SearchStats(
        expansions=expansions,
        generated=generated,
        pruned_on_generation=pruned_on_generation,
        # The focal queue drops dead states itself, so its discards never reach
        # the loop's own counter. Adding them keeps the number comparable
        # between the two variants instead of reading zero for one of them.
        pruned_on_pop=pruned_on_pop + getattr(queue, "discarded", 0),
        peak_frontier=peak_frontier,
        search_s=time.monotonic() - started,
    )
    return _finish(problem, matrix, queue, incumbent, status, stats)


def _finish(problem, matrix, queue, incumbent, status, stats):
    # type: (object, Sequence[Sequence[float]], object, Optional[SearchState], str, SearchStats) -> SearchOutcome
    """Turn what the loop ended up holding into an outcome.

    Three cases, and the middle one is the reason this is not a one-liner.

    Args:
        problem: The problem, for the fallback route.
        matrix: The cost matrix.
        queue: The open list, still holding whatever was never expanded.
        incumbent: The best complete ordering the search reached, if any.
        status: How the loop ended.
        stats: What it did.

    Returns:
        The outcome.
    """
    lower_bound = queue.best_estimate()
    if status == STATUS_SOLVED and incumbent is not None:
        # The popped goal state is optimal for RPT*, and within the band for
        # F-RPT*. Either way its own cost is the tightest bound we can state.
        lower_bound = min(lower_bound, incumbent.cost)
    elif lower_bound == float("inf"):
        lower_bound = 0.0

    if incumbent is not None:
        return SearchOutcome(reconstruct(incumbent), incumbent.cost,
                             lower_bound, status, ROUTE_FROM_SEARCH, stats)

    if status == STATUS_BUDGET_EXCEEDED:
        # The budget expired before any complete ordering was even generated,
        # which is the normal outcome on a hard instance -- the search spends
        # its whole allowance on partial routes. Returning nothing here would
        # hand a mission node None at the exact moment it needs somewhere to
        # go, so a route is produced by the cheapest means available instead.
        # It changes no expansion decision and claims nothing: the status still
        # says the budget ran out, the guarantee is still withdrawn, and the
        # source says where the route came from.
        order = nearest_neighbour_order(problem, matrix)
        return SearchOutcome(order,
                             expected_cost(order, problem.probs, matrix),
                             lower_bound, status, ROUTE_FROM_FALLBACK, stats)

    return SearchOutcome((), float("inf"), lower_bound, STATUS_NO_ROUTE,
                         ROUTE_FROM_SEARCH, stats)


def _make_queue(params, frontiers):
    # type: (RptStarParams, List[VertexFrontier]) -> object
    """The exact queue, or the focal one when an epsilon was asked for."""
    if params.epsilon is None:
        return OpenList()

    def is_dead(state):
        # type: (SearchState) -> bool
        """Whether a vertex frontier has already evicted this state."""
        return frontiers[state.vertex].is_pruned(state.cost, state.visited)

    return FocalList(params.epsilon, is_dead)


def _out_of_budget(params, expansions, deadline):
    # type: (RptStarParams, int, Optional[float]) -> bool
    """Whether the expansion cap or the clock has stopped the search.

    The clock is read once every :data:`BUDGET_CHECK_INTERVAL` expansions --
    on an easy instance the whole search costs less than a few thousand clock
    reads would.
    """
    if params.max_expansions is not None and expansions >= params.max_expansions:
        return True
    if deadline is None:
        return False
    if expansions % BUDGET_CHECK_INTERVAL:
        return False
    return time.monotonic() >= deadline
