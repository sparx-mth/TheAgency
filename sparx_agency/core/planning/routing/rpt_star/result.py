"""What a solve hands back, and how much of it you are allowed to believe.

Two fields say what happened and they are deliberately not one field:

* :attr:`RouteSolution.status` -- how the search terminated. Did it prove it
  was done, or did it run out of time?
* :attr:`RouteSolution.guarantee` -- what the answer is worth. Optimal, within
  a factor, or nothing at all?

Collapsing them loses the case that matters: a search can terminate perfectly
cleanly, having explored everything it meant to, and still owe no guarantee --
because the caller waived the triangle-inequality check, or because the budget
expired and the route being returned is the best complete one that happened to
turn up. A single "success" flag reports that as success, and a route that is
quietly forty percent worse than it should be is exactly the failure this
package is built to make impossible.

Every result also carries :attr:`RouteSolution.lower_bound`, so it certifies
itself: no ordering of these places can cost less than that, whatever happened
during the search.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from sparx_agency.core.planning.routing.rpt_star.objective import RouteLeg

# -- how the search ended -------------------------------------------------

#: The search emptied its queue or reached a goal state having proved it best.
STATUS_SOLVED = "solved"
#: The expansion cap or the clock stopped it. A route may still be returned.
STATUS_BUDGET_EXCEEDED = "budget_exceeded"
#: The queue emptied with no complete route. The graph admits no Hamiltonian
#: path, which for a complete graph means an edge was missing.
STATUS_NO_ROUTE = "no_route"

# -- what the answer is worth ---------------------------------------------

#: Provably the cheapest ordering there is.
GUARANTEE_OPTIMAL = "optimal"
#: Within ``(1 + epsilon)`` of the cheapest -- see :attr:`RouteSolution.epsilon`.
GUARANTEE_BOUNDED = "bounded"
#: A valid route with no quality claim attached.
GUARANTEE_NONE = "none"

# -- where the route came from --------------------------------------------

#: The search found it.
ROUTE_FROM_SEARCH = "search"
#: The budget expired before the search completed any ordering, so a cheap
#: constructive fallback produced one. Always paired with a withdrawn
#: guarantee -- it is somewhere to fly, not an answer.
ROUTE_FROM_FALLBACK = "fallback"


@dataclass(frozen=True)
class SearchStats:
    """What the search did, for a log or a budget that needs sizing.

    Attributes:
        expansions: States expanded -- the honest measure of search effort,
            and the one a test should pin rather than a wall-clock time.
        generated: Successor states created.
        pruned_on_generation: Successors discarded before ever being queued.
        pruned_on_pop: States discarded when they surfaced, because something
            better had reached their vertex in the meantime. Includes the ones
            the focal queue drops internally, so the figure means the same
            thing for both variants.
        peak_frontier: The largest number of mutually non-dominated routes held
            at any single vertex. The memory term, and the thing that grows
            when an instance is about to become hopeless.
        heuristic_build_s: Seconds spent building the gamma table. Cubic in the
            vertex count, and on a large instance it can dominate.
        search_s: Seconds spent searching.
    """

    expansions: int = 0
    generated: int = 0
    pruned_on_generation: int = 0
    pruned_on_pop: int = 0
    peak_frontier: int = 0
    heuristic_build_s: float = 0.0
    search_s: float = 0.0


@dataclass(frozen=True)
class RouteSolution:
    """A visiting order, what it is expected to cost, and what that is worth.

    Attributes:
        order: The places to visit, as the caller's own ids, starting with the
            start vertex.
        order_indices: The same route as dense indices, for comparing against
            the paper or a brute-force oracle.
        next_id: The id to actually go to now -- ``order[1]``, or None when
            there is nowhere to go. The whole ordering is planned and only this
            is executed; the rest is replanned once something has been learned.
        expected_cost: The expected cost of the whole ordering, by Lemma 1.
        lower_bound: No ordering of these places can cost less than this. Equal
            to :attr:`expected_cost` when the guarantee is optimal, and the
            only thing a timed-out search can still say for certain.
        bound_ratio: :attr:`expected_cost` divided by :attr:`lower_bound` -- the
            sub-optimality actually realised, which is usually far tighter than
            the epsilon that was allowed. The number worth logging.
        status: One of :data:`STATUS_SOLVED`, :data:`STATUS_BUDGET_EXCEEDED`,
            :data:`STATUS_NO_ROUTE`.
        guarantee: One of :data:`GUARANTEE_OPTIMAL`, :data:`GUARANTEE_BOUNDED`,
            :data:`GUARANTEE_NONE`.
        route_source: :data:`ROUTE_FROM_SEARCH` or :data:`ROUTE_FROM_FALLBACK`
            -- whether the search produced this route or, having run out of
            budget before completing any ordering, a constructive fallback did.
        epsilon: The bound that was asked for, or None for an exact search.
        legs: Per-hop breakdown, so a caller can see where the cost went.
        warnings: Things validation thought were worth saying but not worth
            refusing -- probabilities that do not sum to one, zero-cost edges.
        stats: What the search did.
    """

    order: Tuple[Any, ...] = ()
    order_indices: Tuple[int, ...] = ()
    next_id: Optional[Any] = None
    expected_cost: float = float("inf")
    lower_bound: float = 0.0
    bound_ratio: float = float("inf")
    status: str = STATUS_NO_ROUTE
    guarantee: str = GUARANTEE_NONE
    route_source: str = ROUTE_FROM_SEARCH
    epsilon: Optional[float] = None
    legs: Tuple[RouteLeg, ...] = ()
    warnings: Tuple[str, ...] = ()
    stats: SearchStats = field(default_factory=SearchStats)

    @property
    def found(self):
        # type: () -> bool
        """Whether there is a route to fly at all."""
        return bool(self.order_indices)

    @property
    def is_optimal(self):
        # type: () -> bool
        """Whether the route is provably the cheapest ordering."""
        return self.guarantee == GUARANTEE_OPTIMAL

    def __repr__(self):
        # type: () -> str
        return ("RouteSolution(%s/%s via %s, next=%r, cost=%.6g, "
                "bound=%.6g, ratio=%.4f, expansions=%d)"
                % (self.status, self.guarantee, self.route_source,
                   self.next_id, self.expected_cost, self.lower_bound,
                   self.bound_ratio, self.stats.expansions))
