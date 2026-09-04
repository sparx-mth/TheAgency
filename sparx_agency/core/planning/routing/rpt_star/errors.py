"""What a caller can get wrong, named so the message says which.

Every one of these is raised at ingest, before a single state is expanded. That
is deliberate: the failure mode this package exists to avoid is a search that
runs to completion, returns a route, and is quietly not optimal because an
assumption the proofs rest on was violated three files away. A loud exception
before the search beats a plausible answer after it.

:class:`RoutingInternalError` sits on a different base class on purpose -- it
means *our* invariant broke, never the caller's input, and a caller who writes
``except RoutingError`` should not swallow it.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations


class RoutingError(ValueError):
    """Base for every input-contract failure.

    A :class:`ValueError` so a caller may catch the whole family broadly
    without also catching a bug in this package.
    """


class InvalidProblemError(RoutingError):
    """The vertex set itself is malformed.

    An empty vertex set, a duplicate vertex id, a start id that is not in the
    set, or a probability outside ``[0, 1)``.
    """


class InvalidCostError(RoutingError):
    """An edge cost is missing, negative, NaN or infinite where one is needed.

    The search reads every off-diagonal entry while building the heuristic
    table, so a hole in the matrix is fatal and is worth saying so up front
    rather than at the expansion that happens to reach it.
    """


class TriangleInequalityError(RoutingError):
    """Some ordered triple violates ``c(a,c) <= c(a,b) + c(b,c)``.

    This is the quiet one. The search still runs, still terminates, and still
    returns a legal route whose reported cost is honestly computed -- it is
    merely no longer guaranteed to be the cheapest, because the dominance rule
    that prunes the search (Def. 3 in the paper) is only sound when a detour
    through a third vertex cannot beat going direct.

    Shortest-path costs satisfy the inequality by construction, so a caller
    whose costs came from A* over a grid will never see this. A caller who
    sees it has approximated something, and wants to know.

    Attributes:
        triple: The worst offending ``(a, b, c)`` as caller-facing vertex ids.
        excess: How much ``c(a,c)`` exceeds ``c(a,b) + c(b,c)``, in cost units.
        count: How many ordered triples violate it in total.
    """

    def __init__(self, triple, excess, count):
        # type: (tuple, float, int) -> None
        message = (
            "edge costs violate the triangle inequality: "
            "c(%r,%r) exceeds c(%r,%r) + c(%r,%r) by %.6g, and %d ordered "
            "triple(s) do this. RPT*'s dominance pruning (Def. 3) is unsound "
            "without it, so the result would not be optimal. Shortest-path "
            "costs never violate it -- see metric_closure() in costs.py."
            % (triple[0], triple[2], triple[0], triple[1], triple[1],
               triple[2], excess, count))
        super(TriangleInequalityError, self).__init__(message)
        self.triple = triple
        self.excess = float(excess)
        self.count = int(count)


class DisconnectedGraphError(RoutingError):
    """Some pair of vertices has no finite cost between them.

    RPT* plans a route through *every* vertex, so one unreachable vertex makes
    the whole instance infeasible -- not merely expensive. A caller must drop
    such a vertex from the problem rather than hand it a large finite cost,
    which would look feasible and silently distort the ordering of everything
    else.

    Attributes:
        pairs: Up to a handful of offending ``(from_id, to_id)`` pairs.
    """

    def __init__(self, pairs):
        # type: (list) -> None
        shown = ", ".join("%r->%r" % (a, b) for a, b in pairs[:5])
        message = (
            "no finite cost for %d vertex pair(s) (%s%s). A Hamiltonian path "
            "must traverse every vertex, so an unreachable vertex makes the "
            "instance infeasible -- drop it from the problem instead of "
            "giving it a large cost."
            % (len(pairs), shown, ", ..." if len(pairs) > 5 else ""))
        super(DisconnectedGraphError, self).__init__(message)
        self.pairs = list(pairs)


class RoutingInternalError(RuntimeError):
    """A search invariant broke. Our bug, never the caller's input.

    Raised by the tripwire that re-scores the reconstructed route against the
    g-value the search terminated on. If those two disagree, something in the
    state machine is wrong and the route must not be flown.
    """
