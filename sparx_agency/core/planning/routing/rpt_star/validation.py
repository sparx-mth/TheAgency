"""Everything checked before a single state is expanded.

The whole point of checking here is that the failures this catches do not look
like failures later. A cost matrix with a missing entry produces a route that
skips somewhere; a matrix where a detour beats going direct produces a route
that is merely not the best one, with nothing anywhere to say so. By the time
either reaches a flight log it is indistinguishable from a planner that simply
made an odd choice.

So the split is: **anything that would invalidate the answer raises; anything
that is merely unusual is returned as a warning.** The line matters most in one
place. The paper notes (Remark 1, p.3) that probabilities summing to one is a
*modelling* statement about the single-target case, and that the algorithm does
not depend on it -- nothing in the search ever sums the probabilities. An LLM
ranking violates it constantly and is still perfectly usable. So a sum that is
not one is a warning, never an error.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from sparx_agency.core.planning.routing.rpt_star.errors import (
    DisconnectedGraphError,
    InvalidCostError,
    TriangleInequalityError,
)

#: How far the probabilities may stray from summing to one before it is worth
#: a warning. Generous, because a normalised ranking that has been filtered is
#: the normal case, not a bug.
SUM_PROB_TOLERANCE = 0.05

#: Relative slack in the triangle-inequality test. Costs that came from a grid
#: search are sums of many floats, so demanding exactness would reject matrices
#: that are correct to every digit that matters.
TRIANGLE_TOLERANCE = 1e-9


def validate(problem, matrix, require_triangle_inequality=True):
    # type: (object, Sequence[Sequence[float]], bool) -> Tuple[str, ...]
    """Check a problem and its costs, raising on anything that breaks the answer.

    Args:
        problem: A
            :class:`~sparx_agency.core.planning.routing.rpt_star.problem.RouteProblem`.
        matrix: The cost matrix.
        require_triangle_inequality: Whether a detour that beats going direct
            is fatal. It is, for optimality -- see
            :class:`~sparx_agency.core.planning.routing.rpt_star.errors.TriangleInequalityError`.

    Returns:
        Warnings: things worth saying that do not invalidate anything.

    Raises:
        InvalidCostError: The matrix is the wrong shape.
        DisconnectedGraphError: Some ordered pair has no finite cost.
        TriangleInequalityError: A detour beats going direct, and it was
            required not to.
    """
    n = problem.n
    if len(matrix) != n:
        raise InvalidCostError(
            "cost matrix has %d rows for a %d-vertex problem"
            % (len(matrix), n))
    for index, row in enumerate(matrix):
        if len(row) != n:
            raise InvalidCostError(
                "cost matrix row %d has %d entries, expected %d"
                % (index, len(row), n))

    _require_usable_costs(problem, matrix)
    if require_triangle_inequality:
        _require_triangle_inequality(problem, matrix)
    return tuple(_warnings(problem, matrix, require_triangle_inequality))


def _require_usable_costs(problem, matrix):
    # type: (object, Sequence[Sequence[float]]) -> None
    """Every ordered pair needs a finite, non-negative cost.

    Both halves matter, and both are checked *here* rather than only in the
    builders in :mod:`.costs`, because ``solve`` accepts any sequence of
    sequences. A caller who assembles a matrix by hand -- or takes one through
    :func:`~sparx_agency.core.planning.routing.rpt_star.costs.metric_closure`,
    which preserves whatever it is given -- never passes through those builders
    and would otherwise get no check at all.

    A negative edge breaks the pruning the same way a triangle violation does,
    and just as silently: it makes flying further able to reduce the expected
    cost, so a shortcut is no longer always an improvement and Lemma 6 fails.

    Raises:
        InvalidCostError: On a negative or NaN cost.
        DisconnectedGraphError: On an infinite one -- a different failure with
            a different remedy, so a different exception.
    """
    missing = []                                # type: List[Tuple[int, int]]
    for u in range(problem.n):
        row = matrix[u]
        for w in range(problem.n):
            if u == w:
                continue
            cost = row[w]
            if math.isnan(cost):
                raise InvalidCostError(
                    "cost c(%r,%r) is NaN"
                    % (problem.id_of(u), problem.id_of(w)))
            if cost < 0.0:
                raise InvalidCostError(
                    "cost c(%r,%r) is negative (%r). The paper's edge costs "
                    "are in R+ (p.3); a negative edge lets the expected cost "
                    "fall by flying further, which makes the dominance "
                    "pruning unsound and the answer silently sub-optimal."
                    % (problem.id_of(u), problem.id_of(w), cost))
            if math.isinf(cost):
                missing.append((problem.id_of(u), problem.id_of(w)))
    if missing:
        raise DisconnectedGraphError(missing)


def _require_triangle_inequality(problem, matrix):
    # type: (object, Sequence[Sequence[float]]) -> None
    """Reject a matrix in which going via somewhere else is cheaper.

    Cubic, which is the same order as building the heuristic table, so it adds
    a constant factor and not a complexity class.

    Only the worst violation is reported: a matrix that fails this usually
    fails it in many places, and a caller needs one concrete example plus a
    count, not a list of thousands.
    """
    n = problem.n
    worst_excess = 0.0
    worst_triple = None                         # type: Tuple[int, int, int]
    count = 0
    for b in range(n):
        row_b = matrix[b]
        for a in range(n):
            if a == b:
                continue
            direct_row = matrix[a]
            via_first = direct_row[b]
            for c in range(n):
                if c == a or c == b:
                    continue
                excess = direct_row[c] - (via_first + row_b[c])
                if excess > TRIANGLE_TOLERANCE * max(1.0, abs(direct_row[c])):
                    count += 1
                    if excess > worst_excess:
                        worst_excess = excess
                        worst_triple = (a, b, c)
    if worst_triple is not None:
        raise TriangleInequalityError(
            (problem.id_of(worst_triple[0]), problem.id_of(worst_triple[1]),
             problem.id_of(worst_triple[2])),
            worst_excess, count)


def _warnings(problem, matrix, require_triangle_inequality):
    # type: (object, Sequence[Sequence[float]], bool) -> List[str]
    """Unusual but legal: things a caller may want to look at."""
    out = []                                    # type: List[str]
    if not require_triangle_inequality:
        out.append(
            "the triangle-inequality check was waived, so the dominance rule "
            "of Def. 3 may discard the optimal route and the search cannot "
            "tell that it did. The result carries guarantee='none' and no "
            "lower bound for this reason.")
    total = sum(problem.probs)
    if abs(total - 1.0) > SUM_PROB_TOLERANCE:
        out.append(
            "probabilities sum to %.4f, not 1.0. This is legal -- the paper's "
            "Remark 1 (p.3) says the algorithm does not depend on it -- but if "
            "the intent was a single target that is certainly present, the "
            "distribution is not normalised." % total)
    if all(p == 0.0 for p in problem.probs):
        out.append(
            "every probability is zero, so the expected cost reduces to the "
            "plain shortest Hamiltonian path and the ordering carries no "
            "belief about where the target is.")
    zero_edges = sum(1 for u in range(problem.n) for w in range(problem.n)
                     if u != w and matrix[u][w] == 0.0)
    if zero_edges:
        out.append(
            "%d edge(s) cost zero, so those places are being treated as the "
            "same location and their visiting order is arbitrary."
            % zero_edges)
    return out
