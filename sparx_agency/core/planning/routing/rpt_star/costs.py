"""Getting an N x N cost matrix in, from whatever shape the caller has it.

The search reads every off-diagonal cost before it expands a single state --
the heuristic table (Eq. 5, p.5) is built by sweeping the whole matrix -- so
there is no laziness to exploit and the internal representation is always a
frozen dense matrix. What varies is the shape the numbers arrive in, and the
one that matters is :func:`costs_from_row_callback`.

**Why the row callback earns its place.** The natural cost between two places
is the length of the path a planner would actually fly, and a grid search from
one place is a wavefront that computes the distance to *every* other place on
its way out. Asking a caller for one cost at a time throws that away: twenty
rooms become 380 separate A* runs instead of 20. A caller who has Dijkstra
should hand back a whole row.

**The diagonal is infinity, never zero.** A zero diagonal is the kind of value
that makes a buggy recurrence look plausible -- ``min`` over a row would happily
return "stay where you are, for free" forever. Infinity makes that bug loud.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sparx_agency.core.planning.routing.rpt_star.errors import InvalidCostError

#: The diagonal. Never zero -- see the module docstring.
NO_EDGE = float("inf")

#: A dense, immutable cost matrix: ``matrix[u][w]`` is ``c(u, w)``. Directed,
#: so ``c(u, w)`` need not equal ``c(w, u)``; the paper's graph is directed
#: (p.3) and nothing in the algorithm assumes symmetry.
CostMatrix = Tuple[Tuple[float, ...], ...]


def dense_costs(rows):
    # type: (Sequence[Sequence[float]]) -> CostMatrix
    """Freeze a square list-of-lists into a cost matrix.

    The diagonal is overwritten with :data:`NO_EDGE` whatever it held, since a
    Hamiltonian path never traverses a self-loop and a finite diagonal is only
    ever a source of confusion.

    Args:
        rows: ``n`` rows of ``n`` numbers.

    Returns:
        The frozen matrix.

    Raises:
        InvalidCostError: If the input is not square, or an off-diagonal entry
            is negative or NaN.
    """
    n = len(rows)
    out = []                                    # type: List[Tuple[float, ...]]
    for u, row in enumerate(rows):
        if len(row) != n:
            raise InvalidCostError(
                "cost matrix must be square: row %d has %d entries, expected %d"
                % (u, len(row), n))
        frozen = []                             # type: List[float]
        for w, value in enumerate(row):
            if u == w:
                frozen.append(NO_EDGE)
                continue
            cost = float(value)
            if math.isnan(cost):
                raise InvalidCostError(
                    "cost c(%d,%d) is NaN" % (u, w))
            if cost < 0.0:
                raise InvalidCostError(
                    "cost c(%d,%d) is negative (%r). The paper's edge costs "
                    "are in R+ (p.3), and a negative edge would let the "
                    "expected cost decrease by flying further." % (u, w, value))
            frozen.append(cost)
        out.append(tuple(frozen))
    return tuple(out)


def costs_from_pairs(n, pairs, default=NO_EDGE):
    # type: (int, Mapping[Tuple[int, int], float], float) -> CostMatrix
    """Build a matrix from a sparse ``{(u, w): cost}`` mapping.

    Anything absent takes ``default``, which is :data:`NO_EDGE` -- so a caller
    who forgets a pair gets :class:`DisconnectedGraphError` from validation
    rather than a route built on a number nobody chose.

    Args:
        n: The vertex count.
        pairs: Known costs, keyed by ordered index pair.
        default: What an absent pair costs.

    Returns:
        The frozen matrix.
    """
    rows = [[default] * n for _ in range(n)]
    for (u, w), cost in pairs.items():
        if not (0 <= u < n and 0 <= w < n):
            raise InvalidCostError(
                "cost pair (%d,%d) is outside a %d-vertex problem" % (u, w, n))
        rows[u][w] = float(cost)
    return dense_costs(rows)


def costs_from_row_callback(n, row_of):
    # type: (int, Callable[[int], Sequence[float]]) -> CostMatrix
    """Build a matrix by asking for one whole row at a time.

    The shape to use when each cost is a graph search: one call per source
    vertex, ``n`` searches rather than ``n * (n - 1)``.

    Args:
        n: The vertex count.
        row_of: Called once per source index ``u``, returns ``n`` costs
            ``c(u, 0) .. c(u, n-1)``. Its own diagonal entry is ignored.

    Returns:
        The frozen matrix.

    Raises:
        InvalidCostError: If a row is the wrong length.
    """
    rows = []                                   # type: List[Sequence[float]]
    for u in range(n):
        row = list(row_of(u))
        if len(row) != n:
            raise InvalidCostError(
                "row callback returned %d costs for vertex %d, expected %d"
                % (len(row), u, n))
        rows.append(row)
    return dense_costs(rows)


def costs_from_points(points, distance=None):
    # type: (Sequence[Sequence[float]], Optional[Callable]) -> CostMatrix
    """Straight-line costs between coordinates, for tests and for a first cut.

    Euclidean distance satisfies the triangle inequality, so a matrix built
    this way always passes validation. It is not what should be flown -- a
    straight line between two rooms goes through the wall between them -- but
    it is the right thing for a unit test and a reasonable placeholder before
    a real planner is wired in.

    **Every point must have the same number of coordinates**, and mixing them
    is refused rather than tolerated. Zipping a three-component robot pose
    against a two-component room centroid would silently measure only the
    shared prefix: the result is still a perfectly good metric over the
    projection, so validation passes and the search returns a provably optimal
    route -- to the wrong problem, stamped optimal. A caller assembling a list
    by hand from poses and map centroids is exactly the case where this
    happens, so it raises.

    Args:
        points: One coordinate tuple per vertex, all the same length.
        distance: Optional replacement metric ``(a, b) -> float``. Supplying
            one waives the length check, since a custom metric may well know
            what to do with mixed inputs.

    Returns:
        The frozen matrix.

    Raises:
        InvalidCostError: If the points have differing lengths, or the
            coordinates are large enough that a squared term overflows.
    """
    if distance is None:
        _require_uniform_dimension(points)
        distance = _euclidean
    n = len(points)
    rows = [[0.0] * n for _ in range(n)]
    for u in range(n):
        for w in range(n):
            if u != w:
                cost = float(distance(points[u], points[w]))
                if math.isinf(cost):
                    raise InvalidCostError(
                        "the distance between points %d and %d overflowed to "
                        "infinity; the coordinates are too large to square"
                        % (u, w))
                rows[u][w] = cost
    return dense_costs(rows)


def _require_uniform_dimension(points):
    # type: (Sequence[Sequence[float]]) -> None
    """Refuse a mix of 2D and 3D coordinates. See :func:`costs_from_points`."""
    if not points:
        return
    expected = len(points[0])
    for index, point in enumerate(points):
        if len(point) != expected:
            raise InvalidCostError(
                "point %d has %d coordinates but point 0 has %d. Mixing "
                "dimensions would silently measure only the shared prefix -- "
                "a valid metric over the wrong problem, which the search "
                "would then solve optimally and report as optimal."
                % (index, len(point), expected))


def metric_closure(matrix):
    # type: (CostMatrix) -> CostMatrix
    """Replace every cost by the cheapest route, direct or indirect.

    The repair for a matrix that fails the triangle-inequality check, and the
    way to make a sparse graph complete. After this every entry is a genuine
    shortest-path cost, which satisfies the inequality by construction and is
    what the paper's own system uses (HATS-L takes edge weights from an A*
    search between each pair of locations, p.12).

    It is offered rather than applied: a cost that gets quietly rewritten under
    a caller is a cost the caller can no longer reason about, and the detour it
    implies is a real detour the robot would have to fly.

    Floyd-Warshall, ``O(n^3)``.

    Args:
        matrix: The costs as given.

    Returns:
        A new matrix of all-pairs shortest costs, diagonal :data:`NO_EDGE`.
    """
    n = len(matrix)
    best = [list(row) for row in matrix]
    for via in range(n):
        row_via = best[via]
        for u in range(n):
            through = best[u][via]
            if through == NO_EDGE:
                continue
            row_u = best[u]
            for w in range(n):
                if u == w:
                    continue
                candidate = through + row_via[w]
                if candidate < row_u[w]:
                    row_u[w] = candidate
    for u in range(n):
        best[u][u] = NO_EDGE
    return tuple(tuple(row) for row in best)


def _euclidean(a, b):
    # type: (Sequence[float], Sequence[float]) -> float
    """Straight-line distance in any dimension."""
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))
