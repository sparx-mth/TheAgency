"""The two orderings RPT* is meant to beat, so that claim can be measured.

The paper compares against exactly these (Sec. VII-C, p.12), and its most
useful result is the one where they *win*: with an accurate prior, greedy
reaches the target faster than RPT* does, because it drives straight at the
answer while RPT* hedges (Table II, p.13). RPT* only pays off when the prior is
wrong (Table III), where greedy nearly triples and RPT* barely moves.

That is a trade, not a free win, so anyone deciding whether to fly this needs
both baselines to hand. They are here for that, and for the tests: an ordering
that RPT* cannot beat on expected cost is an ordering that says the search is
broken.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def greedy_probability_order(problem, matrix=None):
    # type: (object, Sequence[Sequence[float]]) -> Tuple[int, ...]
    """Always go to the most likely place left, whatever it costs to get there.

    The paper's ``Greedy`` baseline (p.12): "move to a vertex with the largest
    probability value among the unvisited". Distance never enters it, which is
    exactly why it doubles or triples the optimal expected cost on the paper's
    own benchmarks (Fig. 11a) -- and also why it wins when the prior happens to
    be right.

    Args:
        problem: A
            :class:`~sparx_agency.core.planning.routing.rpt_star.problem.RouteProblem`.
        matrix: Unused. Accepted so the baselines share one signature.

    Returns:
        The visiting order, as indices, starting at the start vertex.
    """
    probs = problem.probs
    remaining = [v for v in range(problem.n) if v != problem.start]
    remaining.sort(key=lambda v: (-probs[v], v))
    return (problem.start,) + tuple(remaining)


def nearest_neighbour_order(problem, matrix):
    # type: (object, Sequence[Sequence[float]]) -> Tuple[int, ...]
    """Always go to the closest place left, whatever the chance it is there.

    Stands in for the paper's ``LKH`` baseline: treat every probability as
    equal and just minimise distance. LKH is a far better tour heuristic than
    nearest-neighbour, so this is the weaker version of that comparison -- but
    it is the same *idea*, which is the part that matters. Distance-only
    ordering is what RPT* degenerates to when every probability is zero, so
    this is also the sanity check for that case.

    Args:
        problem: A
            :class:`~sparx_agency.core.planning.routing.rpt_star.problem.RouteProblem`.
        matrix: The cost matrix.

    Returns:
        The visiting order, as indices, starting at the start vertex.
    """
    remaining = set(range(problem.n))
    remaining.discard(problem.start)
    return tuple(_extend_greedily([problem.start], remaining, matrix))


def _extend_greedily(order, remaining, matrix):
    # type: (List[int], set, Sequence[Sequence[float]]) -> List[int]
    """Walk from the end of ``order`` to the closest place left, repeatedly.

    Shared by both nearest-neighbour constructions below, which differ only in
    the prefix they start from. Ties break on the lower index so the result is
    reproducible.

    Args:
        order: The route so far; extended in place and returned.
        remaining: Places not yet visited; emptied.
        matrix: The cost matrix.

    Returns:
        ``order``, now covering everything.
    """
    current = order[-1]
    while remaining:
        row = matrix[current]
        current = min(remaining, key=lambda v: (row[v], v))
        remaining.discard(current)
        order.append(current)
    return order


def path_length(order, matrix):
    # type: (Sequence[int], Sequence[Sequence[float]]) -> float
    """The raw travel distance of a route, with no probability weighting.

    This is the objective a plain HPP minimises, and therefore the one the
    ``LKH`` baseline optimises. It is *not* the HPP-PT objective; see
    :func:`~sparx_agency.core.planning.routing.rpt_star.objective.expected_cost`
    for that.

    Args:
        order: The visiting order, as vertex indices.
        matrix: The cost matrix.

    Returns:
        The sum of the edge costs along the route.
    """
    return sum(matrix[order[i]][order[i + 1]] for i in range(len(order) - 1))


def lkh_style_order(problem, matrix, max_passes=200):
    # type: (object, Sequence[Sequence[float]], int) -> Tuple[int, ...]
    """Ignore the belief and fly the shortest route that covers everywhere.

    The paper's ``LKH`` baseline (Sec. VII-C-1, p.12): "ignores the probability
    by treating the probability of all vertices simply as one. As a result,
    HPP-PT becomes a regular HPP", solved with LKH [15]. So the objective
    optimised here is :func:`path_length` -- raw distance, no ``q`` weighting --
    and the resulting order is then scored under the real expected-cost
    objective, which is what makes the comparison meaningful.

    **This is a stand-in for LKH, not LKH.** LKH is Lin-Kernighan-Helsgaun, a
    sequential k-opt heuristic that is very close to optimal on instances of
    this size. Vendoring it is out of scope, so this runs the strongest local
    search that stays in the standard library: nearest-neighbour construction
    from every possible first vertex, each improved to a local optimum under
    both 2-opt (segment reversal) and Or-opt (segment relocation, lengths 1-3),
    keeping the shortest.

    That matters for honesty of the comparison. Plain nearest-neighbour --
    which :func:`nearest_neighbour_order` provides, and which is what a weaker
    stand-in would use -- typically lands 15-25% above the optimal tour, so
    substituting it would credit RPT\\* with a win that belongs to the baseline
    being bad. This lands close enough to optimal on 20-40 vertices that the
    remaining gap is the algorithmic difference the paper is claiming, which is
    the only reason to measure it at all.

    The start vertex is held fixed at position zero throughout, so this solves
    the fixed-start, free-end Hamiltonian path directly. That is what Def. 2
    asks for. The paper's footnote 3 instead reaches it by adding a zero-cost
    copy of the start and handing the result to a tour solver, which is the
    conversion LKH needs because LKH only solves closed tours; optimising the
    open path directly is the same problem without the encoding step.

    Args:
        problem: A
            :class:`~sparx_agency.core.planning.routing.rpt_star.problem.RouteProblem`.
        matrix: The cost matrix.
        max_passes: Safety cap on improvement sweeps per restart. The local
            search converges long before this on the sizes used here; the cap
            only stops a pathological matrix spinning forever.

    Returns:
        The visiting order, as indices, starting at the start vertex.
    """
    n = problem.n
    start = problem.start
    if n <= 2:
        return (start,) + tuple(v for v in range(n) if v != start)

    best = None                                 # type: Optional[List[int]]
    best_length = float("inf")
    # Restarting from every second vertex costs |V| local searches and removes
    # the dependence on which greedy start happened to be lucky. At the paper's
    # sizes this is milliseconds and it is what keeps the baseline strong.
    for first_hop in range(n):
        if first_hop == start:
            continue
        order = _nearest_neighbour_from(start, first_hop, matrix, n)
        order = _local_search(order, matrix, max_passes)
        length = path_length(order, matrix)
        if length < best_length:
            best_length = length
            best = order
    return tuple(best if best is not None else [start])


def _nearest_neighbour_from(start, first_hop, matrix, n):
    # type: (int, int, Sequence[Sequence[float]], int) -> List[int]
    """Nearest-neighbour construction forced through a given first hop."""
    remaining = set(range(n))
    remaining.discard(start)
    remaining.discard(first_hop)
    return _extend_greedily([start, first_hop], remaining, matrix)


def _local_search(order, matrix, max_passes):
    # type: (List[int], Sequence[Sequence[float]], int) -> List[int]
    """Improve an open path to a 2-opt and Or-opt local optimum.

    Position zero is never moved, so the start stays the start. Both moves are
    applied until a full sweep finds nothing, which is the standard definition
    of a local optimum under the combined neighbourhood.
    """
    for _ in range(max_passes):
        if not (_two_opt_pass(order, matrix) or _or_opt_pass(order, matrix)):
            return order
    return order


def _two_opt_pass(order, matrix):
    # type: (List[int], Sequence[Sequence[float]]) -> bool
    """Reverse any segment that shortens the path. True if anything improved.

    For an open path, reversing ``order[i:j+1]`` replaces the edge entering the
    segment and the edge leaving it. When the segment runs to the end of the
    path there is no leaving edge, which is the case a tour-shaped 2-opt gets
    wrong and an open-path one must handle.
    """
    n = len(order)
    improved = False
    for i in range(1, n - 1):
        before = order[i - 1]
        for j in range(i + 1, n):
            delta = (matrix[before][order[j]] - matrix[before][order[i]])
            if j + 1 < n:
                delta += (matrix[order[i]][order[j + 1]]
                          - matrix[order[j]][order[j + 1]])
            if delta < -1e-12:
                order[i:j + 1] = reversed(order[i:j + 1])
                improved = True
                before = order[i - 1]
    return improved


def _or_opt_pass(order, matrix):
    # type: (List[int], Sequence[Sequence[float]]) -> bool
    """Relocate short runs of 1-3 vertices elsewhere. True if anything improved.

    2-opt cannot move a single misplaced vertex without reversing everything
    between it and its destination, so the two neighbourhoods catch different
    mistakes and a search using only one of them stops well short.
    """
    n = len(order)
    improved = False
    for span in (1, 2, 3):
        i = 1
        while i + span <= n:
            segment = order[i:i + span]
            rest = order[:i] + order[i + span:]
            current = path_length(order, matrix)
            for position in range(1, len(rest) + 1):
                if position == i:
                    continue
                candidate = rest[:position] + segment + rest[position:]
                if path_length(candidate, matrix) < current - 1e-12:
                    order[:] = candidate
                    improved = True
                    break
            i += 1
    return improved


