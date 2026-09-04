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

from typing import List, Sequence, Tuple


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
    current = problem.start
    remaining.discard(current)
    order = [current]                           # type: List[int]
    while remaining:
        row = matrix[current]
        current = min(remaining, key=lambda v: (row[v], v))
        remaining.discard(current)
        order.append(current)
    return tuple(order)
