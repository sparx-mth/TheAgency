"""The search state, and the one substitution that makes it a search at all.

A route's expected cost depends on the *order* it was built in, not just on the
set of edges it used -- an edge late in the route is only paid for if nothing
was found earlier. That history dependence is what stops ordinary shortest-path
machinery working, and the paper's answer (Lemma 1, p.4) is to carry the
history as a single number: ``q``, the probability of still having found
nothing.

With ``q`` in the state, the successor rules (Eqs. 2-4, p.4) read only the
state and never its ancestors, so the state is Markovian (Lemma 2) and A*
applies. That is the entire idea. Everything else in the paper is standard
search machinery wrapped around it.

**The visited set is a bitmask.** ``A`` is compared on every dominance test,
which is the hot loop, so it is a Python ``int`` used as a bit set: the superset
test ``A1 >= A2`` becomes ``a1 & a2 == a2``, one machine instruction on small
sets. It also makes the state hashable and cheap to copy.

**The start vertex is in ``A`` from the beginning.** See the package README for
why the paper's ``A = {}`` cannot be right.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class SearchState(object):
    """One partial route: where it is, what it cost, and what it has visited.

    Attributes:
        vertex: ``v(s)``, the vertex the partial route ends at.
        cost: ``g(s)``, the expected cost of the partial route so far. Equal to
            the Lemma 1 sum over the edges taken.
        survival: ``q(s)``, the probability of having found nothing at any of
            the visited vertices.
        visited: ``A(s)`` as a bitmask, including ``vertex`` itself.
        visited_count: ``|A(s)|``. Cached because the heuristic needs it on
            every evaluation and ``bin(x).count('1')`` is not free.
        parent: The state this one was expanded from, or None for the start.
    """

    __slots__ = ("vertex", "cost", "survival", "visited", "visited_count",
                 "parent")

    def __init__(self, vertex, cost, survival, visited, visited_count, parent):
        # type: (int, float, float, int, int, Optional[SearchState]) -> None
        self.vertex = vertex
        self.cost = cost
        self.survival = survival
        self.visited = visited
        self.visited_count = visited_count
        self.parent = parent

    def __repr__(self):
        # type: () -> str
        return ("SearchState(v=%d, g=%.6g, q=%.6g, |A|=%d)"
                % (self.vertex, self.cost, self.survival, self.visited_count))


def initial_state(start, probs):
    # type: (int, Sequence[float]) -> SearchState
    """The state the search begins from (Algorithm 1, line 1, corrected).

    ``g = 0`` -- nothing has been flown yet. ``q = 1 - p(v_s)`` -- the start
    has already been searched, so its probability is consumed, which matches
    ``q_1`` in Lemma 1. ``A = {v_s}`` -- the start counts as visited, which is
    the correction to the paper's printed ``A = {}``; without it the goal test
    ``A(s) = V`` is unreachable, the start could be revisited, and the
    heuristic indexes one row past the end of its own table.

    Args:
        start: The start vertex index.
        probs: ``p(v)`` by index.

    Returns:
        The initial state.
    """
    return SearchState(
        vertex=start,
        cost=0.0,
        survival=1.0 - probs[start],
        visited=1 << start,
        visited_count=1,
        parent=None,
    )


def successors(state, probs, matrix, n):
    # type: (SearchState, Sequence[float], Sequence[Sequence[float]], int) -> List[SearchState]
    """Extend a partial route to every vertex it has not visited (Eqs. 2-4).

    ``g' = g + q * c(v, v')`` -- the new edge is charged at the *current*
    survival probability, not the new one, because the robot only flies it if
    it has not already found the target.

    ``q' = q * (1 - p(v'))`` and ``A' = A + {v'}``.

    Args:
        state: The state to expand.
        probs: ``p(v)`` by index.
        matrix: The cost matrix.
        n: The vertex count.

    Returns:
        One successor per unvisited vertex, in index order. Edges of infinite
        cost are skipped -- validation should already have rejected them, but a
        search that silently propagates infinity is worse than one that does
        not generate the child.
    """
    out = []                                    # type: List[SearchState]
    visited = state.visited
    row = matrix[state.vertex]
    for target in range(n):
        if visited >> target & 1:
            continue
        edge = row[target]
        if edge == float("inf"):
            continue
        out.append(SearchState(
            vertex=target,
            cost=state.cost + state.survival * edge,
            survival=state.survival * (1.0 - probs[target]),
            visited=visited | (1 << target),
            visited_count=state.visited_count + 1,
            parent=state,
        ))
    return out


def reconstruct(state):
    # type: (SearchState) -> Tuple[int, ...]
    """Walk the parent pointers back to the start and return the route forwards.

    Args:
        state: A goal state.

    Returns:
        The visiting order as vertex indices, starting at the start vertex.
    """
    order = []                                  # type: List[int]
    current = state                             # type: Optional[SearchState]
    while current is not None:
        order.append(current.vertex)
        current = current.parent
    order.reverse()
    return tuple(order)
