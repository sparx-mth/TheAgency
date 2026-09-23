"""F-RPT*'s queue: dive for a complete route, but never leave the cost band.

The exact search expands whatever looks cheapest, which on a large graph means
exploring an enormous number of half-finished routes before it ever completes
one. The paper's observation (p.6) is that this is largely wasted: the survival
probability ``q`` is a product of ``(1 - p)`` terms, so it decays along every
route, the cost stops growing, and thousands of orderings end up within a hair
of the optimum. Paying an exponential price to separate them is a poor trade.

So F-RPT* keeps a second queue. FOCAL holds those states whose ``f`` is within
a factor ``(1 + eps)`` of the best bound the search has proved so far, and
among those it expands the one with the *fewest places left to visit*. That
drives straight at a complete route. Because everything popped was inside the
band, the route that comes out is within ``(1 + eps)`` of optimal (Theorem 3,
p.7).

**The paper defines FOCAL and does not say how to maintain it.** Section IV-D
gives the membership rule and stops. What follows is the standard construction,
and the ordering of its four steps is the whole of its correctness:

1. Drop dead states from the front of FOCAL -- ones a vertex frontier has since
   evicted, or that fell out of the band when it moved.
2. Read the bound from OPEN. This must happen *before* admitting, or the band
   is computed from a stale minimum.
3. Admit newly-eligible states from the pending heap, which is ordered by ``f``
   so admission stops at the first state above the band and resumes there next
   time. The band only ever moves up -- ``f_min`` is non-decreasing under an
   admissible heuristic -- so a state admitted once never has to be
   reconsidered.
4. Pop from FOCAL. If FOCAL is somehow empty, fall back to OPEN, which is still
   correct: the state with the minimum ``f`` is trivially inside its own band.

Measured against exhaustive search on several hundred random instances, this
honours the bound at every epsilon tried, and ``eps = 0`` reproduces the exact
optimum.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import heapq
from typing import Callable, List, Optional, Tuple

from sparx_agency.core.planning.routing.rpt_star.state import SearchState


class FocalList(object):
    """OPEN and FOCAL together, behind the same contract as :class:`OpenList`.

    Args:
        epsilon: The sub-optimality factor. ``0.0`` makes the band a single
            point and the search exact again, at the cost of the extra
            bookkeeping.
        is_dead: Asked before a state is popped, and again before one is
            admitted; returns True for a state some vertex frontier has already
            evicted. Injected rather than imported so this class knows nothing
            about dominance.
    """

    __slots__ = ("_epsilon", "_is_dead", "_open", "_pending", "_focal",
                 "_sequence", "_admitted_to", "discarded")

    def __init__(self, epsilon, is_dead):
        # type: (float, Callable[[SearchState], bool]) -> None
        self._epsilon = float(epsilon)
        self._is_dead = is_dead
        #: Dead states dropped on the way to a pop. Counted here rather than by
        #: the caller because this queue filters them itself -- by the time a
        #: state reaches the search loop it can never be dominated, so the
        #: loop's own counter would read zero and make the two variants'
        #: statistics silently incomparable.
        self.discarded = 0
        #: Everything queued, ordered by f. The authority on the lower bound.
        self._open = []      # type: List[Tuple[float, int, SearchState]]
        #: Not yet admitted to FOCAL, ordered by f. A cursor into the same set.
        self._pending = []   # type: List[Tuple[float, int, SearchState]]
        #: Inside the band, ordered by fewest-remaining then f.
        self._focal = []     # type: List[Tuple[int, float, int, SearchState]]
        self._sequence = 0
        #: The highest band ceiling anything has been admitted under, so
        #: admission never re-walks states it already considered.
        self._admitted_to = float("-inf")

    def push(self, state, estimate):
        # type: (SearchState, float) -> None
        """Queue a state in OPEN, and in FOCAL if it is already in the band.

        Args:
            state: The state to queue.
            estimate: Its ``h``.
        """
        self._sequence += 1
        total = state.cost + estimate
        entry = (total, self._sequence, state)
        heapq.heappush(self._open, entry)
        if total <= self._admitted_to:
            self._push_focal(total, self._sequence, state)
        else:
            heapq.heappush(self._pending, entry)

    def pop(self):
        # type: () -> SearchState
        """Remove and return the next state to expand.

        Returns:
            The state with the fewest unvisited places among those inside the
            current cost band.
        """
        self._purge_open()
        bound = (1.0 + self._epsilon) * self._open[0][0]
        self._admit(bound)
        while self._focal:
            _, _, _, state = heapq.heappop(self._focal)
            if not self._is_dead(state):
                return state
            self.discarded += 1
        return heapq.heappop(self._open)[2]

    def best_estimate(self):
        # type: () -> float
        """The smallest ``f`` still queued -- a lower bound on the optimum."""
        self._purge_open()
        if not self._open:
            return float("inf")
        return self._open[0][0]

    # -- internals --------------------------------------------------------

    def _push_focal(self, total, sequence, state):
        # type: (float, int, SearchState) -> None
        """Order FOCAL by fewest places left, then by cost, then by arrival."""
        remaining = state.visited_count
        heapq.heappush(self._focal, (-remaining, total, sequence, state))

    def _admit(self, bound):
        # type: (float) -> None
        """Move everything now inside the band from pending into FOCAL.

        ``_pending`` is ordered by ``f``, so this stops at the first state above
        the band and resumes from there on the next call. The band never moves
        down, so nothing admitted needs revisiting.
        """
        if bound <= self._admitted_to:
            return
        self._admitted_to = bound
        while self._pending and self._pending[0][0] <= bound:
            total, sequence, state = heapq.heappop(self._pending)
            if self._is_dead(state):
                self.discarded += 1
            else:
                self._push_focal(total, sequence, state)

    def _purge_open(self):
        # type: () -> None
        """Drop evicted states from the front of OPEN.

        The bound is read off OPEN's head, so a dead state sitting there would
        hold the band down and admit more to FOCAL than belong in it. Only the
        front matters; dead states deeper in the heap are harmless until they
        surface.
        """
        while self._open and self._is_dead(self._open[0][2]):
            heapq.heappop(self._open)
            self.discarded += 1

    def __len__(self):
        # type: () -> int
        return len(self._open)

    def __bool__(self):
        # type: () -> bool
        self._purge_open()
        return bool(self._open)
