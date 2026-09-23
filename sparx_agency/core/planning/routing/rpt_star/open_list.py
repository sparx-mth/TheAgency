"""The exact search's queue: cheapest estimated total first.

A plain binary heap ordered on ``f = g + h``, which is what makes RPT* an A*
rather than an enumeration. Two details are worth stating because they are the
difference between a reproducible planner and one that returns a different
route each time it is asked the same question:

* **Ties are broken by insertion order.** Equal ``f`` values are common here --
  many routes cost the same once the survival probability has decayed -- and
  Python's heap is not stable, so without an explicit sequence number the
  answer would depend on memory addresses. A route that changes between runs of
  the same input cannot be debugged from a recording.

* **There is no removal.** A state evicted from a vertex frontier stays in the
  heap and is discarded when it surfaces. Lazy deletion costs one dominance
  check on pop and saves the whole bookkeeping of an indexed heap.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

from sparx_agency.core.planning.routing.rpt_star.state import SearchState


class OpenList(object):
    """States waiting to be expanded, cheapest estimated total first."""

    __slots__ = ("_heap", "_sequence", "discarded")

    def __init__(self):
        # type: () -> None
        self._heap = []                 # type: List[Tuple[float, int, SearchState]]
        self._sequence = 0
        #: Always zero: this queue filters nothing, so every dead state is
        #: discarded by the search loop and counted there. Present so both
        #: queues answer the same question and the caller needs no special
        #: case.
        self.discarded = 0

    def push(self, state, estimate):
        # type: (SearchState, float) -> None
        """Queue a state under ``f = g + h``.

        Args:
            state: The state to queue.
            estimate: Its ``h``.
        """
        self._sequence += 1
        heapq.heappush(self._heap, (state.cost + estimate, self._sequence, state))

    def pop(self):
        # type: () -> SearchState
        """Remove and return the state with the smallest ``f``."""
        return heapq.heappop(self._heap)[2]

    def best_estimate(self):
        # type: () -> float
        """The smallest ``f`` still queued -- a lower bound on the optimum.

        With an admissible heuristic no route can cost less than this, so it is
        what a search that runs out of budget reports as its lower bound.

        Returns:
            The bound, or infinity when the queue is empty.
        """
        if not self._heap:
            return float("inf")
        return self._heap[0][0]

    def __len__(self):
        # type: () -> int
        return len(self._heap)

    def __bool__(self):
        # type: () -> bool
        return bool(self._heap)
