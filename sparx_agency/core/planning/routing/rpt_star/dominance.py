"""Which partial routes can be thrown away, and the reason it is safe.

Two partial routes that have arrived at the same vertex are comparable. If one
of them cost less *and* has already visited a superset of the other's places,
then anything the second could go on to do, the first can do at least as
cheaply -- so the second is discarded. That is Definition 3 (p.4), and Lemma 6
(p.7) is the proof: take any completion of the worse route, graft it onto the
better one, and cut out the vertices that now repeat. Cutting a vertex out of a
route cannot make it longer, *provided a detour through a vertex is never
cheaper than going direct* -- which is precisely the triangle inequality, and
precisely why this package refuses a cost matrix that violates it.

Note that a superset of visited places implies a smaller survival probability
(Lemma 3, p.4) -- ``q`` is a product of ``(1 - p)`` terms, so more terms means
a smaller product. So ``q`` never needs to be compared; it comes along for free.

TWO SUBTLETIES WORTH THE WORDS:

* **The pruning runs in both directions and they are not the same test.**
  ``is_pruned`` asks "does anything already here beat the newcomer?";
  ``filter_and_add`` asks "does the newcomer beat anything already here?".
  Algorithm 1 (p.5) calls them at different moments and with the arguments the
  other way round, and swapping them silently turns an exact search into a
  wrong one. Both call :func:`dominates` rather than inlining the comparison,
  so there is exactly one definition of the rule to get right -- a second copy
  in the hot loop would be marginally faster and would eventually disagree with
  the first, which is the failure this codebase has had before.

* **Do not add a plain "same vertex, same visited set" duplicate check.** It
  looks like a free optimisation and is redundant here anyway, but under focal
  search it is actively unsafe: F-RPT* pops by fewest-remaining rather than by
  cost, so it can legitimately generate a *cheaper* duplicate of a state it has
  already expanded, and a naive duplicate guard would drop it.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from typing import List, Tuple


def dominates(cost_a, visited_a, cost_b, visited_b):
    # type: (float, int, float, int) -> bool
    """Whether route A dominates route B at a shared vertex (Definition 3).

    True when A costs no more and has visited everything B has.

    The comparison on cost is exact rather than tolerant. A tolerance would
    break transitivity -- three routes could each "dominate" the next around a
    cycle -- and an intransitive dominance relation makes the frontier's own
    contents depend on insertion order, which is how an exact search quietly
    stops being one. The price is failing to prune a route that is better by
    one part in ten thousand billion, which costs one extra expansion.

    Args:
        cost_a: ``g`` of the candidate dominator.
        visited_a: ``A`` of the candidate dominator, as a bitmask.
        cost_b: ``g`` of the candidate victim.
        visited_b: ``A`` of the candidate victim, as a bitmask.

    Returns:
        True if A dominates B.
    """
    return cost_a <= cost_b and (visited_a & visited_b) == visited_b


class VertexFrontier(object):
    """The mutually non-dominated partial routes arriving at one vertex.

    ``F(v)`` in the paper (p.4). Held as a plain list: a sorted or indexed
    structure was measured and bought about ten percent, which is not worth the
    loss of obviousness in the one piece of code whose correctness the whole
    optimality proof rests on.

    Attributes:
        entries: ``(cost, visited)`` pairs, mutually non-dominated.
    """

    __slots__ = ("entries",)

    def __init__(self):
        # type: () -> None
        self.entries = []                       # type: List[Tuple[float, int]]

    def is_pruned(self, cost, visited):
        # type: (float, int) -> bool
        """Whether an incoming route is dominated by one already here.

        ``IsPruned`` in Algorithm 1. Called twice per state -- once when it is
        generated, to avoid queueing it at all, and once when it is popped,
        because something better may have arrived in the meantime.

        Args:
            cost: ``g`` of the incoming route.
            visited: ``A`` of the incoming route.

        Returns:
            True if it can be discarded.
        """
        for held_cost, held_visited in self.entries:
            if dominates(held_cost, held_visited, cost, visited):
                return True
        return False

    def filter_and_add(self, cost, visited):
        # type: (float, int) -> None
        """Admit a route, evicting any it dominates.

        ``FilterAndAddFront`` in Algorithm 1. The evicted routes may still be
        sitting in the open list; they are not hunted down there, because
        :meth:`is_pruned` is checked again on pop and will catch them. Lazy
        deletion, and the reason the open list needs no removal support.

        Args:
            cost: ``g`` of the route being admitted.
            visited: ``A`` of the route being admitted.
        """
        self.entries = [
            (held_cost, held_visited)
            for held_cost, held_visited in self.entries
            if not dominates(cost, visited, held_cost, held_visited)
        ]
        self.entries.append((cost, visited))

    def __len__(self):
        # type: () -> int
        return len(self.entries)
