"""The admissible estimate of what is left to fly, and the table behind it.

The estimate answers: from here, with ``k`` places still unvisited, what is the
least this can cost? Answering it exactly is the original problem again, so the
paper relaxes one constraint (p.6): the remaining ``k`` places become ``k``
*steps*, and a step may return to somewhere already visited. Only an immediate
self-loop stays forbidden. Allowing repeats can only make a route cheaper, so
the relaxed answer is a lower bound and the estimate is admissible (Lemma 5).

That relaxation is what makes it precomputable. ``gamma(v, k)`` -- the cheapest
``k`` steps out of ``v`` -- obeys a recurrence over ``k`` (Eq. 5, p.5) that
knows nothing about which places have actually been visited, so the whole table
is built once before the search starts and read in constant time thereafter.

TWO CORRECTIONS TO THE PAPER, both load-bearing:

* **Lemma 4 claims the table costs O(|V|^2). It costs O(|V|^3).** The table has
  ``|V|`` rows of ``|V|`` entries, and *each entry* is a minimum over the other
  ``|V| - 1`` vertices. The proof (p.5) counts the entries and forgets the
  minimisation inside them. Harmless at the sizes this is used at -- 200
  vertices is eight million operations -- but a caller sizing a budget from the
  paper's figure would be wrong by two orders of magnitude.

* **The table's row index is why Algorithm 1's line 1 must be a typo.** Eq. 5
  defines rows ``k = 0 .. |V| - 1``, and Eq. 6 indexes it with
  ``k(s) = |V| - |A(s)|``. Line 1 of Algorithm 1 sets ``A = {}`` for the
  initial state, which asks for row ``|V|`` -- one past the last row the paper
  itself defines. The start vertex must be in ``A``, and then everything lines
  up: ``k = |V| - 1`` places remain, which is exactly right. See the package
  README.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


class GammaTable(object):
    """``gamma(v, k)``: the cheapest ``k`` steps out of ``v``, repeats allowed.

    Built once per solve, then read in constant time. Row ``0`` is all zeros --
    no steps left, nothing more to fly.

    Args:
        probs: ``p(v)`` by index.
        matrix: The cost matrix. Its diagonal must not be finite, or the
            minimisation would be free to stand still.

    Attributes:
        n: The vertex count.
    """

    def __init__(self, probs, matrix):
        # type: (Sequence[float], Sequence[Sequence[float]]) -> None
        self.n = len(probs)
        self._probs = tuple(float(p) for p in probs)
        self._rows = self._build(matrix)

    def _build(self, matrix):
        # type: (Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]
        """Fill the table bottom-up, exactly as Eq. (5) prescribes.

        ``gamma(v, i+1) = min over u != v of (1 - p(v)) * (c(v, u) + gamma(u, i))``

        The ``(1 - p(v))`` factor sits outside the minimisation, so it can be
        hoisted: it does not depend on which ``u`` wins.
        """
        n = self.n
        survive = tuple(1.0 - p for p in self._probs)
        rows = [tuple([0.0] * n)]               # gamma(v, 0) = 0 for every v
        for _ in range(1, n):
            previous = rows[-1]
            row = []                            # type: List[float]
            for v in range(n):
                costs_from_v = matrix[v]
                best = float("inf")
                for u in range(n):
                    if u == v:
                        continue
                    candidate = costs_from_v[u] + previous[u]
                    if candidate < best:
                        best = candidate
                row.append(survive[v] * best)
            rows.append(tuple(row))
        return tuple(rows)

    def steps(self, vertex, remaining):
        # type: (int, int) -> float
        """``gamma(vertex, remaining)`` -- the raw table entry.

        Args:
            vertex: Where the remaining steps start.
            remaining: How many steps, in ``0 .. n - 1``.

        Returns:
            The relaxed cost of that many steps.
        """
        return self._rows[remaining][vertex]

    def estimate(self, vertex, survival, visited_count):
        # type: (int, float, int) -> float
        """``h(s)``, the admissible cost-to-go for a state (Eq. 6, p.5).

        ``h(s) = q(s) / (1 - p(v(s))) * gamma(v(s), |V| - |A(s)|)``

        The leading factor is the survival probability *before* the current
        vertex's own probability was applied -- the parent's ``q``. It belongs
        there because ``gamma`` reapplies ``(1 - p(v))`` itself at its first
        step, and charging it twice would inflate the estimate and destroy
        admissibility.

        Args:
            vertex: The state's current vertex.
            survival: The state's ``q``.
            visited_count: ``|A(s)|``, the number of vertices already visited,
                the current one included.

        Returns:
            The estimate, or ``0.0`` at a goal state where nothing remains.
        """
        remaining = self.n - visited_count
        if remaining <= 0:
            return 0.0
        parent_survival = survival / (1.0 - self._probs[vertex])
        return parent_survival * self._rows[remaining][vertex]
