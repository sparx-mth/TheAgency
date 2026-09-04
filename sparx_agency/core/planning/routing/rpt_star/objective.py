"""What a route costs -- written twice, because the paper's whole trick is that
the two agree.

Definition 1 (p.3) states the expected cost as a sum of terms, one per place
the search might end at, each a long product of probabilities multiplied by the
distance flown in that event. Lemma 1 (p.4) collapses it to a sum in which each
edge is simply weighted by the probability of still having found nothing when
you traverse it. That collapse is what makes the problem searchable at all: the
first form depends on the whole ordering at once, the second is additive along
the route, which is what an A* g-value has to be.

Both are implemented. :func:`expected_cost` is the one the search uses;
:func:`expected_cost_literal` is Definition 1 transcribed as printed, and it
exists so a test can assert the lemma rather than trust it. They agree to
floating-point noise on every route.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class RouteLeg:
    """One hop of a route, with the probabilities that make it cost what it does.

    Attributes:
        from_index: Where the hop starts.
        to_index: Where it ends.
        cost: The raw edge cost ``c(from, to)``, in the caller's units.
        survival_prob: ``q``, the probability of having found nothing at any
            vertex up to and including ``from_index``. This is the weight the
            edge is charged at.
        weighted_cost: ``survival_prob * cost`` -- what this hop contributes to
            the expected total.
        cumulative_cost: The expected cost of the route up to and including
            this hop.
        find_prob: ``p`` at ``to_index``: the chance the search ends here.
    """

    from_index: int
    to_index: int
    cost: float
    survival_prob: float
    weighted_cost: float
    cumulative_cost: float
    find_prob: float


def expected_cost(order, probs, matrix):
    # type: (Sequence[int], Sequence[float], Sequence[Sequence[float]]) -> float
    """The expected cost of a route, by Lemma 1 (p.4).

    ``xi(pi) = sum over i of q_i * c(v_i, v_i+1)``, where
    ``q_i = (1 - p(v_1)) * ... * (1 - p(v_i))``.

    Note that ``q`` includes the start vertex's own probability, which is why a
    start the robot is already standing on should be given ``p = 0``.

    Args:
        order: The visiting order, as vertex indices. A single vertex costs
            zero -- there is nowhere to fly.
        probs: ``p(v)`` by index.
        matrix: The cost matrix.

    Returns:
        The expected cost.
    """
    total = 0.0
    survival = 1.0
    for position in range(len(order) - 1):
        survival *= (1.0 - probs[order[position]])
        total += survival * matrix[order[position]][order[position + 1]]
    return total


def expected_cost_literal(order, probs, matrix):
    # type: (Sequence[int], Sequence[float], Sequence[Sequence[float]]) -> float
    """The expected cost of a route, by Definition 1 / Eq. (1) as printed (p.3).

    Kept so that Lemma 1 is a claim this package tests rather than a claim it
    assumes. Quadratic in the route length and never used by the search.

    The last term deliberately carries no ``p`` factor: if the target was not
    found at any of the first ``N-1`` places, the robot flies to the last one
    regardless of how likely it is to be there.

    Args:
        order: The visiting order, as vertex indices.
        probs: ``p(v)`` by index.
        matrix: The cost matrix.

    Returns:
        The expected cost. Equal to :func:`expected_cost` up to rounding.
    """
    count = len(order)
    total = 0.0
    for k in range(1, count):
        survival = 1.0
        for position in range(k):
            survival *= (1.0 - probs[order[position]])
        travel = 0.0
        for position in range(k):
            travel += matrix[order[position]][order[position + 1]]
        if k < count - 1:
            total += survival * probs[order[k]] * travel
        else:
            total += survival * travel
    return total


def decompose(order, probs, matrix):
    # type: (Sequence[int], Sequence[float], Sequence[Sequence[float]]) -> Tuple[RouteLeg, ...]
    """Break a route into legs, so a caller can see where the cost went.

    The last leg's ``cumulative_cost`` equals :func:`expected_cost` for the same
    route.

    Args:
        order: The visiting order, as vertex indices.
        probs: ``p(v)`` by index.
        matrix: The cost matrix.

    Returns:
        One :class:`RouteLeg` per hop; empty for a single-vertex route.
    """
    legs = []                                   # type: List[RouteLeg]
    survival = 1.0
    cumulative = 0.0
    for position in range(len(order) - 1):
        source = order[position]
        target = order[position + 1]
        survival *= (1.0 - probs[source])
        cost = matrix[source][target]
        weighted = survival * cost
        cumulative += weighted
        legs.append(RouteLeg(
            from_index=source,
            to_index=target,
            cost=cost,
            survival_prob=survival,
            weighted_cost=weighted,
            cumulative_cost=cumulative,
            find_prob=probs[target],
        ))
    return tuple(legs)
