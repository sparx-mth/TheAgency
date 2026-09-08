"""How far the robot actually flies before it finds the thing.

**The metric is computed under the truth, not under the belief.** A planner is
handed the oracle's belief and optimises against it; it is scored on where the
object really is. Any other choice measures the wrong thing -- RPT* provably
minimises expected cost under whatever distribution it is given, so grading it
against its own input would only confirm that it is not broken.

**And it is exact, not sampled.** Once an ordering is fixed, the expected
distance is a finite sum: for every room, the probability the object is there
times the distance flown to reach it. There is no Monte-Carlo error in any
number this benchmark reports, so a difference of one percent between two
planners is a real difference and not noise. This is worth more than a large
number of sampled trials: a thousand samples would still leave several percent
of standard error, which is the size of the effects being measured.

Three numbers per ordering:

* :attr:`Outcome.distance` -- expected metres flown until the object is found.
  The headline.
* :attr:`Outcome.rooms_searched` -- expected number of rooms entered. What
  matters when searching a room costs more than reaching it.
* :attr:`Outcome.mission_time` -- the two combined, given a flying speed and a
  per-room dwell. What an operator would actually stopwatch.

Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class Outcome:
    """What one ordering is expected to cost, under the truth.

    Attributes:
        distance: Expected metres flown before the object is found.
        rooms_searched: Expected number of rooms entered, the first included.
        mission_time: Expected seconds, flying plus searching.
        regret: ``distance`` minus what a planner that already knew the answer
            would fly. Zero is unbeatable; it is the honest scale on which to
            compare planners, because part of every route is unavoidable.
        ratio_to_best: ``distance`` divided by the best distance any planner
            achieved on this same scenario. One means it won.
    """

    distance: float
    rooms_searched: float
    mission_time: float
    regret: float = 0.0
    ratio_to_best: float = 1.0


def evaluate(order, truth, distance, speed_mps=1.0, dwell_s=0.0):
    # type: (Sequence[int], Sequence[float], Sequence[Sequence[float]], float, float) -> Outcome
    """Score a visiting order against where the object really is.

    Args:
        order: Room indices in visiting order. Index ``len(truth)`` is the
            entrance, which is where the robot starts and is not searchable.
        truth: The real probability the object is in each room. Should sum to
            one; if it does not, the result is scaled as if it did.
        distance: Walking distance between places, entrance last.
        speed_mps: Flying speed, for the time figure.
        dwell_s: How long searching one room takes, for the time figure.

    Returns:
        The expected outcome. ``regret`` and ``ratio_to_best`` are filled in
        later by :func:`add_comparisons`, which needs the whole field.
    """
    total = float(sum(truth)) or 1.0
    travelled = 0.0
    expected_distance = 0.0
    expected_rooms = 0.0
    searched = 0
    for position, room in enumerate(order):
        if position:
            travelled += distance[order[position - 1]][room]
        if room >= len(truth):
            continue                            # the entrance is not searched
        searched += 1
        weight = truth[room] / total
        expected_distance += weight * travelled
        expected_rooms += weight * searched
    return Outcome(
        distance=expected_distance,
        rooms_searched=expected_rooms,
        mission_time=expected_distance / speed_mps + expected_rooms * dwell_s,
    )


def clairvoyant_distance(truth, distance, start):
    # type: (Sequence[float], Sequence[Sequence[float]], int) -> float
    """What a planner that already knew the answer would fly.

    Straight to the right room, every time. Nothing can beat it, so it is the
    zero of the regret scale -- and the gap between it and the best real
    planner is the part of the problem that is irreducible rather than badly
    solved.
    """
    total = float(sum(truth)) or 1.0
    return sum((probability / total) * distance[start][room]
               for room, probability in enumerate(truth))


def add_comparisons(outcomes, clairvoyant):
    # type: (dict, float) -> dict
    """Fill in regret and the ratio to the winner, now the field is known.

    Args:
        outcomes: ``{planner name: Outcome}`` for one scenario.
        clairvoyant: The unbeatable distance for that scenario.

    Returns:
        The same mapping with both comparison fields populated.
    """
    best = min(outcome.distance for outcome in outcomes.values())
    filled = {}
    for name, outcome in outcomes.items():
        filled[name] = Outcome(
            distance=outcome.distance,
            rooms_searched=outcome.rooms_searched,
            mission_time=outcome.mission_time,
            regret=outcome.distance - clairvoyant,
            ratio_to_best=(outcome.distance / best) if best > 0.0 else 1.0,
        )
    return filled
