"""What a route costs: what the planner thought, and what it really costs.

Four numbers, and the distinction between the first two is the whole point of
the study.

* :func:`expected_cost_under_belief` -- ``xi(pi)`` computed against the
  probabilities the planner was handed. This is the quantity RPT* minimises,
  and therefore the quantity RPT* is guaranteed to win on. Scoring planners by
  it alone would prove nothing except that the optimiser optimises.
* :func:`expected_distance_under_truth` -- the same route scored against where
  the target actually is. **This is the flight distance**: the metres a robot
  really expects to cover before it finds the thing. When the belief is
  perfect the two agree exactly; as reliability drops they separate, and the
  separation is what the reliability sweep exists to expose.
* :func:`expected_flight_time` -- that distance converted to seconds and
  charged a fixed dwell for each place actually searched. A route that finds
  the target in two long hops can beat one that finds it in six short ones,
  which pure distance hides.
* :func:`expected_places_searched` -- how many places get opened before the
  target turns up.

**Every one of these is an exact expectation, not a sample.** Once the ordering
is fixed the target's location is the only random quantity left, and it ranges
over a finite vertex set, so each metric is a finite weighted sum. There is no
Monte-Carlo error anywhere and a one-percent difference between two planners is
a real one.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

#: Metres per second the robot travels. Only converts distance into time; any
#: constant leaves the ranking of planners unchanged.
DEFAULT_SPEED_MPS = 1.2

#: Seconds spent searching each place on arrival. Charged per place opened, so
#: it penalises long routes independently of how far apart the places are.
DEFAULT_DWELL_S = 30.0


def expected_cost_under_belief(order, belief, matrix):
    # type: (Sequence[int], Sequence[float], Sequence[Sequence[float]]) -> float
    """``xi(pi)`` under the planner's own belief -- Lemma 1 (p.4).

    Delegates to the solver's objective rather than restating it, so the
    benchmark cannot drift from the thing it is benchmarking.

    Args:
        order: The visiting order, as vertex indices.
        belief: ``p(v)`` as given to the planner.
        matrix: The cost matrix.

    Returns:
        The expected cost the planner believed it was paying.
    """
    from sparx_agency.core.planning.routing.rpt_star import expected_cost
    return expected_cost(order, belief, matrix)


def cumulative_distances(order, matrix):
    # type: (Sequence[int], Sequence[Sequence[float]]) -> List[float]
    """Distance travelled by the time each place in the route is reached.

    Args:
        order: The visiting order, as vertex indices.
        matrix: The cost matrix.

    Returns:
        One entry per place, the first being zero -- the robot starts there.
    """
    out = [0.0]
    for position in range(len(order) - 1):
        out.append(out[-1] + matrix[order[position]][order[position + 1]])
    return out


def expected_distance_under_truth(order, truth, matrix):
    # type: (Sequence[int], Sequence[float], Sequence[Sequence[float]]) -> float
    """The distance really flown, in expectation over where the target is.

    If the target is at the ``k``-th place in the route, the robot flies the
    first ``k - 1`` hops and stops. Averaging over ``k`` with the *true*
    distribution gives the honest cost of that ordering, whatever the planner
    believed when it chose it.

    Args:
        order: The visiting order, as vertex indices.
        truth: The true distribution over vertices. Expected to sum to one --
            a single target that is definitely somewhere, which is Remark 1's
            single-object case (p.3).
        matrix: The cost matrix.

    Returns:
        The expected distance.
    """
    reached = cumulative_distances(order, matrix)
    return sum(truth[vertex] * reached[position]
               for position, vertex in enumerate(order))


def expected_cost_single_target(order, probs, matrix):
    # type: (Sequence[int], Sequence[float], Sequence[Sequence[float]]) -> float
    """The expected cost when exactly one target exists -- the corrected ``q``.

    **The paper's objective is not this, and Remark 1 claims it is.**

    Equation 1 weights the ``i``-th edge by ``q_i = prod_{k<=i} (1 - p_k)``,
    the probability of independently failing to find a target at each of the
    first ``i`` places. That is exactly right when every vertex independently
    may hold a target -- Remark 1's "multiple target objects (such as trash
    bins)" case, where the ``p(v)`` need not sum to anything.

    It is *wrong* for the other case Remark 1 names, and endorses: "If there is
    only one target object (such as an ID card) in G and it is known that the
    object must exist in G, then an additional constraint that sum p(v) = 1
    should be imposed. Our approaches ... do not rely on those constraints and
    is applicable to all these problem variants."

    With one target that certainly exists, the events "it is at ``v_1``" and
    "it is at ``v_2``" are mutually exclusive, not independent. The probability
    of still searching after ``i`` places is then

        ``1 - sum_{k<=i} p_k``

    which is not ``prod_{k<=i} (1 - p_k)``. Two places at ``p = 0.3`` give
    ``1 - 0.6 = 0.40`` against ``0.7 * 0.7 = 0.49``: the paper's weight is
    larger, and it is larger by more the further along the route it is
    evaluated. So Eq. 1 systematically over-charges the tail of the route, and
    the ordering that minimises it is not in general the ordering that finds a
    single object soonest.

    This function is the corrected objective, provided for measurement only.
    The solver deliberately keeps Eq. 1 exactly as published -- see the package
    README for what that costs, measured.

    Args:
        order: The visiting order, as vertex indices.
        probs: ``p(v)`` by index, summing to one.
        matrix: The cost matrix.

    Returns:
        The expected travel cost under the single-target model. Equal to
        :func:`expected_distance_under_truth` when ``probs`` is the truth.
    """
    total = 0.0
    remaining = 1.0
    for position in range(len(order) - 1):
        remaining -= probs[order[position]]
        if remaining < 0.0:
            remaining = 0.0
        total += remaining * matrix[order[position]][order[position + 1]]
    return total


def expected_places_searched(order, truth):
    # type: (Sequence[int], Sequence[float]) -> float
    """How many places are opened before the target is found, in expectation.

    The place the target is found at counts, so this is at least one whenever
    the route starts somewhere the target might be.

    Args:
        order: The visiting order, as vertex indices.
        truth: The true distribution over vertices.

    Returns:
        The expected count.
    """
    return sum(truth[vertex] * (position + 1)
               for position, vertex in enumerate(order))


def expected_flight_time(order, truth, matrix, speed_mps=DEFAULT_SPEED_MPS,
                         dwell_s=DEFAULT_DWELL_S):
    # type: (Sequence[int], Sequence[float], Sequence[Sequence[float]], float, float) -> float
    """Seconds until the target is found, in expectation.

    Travel at a constant speed plus a fixed dwell for each place searched. The
    dwell is why this is not just a rescaled distance: it charges the *number*
    of places opened, so a planner that saves metres by opening more rooms can
    still lose.

    The paper charges travel only (Def. 1, p.3) -- folding a constant dwell
    into the objective would change Eq. 1 -- so this is a reporting metric, not
    something any planner here optimises.

    Args:
        order: The visiting order, as vertex indices.
        truth: The true distribution over vertices.
        matrix: The cost matrix.
        speed_mps: Travel speed.
        dwell_s: Seconds spent searching each place.

    Returns:
        The expected time in seconds.
    """
    reached = cumulative_distances(order, matrix)
    return sum(truth[vertex] * (reached[position] / speed_mps
                                + dwell_s * (position + 1))
               for position, vertex in enumerate(order))


class Outcome(object):
    """Everything measured about one planner on one scenario.

    Attributes:
        planner: Which planner produced the route.
        order: The visiting order it produced.
        belief_cost: :func:`expected_cost_under_belief`.
        distance: :func:`expected_distance_under_truth` -- the flight distance.
        flight_time_s: :func:`expected_flight_time`.
        places_searched: :func:`expected_places_searched`.
        planning_s: Wall-clock seconds spent deciding the order.
        expansions: States expanded, for the search-based planners; zero for
            the others.
        status: The solver's status string, or ``"n/a"`` for a baseline.
        guarantee: The solver's guarantee string, or ``"n/a"``.
        solved: Whether the planner returned a route it could stand behind --
            the paper's success-rate criterion (Sec. VII-B, p.12).
    """

    __slots__ = ("planner", "order", "belief_cost", "distance",
                 "flight_time_s", "places_searched", "planning_s",
                 "expansions", "status", "guarantee", "solved")

    def __init__(self, planner, order, belief_cost, distance, flight_time_s,
                 places_searched, planning_s, expansions, status, guarantee,
                 solved):
        # type: (str, Tuple[int, ...], float, float, float, float, float, int, str, str, bool) -> None
        self.planner = planner
        self.order = order
        self.belief_cost = belief_cost
        self.distance = distance
        self.flight_time_s = flight_time_s
        self.places_searched = places_searched
        self.planning_s = planning_s
        self.expansions = expansions
        self.status = status
        self.guarantee = guarantee
        self.solved = solved

    def as_row(self):
        # type: () -> dict
        """A flat dict, for JSON and for tabulation."""
        return {
            "planner": self.planner,
            "order": list(self.order),
            "belief_cost": self.belief_cost,
            "distance": self.distance,
            "flight_time_s": self.flight_time_s,
            "places_searched": self.places_searched,
            "planning_s": self.planning_s,
            "expansions": self.expansions,
            "status": self.status,
            "guarantee": self.guarantee,
            "solved": self.solved,
        }


def evaluate(planner, order, belief, truth, matrix, planning_s, expansions,
             status, guarantee, solved, speed_mps=DEFAULT_SPEED_MPS,
             dwell_s=DEFAULT_DWELL_S):
    # type: (str, Sequence[int], Sequence[float], Sequence[float], Sequence[Sequence[float]], float, int, str, str, bool, float, float) -> Outcome
    """Score one route every way at once.

    Args:
        planner: The planner's name.
        order: The visiting order it produced.
        belief: What the planner was told.
        truth: Where the target actually is.
        matrix: The cost matrix.
        planning_s: Seconds spent planning.
        expansions: States expanded.
        status: Solver status, or ``"n/a"``.
        guarantee: Solver guarantee, or ``"n/a"``.
        solved: Whether the planner stands behind the route.
        speed_mps: Travel speed.
        dwell_s: Seconds per place searched.

    Returns:
        The filled-in :class:`Outcome`.
    """
    order = tuple(order)
    return Outcome(
        planner=planner,
        order=order,
        belief_cost=expected_cost_under_belief(order, belief, matrix),
        distance=expected_distance_under_truth(order, truth, matrix),
        flight_time_s=expected_flight_time(order, truth, matrix, speed_mps,
                                           dwell_s),
        places_searched=expected_places_searched(order, truth),
        planning_s=planning_s,
        expansions=expansions,
        status=status,
        guarantee=guarantee,
        solved=solved,
    )

