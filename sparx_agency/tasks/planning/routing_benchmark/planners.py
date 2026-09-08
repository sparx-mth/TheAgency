"""The competitors, behind one signature so the runner can be dumb.

Every planner here gets the same thing -- the rooms, the oracle's belief, the
walking distances, and where the robot is standing -- and returns a visiting
order plus how long it took to decide. What separates them is only what they
pay attention to:

============== ==========================================================
``rpt_star``   Belief and distance, weighed against each other optimally.
``f_rpt_star`` The same, allowed to stop a little short of optimal.
``greedy``     Belief only. The paper's ``Greedy`` baseline: always go to the
               most likely room left, however far away it is.
``nearest``    Distance only. Stands in for the paper's ``LKH``: ignore the
               belief entirely and just walk the shortest sensible round.
``nearest_2opt`` Distance only, but a genuinely good tour -- the solver's own
               LKH stand-in: nearest-neighbour restarted from every first hop,
               each improved to a local optimum under 2-opt and Or-opt. This
               is the honest version of the distance-only baseline, and the
               one to beat.
``random``     Neither. The floor, so every other number has a scale.
============== ==========================================================

``clairvoyant`` is not here because it is not a planner -- it is the bound that
knows the answer, and it lives in :mod:`.metrics`.

**Why a strong distance-only baseline matters.** The paper compares against
LKH, a very good tour heuristic, and reports it 50-80% worse than optimal. A
plain nearest-neighbour tour is far weaker than LKH, so beating *that* would
credit RPT* with a win belonging to the baseline being bad. The ordering used
here is close enough to optimal on these sizes that the remaining gap is the
algorithmic difference worth measuring.

**Every ordering here comes from the solver package**, not from a local copy.
The baselines are part of what is being tested -- the paper's own Greedy and
LKH comparators -- so they live in ``core`` beside the algorithm and are
called from here.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.planning.routing.rpt_star import (
    RouteProblem,
    RouteVertex,
    RptStarParams,
    dense_costs,
    greedy_probability_order,
    lkh_style_order,
    nearest_neighbour_order,
    solve,
)
from sparx_agency.core.planning.routing.rpt_star.result import (
    ROUTE_FROM_SEARCH,
)

#: How long a planner may think, in seconds. Two, because that is roughly what
#: a robot replanning between rooms can actually afford -- it decides where to
#: go next while hovering, not overnight. Ten would flatter the search with
#: time no mission would grant it, and the point of measuring runtime is to
#: find where the method stops being usable, not to prove it finishes
#: eventually.
PLAN_BUDGET_S = 2.0


@dataclass(frozen=True)
class Plan:
    """One planner's answer, and what it cost to produce.

    Attributes:
        order: Visiting order over room indices, starting at the entrance.
        seconds: Wall-clock time spent deciding.
        solved: Whether the planner produced this itself, as opposed to
            falling back. Only ever False for the RPT* family.
        guarantee: What the planner claims about the answer, or ``""``.
        expansions: Search effort, where the planner has a notion of it.
    """

    order: Tuple[int, ...]
    seconds: float
    solved: bool = True
    guarantee: str = ""
    expansions: int = 0


def _problem(belief, distance):
    """Wrap a scenario as a routing problem: rooms, then the entrance as start.

    The entrance is index ``n`` in the distance matrix and carries probability
    zero -- the robot is standing there, it is not a place to search, and it
    must still be visited first. That is exactly what an external start is for.
    """
    vertices = [RouteVertex(id=index, prob=probability, label="room %d" % index)
                for index, probability in enumerate(belief)]
    entrance = len(belief)
    vertices.append(RouteVertex(id=entrance, prob=0.0, label="entrance"))
    return RouteProblem(vertices, start_id=entrance), dense_costs(distance)


def plan_rpt_star(belief, distance, params=None):
    """Optimal ordering by expected cost, from the package under test."""
    problem, matrix = _problem(belief, distance)
    started = time.perf_counter()
    solution = solve(problem, matrix,
                     params or RptStarParams(
                         epsilon=None, time_budget_s=PLAN_BUDGET_S))
    seconds = time.perf_counter() - started
    return Plan(order=tuple(solution.order_indices), seconds=seconds,
                solved=solution.route_source == ROUTE_FROM_SEARCH,
                guarantee=solution.guarantee,
                expansions=solution.stats.expansions)


def plan_f_rpt_star(belief, distance, epsilon=0.5):
    """The bounded-suboptimal variant, for when exact will not finish."""
    return plan_rpt_star(
        belief, distance,
        RptStarParams(epsilon=epsilon, time_budget_s=PLAN_BUDGET_S))


def plan_greedy(belief, distance):
    """Always the most likely room left, whatever it costs to get there."""
    problem, _ = _problem(belief, distance)
    started = time.perf_counter()
    order = greedy_probability_order(problem, distance)
    return Plan(order=order, seconds=time.perf_counter() - started)


def plan_nearest(belief, distance):
    """Always the closest room left, ignoring the belief entirely."""
    problem, _ = _problem(belief, distance)
    started = time.perf_counter()
    order = nearest_neighbour_order(problem, distance)
    return Plan(order=order, seconds=time.perf_counter() - started)


def plan_nearest_2opt(belief, distance):
    """A genuinely short round trip, still ignoring the belief."""
    problem, _ = _problem(belief, distance)
    started = time.perf_counter()
    order = lkh_style_order(problem, distance)
    return Plan(order=order, seconds=time.perf_counter() - started)


def plan_random(belief, distance, rng=None):
    """No idea at all. The floor that gives the other numbers a scale."""
    started = time.perf_counter()
    generator = rng or random.Random(0)
    order = list(range(len(belief)))
    generator.shuffle(order)
    return Plan(order=(len(belief),) + tuple(order),
                seconds=time.perf_counter() - started)


#: Everything the runner sweeps, in the order results should be presented.
PLANNERS = (
    ("rpt_star", plan_rpt_star),
    ("f_rpt_star", plan_f_rpt_star),
    ("nearest_2opt", plan_nearest_2opt),
    ("nearest", plan_nearest),
    ("greedy", plan_greedy),
    ("random", plan_random),
)


# -- why there are no local implementations below this line ---------------
#
# The greedy, nearest-neighbour and tour-improvement orderings all live in
# core/planning/routing/rpt_star/baselines.py and are called from there. An
# earlier version of this file reimplemented all three inline, which is the
# accidental-second-copy this repository has been bitten by before: the two
# copies agree until one of them is fixed.
