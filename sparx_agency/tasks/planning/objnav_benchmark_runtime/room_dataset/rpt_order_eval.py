"""Offline test of room ordering: RPT* (the planner's own solver) given a score per room, the walk it implies until the first room that holds the target. Standard library only."""
from sparx_agency.core.planning.routing.rpt_star import RouteProblem, RouteVertex, solve
from sparx_agency.core.planning.routing.rpt_star.baselines import nearest_neighbour_order

NEVER_ENTERED_FLOOR = 0.15
NEVER_ENTERED_GAIN = 1.5
BELOW_ONE = 1e-6


def adjusted_probability(percent):
    """A room's score as the planner states it for a room never entered: scaled by 1.5 and floored at 0.15, below 1."""
    p = min(float(percent) / 100.0, 1.0 - BELOW_ONE)
    return min(1.0 - BELOW_ONE, max(NEVER_ENTERED_FLOOR, NEVER_ENTERED_GAIN * p))


def rpt_order(probabilities, matrix):
    """Room indices (0..n-1) in the order RPT* visits them; ``matrix`` has the start at index 0 and room ``i`` at ``i + 1``."""
    problem = RouteProblem.with_external_start([RouteVertex(id=i, prob=float(p), label=str(i)) for i, p in enumerate(probabilities)])
    return [i - 1 for i in solve(problem, matrix).order_indices[1:]]


def nearest_first(matrix):
    """Room indices in the order of always going to the closest room left, whatever the belief."""
    problem = RouteProblem.with_external_start([RouteVertex(id=i, prob=0.1, label=str(i)) for i in range(len(matrix) - 1)])
    return [i - 1 for i in nearest_neighbour_order(problem, matrix)[1:]]


def walk_cost(order, matrix, holders):
    """``(distance, rooms visited)`` walking ``order`` from the start until the first room in ``holders``; ``(None, None)`` if none is reached."""
    here, total = 0, 0.0
    for visited, room in enumerate(order, start=1):
        total += matrix[here][room + 1]
        here = room + 1
        if room in holders:
            return total, visited
    return None, None


def efficiency(distance, matrix, holders):
    """The shortest trip from the start to a holder over the distance walked, at most 1."""
    best = min(matrix[0][h + 1] for h in holders)
    return 1.0 if distance <= best else best / distance
