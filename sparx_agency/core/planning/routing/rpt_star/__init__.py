"""RPT* -- the visiting order that finds a thing soonest.

Given a set of places, how likely the target is to be at each, and what it
costs to fly between them, this returns the order to visit them in that
minimises the *expected* distance flown before the target is found. Not the
shortest tour, and not "go to the most likely place" -- the ordering that
balances the two, which is neither of the obvious answers and is what the
paper is about.

A faithful implementation of:

    Lyu, Cao, Zhang, Choset, Ren, "RPT*: Global Planning with Probabilistic
    Terminals for Target Search in Complex Environments", arXiv:2601.12701.

No code was ever released with that paper. This is a clean-room implementation
from the text; the README maps every equation and lemma to the file that
implements it, and records the places where the paper is wrong.

Typical use::

    from sparx_agency.core.planning.routing.rpt_star import (
        RouteProblem, RouteVertex, RptStarParams, costs_from_points, solve)

    rooms = [RouteVertex(id="kitchen", prob=0.5, label="kitchen"),
             RouteVertex(id="ward", prob=0.3, label="ward"),
             RouteVertex(id="store", prob=0.2, label="store")]
    problem = RouteProblem.with_external_start(rooms, payload=robot_pose)
    matrix = costs_from_points([robot_xy, kitchen_xy, ward_xy, store_xy])

    solution = solve(problem, matrix)
    fly_to(solution.next_id)          # then learn something, and solve again

Python 3.8 syntax, standard library only, no ROS and no numpy.
"""
from sparx_agency.core.planning.routing.rpt_star.baselines import (
    greedy_probability_order,
    nearest_neighbour_order,
)
from sparx_agency.core.planning.routing.rpt_star.brute_force import (
    brute_force_order,
)
from sparx_agency.core.planning.routing.rpt_star.costs import (
    NO_EDGE,
    costs_from_pairs,
    costs_from_points,
    costs_from_row_callback,
    dense_costs,
    metric_closure,
)
from sparx_agency.core.planning.routing.rpt_star.errors import (
    DisconnectedGraphError,
    InvalidCostError,
    InvalidProblemError,
    RoutingError,
    RoutingInternalError,
    TriangleInequalityError,
)
from sparx_agency.core.planning.routing.rpt_star.heuristic import GammaTable
from sparx_agency.core.planning.routing.rpt_star.objective import (
    RouteLeg,
    decompose,
    expected_cost,
    expected_cost_literal,
)
from sparx_agency.core.planning.routing.rpt_star.params import RptStarParams
from sparx_agency.core.planning.routing.rpt_star.problem import (
    RouteProblem,
    RouteVertex,
)
from sparx_agency.core.planning.routing.rpt_star.result import (
    GUARANTEE_BOUNDED,
    GUARANTEE_NONE,
    GUARANTEE_OPTIMAL,
    STATUS_BUDGET_EXCEEDED,
    STATUS_NO_ROUTE,
    STATUS_SOLVED,
    RouteSolution,
    SearchStats,
)
from sparx_agency.core.planning.routing.rpt_star.solver import solve
from sparx_agency.core.planning.routing.rpt_star.validation import validate

__all__ = [
    # the problem
    "RouteProblem",
    "RouteVertex",
    # the costs
    "dense_costs",
    "costs_from_pairs",
    "costs_from_points",
    "costs_from_row_callback",
    "metric_closure",
    "NO_EDGE",
    # solving
    "solve",
    "validate",
    "RptStarParams",
    # the answer
    "RouteSolution",
    "RouteLeg",
    "SearchStats",
    "STATUS_SOLVED",
    "STATUS_BUDGET_EXCEEDED",
    "STATUS_NO_ROUTE",
    "GUARANTEE_OPTIMAL",
    "GUARANTEE_BOUNDED",
    "GUARANTEE_NONE",
    # the objective, and the ways to check it
    "expected_cost",
    "expected_cost_literal",
    "decompose",
    "GammaTable",
    "brute_force_order",
    "greedy_probability_order",
    "nearest_neighbour_order",
    # failures
    "RoutingError",
    "InvalidProblemError",
    "InvalidCostError",
    "TriangleInequalityError",
    "DisconnectedGraphError",
    "RoutingInternalError",
]
