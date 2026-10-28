"""Parameters of the discrete-action state-lattice planner (Python 3.8 syntax, numpy-free)."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers


def _positive(name, value, allow_zero=False):
    if (not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(value)
            or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError("LatticePlannerParams.%s must be a %s finite number, got %r"
                         % (name, "non-negative" if allow_zero else "positive", value))


@dataclass(frozen=True)
class LatticePlannerParams:
    """How the lattice search prices and bounds a plan. Metres, radians, actions.

    The search minimises ACTIONS, not metres: on a benchmark with a step
    budget every TURN costs what a MOVE_FORWARD does, and a route that saves
    a metre by turning four times is the longer one. Each MOVE_FORWARD is
    charged ``forward_action_cost`` times the mean per-cell cost it crosses
    (the 2D planner's cost grid, 1.0 on open floor, more near walls and in
    the gray), each turn ``turn_action_cost``.

    Attributes:
        forward_action_cost: What one MOVE_FORWARD over open floor costs.
        turn_action_cost: What one TURN costs. 1.0 prices actions equally,
            the benchmark's own accounting; below 1 the search prefers
            tighter, better-aligned routes with more turns.
        corridor_m: How far from the 2D route the search may wander. The
            route's cost-to-go field (``route_field.route_cost_to_go``) is
            the heuristic and the search domain at once: a state farther
            than this from every route point is never expanded. 0.4 m
            (since the strict-tracking revision of 2026-10-28, down from
            1.5): the 2D route is the medial-axis path the plan is to keep,
            lining up a lattice heading through a doorway takes a quarter
            of a metre at most, and the policy's route memory drops a route
            the agent strays half a metre from.
        route_offset_cost: Actions charged per metre of lateral offset from
            the route at the end of every MOVE_FORWARD. Inside the corridor
            the search may still prefer the route's own line to a shorter
            cut; 2.0 makes a step a quarter of a metre off the route cost
            half a turn more than one on it, so a plan leaves the route
            only to line up a heading and returns at once. 0 disables the
            charge (the corridor alone bounds the plan).
        sample_spacing_m: Spacing of the collision samples along one forward
            step. Half a cell at the default 0.1 m grid: no cell a step
            crosses is skipped.
        goal_tolerance_m: A state within this distance of the goal point is a
            goal. The converter's own ``goal_tolerance_m`` (0.25 m), so the
            route memory's arrival rule and the plan agree on where the
            route ends.
        goal_heading_tolerance_rad: With a goal heading, how far the final
            heading may be from it. Half a turn plus slack: the nearest
            lattice heading.
        max_expansions: Cap on state pops before the search gives up with
            ``expansion_cap``. A 30 m route needs a few thousand; the cap
            bounds a search that found no way through.
        heuristic_weight: Weight on the cost-to-go heuristic. 1.0 is A*;
            above 1 the search is greedier along the route and expands
            fewer states for a plan at most that factor longer. 1.5 by
            default: on a 32 m synthetic route through three 0.8 m doorways
            plain A* passed 40,000 states without a plan, weighted it
            planned in a few thousand.
        refused_move_actions: How long a refused MOVE_FORWARD (the simulator
            turned the step down) excludes that pose's forward primitive
            from every later plan, in actions; 0 keeps it for the episode.
        refused_move_radius_m: Within this distance of a refused step's
            start, at the same lattice heading, the forward primitive is
            excluded too: the body that was refused here is refused a
            centimetre to the side.
    """

    forward_action_cost: float = 1.0
    turn_action_cost: float = 1.0
    corridor_m: float = 0.4
    route_offset_cost: float = 2.0
    sample_spacing_m: float = 0.05
    goal_tolerance_m: float = 0.25
    goal_heading_tolerance_rad: float = math.radians(15.0) + 1e-6
    max_expansions: int = 60000
    heuristic_weight: float = 1.5
    refused_move_actions: int = 120
    refused_move_radius_m: float = 0.15

    def __post_init__(self):
        for name in ("forward_action_cost", "turn_action_cost", "corridor_m", "sample_spacing_m",
                     "goal_tolerance_m", "goal_heading_tolerance_rad", "heuristic_weight"):
            _positive(name, getattr(self, name))
        _positive("refused_move_radius_m", self.refused_move_radius_m, allow_zero=True)
        _positive("route_offset_cost", self.route_offset_cost, allow_zero=True)
        if type(self.max_expansions) is not int or self.max_expansions < 1:
            raise ValueError("LatticePlannerParams.max_expansions must be a positive integer")
        if type(self.refused_move_actions) is not int or self.refused_move_actions < 0:
            raise ValueError("LatticePlannerParams.refused_move_actions must be a non-negative integer")
        if self.heuristic_weight < 1.0:
            raise ValueError("LatticePlannerParams.heuristic_weight below 1 makes the search slower, never better")
