"""
RRT* path planning using OMPL.

Implements RRT* with optional clearance-based cost optimization,
adaptive waypoint reduction, and world-space interpolation.
"""
from __future__ import annotations

from math import hypot
from typing import List, Optional, TYPE_CHECKING

from sparx_agency.core.common.types import Path2D, Pose2D, PlanResult, PlanStatus
from sparx_agency.core.planning.environment import Costmap2D

from .params import RRTStarOmplParams

if TYPE_CHECKING:
    from ompl import base as ob

# Lazy OMPL import with error capture
try:
    from ompl import base as ob
    from ompl import geometric as og
    OMPL_AVAILABLE = True
except ImportError as e:
    ob = None  # type: ignore
    og = None  # type: ignore
    _OMPL_ERROR = str(e)
    OMPL_AVAILABLE = False
else:
    _OMPL_ERROR = None


def _make_clearance_objective(si, costmap: Costmap2D, weight: float):
    """Create clearance objective (only called when OMPL is available)."""
    class _ClearanceObjective(ob.StateCostIntegralObjective):
        """Cost objective that penalizes states close to obstacles."""

        def __init__(self, si, costmap: Costmap2D, weight: float) -> None:
            super().__init__(si, True)
            self._costmap = costmap
            self._weight = weight

        def stateCost(self, state) -> ob.Cost:
            x, y = state[0], state[1]
            clearance = self._costmap.world_clearance(x, y)
            return ob.Cost(self._weight / (clearance + 1.0))

    return _ClearanceObjective(si, costmap, weight)


def _interpolate_path(points: List[Pose2D], spacing: float) -> List[Pose2D]:
    """
    Interpolate path at uniform spacing.

    Args:
        points: Waypoints to interpolate between.
        spacing: Target distance between output points (meters).

    Returns:
        Interpolated path including all original waypoints.
    """
    if len(points) < 2 or spacing <= 0:
        return points

    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        dist = hypot(dx, dy)

        if dist > spacing:
            n_segments = int(dist / spacing)
            for i in range(1, n_segments + 1):
                t = i / (n_segments + 1)
                result.append(Pose2D(a.x + t * dx, a.y + t * dy))

        result.append(b)

    return result


def _reduce_path(si, costmap: Costmap2D, states: List, min_clearance: float) -> List:
    """
    Adaptive waypoint reduction preserving path validity.

    Keeps waypoints in tight spaces (low clearance) or when direct
    shortcuts are blocked by obstacles.

    Args:
        si: OMPL space information for motion validation.
        costmap: Occupancy grid with clearance.
        states: Original path states.
        min_clearance: Clearance threshold (meters) below which to keep waypoint.

    Returns:
        Reduced list of states (cloned, caller must free).
    """
    if len(states) < 3:
        return [si.cloneState(s) for s in states]

    kept = [si.cloneState(states[0])]

    for i in range(1, len(states) - 1):
        x, y = states[i][0], states[i][1]
        clearance = costmap.world_clearance(x, y)
        can_skip = si.checkMotion(kept[-1], states[i + 1])

        if clearance < min_clearance or not can_skip:
            kept.append(si.cloneState(states[i]))

    kept.append(si.cloneState(states[-1]))
    return kept


def plan_rrtstar(
    start: Pose2D,
    goal: Pose2D,
    costmap: Costmap2D,
    params: RRTStarOmplParams,
) -> PlanResult:
    """
    Plan a collision-free path using RRT* with clearance optimization.

    Args:
        start: Start pose in world frame (meters).
        goal: Goal pose in world frame (meters).
        costmap: Occupancy grid with optional clearance field.
        params: Algorithm configuration (timeout, weights, spacing).

    Returns:
        PlanResult containing status and Path2D if successful.
        The path is interpolated at `params.interpolation_spacing` intervals.
    """
    if not OMPL_AVAILABLE:
        return PlanResult(
            status=PlanStatus.ERROR,
            message=f"OMPL not available: {_OMPL_ERROR}",
        )

    # Validate start/goal
    if not costmap.is_free(*costmap.world_to_grid(start.x, start.y)):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision")

    if not costmap.is_free(*costmap.world_to_grid(goal.x, goal.y)):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision")

    # Configure state space (world coordinates)
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, costmap.origin_x)
    bounds.setHigh(0, costmap.origin_x + costmap.width * costmap.resolution)
    bounds.setLow(1, costmap.origin_y)
    bounds.setHigh(1, costmap.origin_y + costmap.height * costmap.resolution)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)

    # State validity checker
    def is_valid(state) -> bool:
        x, y = state[0], state[1]
        return costmap.is_free(*costmap.world_to_grid(x, y))

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))

    # Set start and goal
    start_state = ob.State(space)
    start_state[0], start_state[1] = start.x, start.y

    goal_state = ob.State(space)
    goal_state[0], goal_state[1] = goal.x, goal.y

    ss.setStartAndGoalStates(start_state, goal_state)

    # Clearance objective
    if params.use_clearance_objective and costmap.clearance is not None:
        objective = _make_clearance_objective(
            ss.getSpaceInformation(),
            costmap,
            params.clearance_weight,
        )
        ss.setOptimizationObjective(objective)

    # Run planner
    ss.setPlanner(og.RRTstar(ss.getSpaceInformation()))
    solved = ss.solve(params.timeout)

    if not solved:
        return PlanResult(status=PlanStatus.NO_PATH, message="No solution found")

    # Extract and process path
    si = ss.getSpaceInformation()
    path = ss.getSolutionPath()
    states = [path.getState(i) for i in range(path.getStateCount())]

    # Reduce waypoints
    reduced = _reduce_path(si, costmap, states, params.min_clearance_for_keep)

    # Convert to world poses
    waypoints = [Pose2D(s[0], s[1]) for s in reduced]

    # Free cloned states
    for s in reduced:
        si.freeState(s)

    # Ensure goal is included
    if waypoints[-1].distance_to(goal) > 0.1:
        waypoints.append(goal)

    # Interpolate
    n_before = len(waypoints)
    waypoints = _interpolate_path(waypoints, params.interpolation_spacing)

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=Path2D(
            points=tuple(waypoints),
            frame_id=costmap.frame_id,
            metadata={
                "planner": "rrtstar",
                "waypoints_raw": n_before,
                "waypoints_interpolated": len(waypoints),
            },
        ),
        message=f"Path found: {n_before} -> {len(waypoints)} waypoints",
    )