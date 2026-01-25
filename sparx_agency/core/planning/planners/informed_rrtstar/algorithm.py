"""
Informed RRT* 3D path planning using OMPL.

Informed RRT* improves on RRT* by:
- Sampling in ellipsoidal region after first solution
- Focusing samples toward goal using admissible heuristic
- Faster convergence in higher dimensions
"""
from __future__ import annotations

from sparx_agency.core.common.types import Pose3D, Path3D, PlanStatus, PlanResult

from .params import InformedRRTStarParams

# Import shared utilities from common
from ..common import (
    ob, og, OMPL_AVAILABLE, OMPL_ERROR,
    dist3d, interpolate_path_3d, reduce_path_3d,
    make_clearance_objective_3d, setup_ompl_space_3d,
)


def plan_informed_rrtstar_3d(
    start: Pose3D,
    goal: Pose3D,
    voxelmap,
    params: InformedRRTStarParams,
) -> PlanResult:
    """
    Plan a 3D path using Informed RRT*.

    Informed RRT* improves on RRT* by:
    - Sampling in ellipsoidal region after first solution
    - Focusing samples toward goal using admissible heuristic
    - Faster convergence in higher dimensions
    """
    if not OMPL_AVAILABLE:
        return PlanResult(status=PlanStatus.ERROR, message=f"OMPL unavailable: {OMPL_ERROR}")

    # Check start/goal validity
    if not voxelmap.is_free_world(start.x, start.y, start.z):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision")
    if not voxelmap.is_free_world(goal.x, goal.y, goal.z):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision")

    # Setup OMPL
    space, ss, si = setup_ompl_space_3d(voxelmap, params)

    # Set start and goal
    start_state = ob.State(space)
    goal_state = ob.State(space)
    start_state[0], start_state[1], start_state[2] = start.x, start.y, start.z
    goal_state[0], goal_state[1], goal_state[2] = goal.x, goal.y, goal.z
    ss.setStartAndGoalStates(start_state, goal_state)

    # Optimization objective
    if params.use_clearance_objective:
        ss.setOptimizationObjective(make_clearance_objective_3d(si, voxelmap, params.clearance_weight))

    # Create Informed RRT* planner
    planner = og.InformedRRTstar(si)

    # Configure range if specified
    if params.range_m is not None:
        planner.setRange(params.range_m)

    ss.setPlanner(planner)

    # Solve
    solved = ss.solve(params.timeout)

    if not solved:
        return PlanResult(status=PlanStatus.NO_PATH, message="Informed RRT* found no solution")

    # Check for exact solution
    try:
        is_exact = bool(ss.haveExactSolutionPath())
    except:
        is_exact = True

    if not is_exact:
        return PlanResult(status=PlanStatus.NO_PATH, message="Informed RRT* found only approximate solution")

    # Extract path
    path = ss.getSolutionPath()
    states = [path.getState(i) for i in range(path.getStateCount())]

    reduced = reduce_path_3d(si, voxelmap, states, params.min_clearance_for_keep)
    waypoints = [Pose3D(float(s[0]), float(s[1]), float(s[2])) for s in reduced]

    for s in reduced:
        si.freeState(s)

    if dist3d(waypoints[-1], goal) > 0.1:
        waypoints.append(goal)

    n_raw = len(waypoints)
    waypoints = interpolate_path_3d(waypoints, params.interpolation_spacing)

    path3d = Path3D(
        points=tuple(waypoints),
        frame_id=params.frame_id,
        metadata={"planner": "informed_rrtstar_3d", "waypoints_raw": n_raw, "waypoints_final": len(waypoints)},
    )

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=path3d,
        message=f"Informed RRT* path: {n_raw} -> {len(waypoints)} waypoints",
        artifacts={"path3d": path3d},
    )
