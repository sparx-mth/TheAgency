"""
BIT* (Batch Informed Trees) 3D path planning using OMPL.

BIT* combines the best of RRT* and graph-based planners:
- Processes samples in batches for efficiency
- Uses heuristics to focus sampling
- Asymptotically optimal with faster convergence than RRT*
"""
from __future__ import annotations

from sparx_agency.core.common.types import Pose3D, Path3D, PlanStatus, PlanResult

from .params import BITStarParams

# Import shared utilities from common
from ..common import (
    ob, og, OMPL_AVAILABLE, OMPL_ERROR,
    dist3d, interpolate_path_3d, reduce_path_3d,
    make_clearance_objective_3d, setup_ompl_space_3d,
)


def plan_bitstar_3d(
    start: Pose3D,
    goal: Pose3D,
    voxelmap,
    params: BITStarParams,
) -> PlanResult:
    """
    Plan a 3D path using BIT* (Batch Informed Trees).

    BIT* combines the best of RRT* and graph-based planners:
    - Processes samples in batches for efficiency
    - Uses heuristics to focus sampling
    - Asymptotically optimal
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

    # Create BIT* planner
    planner = og.BITstar(si)

    # Configure BIT* parameters
    planner.setSamplesPerBatch(params.samples_per_batch)
    planner.setUseKNearest(params.use_k_nearest)
    planner.setRewireFactor(params.rewire_factor)

    ss.setPlanner(planner)

    # Solve
    solved = ss.solve(params.timeout)

    if not solved:
        return PlanResult(status=PlanStatus.NO_PATH, message="BIT* found no solution")

    # Check for exact solution
    try:
        is_exact = bool(ss.haveExactSolutionPath())
    except:
        is_exact = True  # Assume exact if check not available

    if not is_exact:
        return PlanResult(status=PlanStatus.NO_PATH, message="BIT* found only approximate solution")

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
        metadata={"planner": "bitstar_3d", "waypoints_raw": n_raw, "waypoints_final": len(waypoints)},
    )

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=path3d,
        message=f"BIT* path: {n_raw} -> {len(waypoints)} waypoints",
        artifacts={"path3d": path3d},
    )
