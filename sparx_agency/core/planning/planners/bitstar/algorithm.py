"""
BIT* (Batch Informed Trees) 3D path planning using OMPL.

BIT* combines the best of RRT* and graph-based planners:
- Processes samples in batches for efficiency
- Uses heuristics to focus sampling
- Asymptotically optimal with faster convergence than RRT*

FIXED: Now properly returns first solution found and tracks improvements.
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


# ============================================================================
# ============================================================================
# BENCHMARK ADDITIONS - DELETE THIS ENTIRE SECTION AFTER EXPERIMENTS
# ============================================================================
# ============================================================================
#
# This section contains functions for benchmarking BIT* performance.
# It tracks intermediate solutions and their timing/path lengths.
#
# To use: from sparx_agency.core.planning.planners.bitstar.algorithm import (
#     plan_bitstar_3d_with_tracking, SolutionSnapshot
# )
#
# To remove: Delete everything from "BENCHMARK ADDITIONS" to "END BENCHMARK ADDITIONS"
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import time


@dataclass
class SolutionSnapshot:
    """A single solution snapshot captured during planning."""
    time_from_start_s: float  # Time since planning started
    path_length_m: float  # Total Euclidean length of path
    num_waypoints: int  # Number of waypoints in raw path
    waypoints: List[List[float]] = field(default_factory=list)  # [[x,y,z], ...]
    cost: float = 0.0  # OMPL cost value


def _compute_path_length_from_waypoints(waypoints: List[List[float]]) -> float:
    """Compute total Euclidean path length from waypoint list."""
    if len(waypoints) < 2:
        return 0.0
    import numpy as np
    total = 0.0
    for i in range(len(waypoints) - 1):
        p1 = np.array(waypoints[i])
        p2 = np.array(waypoints[i + 1])
        total += float(np.linalg.norm(p2 - p1))
    return total


def plan_bitstar_3d_with_tracking(
        start: Pose3D,
        goal: Pose3D,
        voxelmap,
        params: BITStarParams,
        poll_interval_s: float = 0.1,
) -> Tuple[PlanResult, List[SolutionSnapshot]]:
    """
    Plan a 3D path using BIT* with solution tracking.

    FIXED VERSION: Uses iterative solve calls to properly capture intermediate
    solutions. The previous polling approach didn't work because OMPL's state
    isn't thread-safe for concurrent access during solve().

    This version:
    1. Calls solve() repeatedly with short intervals
    2. Captures each improved solution as it's found
    3. Continues until timeout expires

    Args:
        start: Start pose
        goal: Goal pose
        voxelmap: Collision checking voxelmap
        params: BIT* parameters
        poll_interval_s: How often to check for improved solutions (default 0.1s)

    Returns:
        Tuple of (PlanResult, List[SolutionSnapshot])
        The snapshots are in chronological order.

    BENCHMARK CODE - DELETE AFTER EXPERIMENTS
    """
    if not OMPL_AVAILABLE:
        return PlanResult(status=PlanStatus.ERROR, message=f"OMPL unavailable: {OMPL_ERROR}"), []

    # Check start/goal validity
    if not voxelmap.is_free_world(start.x, start.y, start.z):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision"), []
    if not voxelmap.is_free_world(goal.x, goal.y, goal.z):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision"), []

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
    planner.setSamplesPerBatch(params.samples_per_batch)
    planner.setUseKNearest(params.use_k_nearest)
    planner.setRewireFactor(params.rewire_factor)

    ss.setPlanner(planner)
    ss.setup()

    # Solution tracking state
    solutions: List[SolutionSnapshot] = []
    last_cost = float('inf')
    planning_start = time.perf_counter()
    timeout_remaining = params.timeout

    def extract_and_record_solution() -> bool:
        """Extract current solution if improved. Returns True if new solution recorded."""
        nonlocal last_cost

        try:
            if not ss.haveSolutionPath():
                return False

            # Get current cost
            pdef = ss.getProblemDefinition()
            opt_obj = pdef.getOptimizationObjective()
            solution_path = pdef.getSolutionPath()

            if solution_path is None:
                return False

            current_cost = solution_path.cost(opt_obj).value()

            # Only record if better
            if current_cost >= last_cost - 1e-9:
                return False

            last_cost = current_cost
            elapsed = time.perf_counter() - planning_start

            # Extract path waypoints
            path = ss.getSolutionPath()
            waypoints = []
            for i in range(path.getStateCount()):
                s = path.getState(i)
                waypoints.append([float(s[0]), float(s[1]), float(s[2])])

            if len(waypoints) < 2:
                return False

            path_length = _compute_path_length_from_waypoints(waypoints)

            solutions.append(SolutionSnapshot(
                time_from_start_s=elapsed,
                path_length_m=path_length,
                num_waypoints=len(waypoints),
                waypoints=waypoints,
                cost=current_cost,
            ))
            return True

        except Exception:
            return False

    # Iterative solving: call solve() repeatedly with short intervals
    # This allows us to check for improved solutions between calls
    solved = False
    while timeout_remaining > 0:
        # Solve for a short interval
        interval = min(poll_interval_s, timeout_remaining)
        result = ss.solve(interval)

        if result:
            solved = True
            # Check if we have a new/improved solution
            extract_and_record_solution()

        timeout_remaining -= interval
        elapsed = time.perf_counter() - planning_start

        # Safety check - break if we've exceeded timeout
        if elapsed >= params.timeout:
            break

    # Final extraction in case we missed the last improvement
    extract_and_record_solution()

    if not solved:
        return PlanResult(status=PlanStatus.NO_PATH, message="BIT* found no solution"), solutions

    # Check for exact solution
    try:
        is_exact = bool(ss.haveExactSolutionPath())
    except:
        is_exact = True

    if not is_exact:
        return PlanResult(status=PlanStatus.NO_PATH, message="BIT* found only approximate solution"), solutions

    # Extract final path for PlanResult
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
        metadata={
            "planner": "bitstar_3d",
            "waypoints_raw": n_raw,
            "waypoints_final": len(waypoints),
            "num_solutions_found": len(solutions),
        },
    )

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=path3d,
        message=f"BIT* path: {n_raw} -> {len(waypoints)} waypoints, {len(solutions)} solutions tracked",
        artifacts={"path3d": path3d, "solution_snapshots": solutions},
    ), solutions


# Alternative: Using OMPL's intermediate solution callback (if available)
def plan_bitstar_3d_with_callback_tracking(
        start: Pose3D,
        goal: Pose3D,
        voxelmap,
        params: BITStarParams,
) -> Tuple[PlanResult, List[SolutionSnapshot]]:
    """
    Plan using OMPL's native intermediate solution callback.

    Note: This may not work with all OMPL Python binding versions.
    Use plan_bitstar_3d_with_tracking() if this doesn't work.

    BENCHMARK CODE - DELETE AFTER EXPERIMENTS
    """
    if not OMPL_AVAILABLE:
        return PlanResult(status=PlanStatus.ERROR, message=f"OMPL unavailable: {OMPL_ERROR}"), []

    if not voxelmap.is_free_world(start.x, start.y, start.z):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision"), []
    if not voxelmap.is_free_world(goal.x, goal.y, goal.z):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision"), []

    space, ss, si = setup_ompl_space_3d(voxelmap, params)

    start_state = ob.State(space)
    goal_state = ob.State(space)
    start_state[0], start_state[1], start_state[2] = start.x, start.y, start.z
    goal_state[0], goal_state[1], goal_state[2] = goal.x, goal.y, goal.z
    ss.setStartAndGoalStates(start_state, goal_state)

    if params.use_clearance_objective:
        ss.setOptimizationObjective(make_clearance_objective_3d(si, voxelmap, params.clearance_weight))

    planner = og.BITstar(si)
    planner.setSamplesPerBatch(params.samples_per_batch)
    planner.setUseKNearest(params.use_k_nearest)
    planner.setRewireFactor(params.rewire_factor)

    ss.setPlanner(planner)

    solutions: List[SolutionSnapshot] = []
    planning_start = time.perf_counter()
    last_cost = float('inf')

    def on_new_solution(planner_ptr, paths, cost):
        """Callback invoked by OMPL when a new solution is found."""
        nonlocal last_cost

        try:
            current_cost = cost.value()
            if current_cost >= last_cost - 1e-9:
                return
            last_cost = current_cost

            elapsed = time.perf_counter() - planning_start

            if ss.haveSolutionPath():
                path = ss.getSolutionPath()
                waypoints = []
                for i in range(path.getStateCount()):
                    s = path.getState(i)
                    waypoints.append([float(s[0]), float(s[1]), float(s[2])])

                if len(waypoints) >= 2:
                    path_length = _compute_path_length_from_waypoints(waypoints)
                    solutions.append(SolutionSnapshot(
                        time_from_start_s=elapsed,
                        path_length_m=path_length,
                        num_waypoints=len(waypoints),
                        waypoints=waypoints,
                        cost=current_cost,
                    ))
        except:
            pass

    # Try to register callback
    try:
        pdef = ss.getProblemDefinition()
        pdef.setIntermediateSolutionCallback(on_new_solution)
    except AttributeError:
        # Callback not available, fall back to iterative version
        return plan_bitstar_3d_with_tracking(start, goal, voxelmap, params)

    ss.setup()
    solved = ss.solve(params.timeout)

    # Capture final solution
    if solved and ss.haveSolutionPath():
        try:
            pdef = ss.getProblemDefinition()
            current_cost = pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()

            if current_cost < last_cost - 1e-9:
                elapsed = time.perf_counter() - planning_start
                path = ss.getSolutionPath()
                waypoints = []
                for i in range(path.getStateCount()):
                    s = path.getState(i)
                    waypoints.append([float(s[0]), float(s[1]), float(s[2])])

                if len(waypoints) >= 2:
                    path_length = _compute_path_length_from_waypoints(waypoints)
                    solutions.append(SolutionSnapshot(
                        time_from_start_s=elapsed,
                        path_length_m=path_length,
                        num_waypoints=len(waypoints),
                        waypoints=waypoints,
                        cost=current_cost,
                    ))
        except:
            pass

    if not solved:
        return PlanResult(status=PlanStatus.NO_PATH, message="BIT* found no solution"), solutions

    try:
        is_exact = bool(ss.haveExactSolutionPath())
    except:
        is_exact = True

    if not is_exact:
        return PlanResult(status=PlanStatus.NO_PATH, message="BIT* found only approximate solution"), solutions

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
        metadata={
            "planner": "bitstar_3d",
            "waypoints_raw": n_raw,
            "waypoints_final": len(waypoints),
            "num_solutions_found": len(solutions),
        },
    )

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=path3d,
        message=f"BIT* path with callback tracking: {len(solutions)} solutions",
        artifacts={"path3d": path3d, "solution_snapshots": solutions},
    ), solutions

# ============================================================================
# END BENCHMARK ADDITIONS - DELETE EVERYTHING ABOVE UP TO "BENCHMARK ADDITIONS"
# ============================================================================