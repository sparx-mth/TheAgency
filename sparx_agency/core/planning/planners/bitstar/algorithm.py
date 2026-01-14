"""
3D Path Planning: BIT* and Informed RRT* using OMPL.

Both algorithms are better suited for 3D planning than standard RRT*:
- BIT*: Batch processing + informed sampling = faster convergence
- Informed RRT*: Ellipsoidal sampling after first solution = focused search
"""

from __future__ import annotations
from math import sqrt
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .params import BITStarParams, InformedRRTStarParams

# OMPL imports
try:
    from ompl import base as ob
    from ompl import geometric as og

    OMPL_AVAILABLE = True
except ImportError as e:
    ob = None
    og = None
    OMPL_AVAILABLE = False
    _OMPL_ERROR = str(e)


# =============================================================================
# Types (simplified - replace with your actual types)
# =============================================================================

@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Path3D:
    points: Tuple[Pose3D, ...]
    frame_id: str = "map"
    metadata: dict = field(default_factory=dict)


class PlanStatus:
    SUCCESS = "success"
    NO_PATH = "no_path"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"
    ERROR = "error"


@dataclass
class PlanResult:
    status: str
    path: Optional[Path3D] = None
    message: str = ""
    artifacts: dict = field(default_factory=dict)


# =============================================================================
# Utilities
# =============================================================================

def _dist3d(a: Pose3D, b: Pose3D) -> float:
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _interpolate_path_3d(points: List[Pose3D], spacing: float) -> List[Pose3D]:
    """Interpolate 3D path at uniform spacing."""
    if len(points) < 2 or spacing <= 0:
        return points

    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dist = _dist3d(a, b)
        if dist > spacing:
            n = int(dist / spacing)
            for i in range(1, n + 1):
                t = i / (n + 1)
                result.append(Pose3D(
                    a.x + t * (b.x - a.x),
                    a.y + t * (b.y - a.y),
                    a.z + t * (b.z - a.z),
                ))
        result.append(b)
    return result


def _reduce_path_3d(si, voxelmap, states: List, min_clearance: float) -> List:
    """Remove redundant waypoints while maintaining collision-free path."""
    if len(states) < 3:
        return [si.cloneState(s) for s in states]

    kept = [si.cloneState(states[0])]
    for i in range(1, len(states) - 1):
        x, y, z = states[i][0], states[i][1], states[i][2]
        clearance = voxelmap.world_clearance(x, y, z)
        can_skip = si.checkMotion(kept[-1], states[i + 1])

        if clearance < min_clearance or not can_skip:
            kept.append(si.cloneState(states[i]))

    kept.append(si.cloneState(states[-1]))
    return kept


def _make_clearance_objective_3d(si, voxelmap, weight: float):
    """Create 3D clearance-based optimization objective."""

    class ClearanceObjective(ob.StateCostIntegralObjective):
        def __init__(self, si, voxelmap, weight):
            super().__init__(si, True)
            self._voxelmap = voxelmap
            self._weight = weight

        def stateCost(self, state) -> ob.Cost:
            clearance = self._voxelmap.world_clearance(state[0], state[1], state[2])
            return ob.Cost(self._weight / (clearance + 1.0))

    return ClearanceObjective(si, voxelmap, weight)


def _vm_dim(voxelmap, a: str, b: str) -> int:
    """
    Return integer dimension from voxelmap, supporting multiple naming conventions:
    - size_x/size_y/size_z
    - width/height/depth
    """
    if hasattr(voxelmap, a):
        return int(getattr(voxelmap, a))
    if hasattr(voxelmap, b):
        return int(getattr(voxelmap, b))
    raise AttributeError(f"Voxelmap missing dimension attributes: '{a}' or '{b}'")


def _vm_res(voxelmap) -> float:
    """
    Return voxel resolution (meters per cell) supporting common field names.
    """
    if hasattr(voxelmap, "resolution"):
        return float(getattr(voxelmap, "resolution"))
    if hasattr(voxelmap, "voxel_size"):
        return float(getattr(voxelmap, "voxel_size"))
    raise AttributeError("Voxelmap missing resolution attribute: 'resolution' or 'voxel_size'")


def _setup_space_3d(voxelmap, params) -> Tuple:
    """Create OMPL state space and SimpleSetup for 3D planning."""
    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)

    sx = _vm_dim(voxelmap, "size_x", "width")
    sy = _vm_dim(voxelmap, "size_y", "height")
    sz = _vm_dim(voxelmap, "size_z", "depth")
    res = _vm_res(voxelmap)

    ox = float(voxelmap.origin_x)
    oy = float(voxelmap.origin_y)
    oz = float(voxelmap.origin_z)

    # World bounds
    bounds.setLow(0, ox)
    bounds.setHigh(0, ox + sx * res)
    bounds.setLow(1, oy)
    bounds.setHigh(1, oy + sy * res)
    bounds.setLow(2, oz)
    bounds.setHigh(2, oz + sz * res)
    space.setBounds(bounds)

    # Longest valid segment fraction (in meters -> fraction of space diagonal)
    if getattr(params, "longest_valid_segment_m", None) is not None:
        diag = sqrt((sx * res) ** 2 + (sy * res) ** 2 + (sz * res) ** 2)
        fraction = max(0.001, min(0.1, float(params.longest_valid_segment_m) / diag))
        space.setLongestValidSegmentFraction(fraction)

    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    si.setStateValidityCheckingResolution(float(params.collision_check_resolution))

    def is_valid(state) -> bool:
        return bool(voxelmap.is_free_world(float(state[0]), float(state[1]), float(state[2])))

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
    return space, ss, si



# =============================================================================
# BIT* Planner
# =============================================================================

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
        return PlanResult(status=PlanStatus.ERROR, message=f"OMPL unavailable: {_OMPL_ERROR}")

    # Check start/goal validity
    if not voxelmap.is_free_world(start.x, start.y, start.z):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision")
    if not voxelmap.is_free_world(goal.x, goal.y, goal.z):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision")

    # Setup OMPL
    space, ss, si = _setup_space_3d(voxelmap, params)

    # Set start and goal
    start_state = ob.State(space)
    goal_state = ob.State(space)
    start_state[0], start_state[1], start_state[2] = start.x, start.y, start.z
    goal_state[0], goal_state[1], goal_state[2] = goal.x, goal.y, goal.z
    ss.setStartAndGoalStates(start_state, goal_state)

    # Optimization objective
    if params.use_clearance_objective:
        ss.setOptimizationObjective(_make_clearance_objective_3d(si, voxelmap, params.clearance_weight))

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

    reduced = _reduce_path_3d(si, voxelmap, states, params.min_clearance_for_keep)
    waypoints = [Pose3D(float(s[0]), float(s[1]), float(s[2])) for s in reduced]

    for s in reduced:
        si.freeState(s)

    if _dist3d(waypoints[-1], goal) > 0.1:
        waypoints.append(goal)

    n_raw = len(waypoints)
    waypoints = _interpolate_path_3d(waypoints, params.interpolation_spacing)

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


# =============================================================================
# Informed RRT* Planner
# =============================================================================

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
        return PlanResult(status=PlanStatus.ERROR, message=f"OMPL unavailable: {_OMPL_ERROR}")

    # Check start/goal validity
    if not voxelmap.is_free_world(start.x, start.y, start.z):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision")
    if not voxelmap.is_free_world(goal.x, goal.y, goal.z):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision")

    # Setup OMPL
    space, ss, si = _setup_space_3d(voxelmap, params)

    # Set start and goal
    start_state = ob.State(space)
    goal_state = ob.State(space)
    start_state[0], start_state[1], start_state[2] = start.x, start.y, start.z
    goal_state[0], goal_state[1], goal_state[2] = goal.x, goal.y, goal.z
    ss.setStartAndGoalStates(start_state, goal_state)

    # Optimization objective
    if params.use_clearance_objective:
        ss.setOptimizationObjective(_make_clearance_objective_3d(si, voxelmap, params.clearance_weight))

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

    reduced = _reduce_path_3d(si, voxelmap, states, params.min_clearance_for_keep)
    waypoints = [Pose3D(float(s[0]), float(s[1]), float(s[2])) for s in reduced]

    for s in reduced:
        si.freeState(s)

    if _dist3d(waypoints[-1], goal) > 0.1:
        waypoints.append(goal)

    n_raw = len(waypoints)
    waypoints = _interpolate_path_3d(waypoints, params.interpolation_spacing)

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