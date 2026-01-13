"""
RRT* path planning using OMPL (2D and 3D).

Implements RRT* with optional clearance-based cost optimization,
adaptive waypoint reduction, and world-space interpolation.
"""
from __future__ import annotations

from math import hypot, sqrt
from typing import List, Optional, TYPE_CHECKING

from sparx_agency.core.common.types import Path2D, Pose2D, Pose3D, PlanResult, PlanStatus
from sparx_agency.core.planning.environment import Costmap2D

from .params import RRTStarOmplParams, RRTStarOmpl3DParams

if TYPE_CHECKING:
    from ompl import base as ob

try:
    from ompl import base as ob
    from ompl import geometric as og
    OMPL_AVAILABLE = True
except ImportError as e:
    ob = None
    og = None
    _OMPL_ERROR = str(e)
    OMPL_AVAILABLE = False
else:
    _OMPL_ERROR = None


# =============================================================================
# Shared utilities
# =============================================================================

def _make_clearance_objective_2d(si, costmap: Costmap2D, weight: float):
    """Create 2D clearance objective."""
    class _ClearanceObjective(ob.StateCostIntegralObjective):
        def __init__(self, si, costmap: Costmap2D, weight: float) -> None:
            super().__init__(si, True)
            self._costmap = costmap
            self._weight = weight

        def stateCost(self, state) -> ob.Cost:
            clearance = self._costmap.world_clearance(state[0], state[1])
            return ob.Cost(self._weight / (clearance + 1.0))

    return _ClearanceObjective(si, costmap, weight)


def _make_clearance_objective_3d(si, voxelmap, weight: float):
    """Create 3D clearance objective."""
    class _ClearanceObjective3D(ob.StateCostIntegralObjective):
        def __init__(self, si, voxelmap, weight: float) -> None:
            super().__init__(si, True)
            self._voxelmap = voxelmap
            self._weight = weight

        def stateCost(self, state) -> ob.Cost:
            clearance = self._voxelmap.world_clearance(state[0], state[1], state[2])
            return ob.Cost(self._weight / (clearance + 1.0))

    return _ClearanceObjective3D(si, voxelmap, weight)


# =============================================================================
# 2D RRT* (unchanged)
# =============================================================================

def _interpolate_path_2d(points: List[Pose2D], spacing: float) -> List[Pose2D]:
    """Interpolate 2D path at uniform spacing."""
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


def _reduce_path_2d(si, costmap: Costmap2D, states: List, min_clearance: float) -> List:
    """Adaptive waypoint reduction for 2D."""
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
    """Plan a 2D collision-free path using RRT*."""
    if not OMPL_AVAILABLE:
        return PlanResult(status=PlanStatus.ERROR, message=f"OMPL not available: {_OMPL_ERROR}")

    if not costmap.is_free(*costmap.world_to_grid(start.x, start.y)):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision")

    if not costmap.is_free(*costmap.world_to_grid(goal.x, goal.y)):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision")

    # 2D state space
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, costmap.origin_x)
    bounds.setHigh(0, costmap.origin_x + costmap.width * costmap.resolution)
    bounds.setLow(1, costmap.origin_y)
    bounds.setHigh(1, costmap.origin_y + costmap.height * costmap.resolution)
    space.setBounds(bounds)

    # Collision checking resolution
    longest_segment = params.longest_valid_segment_m or costmap.resolution * 0.5
    space_diagonal = hypot(costmap.width * costmap.resolution, costmap.height * costmap.resolution)
    longest_segment_fraction = max(0.001, min(0.1, longest_segment / space_diagonal))
    space.setLongestValidSegmentFraction(longest_segment_fraction)

    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    si.setStateValidityCheckingResolution(params.collision_check_resolution)

    def is_valid(state) -> bool:
        return costmap.is_free(*costmap.world_to_grid(state[0], state[1]))

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))

    start_state, goal_state = ob.State(space), ob.State(space)
    start_state[0], start_state[1] = start.x, start.y
    goal_state[0], goal_state[1] = goal.x, goal.y
    ss.setStartAndGoalStates(start_state, goal_state)

    if params.use_clearance_objective and costmap.clearance is not None:
        ss.setOptimizationObjective(_make_clearance_objective_2d(si, costmap, params.clearance_weight))

    ss.setPlanner(og.RRTstar(si))
    if not ss.solve(params.timeout):
        return PlanResult(status=PlanStatus.NO_PATH, message="No solution found")

    path = ss.getSolutionPath()
    states = [path.getState(i) for i in range(path.getStateCount())]
    reduced = _reduce_path_2d(si, costmap, states, params.min_clearance_for_keep)
    waypoints = [Pose2D(s[0], s[1]) for s in reduced]

    for s in reduced:
        si.freeState(s)

    if waypoints[-1].distance_to(goal) > 0.1:
        waypoints.append(goal)

    n_before = len(waypoints)
    waypoints = _interpolate_path_2d(waypoints, params.interpolation_spacing)

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=Path2D(points=tuple(waypoints), frame_id=costmap.frame_id,
                    metadata={"planner": "rrtstar", "waypoints_raw": n_before, "waypoints_interpolated": len(waypoints)}),
        message=f"Path found: {n_before} -> {len(waypoints)} waypoints",
    )


# =============================================================================
# 3D RRT* (new)
# =============================================================================

# Import Path3D - assume it's defined in types or we define locally
try:
    from sparx_agency.core.common.types import Path3D
except ImportError:
    from dataclasses import dataclass, field
    from typing import Any, Dict, Tuple

    @dataclass(frozen=True)
    class Path3D:
        """3D geometric path."""
        points: Tuple[Pose3D, ...]
        frame_id: str = "map"
        metadata: Dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            if len(self.points) < 2:
                raise ValueError("Path3D requires at least 2 points")

        def __len__(self) -> int:
            return len(self.points)

        @property
        def start(self) -> Pose3D:
            return self.points[0]

        @property
        def goal(self) -> Pose3D:
            return self.points[-1]


def _dist3d(a: Pose3D, b: Pose3D) -> float:
    return sqrt((b.x - a.x)**2 + (b.y - a.y)**2 + (b.z - a.z)**2)


def _interpolate_path_3d(points: List[Pose3D], spacing: float) -> List[Pose3D]:
    """Interpolate 3D path at uniform spacing."""
    if len(points) < 2 or spacing <= 0:
        return points

    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
        dist = sqrt(dx*dx + dy*dy + dz*dz)

        if dist > spacing:
            n_segments = int(dist / spacing)
            for i in range(1, n_segments + 1):
                t = i / (n_segments + 1)
                result.append(Pose3D(a.x + t*dx, a.y + t*dy, a.z + t*dz))
        result.append(b)
    return result


def _reduce_path_3d(si, voxelmap, states: List, min_clearance: float) -> List:
    """Adaptive waypoint reduction for 3D."""
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


def plan_rrtstar_3d(
    start: Pose3D,
    goal: Pose3D,
    voxelmap,  # VoxelMap3D with: is_free(i,j,k), world_to_grid(x,y,z), world_clearance(x,y,z)
    params: RRTStarOmpl3DParams,
) -> PlanResult:
    """
    Plan a 3D collision-free path using RRT*.

    Args:
        start: Start pose in world frame (meters).
        goal: Goal pose in world frame (meters).
        voxelmap: 3D voxel grid with is_free(i,j,k), world_to_grid(x,y,z),
                  world_clearance(x,y,z), and bounds properties.
        params: Algorithm configuration.

    Returns:
        PlanResult with path3d in artifacts if successful.
    """
    if not OMPL_AVAILABLE:
        return PlanResult(status=PlanStatus.ERROR, message=f"OMPL not available: {_OMPL_ERROR}")

    # Validate start/goal
    si_idx, sj_idx, sk_idx = voxelmap.world_to_grid(start.x, start.y, start.z)
    if not voxelmap.is_free(si_idx, sj_idx, sk_idx):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision")

    gi_idx, gj_idx, gk_idx = voxelmap.world_to_grid(goal.x, goal.y, goal.z)
    if not voxelmap.is_free(gi_idx, gj_idx, gk_idx):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision")

    # 3D state space
    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, voxelmap.origin_x)
    bounds.setHigh(0, voxelmap.origin_x + voxelmap.width * voxelmap.resolution)
    bounds.setLow(1, voxelmap.origin_y)
    bounds.setHigh(1, voxelmap.origin_y + voxelmap.height * voxelmap.resolution)
    bounds.setLow(2, voxelmap.origin_z)
    bounds.setHigh(2, voxelmap.origin_z + voxelmap.depth * voxelmap.resolution)
    space.setBounds(bounds)

    # Collision checking resolution
    longest_segment = params.longest_valid_segment_m or voxelmap.resolution * 0.5
    space_diagonal = sqrt(
        (voxelmap.width * voxelmap.resolution)**2 +
        (voxelmap.height * voxelmap.resolution)**2 +
        (voxelmap.depth * voxelmap.resolution)**2
    )
    longest_segment_fraction = max(0.001, min(0.1, longest_segment / space_diagonal))
    space.setLongestValidSegmentFraction(longest_segment_fraction)

    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    si.setStateValidityCheckingResolution(params.collision_check_resolution)

    def is_valid(state) -> bool:
        i, j, k = voxelmap.world_to_grid(state[0], state[1], state[2])
        return voxelmap.is_free(i, j, k)

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))

    start_state, goal_state = ob.State(space), ob.State(space)
    start_state[0], start_state[1], start_state[2] = start.x, start.y, start.z
    goal_state[0], goal_state[1], goal_state[2] = goal.x, goal.y, goal.z
    ss.setStartAndGoalStates(start_state, goal_state)

    if params.use_clearance_objective and hasattr(voxelmap, 'clearance') and voxelmap.clearance is not None:
        ss.setOptimizationObjective(_make_clearance_objective_3d(si, voxelmap, params.clearance_weight))

    ss.setPlanner(og.RRTstar(si))
    if not ss.solve(params.timeout):
        return PlanResult(status=PlanStatus.NO_PATH, message="No solution found")

    path = ss.getSolutionPath()
    states = [path.getState(i) for i in range(path.getStateCount())]
    reduced = _reduce_path_3d(si, voxelmap, states, params.min_clearance_for_keep)
    waypoints = [Pose3D(s[0], s[1], s[2]) for s in reduced]

    for s in reduced:
        si.freeState(s)

    if _dist3d(waypoints[-1], goal) > 0.1:
        waypoints.append(goal)

    n_before = len(waypoints)
    waypoints = _interpolate_path_3d(waypoints, params.interpolation_spacing)

    path3d = Path3D(points=tuple(waypoints), frame_id=getattr(voxelmap, 'frame_id', 'map'),
                    metadata={"planner": "rrtstar_3d", "waypoints_raw": n_before, "waypoints_interpolated": len(waypoints)})

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=None,  # 2D path field
        message=f"3D path found: {n_before} -> {len(waypoints)} waypoints",
        artifacts={"path3d": path3d},
    )