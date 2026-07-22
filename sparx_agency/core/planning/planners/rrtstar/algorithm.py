"""
RRT* path planning using OMPL (2D and 3D).

Key fixes:
- 3D: clearance objective is enabled correctly (uses voxelmap.world_clearance()).
- 3D: validity checks evaluate mesh constraints at the REAL state (x,y,z),
      not at voxel-center (prevents voxel-center mismatch).
- 3D: reject approximate solutions (OMPL can return "closest found" path that
      can violate collision). We only accept exact solutions.
- Adds throttled, high-signal debug logs for validity and solution quality.
"""

from __future__ import annotations

from math import hypot, sqrt
from typing import List, Optional, TYPE_CHECKING, Any, Dict

from sparx_agency.core.common.types import Path2D, Pose2D, Pose3D, PlanResult, PlanStatus
from sparx_agency.core.planning.environment import Costmap2D

from .params import RRTStarOmplParams, RRTStarOmpl3DParams

# Import shared utilities from common
from ..common import (
    ob, og, OMPL_AVAILABLE, OMPL_ERROR,
    interpolate_path_2d, reduce_path_2d, make_clearance_objective_2d,
    dist3d, interpolate_path_3d, reduce_path_3d, make_clearance_objective_3d,
)

if TYPE_CHECKING:
    from ompl import base as ob


# =============================================================================
# 2D RRT*
# =============================================================================

def plan_rrtstar(
    start: Pose2D,
    goal: Pose2D,
    costmap: Costmap2D,
    params: RRTStarOmplParams,
) -> PlanResult:
    """Plan a 2D collision-free path using RRT*."""
    if not OMPL_AVAILABLE:
        return PlanResult(status=PlanStatus.ERROR, message=f"OMPL not available: {OMPL_ERROR}")

    if not costmap.is_free(*costmap.world_to_grid(start.x, start.y)):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision")

    if not costmap.is_free(*costmap.world_to_grid(goal.x, goal.y)):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision")

    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, costmap.origin_x)
    bounds.setHigh(0, costmap.origin_x + costmap.width * costmap.resolution)
    bounds.setLow(1, costmap.origin_y)
    bounds.setHigh(1, costmap.origin_y + costmap.height * costmap.resolution)
    space.setBounds(bounds)

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

    if params.use_clearance_objective and getattr(costmap, "clearance", None) is not None:
        ss.setOptimizationObjective(make_clearance_objective_2d(si, costmap, params.clearance_weight))

    ss.setPlanner(og.RRTstar(si))
    if not ss.solve(params.timeout):
        return PlanResult(status=PlanStatus.NO_PATH, message="No solution found")

    path = ss.getSolutionPath()
    states = [path.getState(i) for i in range(path.getStateCount())]
    reduced = reduce_path_2d(si, costmap, states, params.min_clearance_for_keep)
    waypoints = [Pose2D(s[0], s[1]) for s in reduced]

    for s in reduced:
        si.freeState(s)

    if waypoints[-1].distance_to(goal) > 0.1:
        waypoints.append(goal)

    n_before = len(waypoints)
    waypoints = interpolate_path_2d(waypoints, params.interpolation_spacing)

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=Path2D(
            points=tuple(waypoints),
            frame_id=costmap.frame_id,
            metadata={"planner": "rrtstar", "waypoints_raw": n_before, "waypoints_interpolated": len(waypoints)},
        ),
        message=f"Path found: {n_before} -> {len(waypoints)} waypoints",
    )


# =============================================================================
# 3D RRT*
# =============================================================================

try:
    from sparx_agency.core.common.types import Path3D
except ImportError:
    from dataclasses import dataclass, field
    from typing import Any as _Any, Dict as _Dict, Tuple as _Tuple

    @dataclass(frozen=True)
    class Path3D:
        """3D geometric path."""
        points: _Tuple[Pose3D, ...]
        frame_id: str = "map"
        metadata: _Dict[str, _Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            if len(self.points) < 2:
                raise ValueError("Path3D requires at least 2 points")

        def __len__(self) -> int:
            return len(self.points)


def plan_rrtstar_3d(
    start: Pose3D,
    goal: Pose3D,
    voxelmap,  # expects: is_free_world(x,y,z), world_to_grid, world_clearance, bounds fields
    params: RRTStarOmpl3DParams,
) -> PlanResult:
    """
    Plan a 3D collision-free path using RRT* (OMPL).

    Fixes / guarantees:
      - Uses voxelmap.is_free_world(x,y,z) for OMPL state validity (REAL point check).
      - Rejects approximate solutions: only exact is accepted.
      - Enables clearance objective whenever requested.

    Debug (core-friendly, optional):
      - Throttled state-validity prints.
      - End-of-solve planner graph stats (nodes/edges/goal hits) + exact/approx + approx distance.
      - Counts motion-check calls/success rate to diagnose "tree not growing".

    Note:
      - RRT* step/extension length is controlled by planner range:
            params.rrt_range_m (meters)  [optional]
      - Goal satisfaction region can be widened via:
            params.goal_tolerance_m (meters)  [optional]

    Returns:
      PlanResult with artifacts["path3d"] on success.
    """
    if not OMPL_AVAILABLE:
        return PlanResult(status=PlanStatus.ERROR, message=f"OMPL not available: {OMPL_ERROR}")

    # Validate start/goal using WORLD validity (important!)
    if not voxelmap.is_free_world(start.x, start.y, start.z):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start in collision (world check)")
    if not voxelmap.is_free_world(goal.x, goal.y, goal.z):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal in collision (world check)")

    # --- Params helpers (no hard dependency on extra fields) -----------------
    def _dbg_enabled() -> bool:
        return bool(getattr(params, "debug_enabled", False))

    def _dbg_every_n() -> int:
        return int(getattr(params, "debug_every_n_validity", 2500))

    def _dbg_max_print() -> int:
        return int(getattr(params, "debug_max_print_validity", 25))

    rrt_range_m = getattr(params, "rrt_range_m", None)
    goal_tolerance_m = getattr(params, "goal_tolerance_m", None)
    # -------------------------------------------------------------------------

    # 3D state space (continuous world)
    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, voxelmap.origin_x)
    bounds.setHigh(0, voxelmap.origin_x + voxelmap.width * voxelmap.resolution)
    bounds.setLow(1, voxelmap.origin_y)
    bounds.setHigh(1, voxelmap.origin_y + voxelmap.height * voxelmap.resolution)
    bounds.setLow(2, voxelmap.origin_z)
    bounds.setHigh(2, voxelmap.origin_z + voxelmap.depth * voxelmap.resolution)
    space.setBounds(bounds)

    # Convert "longest_valid_segment_m" to OMPL fraction-of-diagonal (for checkMotion discretization)
    longest_segment = params.longest_valid_segment_m or voxelmap.resolution * 0.5
    space_diagonal = sqrt(
        (voxelmap.width * voxelmap.resolution) ** 2 +
        (voxelmap.height * voxelmap.resolution) ** 2 +
        (voxelmap.depth * voxelmap.resolution) ** 2
    )
    longest_segment_fraction = max(0.001, min(0.1, float(longest_segment) / float(space_diagonal)))
    space.setLongestValidSegmentFraction(longest_segment_fraction)

    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    si.setStateValidityCheckingResolution(params.collision_check_resolution)

    # --- Debug counters (throttled) ------------------------------------------
    dbg_valid: Dict[str, int] = {"calls": 0, "printed": 0}
    mv_stats: Dict[str, int] = {"calls": 0, "ok": 0}
    # -------------------------------------------------------------------------

    def _log_validity_line(x: float, y: float, z: float, ok: bool) -> None:
        i, j, k = voxelmap.world_to_grid(x, y, z)

        # Diagnostic-only: compare real point vs voxel center (helps catch "center-valid but real-invalid")
        cx = voxelmap.origin_x + (i + 0.5) * voxelmap.resolution
        cy = voxelmap.origin_y + (j + 0.5) * voxelmap.resolution
        cz = voxelmap.origin_z + (k + 0.5) * voxelmap.resolution

        inside_real = None
        inside_ctr = None
        d_real = None
        d_ctr = None

        # Keep debug robust and non-invasive: only if helpers exist
        if hasattr(voxelmap, "_mesh_inside_world"):
            try:
                inside_real = bool(voxelmap._mesh_inside_world(x, y, z))
                inside_ctr = bool(voxelmap._mesh_inside_world(cx, cy, cz))
            except Exception:
                inside_real = inside_ctr = None

        if hasattr(voxelmap, "_mesh_distance_world"):
            try:
                d_real = float(voxelmap._mesh_distance_world(x, y, z))
                d_ctr = float(voxelmap._mesh_distance_world(cx, cy, cz))
            except Exception:
                d_real = d_ctr = None

        parts = [
            f"[DEBUG][OMPL][is_valid] world=({x:.3f},{y:.3f},{z:.3f})",
            f"grid=({i},{j},{k})",
            f"ok={ok}",
        ]
        if inside_real is not None:
            parts.append(f"inside(real)={inside_real}")
        if d_real is not None:
            parts.append(f"d(real)={d_real:.3f}m")
        if inside_ctr is not None:
            parts.append(f"inside(center)={inside_ctr}")
        if d_ctr is not None:
            parts.append(f"d(center)={d_ctr:.3f}m")
        print(" | ".join(parts))

    def is_valid(state) -> bool:
        """State validity for OMPL (world point)."""
        dbg_valid["calls"] += 1
        x, y, z = float(state[0]), float(state[1]), float(state[2])
        ok = bool(voxelmap.is_free_world(x, y, z))

        if _dbg_enabled():
            every = max(1, _dbg_every_n())
            if dbg_valid["printed"] < _dbg_max_print() and (dbg_valid["calls"] % every == 0):
                dbg_valid["printed"] += 1
                _log_validity_line(x, y, z, ok)

        return ok

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))

    # --- Count motion-check success rate (very useful when tree doesn't grow) -
    # Wrap the existing MotionValidator without changing semantics.
    try:
        base_mv = si.getMotionValidator()

        class _CountingMotionValidator(ob.MotionValidator):
            def __init__(self, si_, wrapped_) -> None:
                super().__init__(si_)
                self._wrapped = wrapped_

            def checkMotion(self, s1, s2) -> bool:
                mv_stats["calls"] += 1
                ok_ = bool(self._wrapped.checkMotion(s1, s2))
                if ok_:
                    mv_stats["ok"] += 1
                return ok_

            def checkMotionWithLastValid(self, s1, s2):
                mv_stats["calls"] += 1
                ok_, last_ = self._wrapped.checkMotionWithLastValid(s1, s2)
                ok_ = bool(ok_)
                if ok_:
                    mv_stats["ok"] += 1
                return ok_, last_

        si.setMotionValidator(_CountingMotionValidator(si, base_mv))
    except Exception:
        # If wrapping is not supported in this OMPL build, just skip stats.
        pass
    # -------------------------------------------------------------------------

    start_state, goal_state = ob.State(space), ob.State(space)
    start_state[0], start_state[1], start_state[2] = start.x, start.y, start.z
    goal_state[0], goal_state[1], goal_state[2] = goal.x, goal.y, goal.z

    # Goal tolerance (optional). If None -> keep original behavior.
    if goal_tolerance_m is not None:
        ss.setStartAndGoalStates(start_state, goal_state, float(goal_tolerance_m))
    else:
        ss.setStartAndGoalStates(start_state, goal_state)

    # Enable clearance objective if requested
    if params.use_clearance_objective:
        ss.setOptimizationObjective(make_clearance_objective_3d(si, voxelmap, params.clearance_weight))

    # Planner
    planner = og.RRTstar(si)
    if rrt_range_m is not None:
        try:
            planner.setRange(float(rrt_range_m))
        except Exception:
            pass
    ss.setPlanner(planner)

    # Solve
    solved = ss.solve(params.timeout)

    # ----------------------- End-of-solve DEBUG/STATS -------------------------
    if _dbg_enabled():
        # Planner graph stats (nodes/edges/goal hits)
        try:
            pdata = ob.PlannerData(si)
            ss.getPlannerData(pdata)
            n_vertices = int(pdata.numVertices())
            n_edges = int(pdata.numEdges())

            goal_obj = ss.getProblemDefinition().getGoal()
            goal_hits = 0
            for vi in range(n_vertices):
                try:
                    st = pdata.getVertex(vi).getState()
                    if goal_obj.isSatisfied(st):
                        goal_hits += 1
                except Exception:
                    pass

            have_solution = bool(ss.haveSolutionPath())
            have_exact = False
            try:
                have_exact = bool(ss.haveExactSolutionPath())
            except Exception:
                have_exact = False

            print(f"[DEBUG][OMPL][stats] graph_vertices={n_vertices}")
            print(f"[DEBUG][OMPL][stats] graph_edges={n_edges}")
            print(f"[DEBUG][OMPL][stats] goal_states_in_tree={goal_hits}")
            print(f"[DEBUG][OMPL][stats] have_solution_path={have_solution}")
            print(f"[DEBUG][OMPL][stats] have_exact_solution={have_exact}")
            print(f"[DEBUG][OMPL][stats] validity_calls={dbg_valid['calls']} printed={dbg_valid['printed']}")
            print(f"[DEBUG][OMPL][stats] longest_valid_segment_fraction={longest_segment_fraction:.6f}")
            print(f"[DEBUG][OMPL][stats] state_validity_resolution={params.collision_check_resolution}")

            if rrt_range_m is not None:
                print(f"[DEBUG][OMPL][stats] rrt_range_m={float(rrt_range_m):.3f}")
            if goal_tolerance_m is not None:
                print(f"[DEBUG][OMPL][stats] goal_tolerance_m={float(goal_tolerance_m):.3f}")

            if mv_stats["calls"] > 0:
                rate = 100.0 * float(mv_stats["ok"]) / float(mv_stats["calls"])
                print(f"[DEBUG][OMPL][stats] motion_check_calls={mv_stats['calls']}")
                print(f"[DEBUG][OMPL][stats] motion_check_ok={mv_stats['ok']} ({rate:.2f}%)")

            if have_solution and not have_exact:
                try:
                    apath = ss.getSolutionPath()
                    last = apath.getState(apath.getStateCount() - 1)
                    dx = float(last[0]) - float(goal.x)
                    dy = float(last[1]) - float(goal.y)
                    dz = float(last[2]) - float(goal.z)
                    dist = float(sqrt(dx * dx + dy * dy + dz * dz))
                    print(f"[DEBUG][OMPL][stats] approximate_distance_to_goal_m={dist:.3f}")
                except Exception:
                    pass

        except Exception as e:
            print(f"[DEBUG][OMPL][stats] failed_to_collect_planner_data: {e}")
    # -------------------------------------------------------------------------

    if not solved:
        return PlanResult(status=PlanStatus.NO_PATH, message="No solution found")

    # Reject approximate solutions
    is_exact = False
    try:
        is_exact = bool(ss.haveExactSolutionPath())
    except Exception:
        try:
            is_exact = not bool(ss.getSolutionPath().isApproximate())
        except Exception:
            is_exact = False

    if not is_exact:
        msg = "OMPL returned an APPROXIMATE solution (rejected). Increase time or relax constraints."
        return PlanResult(status=PlanStatus.NO_PATH, message=msg)

    # Extract solution
    path = ss.getSolutionPath()
    states = [path.getState(i) for i in range(path.getStateCount())]

    # Reduce waypoints using OMPL motion checks
    reduced = reduce_path_3d(si, voxelmap, states, params.min_clearance_for_keep)
    waypoints = [Pose3D(float(s[0]), float(s[1]), float(s[2])) for s in reduced]

    for s in reduced:
        si.freeState(s)

    if dist3d(waypoints[-1], goal) > 0.1:
        waypoints.append(goal)

    n_before = len(waypoints)
    waypoints = interpolate_path_3d(waypoints, params.interpolation_spacing)

    path3d = Path3D(
        points=tuple(waypoints),
        frame_id=getattr(voxelmap, "frame_id", "map"),
        metadata={
            "planner": "rrtstar_3d",
            "waypoints_raw": n_before,
            "waypoints_interpolated": len(waypoints),
            "longest_valid_segment_fraction": longest_segment_fraction,
            "state_validity_resolution": params.collision_check_resolution,
            **({"rrt_range_m": float(rrt_range_m)} if rrt_range_m is not None else {}),
            **({"goal_tolerance_m": float(goal_tolerance_m)} if goal_tolerance_m is not None else {}),
        },
    )

    # Optional integrity check (world validity) - debug only
    if _dbg_enabled():
        bad = 0
        for p in waypoints:
            if not voxelmap.is_free_world(p.x, p.y, p.z):
                bad += 1
        if bad > 0:
            print(f"[DEBUG][path] invalid_waypoints_world={bad}/{len(waypoints)} (investigate collision model)")

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=None,
        message=f"3D path found: {n_before} -> {len(waypoints)} waypoints",
        artifacts={"path3d": path3d},
    )
