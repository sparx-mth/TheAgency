"""
OMPL-backed RRT* planning algorithm (ROS-free).

Implements:
1) OMPL RRT* planning on a 2D grid/costmap
2) Adaptive waypoint reduction (keep points in tight spaces / when shortcut invalid)
3) World-space interpolation at fixed spacing

Outputs:
- Path2D (NO velocities)
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, List, Optional, Sequence, Tuple

from core.common.types import Pose2D, Path2D, PlanResult, PlanStatus

from .params import RRTStarOmplParams

try:
    from ompl import base as ob
    from ompl import geometric as og
except Exception as e:  # pragma: no cover
    ob = None
    og = None
    _OMPL_IMPORT_ERROR = e
else:
    _OMPL_IMPORT_ERROR = None


# ---------------------------------------------------------------------
# World adapter (duck-typed)
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GridWorldView:
    """
    Minimal interface the planner needs from the environment.

    Required attributes / methods:
      - width: int   (#cells)
      - height: int  (#cells)
      - resolution: float (meters/cell)
      - origin_x: float (world)
      - origin_y: float (world)
      - is_free(ix:int, iy:int) -> bool

    Optional:
      - clearance_at_world(x:float, y:float) -> float  (meters)
        OR
      - clearance_at_cell(ix:int, iy:int) -> float     (cells or meters)
    """
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    is_free_fn: Any
    clearance_world_fn: Optional[Any] = None
    clearance_cell_fn: Optional[Any] = None

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(round((x - self.origin_x) / self.resolution))
        iy = int(round((y - self.origin_y) / self.resolution))
        return ix, iy

    def cell_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        x = ix * self.resolution + self.origin_x
        y = iy * self.resolution + self.origin_y
        return x, y

    def is_free(self, ix: int, iy: int) -> bool:
        if ix < 0 or ix >= self.width or iy < 0 or iy >= self.height:
            return False
        return bool(self.is_free_fn(ix, iy))

    def clearance_at(self, ix: int, iy: int, wx: Optional[float] = None, wy: Optional[float] = None) -> Optional[float]:
        # Prefer world clearance if available
        if self.clearance_world_fn is not None and wx is not None and wy is not None:
            try:
                return float(self.clearance_world_fn(wx, wy))
            except Exception:
                return None
        if self.clearance_cell_fn is not None:
            try:
                return float(self.clearance_cell_fn(ix, iy))
            except Exception:
                return None
        return None


def world_view_from_costmap(world: Any) -> GridWorldView:
    """
    Build a GridWorldView from your environment object (e.g., Costmap2D).

    This is intentionally permissive (duck typing) so you can adapt without
    pulling ROS into core.
    """
    # Required
    width = int(getattr(world, "width"))
    height = int(getattr(world, "height"))
    resolution = float(getattr(world, "resolution"))
    origin_x = float(getattr(world, "origin_x"))
    origin_y = float(getattr(world, "origin_y"))

    # Required method: is_free(ix, iy)
    if hasattr(world, "is_free") and callable(world.is_free):
        is_free_fn = world.is_free
    elif hasattr(world, "is_occupied") and callable(world.is_occupied):
        is_free_fn = lambda ix, iy: not bool(world.is_occupied(ix, iy))
    else:
        raise AttributeError("World must expose is_free(ix,iy) or is_occupied(ix,iy)")

    # Optional clearance
    clearance_world_fn = getattr(world, "clearance_at_world", None)
    if not callable(clearance_world_fn):
        clearance_world_fn = None

    clearance_cell_fn = getattr(world, "clearance_at_cell", None)
    if not callable(clearance_cell_fn):
        clearance_cell_fn = None

    return GridWorldView(
        width=width,
        height=height,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        is_free_fn=is_free_fn,
        clearance_world_fn=clearance_world_fn,
        clearance_cell_fn=clearance_cell_fn,
    )


# ---------------------------------------------------------------------
# OMPL objective (clearance)
# ---------------------------------------------------------------------

class _ClearanceObjective(ob.StateCostIntegralObjective):
    def __init__(self, si: "ob.SpaceInformation", view: GridWorldView, weight: float):
        super().__init__(si, True)
        self._view = view
        self._w = float(weight)

    def stateCost(self, state: "ob.State") -> "ob.Cost":
        st = state.get()
        # Python bindings: RealVectorStateSpace state supports indexing
        x = float(st[0])
        y = float(st[1])
        ix = int(round(x))
        iy = int(round(y))

        # Convert cell->world for optional clearance world function
        wx, wy = self._view.cell_to_world(ix, iy)
        c = self._view.clearance_at(ix, iy, wx=wx, wy=wy)
        if c is None:
            # fallback: no clearance information
            return ob.Cost(0.0)

        # Similar to your: weight/(clearance+1)
        return ob.Cost(self._w / (float(c) + 1.0))


# ---------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------

def _interpolate_world(points: Sequence[Pose2D], spacing_m: float) -> List[Pose2D]:
    if len(points) < 2 or spacing_m <= 0:
        return list(points)

    out: List[Pose2D] = [points[0]]

    for a, b in zip(points[:-1], points[1:]):
        dx = b.x - a.x
        dy = b.y - a.y
        seg_len = hypot(dx, dy)
        if seg_len < 1e-9:
            continue

        n_mid = int(seg_len // spacing_m)
        for j in range(1, n_mid + 1):
            t = j / (n_mid + 1)
            out.append(Pose2D(a.x + t * dx, a.y + t * dy, 0.0))

        out.append(Pose2D(b.x, b.y, 0.0))

    return out


def _adaptive_reduce(
    si: "ob.SpaceInformation",
    view: GridWorldView,
    path_states: Sequence["ob.State"],
    min_clearance_keep: float,
) -> List["ob.State"]:
    """
    Similar to your C++:
    - Keep the first state
    - For each intermediate i, try to skip it if:
        clearance >= threshold AND si.checkMotion(last_kept, state[i+1]) is True
      otherwise keep it.
    """
    if len(path_states) < 2:
        return list(path_states)

    kept: List["ob.State"] = [si.cloneState(path_states[0])]

    for i in range(1, len(path_states) - 1):
        curr = path_states[i].get()
        x = float(curr[0])
        y = float(curr[1])
        ix = int(round(x))
        iy = int(round(y))

        wx, wy = view.cell_to_world(ix, iy)
        clearance = view.clearance_at(ix, iy, wx=wx, wy=wy)

        can_skip = si.checkMotion(kept[-1], path_states[i + 1])

        # If no clearance exists, behave conservatively: keep when shortcut is invalid
        if clearance is None:
            if not can_skip:
                kept.append(si.cloneState(path_states[i]))
            continue

        if (clearance < min_clearance_keep) or (not can_skip):
            kept.append(si.cloneState(path_states[i]))

    kept.append(si.cloneState(path_states[-1]))
    return kept


# ---------------------------------------------------------------------
# Main planning entry
# ---------------------------------------------------------------------

def plan_rrtstar_ompl(
    start: Pose2D,
    goal: Pose2D,
    world: Any,
    params: RRTStarOmplParams,
) -> PlanResult:
    if ob is None or og is None:
        return PlanResult(
            status=PlanStatus.ERROR,
            message=(
                "OMPL python bindings are not available (import ompl failed). "
                f"Import error: {_OMPL_IMPORT_ERROR}"
            ),
        )

    view = world_view_from_costmap(world)

    # Map start/goal to grid cells
    sx, sy = view.world_to_cell(start.x, start.y)
    gx, gy = view.world_to_cell(goal.x, goal.y)

    if not view.is_free(sx, sy):
        return PlanResult(status=PlanStatus.INVALID_START, message="Start is in obstacle")
    if not view.is_free(gx, gy):
        return PlanResult(status=PlanStatus.INVALID_GOAL, message="Goal is in obstacle")

    # OMPL setup (cell-space)
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, 0.0)
    bounds.setHigh(0, float(view.width - 1))
    bounds.setLow(1, 0.0)
    bounds.setHigh(1, float(view.height - 1))
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)

    def is_state_valid(state: "ob.State") -> bool:
        st = state.get()
        ix = int(round(float(st[0])))
        iy = int(round(float(st[1])))
        return view.is_free(ix, iy)

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))

    start_state = ob.State(space)
    start_state()[0] = float(sx)
    start_state()[1] = float(sy)

    goal_state = ob.State(space)
    goal_state()[0] = float(gx)
    goal_state()[1] = float(gy)

    ss.setStartAndGoalStates(start_state, goal_state)

    # Objective (clearance) - if available
    if params.use_clearance_objective:
        si = ss.getSpaceInformation()
        ss.setOptimizationObjective(_ClearanceObjective(si, view, params.clearance_weight))

    # Planner
    ss.setPlanner(og.RRTstar(ss.getSpaceInformation()))

    solved = ss.solve(float(params.planning_timeout_s))
    if not solved:
        return PlanResult(status=PlanStatus.NO_PATH, message="No path found")

    path = ss.getSolutionPath()
    si = ss.getSpaceInformation()

    # Collect states
    states: List[ob.State] = []
    for i in range(path.getStateCount()):
        states.append(path.getState(i))

    # Step 1: Adaptive reduction (like your smoothing stage)
    reduced = _adaptive_reduce(
        si=si,
        view=view,
        path_states=states,
        min_clearance_keep=params.min_clearance_for_keep,
    )

    # Step 2: Convert to world coordinates + validate
    world_points: List[Pose2D] = []
    for s in reduced:
        st = s.get()
        ix = int(round(float(st[0])))
        iy = int(round(float(st[1])))

        if not view.is_free(ix, iy):
            # Free cloned states
            for cs in reduced:
                si.freeState(cs)
            return PlanResult(status=PlanStatus.ERROR, message="Path validation failed (invalid waypoint)")

        wx, wy = view.cell_to_world(ix, iy)
        world_points.append(Pose2D(wx, wy, 0.0))

    # Free cloned states
    for cs in reduced:
        si.freeState(cs)

    # Ensure exact goal is included (like your dist_to_goal check)
    if hypot(world_points[-1].x - goal.x, world_points[-1].y - goal.y) > 0.1:
        world_points.append(Pose2D(goal.x, goal.y, 0.0))

    before_interp = len(world_points)

    # Step 3: Interpolation in world meters
    world_points = _interpolate_world(world_points, params.interpolation_spacing_m)

    result_path = Path2D(points=tuple(world_points), frame_id=params.frame_id, metadata={
        "planner": "rrtstar_ompl",
        "points_before_interpolation": before_interp,
        "points_after_interpolation": len(world_points),
        "planning_timeout_s": params.planning_timeout_s,
        "used_clearance_objective": bool(params.use_clearance_objective),
    })

    return PlanResult(
        status=PlanStatus.SUCCESS,
        path=result_path,
        message=f"Path: {before_interp} -> {len(world_points)} pts (interpolated)",
        artifacts={"grid_start": (sx, sy), "grid_goal": (gx, gy)},
    )
