"""A* over the discrete-action state lattice ``(x, y, heading)``: the plan is the action sequence.

The search is Hybrid A* with the benchmark's own primitives for motions:
every state carries its exact continuous pose, states are merged for the
closed set by the grid cell of their position and their lattice heading, and
the only edges are one MOVE_FORWARD (collision-checked at every sample along
the step against the footprint radius on the clearance field, priced by the
2D cost grid it crosses) and one TURN each way (always feasible, priced as an
action). The heuristic and the search domain are the 2D route's cost-to-go
field (:mod:`.route_field`), so the search refines the route the global
planner found instead of re-deriving it, and a 30 m route is a few thousand
states. What comes back is executable as it is: the poses are the ones the
agent will stand at after each action, to the millimetre, if the simulator
executes every action.

Python 3.8 syntax; numpy 1.17 API only.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable, Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction

from .params import LatticePlannerParams
from .primitives import MotionPrimitives

STATUS_SUCCESS = "success"
STATUS_NO_PLAN = "no_plan"
STATUS_EXPANSION_CAP = "expansion_cap"
STATUS_START_OUTSIDE = "start_outside_corridor"
STATUS_GOAL_OUTSIDE = "goal_outside_grid"
STATUSES = (STATUS_SUCCESS, STATUS_NO_PLAN, STATUS_EXPANSION_CAP, STATUS_START_OUTSIDE, STATUS_GOAL_OUTSIDE)

#: A pose on the lattice: ``(x, y, yaw)``, world metres and radians.
LatticePose = Tuple[float, float, float]


@dataclass(frozen=True)
class LatticePlan:
    """The outcome of one lattice search.

    Attributes:
        actions: The body actions to execute, in order. Empty on failure, and
            on success when the start already satisfies the goal.
        poses: The pose before each action and the final pose: one more than
            the actions. On failure, the start alone.
        cost: The plan's cost in action units (``inf`` on failure).
        expanded: States popped from the open list.
        status: One of :data:`STATUSES`.
        message: Why, in one line.
    """

    actions: Tuple[DiscreteAction, ...]
    poses: Tuple[LatticePose, ...]
    cost: float
    expanded: int
    status: str
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SUCCESS

    @property
    def forward_steps(self) -> int:
        return sum(1 for a in self.actions if a == DiscreteAction.MOVE_FORWARD)

    @property
    def turns(self) -> int:
        return len(self.actions) - self.forward_steps


def _failure(status: str, message: str, start: LatticePose, expanded: int = 0) -> LatticePlan:
    return LatticePlan((), (start,), math.inf, expanded, status, message)


def lattice_search(start: LatticePose, goal_xy: Tuple[float, float], cost: np.ndarray, clearance: np.ndarray,
                   footprint_radius_m: float, field: np.ndarray, origin_x: float, origin_y: float, resolution: float,
                   primitives: MotionPrimitives, params: LatticePlannerParams, goal_yaw: Optional[float] = None,
                   move_refused: Optional[Callable[[float, float, float], bool]] = None,
                   start_free_radius_m: float = 0.0, offset: Optional[np.ndarray] = None) -> LatticePlan:
    """Search the lattice from ``start`` to within the goal tolerance of ``goal_xy``.

    Args:
        start: The agent's pose ``(x, y, yaw)``.
        goal_xy: The goal point, world metres.
        cost: ``(H, W)`` per-cell traversal cost of the 2D planner (``inf`` blocked).
        clearance: ``(H, W)`` distance to the nearest confirmed obstacle, metres.
        footprint_radius_m: A sample is feasible only where the clearance is
            at least this: the body radius plus the standoff the route was
            planned at.
        field: The route's cost-to-go field (:func:`.route_field.route_cost_to_go`).
        origin_x: World x of cell ``(0, 0)``'s corner.
        origin_y: World y of that corner.
        resolution: Metres per cell.
        primitives: The benchmark's step and turn.
        params: Costs, tolerances and the expansion cap.
        goal_yaw: A heading to end at, or None for any.
        move_refused: ``f(x, y, yaw) -> bool``: whether a MOVE_FORWARD from
            that pose is excluded (the simulator refused one there).
        start_free_radius_m: Within this distance of the start a sample in a
            lethal-but-not-occupied cell is passable: the body stands there,
            inside the inflation skirt of a wall it is not touching.
        offset: The route's lateral-offset field (``route_corridor_fields``),
            charged ``params.route_offset_cost`` per metre at the end of
            every forward step; None charges nothing.

    Returns:
        The plan.
    """
    h, w = cost.shape
    res = float(resolution)
    x0, y0, yaw0 = float(start[0]), float(start[1]), float(start[2])
    gx_goal, gy_goal = float(goal_xy[0]), float(goal_xy[1])
    step, turn = primitives.step_m, primitives.turn_rad
    tol = float(params.goal_tolerance_m)
    tol_yaw = float(params.goal_heading_tolerance_rad)
    forward_cost, turn_cost = float(params.forward_action_cost), float(params.turn_action_cost)
    offset_cost = float(params.route_offset_cost) if offset is not None else 0.0
    h_scale = float(params.heuristic_weight) * forward_cost / step
    samples = primitives.samples
    inv_count = 1.0 / len(samples)

    def cell(x, y):
        return int(math.floor((x - origin_x) / res)), int(math.floor((y - origin_y) / res))

    sgx, sgy = cell(x0, y0)
    if not (0 <= sgx < w and 0 <= sgy < h) or not math.isfinite(field[sgy, sgx]):
        return _failure(STATUS_START_OUTSIDE, "the start lies outside the route's corridor", start)
    ggx, ggy = cell(gx_goal, gy_goal)
    if not (0 <= ggx < w and 0 <= ggy < h):
        return _failure(STATUS_GOAL_OUTSIDE, "the goal lies outside the grid", start)

    def at_goal(x, y, yaw):
        if math.hypot(x - gx_goal, y - gy_goal) > tol:
            return False
        return goal_yaw is None or abs(normalize_angle(yaw - goal_yaw)) <= tol_yaw

    if at_goal(x0, y0, yaw0):
        return LatticePlan((), (start,), 0.0, 0, STATUS_SUCCESS, "the start satisfies the goal")

    start_relax = float(start_free_radius_m)
    # Per lattice heading: the unit vector, cached once; headings are yaw0 + k * turn.
    unit = {}

    def heading_unit(k):
        value = unit.get(k)
        if value is None:
            yaw = primitives.heading(yaw0, k)
            value = (yaw, math.cos(yaw), math.sin(yaw))
            unit[k] = value
        return value

    def forward_edge(x, y, k):
        """``(nx, ny, edge_cost)`` for one MOVE_FORWARD from ``(x, y)`` at heading ``k``, or None when infeasible."""
        _yaw, c, s = heading_unit(k)
        total = 0.0
        for f in samples:
            sx, sy = x + f * step * c, y + f * step * s
            cx, cy = int(math.floor((sx - origin_x) / res)), int(math.floor((sy - origin_y) / res))
            if not (0 <= cx < w and 0 <= cy < h):
                return None
            value = cost[cy, cx]
            if math.isfinite(value) and clearance[cy, cx] >= footprint_radius_m:
                total += value
                continue
            # The relaxed start: a cell under the body's own skirt, never an obstacle cell.
            if start_relax > 0.0 and clearance[cy, cx] > 0.0 and math.hypot(sx - x0, sy - y0) <= start_relax:
                total += value if math.isfinite(value) else 1.0
                continue
            return None
        return x + step * c, y + step * s, forward_cost * total * inv_count

    # Nodes: parallel lists indexed by node id (one id per push). The open heap holds (f, g, id); ``best``
    # holds the cheapest g per (cell x, cell y, heading index) and ``closed`` the keys already expanded.
    xs, ys, ks, parents, acts = [x0], [y0], [0], [-1], [None]
    best = {(sgx, sgy, 0): 0.0}
    # Ties on f go to the deeper node (larger g): along a corridor that is the one nearer the goal, and
    # it keeps the search from fanning out across every heading of every cell at the same f.
    open_heap = [(h_scale * field[sgy, sgx], -0.0, 0)]
    expanded = 0
    closed = set()
    while open_heap:
        _f, neg_g, node = heapq.heappop(open_heap)
        g = -neg_g
        x, y, k = xs[node], ys[node], ks[node]
        key_cell = cell(x, y)
        key = (key_cell[0], key_cell[1], k)
        if key in closed:
            continue
        closed.add(key)
        expanded += 1
        yaw, _c, _s = heading_unit(k)
        if at_goal(x, y, yaw):
            actions, poses = [], [(x, y, yaw)]
            cur = node
            while parents[cur] >= 0:
                actions.append(acts[cur])
                cur = parents[cur]
                poses.append((xs[cur], ys[cur], heading_unit(ks[cur])[0]))
            actions.reverse()
            poses.reverse()
            return LatticePlan(tuple(actions), tuple(poses), g, expanded, STATUS_SUCCESS,
                               "%d actions (%d forward, %d turns), %d states expanded"
                               % (len(actions), sum(1 for a in actions if a == DiscreteAction.MOVE_FORWARD),
                                  sum(1 for a in actions if a != DiscreteAction.MOVE_FORWARD), expanded))
        if expanded > params.max_expansions:
            return _failure(STATUS_EXPANSION_CAP, "no plan within %d expansions" % params.max_expansions, start, expanded)
        # MOVE_FORWARD first: a plan with fewer turns is found sooner.
        if move_refused is None or not move_refused(x, y, yaw):
            edge = forward_edge(x, y, k)
            if edge is not None:
                nx, ny, edge_cost = edge
                ngx, ngy = cell(nx, ny)
                if 0 <= ngx < w and 0 <= ngy < h:
                    rest = field[ngy, ngx]
                    if math.isfinite(rest):
                        nkey = (ngx, ngy, k)
                        ng = g + edge_cost
                        if offset_cost > 0.0:
                            ng += offset_cost * offset[ngy, ngx]
                        if ng < best.get(nkey, math.inf):
                            best[nkey] = ng
                            xs.append(nx); ys.append(ny); ks.append(k)
                            parents.append(node); acts.append(DiscreteAction.MOVE_FORWARD)
                            heapq.heappush(open_heap, (ng + h_scale * rest, -ng, len(xs) - 1))
        rest = field[key_cell[1], key_cell[0]]
        came_by = acts[node]
        for action in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT):
            # A turn that undoes the previous one is never part of a shortest plan: pruned, like the
            # mirror sequences it would generate, before it costs a state.
            if came_by is not None and came_by != DiscreteAction.MOVE_FORWARD and came_by != action:
                continue
            nk = primitives.turned(k, action)
            nkey = (key_cell[0], key_cell[1], nk)
            ng = g + turn_cost
            if ng < best.get(nkey, math.inf):
                best[nkey] = ng
                xs.append(x); ys.append(y); ks.append(nk)
                parents.append(node); acts.append(action)
                heapq.heappush(open_heap, (ng + h_scale * rest, -ng, len(xs) - 1))
    return _failure(STATUS_NO_PLAN, "no action sequence reaches the goal inside the corridor (%d states)" % expanded,
                    start, expanded)
