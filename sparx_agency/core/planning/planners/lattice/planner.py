"""The discrete-action lattice planner: a 2D route in, the executable action sequence out.

:class:`DiscreteActionLatticePlanner` owns what the bare search does not: the
benchmark's primitives, the parameters, the memory of refused moves and the
running statistics. It is not a :class:`BasePlanner`: it refines a route the
2D planner has already found through the same cost grid, so its input is the
route and the grid's cost and clearance fields rather than a start and a
goal on a map.

**Refused moves** (the simulator turned a MOVE_FORWARD down) are remembered
as poses, not as cells: painting the cell ahead occupied blocks every heading
through it and the way back, and inside a doorway it seals the doorway. A
remembered pose excludes exactly one primitive -- the forward step from
there, and from within ``refused_move_radius_m`` of there at the same
lattice heading -- for ``refused_move_actions`` actions, so the next plan
turns first or lines up a few centimetres over instead of trying the same
step again.

Python 3.8 syntax; numpy 1.17 API only.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.actions import DiscreteActionSpec

from .params import LatticePlannerParams
from .primitives import MotionPrimitives
from .route_field import route_corridor_fields
from .search import STATUS_SUCCESS, LatticePlan, LatticePose, lattice_search


class DiscreteActionLatticePlanner:
    """Plan executable action sequences along 2D routes, remembering where the simulator refused a step.

    Args:
        spec: The benchmark's action geometry.
        params: The search's costs, tolerances and caps; the defaults when None.
    """

    name = "discrete_action_lattice"

    def __init__(self, spec: DiscreteActionSpec, params: Optional[LatticePlannerParams] = None):
        self.params = params or LatticePlannerParams()
        self.primitives = MotionPrimitives(spec, self.params.sample_spacing_m)
        self._refused: Dict[Tuple[int, int], List[Tuple[float, float, float, int]]] = {}
        self._refused_cell_m = max(self.params.refused_move_radius_m, 1e-3)
        self.stats = {"plans": 0, "successes": 0, "failures": 0, "expanded": 0, "actions": 0, "turns": 0,
                      "refused_moves": 0, "refused_expired": 0, "plan_ms_total": 0.0, "plan_ms_max": 0.0}
        self.last: Optional[LatticePlan] = None

    # ------------------------------------------------------------------
    # Refused moves
    # ------------------------------------------------------------------
    def _bucket(self, x: float, y: float) -> Tuple[int, int]:
        return int(math.floor(x / self._refused_cell_m)), int(math.floor(y / self._refused_cell_m))

    def note_refused(self, x: float, y: float, yaw: float, step: int) -> None:
        """Remember that a MOVE_FORWARD from ``(x, y, yaw)`` was refused at action ``step``."""
        self._refused.setdefault(self._bucket(x, y), []).append((float(x), float(y), float(yaw), int(step)))
        self.stats["refused_moves"] += 1

    def expire(self, step: int) -> None:
        """Forget refused moves older than ``refused_move_actions`` (never, when that is 0)."""
        horizon = self.params.refused_move_actions
        if horizon <= 0:
            return
        for key in list(self._refused):
            kept = [entry for entry in self._refused[key] if step - entry[3] <= horizon]
            self.stats["refused_expired"] += len(self._refused[key]) - len(kept)
            if kept:
                self._refused[key] = kept
            else:
                del self._refused[key]

    def move_refused(self, x: float, y: float, yaw: float) -> bool:
        """Whether a MOVE_FORWARD from ``(x, y, yaw)`` is excluded by a remembered refusal."""
        if not self._refused:
            return False
        bx, by = self._bucket(x, y)
        radius = self.params.refused_move_radius_m
        half_turn = self.primitives.turn_rad / 2.0 + 1e-6
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for rx, ry, ryaw, _step in self._refused.get((bx + dx, by + dy), ()):
                    if math.hypot(rx - x, ry - y) <= radius and abs(normalize_angle(ryaw - yaw)) <= half_turn:
                        return True
        return False

    @property
    def refused(self) -> List[Tuple[float, float, float, int]]:
        """Every remembered refusal, ``(x, y, yaw, step)``."""
        return [entry for entries in self._refused.values() for entry in entries]

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan(self, start: LatticePose, route: Sequence[Tuple[float, float]], origin_x: float, origin_y: float,
             resolution: float, cost: np.ndarray, clearance: np.ndarray, footprint_radius_m: float,
             goal_yaw: Optional[float] = None, step: int = 0, start_free_radius_m: float = 0.0) -> LatticePlan:
        """The action sequence that walks ``route`` from ``start``.

        Args:
            start: The agent's pose ``(x, y, yaw)``.
            route: The 2D route, world ``(x, y)`` pairs, goal last.
            origin_x: World x of the grid's cell ``(0, 0)`` corner.
            origin_y: World y of that corner.
            resolution: Metres per cell.
            cost: The 2D planner's cost grid the route was planned on.
            clearance: Its distance-to-confirmed-obstacle field.
            footprint_radius_m: The standoff the route was planned at.
            goal_yaw: A heading to end at, or None.
            step: The current action count, for the refused-move horizon.
            start_free_radius_m: See :func:`.search.lattice_search`.

        Returns:
            The plan; ``ok`` False with a status and message when none exists.
        """
        if cost.shape != clearance.shape:
            raise ValueError("cost %s and clearance %s are not co-registered" % (cost.shape, clearance.shape))
        self.expire(step)
        started = time.monotonic()
        field, offset = route_corridor_fields(route, origin_x, origin_y, resolution, cost.shape, self.params.corridor_m,
                                              cost=cost)
        plan = lattice_search(start, tuple(route[-1]), cost, clearance, footprint_radius_m, field, origin_x, origin_y,
                              resolution, self.primitives, self.params, goal_yaw=goal_yaw,
                              move_refused=self.move_refused if self._refused else None,
                              start_free_radius_m=start_free_radius_m,
                              offset=offset if self.params.route_offset_cost > 0.0 else None)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        s = self.stats
        s["plans"] += 1
        s["expanded"] += plan.expanded
        s["plan_ms_total"] += elapsed_ms
        s["plan_ms_max"] = max(s["plan_ms_max"], elapsed_ms)
        if plan.status == STATUS_SUCCESS:
            s["successes"] += 1
            s["actions"] += len(plan.actions)
            s["turns"] += plan.turns
        else:
            s["failures"] += 1
            s["failed:" + plan.status] = s.get("failed:" + plan.status, 0) + 1
        self.last = plan
        return plan

    def diagnostics(self) -> dict:
        """The running statistics and the refusals in force, JSON-serialisable."""
        out = dict(self.stats)
        out["plan_ms_total"] = round(out["plan_ms_total"], 1)
        out["plan_ms_max"] = round(out["plan_ms_max"], 1)
        out["refused_in_force"] = len(self.refused)
        return out
