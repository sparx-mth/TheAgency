"""
RRT* planner implementation using OMPL Python bindings.

Implements BasePlanner:
  plan(PlanRequest, world) -> PlanResult
"""
from __future__ import annotations

from dataclasses import dataclass

from core.planning.interfaces.planner import BasePlanner, PlanRequest
from core.common.types import PlanResult

from .params import RRTStarOmplParams
from .algorithm import plan_rrtstar_ompl


@dataclass
class RRTStarOmplPlanner(BasePlanner):
    """
    OMPL RRT* planner that returns a geometric Path2D (no velocities).

    The 'world' is expected to behave like Costmap2D (duck typing):
      - width, height, resolution, origin_x, origin_y
      - is_free(ix, iy) -> bool
      - optional clearance functions (see algorithm.py)
    """
    params: RRTStarOmplParams = RRTStarOmplParams()
    name: str = "rrtstar_ompl"

    def plan(self, request: PlanRequest, world: object) -> PlanResult:
        return plan_rrtstar_ompl(
            start=request.start,
            goal=request.goal,
            world=world,
            params=self.params,
        )
