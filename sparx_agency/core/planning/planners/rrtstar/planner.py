"""RRT* planner implementing BasePlanner protocol."""
from __future__ import annotations

from dataclasses import dataclass, field

from sparx_agency.core.common.types import PlanResult
from sparx_agency.core.planning.interfaces.planner import BasePlanner, PlanRequest
from sparx_agency.core.planning.environment import Costmap2D

from .params import RRTStarParams
from .algorithm import plan_rrtstar


@dataclass
class RRTStarPlanner(BasePlanner):
    """
    RRT* planner that produces geometric paths.

    Uses OMPL's RRT* implementation with optional clearance-based
    cost optimization. Output paths are interpolated at uniform spacing.
    """
    name: str = field(default="rrtstar", init=False)
    params: RRTStarParams = field(default_factory=RRTStarParams)

    def plan(self, request: PlanRequest, world: Costmap2D) -> PlanResult:
        """
        Compute a collision-free path from start to goal.

        Args:
            request: Planning request with start/goal poses.
            world: Costmap2D occupancy grid.

        Returns:
            PlanResult with status and Path2D if successful.
        """
        return plan_rrtstar(
            start=request.start,
            goal=request.goal,
            costmap=world,
            params=self.params,
        )