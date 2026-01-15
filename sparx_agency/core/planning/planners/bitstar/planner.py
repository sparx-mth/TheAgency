"""
BIT* (Batch Informed Trees) 3D planner.

Usage:
    from planners.bitstar import BITStarPlanner, PlanRequest3D

    planner = BITStarPlanner()
    result = planner.plan(request, voxelmap)
"""

from dataclasses import dataclass, field

from sparx_agency.core.common.types import PlanResult
from sparx_agency.core.planning.interfaces.planner import PlanRequest3D

from .params import BITStarParams
from .algorithm import plan_bitstar_3d


@dataclass
class BITStarPlanner:
    """
    BIT* (Batch Informed Trees) 3D planner.

    Best for:
    - Complex 3D environments
    - When you need fast convergence to optimal path
    - High-dimensional spaces

    Example:
        planner = BITStarPlanner()
        request = PlanRequest3D(start=Pose3D(0,0,1), goal=Pose3D(10,10,2))
        result = planner.plan(request, voxelmap)
    """
    name: str = field(default="bitstar_3d", init=False)
    params: BITStarParams = field(default_factory=BITStarParams)

    def plan(self, request: PlanRequest3D, world) -> PlanResult:
        return plan_bitstar_3d(
            start=request.start,
            goal=request.goal,
            voxelmap=world,
            params=self.params,
        )
