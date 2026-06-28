"""
Informed RRT* 3D planner.

Usage:
    from planners.informed_rrtstar import InformedRRTStarPlanner, PlanRequest3D

    planner = InformedRRTStarPlanner()
    result = planner.plan(request, voxelmap)
"""

from dataclasses import dataclass, field

from sparx_agency.core.common.types import PlanResult
from sparx_agency.core.planning.interfaces.planner import PlanRequest3D

from .params import InformedRRTStarParams
from .algorithm import plan_informed_rrtstar_3d


@dataclass
class InformedRRTStarPlanner:
    """
    Informed RRT* 3D planner.

    Best for:
    - 3D environments with clear paths
    - When you want good solutions quickly
    - Simpler than BIT* but still informed

    Example:
        planner = InformedRRTStarPlanner()
        request = PlanRequest3D(start=Pose3D(0,0,1), goal=Pose3D(10,10,2))
        result = planner.plan(request, voxelmap)
    """
    name: str = field(default="informed_rrtstar_3d", init=False)
    params: InformedRRTStarParams = field(default_factory=InformedRRTStarParams)

    def plan(self, request: PlanRequest3D, world) -> PlanResult:
        return plan_informed_rrtstar_3d(
            start=request.start,
            goal=request.goal,
            voxelmap=world,
            params=self.params,
        )
