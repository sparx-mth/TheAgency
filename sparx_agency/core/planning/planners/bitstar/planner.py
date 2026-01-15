"""
3D Planners: BIT* and Informed RRT*

Usage:
    from planners import BITStarPlanner, InformedRRTStarPlanner

    # BIT* (recommended for 3D)
    planner = BITStarPlanner()
    result = planner.plan(request, voxelmap)

    # Informed RRT*
    planner = InformedRRTStarPlanner()
    result = planner.plan(request, voxelmap)
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from .params import BITStarParams, InformedRRTStarParams
from .algorithm import plan_bitstar_3d, plan_informed_rrtstar_3d, Pose3D, PlanResult


@dataclass(frozen=True)
class PlanRequest3D:
    """3D planning request."""
    start: Pose3D
    goal: Pose3D
    frame_id: str = "map"
    options: Dict[str, Any] = field(default_factory=dict)


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