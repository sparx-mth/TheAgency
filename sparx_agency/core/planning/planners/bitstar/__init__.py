"""
BIT* (Batch Informed Trees) 3D path planning.

BIT* is excellent for 3D environments because it:
- Uses batch processing for faster convergence
- Focuses search in promising regions using heuristics
- Provides asymptotic optimality with better performance than RRT*

Quick Start:
    from planners.bitstar import BITStarPlanner, PlanRequest3D

    planner = BITStarPlanner()
    request = PlanRequest3D(
        start=Pose3D(0, 0, 1),
        goal=Pose3D(10, 10, 2)
    )
    result = planner.plan(request, your_voxelmap)
"""

from .params import BITStarParams
from .planner import BITStarPlanner
from .algorithm import plan_bitstar_3d

# Re-export interface type from sparx_agency
from sparx_agency.core.planning.interfaces.planner import PlanRequest3D

__all__ = [
    # Parameters
    "BITStarParams",
    # Planner
    "BITStarPlanner",
    # Request type (from sparx_agency)
    "PlanRequest3D",
    # Function
    "plan_bitstar_3d",
]
