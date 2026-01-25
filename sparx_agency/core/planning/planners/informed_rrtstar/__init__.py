"""
Informed RRT* 3D path planning.

Informed RRT* is excellent for 3D environments because it:
- Samples in an ellipsoidal subset once initial solution found
- Focuses search toward the goal using heuristics
- Converges faster than standard RRT* in higher dimensions

Quick Start:
    from planners.informed_rrtstar import InformedRRTStarPlanner, PlanRequest3D

    planner = InformedRRTStarPlanner()
    request = PlanRequest3D(
        start=Pose3D(0, 0, 1),
        goal=Pose3D(10, 10, 2)
    )
    result = planner.plan(request, your_voxelmap)
"""

from .params import InformedRRTStarParams
from .planner import InformedRRTStarPlanner
from .algorithm import plan_informed_rrtstar_3d

# Re-export interface type from sparx_agency
from sparx_agency.core.planning.interfaces.planner import PlanRequest3D

__all__ = [
    # Parameters
    "InformedRRTStarParams",
    # Planner
    "InformedRRTStarPlanner",
    # Request type (from sparx_agency)
    "PlanRequest3D",
    # Function
    "plan_informed_rrtstar_3d",
]
