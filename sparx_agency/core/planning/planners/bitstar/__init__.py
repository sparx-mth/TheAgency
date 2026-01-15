"""
3D Path Planning: BIT* and Informed RRT*

Both algorithms are superior to standard RRT* for 3D planning.

Quick Start:
    from rrtstar import BITStarPlanner, InformedRRTStarPlanner, PlanRequest3D, Pose3D

    # Use BIT* (recommended)
    planner = BITStarPlanner()
    request = PlanRequest3D(
        start=Pose3D(0, 0, 1),
        goal=Pose3D(10, 10, 2)
    )
    result = planner.plan(request, your_voxelmap)

    if result.status == "success":
        path = result.artifacts["path3d"]
        for point in path.points:
            print(f"  ({point.x}, {point.y}, {point.z})")
"""

from .params import (
    BITStarParams,
    InformedRRTStarParams,
    RRTStarOmplParams,
    RRTStarOmpl3DParams,
)
from .planner import (
    BITStarPlanner,
    InformedRRTStarPlanner,
    PlanRequest3D,
)
from .algorithm import (
    plan_bitstar_3d,
    plan_informed_rrtstar_3d,
    Pose3D,
    Path3D,
    PlanResult,
    PlanStatus,
)

__all__ = [
    # Planners
    "BITStarPlanner",
    "InformedRRTStarPlanner",
    # Parameters
    "BITStarParams",
    "InformedRRTStarParams",
    "RRTStarOmplParams",
    "RRTStarOmpl3DParams",
    # Types
    "PlanRequest3D",
    "Pose3D",
    "Path3D",
    "PlanResult",
    "PlanStatus",
    # Functions
    "plan_bitstar_3d",
    "plan_informed_rrtstar_3d",
]