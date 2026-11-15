"""
Path Planning Algorithms

Available planners:
- RRT* (2D and 3D): Standard RRT* with extensive debug support
- BIT* (3D): Batch Informed Trees - faster convergence
- Informed RRT* (3D): Ellipsoidal sampling after first solution

Quick Start:
    # RRT* 2D
    from planners import RRTStarOmplPlanner, PlanRequest
    planner = RRTStarOmplPlanner()
    result = planner.plan(request, costmap)

    # RRT* 3D
    from planners import RRTStarOmpl3DPlanner, PlanRequest3D
    planner = RRTStarOmpl3DPlanner()
    result = planner.plan(request, voxelmap)

    # BIT* 3D (recommended for complex 3D)
    from planners import BITStarPlanner
    planner = BITStarPlanner()
    result = planner.plan(request, voxelmap)

    # Informed RRT* 3D
    from planners import InformedRRTStarPlanner
    planner = InformedRRTStarPlanner()
    result = planner.plan(request, voxelmap)
"""

# Interface types (from sparx_agency - single source of truth)
from sparx_agency.core.planning.interfaces.planner import (
    PlanRequest,
    PlanRequest3D,
    BasePlanner,
    BasePlanner3D,
)

from typing import TYPE_CHECKING

# Importing any child package executes this module. Pure A*/ObjectNav must
# not load optional native sampling planners merely to import grid geometry.
if TYPE_CHECKING:
    from .rrtstar import (
        RRTStarOmplParams, RRTStarOmpl3DParams, RRTStarOmplPlanner,
        RRTStarOmpl3DPlanner, plan_rrtstar, plan_rrtstar_3d,
    )
    from .bitstar import BITStarParams, BITStarPlanner, plan_bitstar_3d
    from .informed_rrtstar import (
        InformedRRTStarParams, InformedRRTStarPlanner, plan_informed_rrtstar_3d,
    )

_OPTIONAL_EXPORTS = {
    name: module
    for module, names in (
        ("rrtstar", ("RRTStarOmplParams", "RRTStarOmpl3DParams", "RRTStarOmplPlanner",
                     "RRTStarOmpl3DPlanner", "plan_rrtstar", "plan_rrtstar_3d")),
        ("bitstar", ("BITStarParams", "BITStarPlanner", "plan_bitstar_3d")),
        ("informed_rrtstar", ("InformedRRTStarParams", "InformedRRTStarPlanner",
                              "plan_informed_rrtstar_3d")),
    )
    for name in names
}


def __getattr__(name):
    """Resolve existing sampling-planner exports only when explicitly requested."""
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    from importlib import import_module
    value = getattr(import_module("." + _OPTIONAL_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value

from .astar import (
    AStarParams,
    AStar3DParams,
    WeightedAStarParams,
    AStarGridPlanner2D,
    AStarVoxelPlanner3D,
    WeightedAStarPlanner2D,
    build_cost_grid,
)


__all__ = [
    # === Interface types (from sparx_agency) ===
    "PlanRequest",
    "PlanRequest3D",
    "BasePlanner",
    "BasePlanner3D",
    # === RRT* (2D) ===
    "RRTStarOmplParams",
    "RRTStarOmplPlanner",
    "plan_rrtstar",
    # === RRT* (3D) ===
    "RRTStarOmpl3DParams",
    "RRTStarOmpl3DPlanner",
    "plan_rrtstar_3d",
    # === BIT* (3D) ===
    "BITStarParams",
    "BITStarPlanner",
    "plan_bitstar_3d",
    # === Informed RRT* (3D) ===
    "InformedRRTStarParams",
    "InformedRRTStarPlanner",
    "plan_informed_rrtstar_3d",
    # === A* (3D & 2D) ===
    "AStarParams",
    "AStar3DParams",
    "WeightedAStarParams",
    "AStarGridPlanner2D",
    "AStarVoxelPlanner3D",
    "WeightedAStarPlanner2D",
    "build_cost_grid",
]
