from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AStarParams:
    """
    Parameters for A* planning on grids.

    Attributes:
        allow_unknown: If True, planner may traverse UNKNOWN cells.
                       Default False (classic exploration planning behavior).
        connectivity: 4 or 8 for 2D. (3D uses 6/18/26 separately in AStar3DParams)
        max_expansions: Safety cap; None means unlimited.
    """
    allow_unknown: bool = False
    connectivity: int = 4
    max_expansions: int | None = 200_000


@dataclass(frozen=True, slots=True)
class AStar3DParams:
    """
    Parameters for voxel-grid A* (3D).

    Attributes:
        allow_unknown: If True, planner may traverse UNKNOWN voxels (if map supports it).
        connectivity: 6, 18, or 26.
        max_expansions: Safety cap; None means unlimited.
    """
    allow_unknown: bool = False
    connectivity: int = 6
    max_expansions: int | None = 800_000
