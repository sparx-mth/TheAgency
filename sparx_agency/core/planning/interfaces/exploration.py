"""
Exploration interfaces.

This is intentionally separate from path-planning interfaces:
- Path planners (BasePlanner/BasePlanner3D) produce Path2D/Path3D.
- Exploration policies choose WHERE to explore next (goal selection) and may
  optionally propose an immediate path.

No ROS. No execution/tracking. Pure decision-making interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Set

from sparx_agency.core.common.types import Pose2D, Path2D


@dataclass(frozen=True)
class ExplorationContext:
    """
    Inputs provided to exploration policies each tick.

    Attributes:
        robot_id: Unique agent identifier.
        pose: Current robot pose (2D world).
        map_frame_id: Frame identifier (usually "map").
        frontiers: Optional frontier set in world frame (Pose2D points with yaw ignored).
        assigned_frontiers: Frontiers already assigned to other robots (world positions).
        options: Free-form extra inputs (camera range, FOV, etc).
    """
    robot_id: int
    pose: Pose2D
    map_frame_id: str = "map"
    frontiers: Set[Pose2D] = field(default_factory=set)
    assigned_frontiers: Set[Pose2D] = field(default_factory=set)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplorationDecision:
    """
    Output of an exploration policy step.

    Attributes:
        goal: Next exploration goal (world pose). yaw may be ignored.
        path: Optional proposed path to that goal (world).
        info: Debug / metadata.
    """
    goal: Optional[Pose2D] = None
    path: Optional[Path2D] = None
    info: Dict[str, Any] = field(default_factory=dict)


class ExplorationPolicy(Protocol):
    """
    Protocol for exploration policies.

    Implementations select a goal (frontier or other) and may propose a path.
    """
    name: str

    def step(self, ctx: ExplorationContext, world: Any) -> ExplorationDecision:
        """
        Args:
            ctx: Current exploration context (pose + frontier candidates).
            world: Map/environment representation (e.g., OccupancyGrid2D).

        Returns:
            ExplorationDecision with optional goal/path.
        """
        ...
