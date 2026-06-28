"""Planner interface definitions (2D and 3D)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from sparx_agency.core.common.types import Pose2D, Pose3D, PlanResult



@dataclass(frozen=True)
class PlanRequest:
    """
    Input to 2D path planners.

    Attributes:
        start: Initial robot pose in world frame.
        goal: Target pose in world frame.
        frame_id: Coordinate frame identifier.
        options: Algorithm-specific options (e.g., timeout override).
    """
    start: Pose2D
    goal: Pose2D
    frame_id: str = "map"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanRequest3D:
    """
    Input to 3D path planners.

    Attributes:
        start: Initial pose in world frame (x, y, z).
        goal: Target pose in world frame (x, y, z).
        frame_id: Coordinate frame identifier.
        options: Algorithm-specific options (e.g., timeout override).
    """
    start: Pose3D
    goal: Pose3D
    frame_id: str = "map"
    options: Dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class BasePlanner(Protocol):
    """
    Protocol for 2D path planners.

    Implementations receive a start/goal request and environment,
    returning a geometric path (no time parameterization).
    """

    name: str

    def plan(self, request: PlanRequest, world: Any) -> PlanResult:
        """
        Compute a geometric path from start to goal.

        Args:
            request: Start and goal poses (2D).
            world: Environment representation (typically Costmap2D).

        Returns:
            PlanResult with status, optional Path2D, and debug info.
        """
        ...


@runtime_checkable
class BasePlanner3D(Protocol):
    """
    Protocol for 3D path planners.

    Implementations receive a start/goal request and 3D environment,
    returning a geometric path (no time parameterization).
    """

    name: str

    def plan(self, request: PlanRequest3D, world: Any) -> PlanResult:
        """
        Compute a geometric path from start to goal in 3D space.

        Args:
            request: Start and goal poses (3D).
            world: Environment representation (typically Voxelmap).

        Returns:
            PlanResult with status, optional path in artifacts["path3d"], and debug info.
        """
        ...