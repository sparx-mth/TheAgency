"""Planner interface definition."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from sparx_agency.core.common.types import Pose2D, PlanResult


@dataclass(frozen=True)
class PlanRequest:
    """
    Input to path planners.

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


@runtime_checkable
class BasePlanner(Protocol):
    """
    Protocol for path planners.

    Implementations receive a start/goal request and environment,
    returning a geometric path (no time parameterization).
    """

    name: str

    def plan(self, request: PlanRequest, world: Any) -> PlanResult:
        """
        Compute a geometric path from start to goal.

        Args:
            request: Start and goal poses.
            world: Environment representation (typically Costmap2D).

        Returns:
            PlanResult with status, optional Path2D, and debug info.
        """
        ...