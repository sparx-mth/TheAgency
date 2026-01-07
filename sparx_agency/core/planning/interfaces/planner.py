"""
Planner interface.

A Planner receives a start/goal and environment context (e.g., Costmap2D)
and returns a PlanResult containing a geometric path (Path2D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from core.common.types import Pose2D, PlanResult


@dataclass(frozen=True)
class PlanRequest:
    """
    Inputs to the planner.

    Notes:
    - Keep this minimal and stable; planners may accept extra options via `options`.
    - `frame_id` is metadata only (no ROS dependency).
    """
    start: Pose2D
    goal: Pose2D
    frame_id: str = "map"
    options: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BasePlanner(Protocol):
    """
    Planner contract.

    `world` is intentionally typed as Any here to avoid import cycles and to
    keep planning interfaces decoupled from a specific environment type.
    In your implementation, it will typically be Costmap2D or similar.
    """

    name: str

    def plan(self, request: PlanRequest, world: Any) -> PlanResult:
        """
        Compute a geometric path from start to goal.

        Returns:
            PlanResult with status + optional Path2D and debug artifacts.
        """
        ...
