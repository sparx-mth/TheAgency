# =========================
# File: sparx_agency/core/planning/planners/rrtstar/planner.py
# =========================
"""
RRT* planners implementing BasePlanner protocol (2D and 3D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from sparx_agency.core.common.types import Pose2D, Pose3D, PlanResult
from sparx_agency.core.planning.environment import Costmap2D

from .params import RRTStarOmplParams, RRTStarOmpl3DParams
from .algorithm import plan_rrtstar, plan_rrtstar_3d


@dataclass(frozen=True)
class PlanRequest:
    """2D planning request."""
    start: Pose2D
    goal: Pose2D
    frame_id: str = "map"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanRequest3D:
    """3D planning request."""
    start: Pose3D
    goal: Pose3D
    frame_id: str = "map"
    options: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BasePlanner(Protocol):
    name: str
    def plan(self, request: PlanRequest, world: Any) -> PlanResult: ...


@runtime_checkable
class BasePlanner3D(Protocol):
    name: str
    def plan(self, request: PlanRequest3D, world: Any) -> PlanResult: ...


@dataclass
class RRTStarOmplPlanner:
    """2D RRT* planner."""
    name: str = field(default="rrtstar_ompl", init=False)
    params: RRTStarOmplParams = field(default_factory=RRTStarOmplParams)

    def plan(self, request: PlanRequest, world: Costmap2D) -> PlanResult:
        return plan_rrtstar(start=request.start, goal=request.goal, costmap=world, params=self.params)


@dataclass
class RRTStarOmpl3DPlanner:
    """3D RRT* planner."""
    name: str = field(default="rrtstar_ompl_3d", init=False)
    params: RRTStarOmpl3DParams = field(default_factory=RRTStarOmpl3DParams)

    def plan(self, request: PlanRequest3D, world) -> PlanResult:
        return plan_rrtstar_3d(start=request.start, goal=request.goal, voxelmap=world, params=self.params)
