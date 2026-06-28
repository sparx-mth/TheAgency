"""
Local A* window planner for 2D occupancy grids.

This is a LocalPlanner wrapper that:
- selects a local goal on the reference
- runs windowed A* (reusing global A*)
- returns LocalPlanOutput with the resulting Path2D in artifacts

Trajectory conversion is intentionally deferred to the integration stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.local_planners.interfaces.local_planner import LocalPlanner
from sparx_agency.core.planning.local_planners.interfaces.types import (
    LocalFailureReason,
    LocalPlanInput,
    LocalPlanOutput,
    LocalPlanStatus,
)
from sparx_agency.core.common.types.geometry import Pose2D

from .algorithm import plan_window_astar_2d
from .params import LocalAStarWindow2DParams


@dataclass
class LocalAStarWindowPlanner2D(LocalPlanner):
    """
    Local replanner using A* inside a small 2D window.

    Notes:
    - Requires `inp.metadata["grid2d"]` to be an OccupancyGrid2D instance.
    - Keeps the planner logic independent of your safety map dispatchers for now.
      Later, the integration layer can provide the correct local map object here.
    """
    params: LocalAStarWindow2DParams = field(default_factory=LocalAStarWindow2DParams)

    def plan(self, inp: LocalPlanInput) -> LocalPlanOutput:
        grid = inp.metadata.get("grid2d", None)
        if not isinstance(grid, OccupancyGrid2D):
            return LocalPlanOutput(
                status=LocalPlanStatus.ERROR,
                reason=LocalFailureReason.INTERNAL_ERROR,
                message='LocalAStarWindowPlanner2D requires metadata["grid2d"]=OccupancyGrid2D',
            )

        # 2D start pose projected from State3D
        start = Pose2D(inp.state.pose.x, inp.state.pose.y, inp.state.pose.yaw)

        win = plan_window_astar_2d(
            grid,
            reference=inp.reference,  # Path2D or Trajectory
            start_world=start,
            params=self.params,
        )

        if win.path is None:
            return LocalPlanOutput(
                status=LocalPlanStatus.NO_SOLUTION,
                reason=LocalFailureReason.GOAL_UNREACHABLE,
                message="Local 2D window A* failed (no path)",
                artifacts=win.artifacts,
            )

        # We return the local path in artifacts; trajectory conversion is later.
        return LocalPlanOutput(
            status=LocalPlanStatus.SUCCESS,
            trajectory=None,
            message="Local 2D window A* success",
            artifacts={**win.artifacts, "path2d": win.path},
        )
