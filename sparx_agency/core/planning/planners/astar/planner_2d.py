from __future__ import annotations

from dataclasses import dataclass, field

from sparx_agency.core.common.types import Pose2D, Path2D, PlanResult, PlanStatus
from sparx_agency.core.planning.interfaces.planner import PlanRequest, BasePlanner
from sparx_agency.core.planning.environment import OccupancyGrid2D

from .params import AStarParams
from .algorithm_2d import astar_grid_2d


@dataclass
class AStarGridPlanner2D:
    """
    A* planner on a 2D OccupancyGrid2D (supports UNKNOWN).

    - Plans in GRID space, outputs in WORLD as Path2D.
    - UNKNOWN is treated as blocked by default (allow_unknown=False).
    """
    name: str = field(default="astar_grid_2d", init=False)
    params: AStarParams = field(default_factory=AStarParams)

    def plan(self, request: PlanRequest, world: OccupancyGrid2D) -> PlanResult:
        start_w = request.start
        goal_w = request.goal

        sx, sy = world.world_to_grid(start_w.x, start_w.y)
        gx, gy = world.world_to_grid(goal_w.x, goal_w.y)

        res = astar_grid_2d(
            world,
            (sx, sy),
            (gx, gy),
            allow_unknown=self.params.allow_unknown,
            connectivity=self.params.connectivity,
            max_expansions=self.params.max_expansions,
        )

        if not res.ok:
            return PlanResult(status=PlanStatus.NO_PATH, message="A* failed (no path)")

        # Convert grid path to world poses (yaw ignored)
        pts = [start_w]
        for cx, cy in res.path:
            wx, wy = world.grid_to_world(cx, cy)
            pts.append(Pose2D(wx, wy, 0.0))
        if pts[-1].distance_to(goal_w) > 1e-3:
            pts.append(goal_w)

        return PlanResult(
            status=PlanStatus.SUCCESS,
            path=Path2D(
                points=tuple(pts),
                frame_id=world.frame_id,
                metadata={"planner": "astar_grid_2d", "expanded": res.expanded, "connectivity": self.params.connectivity},
            ),
            message=f"A* success, expanded={res.expanded}, points={len(pts)}",
        )
