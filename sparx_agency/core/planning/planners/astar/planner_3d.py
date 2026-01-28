from __future__ import annotations

from dataclasses import dataclass, field

from sparx_agency.core.common.types import Pose3D, Path3D, PlanResult, PlanStatus
from sparx_agency.core.planning.interfaces.planner import PlanRequest3D, BasePlanner3D
from sparx_agency.core.planning.environment.voxelmap3d import VoxelMap3D

from .params import AStar3DParams
from .algorithm_3d import astar_voxel_3d


@dataclass
class AStarVoxelPlanner3D:
    """
    A* planner on 3D voxel grids (VoxelMap3D).

    - Plans in voxel INDEX space, outputs Path3D in world space.
    - Uses voxelmap.world_to_grid + voxelmap.is_free(i,j,k).
    """
    name: str = field(default="astar_voxel_3d", init=False)
    params: AStar3DParams = field(default_factory=AStar3DParams)

    def plan(self, request: PlanRequest3D, world: VoxelMap3D) -> PlanResult:
        start_w = request.start
        goal_w = request.goal

        si, sj, sk = world.world_to_grid(start_w.x, start_w.y, start_w.z)
        gi, gj, gk = world.world_to_grid(goal_w.x, goal_w.y, goal_w.z)

        res = astar_voxel_3d(
            world,
            (si, sj, sk),
            (gi, gj, gk),
            allow_unknown=self.params.allow_unknown,
            connectivity=self.params.connectivity,
            max_expansions=self.params.max_expansions,
        )

        if not res.ok:
            return PlanResult(status=PlanStatus.NO_PATH, message="Voxel A* failed (no path)")

        # Convert voxel path to world poses using voxel centers
        pts = [start_w]
        for i, j, k in res.path:
            x = (i + 0.5) * world.resolution + world.origin_x
            y = (j + 0.5) * world.resolution + world.origin_y
            z = (k + 0.5) * world.resolution + world.origin_z
            pts.append(Pose3D(float(x), float(y), float(z), 0.0))
        # Ensure final goal appended if not close
        if pts[-1].distance_to(goal_w) > world.resolution * 0.75:
            pts.append(goal_w)

        path3d = Path3D(
            points=tuple(pts),
            frame_id=getattr(world, "frame_id", "map"),
            metadata={"planner": "astar_voxel_3d", "expanded": res.expanded, "connectivity": self.params.connectivity},
        )

        # For 3D planners in your codebase: you sometimes put Path3D into artifacts.
        # But PlanResult already supports Path3D directly now, so we return it as path.
        return PlanResult(
            status=PlanStatus.SUCCESS,
            path=path3d,
            message=f"Voxel A* success, expanded={res.expanded}, points={len(pts)}",
        )
