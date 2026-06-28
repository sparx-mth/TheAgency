"""
Local A* window planner for 3D voxel maps.

This is a LocalPlanner wrapper that:
- selects a local goal on the reference
- runs windowed voxel A* (reusing global A*)
- returns LocalPlanOutput with the resulting Path3D in artifacts

Trajectory conversion is intentionally deferred to the integration stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sparx_agency.core.planning.environment.voxelmap3d import VoxelMap3D
from sparx_agency.core.planning.local_planners.interfaces.local_planner import LocalPlanner
from sparx_agency.core.planning.local_planners.interfaces.types import (
    LocalFailureReason,
    LocalPlanInput,
    LocalPlanOutput,
    LocalPlanStatus,
)
from sparx_agency.core.common.types.geometry import Pose3D
from sparx_agency.core.common.types.planning import Path2D, Path3D, Trajectory

from .algorithm import plan_window_astar_3d
from .params import LocalAStarWindow3DParams
from .reference_utils import extract_reference_points_2d, select_goal_on_reference_2d


@dataclass
class LocalAStarWindowPlanner3D(LocalPlanner):
    """
    Local replanner using voxel A* inside a small 3D window.

    Notes:
    - Requires `inp.metadata["voxelmap3d"]` to be a VoxelMap3D instance.
    - Supports reference as Path3D/Trajectory. If reference is Path2D, we "lift" the goal
      to the drone's current z (simple and robust for the first iteration).
    """
    params: LocalAStarWindow3DParams = field(default_factory=LocalAStarWindow3DParams)

    def plan(self, inp: LocalPlanInput) -> LocalPlanOutput:
        vox = inp.metadata.get("voxelmap3d", None)
        if not isinstance(vox, VoxelMap3D):
            return LocalPlanOutput(
                status=LocalPlanStatus.ERROR,
                reason=LocalFailureReason.INTERNAL_ERROR,
                message='LocalAStarWindowPlanner3D requires metadata["voxelmap3d"]=VoxelMap3D',
            )

        start = Pose3D(inp.state.pose.x, inp.state.pose.y, inp.state.pose.z, inp.state.pose.yaw)

        # If the reference is 2D, lift it to a 3D goal at current altitude
        ref = inp.reference
        if isinstance(ref, Path2D):
            ref2d = extract_reference_points_2d(ref, sample_dt=0.2)
            goal_xy = select_goal_on_reference_2d(
                ref2d,
                (start.x, start.y),
                lookahead_m=self.params.goal_lookahead_m,
                min_sep_m=self.params.min_goal_separation_m,
            )
            if goal_xy is None:
                return LocalPlanOutput(
                    status=LocalPlanStatus.NO_SOLUTION,
                    reason=LocalFailureReason.REFERENCE_TOO_SHORT,
                    message="Local 3D window A*: reference too short",
                )

            # Build a minimal Path3D reference: start -> lifted goal
            lifted = Pose3D(goal_xy[0], goal_xy[1], start.z, 0.0)
            ref3d = Path3D(points=(start, lifted), frame_id=getattr(vox, "frame_id", "map"))
            ref = ref3d

        win = plan_window_astar_3d(
            vox,
            reference=ref,  # Path3D or Trajectory
            start_world=start,
            params=self.params,
        )

        if win.path is None:
            return LocalPlanOutput(
                status=LocalPlanStatus.NO_SOLUTION,
                reason=LocalFailureReason.GOAL_UNREACHABLE,
                message="Local 3D window A* failed (no path)",
                artifacts=win.artifacts,
            )

        return LocalPlanOutput(
            status=LocalPlanStatus.SUCCESS,
            trajectory=None,
            message="Local 3D window A* success",
            artifacts={**win.artifacts, "path3d": win.path},
        )
