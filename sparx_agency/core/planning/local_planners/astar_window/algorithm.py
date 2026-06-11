"""
Windowed local replanning workflow.

This module:
- builds a small planning window around the current position
- chooses a local goal ahead on the reference
- runs the existing A* search on a window view
- returns a short Path2D/Path3D (conversion to trajectory is handled later)

No A* implementation lives here; we reuse the global A* functions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import ceil
from typing import Dict, Optional, Tuple

from sparx_agency.core.common.types.geometry import Pose2D, Pose3D
from sparx_agency.core.common.types.planning import Path2D, Path3D, Trajectory
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.environment.voxelmap3d import VoxelMap3D
from sparx_agency.core.planning.planners.astar.algorithm_2d import astar_grid_2d
from sparx_agency.core.planning.planners.astar.algorithm_3d import astar_voxel_3d

from .params import LocalAStarWindow2DParams, LocalAStarWindow3DParams
from .reference_utils import (
    extract_reference_points_2d,
    extract_reference_points_3d,
    select_goal_on_reference_2d,
    select_goal_on_reference_3d,
)
from .views import WindowGridView2D, WindowVoxelView3D


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class WindowPlan2D:
    path: Optional[Path2D]
    artifacts: Dict


@dataclass(frozen=True)
class WindowPlan3D:
    path: Optional[Path3D]
    artifacts: Dict


def plan_window_astar_2d(
    grid: OccupancyGrid2D,
    *,
    reference: Path2D | Trajectory,
    start_world: Pose2D,
    params: LocalAStarWindow2DParams,
) -> WindowPlan2D:
    t0 = time.perf_counter()

    # --- choose local goal in world ---
    ref_pts = extract_reference_points_2d(reference, sample_dt=0.2)
    goal_xy = select_goal_on_reference_2d(
        ref_pts,
        (start_world.x, start_world.y),
        lookahead_m=params.goal_lookahead_m,
        min_sep_m=params.min_goal_separation_m,
    )
    if goal_xy is None:
        return WindowPlan2D(
            path=None,
            artifacts={"reason": "reference_too_short", "runtime_ms": (time.perf_counter() - t0) * 1000.0},
        )

    # --- build window in grid indices ---
    # assumes OccupancyGrid2D exposes resolution
    res = float(getattr(grid, "resolution", 1.0))
    half_m = params.window_size_m * 0.5
    rad = max(1, int(ceil(half_m / res)))

    sx, sy = grid.world_to_grid(start_world.x, start_world.y)
    gx, gy = grid.world_to_grid(goal_xy[0], goal_xy[1])

    # window bounds in global grid indices
    x0 = sx - rad
    y0 = sy - rad
    w = 2 * rad + 1
    h = 2 * rad + 1

    view = WindowGridView2D(base=grid, x0=x0, y0=y0, w=w, h=h)

    # convert start/goal to local indices and clamp into window
    lsx, lsy = view.global_to_local(sx, sy)
    lgx, lgy = view.global_to_local(gx, gy)
    lsx = _clamp(lsx, 0, w - 1)
    lsy = _clamp(lsy, 0, h - 1)
    lgx = _clamp(lgx, 0, w - 1)
    lgy = _clamp(lgy, 0, h - 1)

    res_astar = astar_grid_2d(
        view,
        (lsx, lsy),
        (lgx, lgy),
        allow_unknown=params.allow_unknown,
        connectivity=params.connectivity,
        max_expansions=params.max_expansions,
    )

    artifacts = {
        "expanded": getattr(res_astar, "expanded", None),
        "window": {"x0": x0, "y0": y0, "w": w, "h": h, "resolution": res},
        "start_grid": (sx, sy),
        "goal_grid": (gx, gy),
        "goal_world": (goal_xy[0], goal_xy[1]),
    }

    if not res_astar.ok:
        artifacts["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
        return WindowPlan2D(path=None, artifacts=artifacts)

    # reconstruct as world poses
    pts = [start_world]
    for lx, ly in res_astar.path:
        gx2, gy2 = view.local_to_global(lx, ly)
        wx, wy = grid.grid_to_world(gx2, gy2)
        pts.append(Pose2D(float(wx), float(wy), 0.0))

    # ensure we end at chosen local goal (world)
    if pts[-1].distance_to(Pose2D(goal_xy[0], goal_xy[1], 0.0)) > res * 0.75:
        pts.append(Pose2D(goal_xy[0], goal_xy[1], 0.0))

    path = Path2D(points=tuple(pts), frame_id=getattr(grid, "frame_id", "map"), metadata={"planner": "local_astar_window_2d"})
    artifacts["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
    return WindowPlan2D(path=path, artifacts=artifacts)


def plan_window_astar_3d(
    voxelmap: VoxelMap3D,
    *,
    reference: Path3D | Trajectory,
    start_world: Pose3D,
    params: LocalAStarWindow3DParams,
) -> WindowPlan3D:
    t0 = time.perf_counter()

    ref_pts = extract_reference_points_3d(reference, sample_dt=0.2)
    goal_xyz = select_goal_on_reference_3d(
        ref_pts,
        (start_world.x, start_world.y, start_world.z),
        lookahead_m=params.goal_lookahead_m,
        min_sep_m=params.min_goal_separation_m,
    )
    if goal_xyz is None:
        return WindowPlan3D(
            path=None,
            artifacts={"reason": "reference_too_short", "runtime_ms": (time.perf_counter() - t0) * 1000.0},
        )

    res = float(getattr(voxelmap, "resolution", 1.0))
    half_xy = params.window_size_xy_m * 0.5
    half_z = params.window_size_z_m * 0.5
    rad_xy = max(1, int(ceil(half_xy / res)))
    rad_z = max(1, int(ceil(half_z / res)))

    si, sj, sk = voxelmap.world_to_grid(start_world.x, start_world.y, start_world.z)
    gi, gj, gk = voxelmap.world_to_grid(goal_xyz[0], goal_xyz[1], goal_xyz[2])

    i0 = si - rad_xy
    j0 = sj - rad_xy
    k0 = sk - rad_z

    W = 2 * rad_xy + 1
    H = 2 * rad_xy + 1
    D = 2 * rad_z + 1

    view = WindowVoxelView3D(base=voxelmap, i0=i0, j0=j0, k0=k0, width=W, height=H, depth=D)

    lsi, lsj, lsk = view.global_to_local(si, sj, sk)
    lgi, lgj, lgk = view.global_to_local(gi, gj, gk)

    lsi = _clamp(lsi, 0, W - 1)
    lsj = _clamp(lsj, 0, H - 1)
    lsk = _clamp(lsk, 0, D - 1)
    lgi = _clamp(lgi, 0, W - 1)
    lgj = _clamp(lgj, 0, H - 1)
    lgk = _clamp(lgk, 0, D - 1)

    res_astar = astar_voxel_3d(
        view,
        (lsi, lsj, lsk),
        (lgi, lgj, lgk),
        allow_unknown=params.allow_unknown,
        connectivity=params.connectivity,
        max_expansions=params.max_expansions,
    )

    artifacts = {
        "expanded": getattr(res_astar, "expanded", None),
        "window": {"i0": i0, "j0": j0, "k0": k0, "W": W, "H": H, "D": D, "resolution": res},
        "start_grid": (si, sj, sk),
        "goal_grid": (gi, gj, gk),
        "goal_world": (goal_xyz[0], goal_xyz[1], goal_xyz[2]),
    }

    if not res_astar.ok:
        artifacts["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
        return WindowPlan3D(path=None, artifacts=artifacts)

    pts = [start_world]
    for li, lj, lk in res_astar.path:
        gi2, gj2, gk2 = view.local_to_global(li, lj, lk)
        x = (gi2 + 0.5) * voxelmap.resolution + voxelmap.origin_x
        y = (gj2 + 0.5) * voxelmap.resolution + voxelmap.origin_y
        z = (gk2 + 0.5) * voxelmap.resolution + voxelmap.origin_z
        pts.append(Pose3D(float(x), float(y), float(z), 0.0))

    goal_pose = Pose3D(float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2]), 0.0)
    if pts[-1].distance_to(goal_pose) > res * 0.75:
        pts.append(goal_pose)

    path = Path3D(points=tuple(pts), frame_id=getattr(voxelmap, "frame_id", "map"), metadata={"planner": "local_astar_window_3d"})
    artifacts["runtime_ms"] = (time.perf_counter() - t0) * 1000.0
    return WindowPlan3D(path=path, artifacts=artifacts)
