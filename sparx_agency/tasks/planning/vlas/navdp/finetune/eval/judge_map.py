"""The independent safety judge: a fused, multi-frame clearance field.

The fine-tune teacher raised trajectory clearance against the *single-frame* ESDF
built by ``common/frames.py``. Scoring the result against that same field would
be circular -- the model is judged with the ruler it was trained on. This module
builds a different, strictly better-informed map instead:

* it fuses a **window of posed depth frames** (via ``core.mapping.costmap``'s
  log-odds grid and raytracing), so occlusions in the single frame that the
  teacher optimized against are filled in by neighbouring viewpoints;
* it keeps an **observed mask**, so a waypoint flying through never-seen space is
  flagged rather than silently scored as wide-open;
* it uses raycast free-space, which the teacher's grid does not do at all
  (``cloud_to_occupancy_grid`` stamps hits and leaves everything else ``free``).

Poses come from :mod:`bag_poses`. Everything is done in the AprilTag world frame,
where the floor sits at ``z == 0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from sparx_agency.core.mapping.costmap.depth_to_grid import update_grid_from_depth
from sparx_agency.core.mapping.costmap.distance_field import compute_clearance_field
from sparx_agency.core.mapping.costmap.log_odds_grid import (
    LogOddsGridConfig,
    LogOddsGridCostmap,
)


@dataclass(frozen=True)
class JudgeMapConfig:
    """Settings for the fused judge map.

    Attributes:
        resolution_m: cell size; matches the teacher's 10 cm so the two fields
            are directly comparable.
        size_m: square world extent centred on the world origin.
        window: number of frames fused on EACH side of the scored frame. The
            scored frame is always included, so ``window=3`` fuses up to 7.
        z_band: world height band kept as obstacle, in meters above the floor.
        depth_range_m: valid depth interval.
        downsample: pixel stride when backprojecting.
        occ_prob: probability above which a fused cell counts as occupied.
    """

    resolution_m: float = 0.10
    size_m: float = 40.0
    window: int = 3
    z_band: Tuple[float, float] = (0.15, 2.0)
    depth_range_m: Tuple[float, float] = (0.2, 5.0)
    downsample: int = 2
    occ_prob: float = 0.65


@dataclass(frozen=True)
class JudgeField:
    """A world-frame clearance field plus the mask of what was actually seen.

    Attributes:
        clearance: ``(H, W)`` meters to the nearest fused obstacle.
        observed: ``(H, W)`` bool; False where no ray ever passed.
        resolution: meters per cell.
        origin_x, origin_y: world coordinate of cell ``[0, 0]``'s lower corner.
        n_frames: how many depth frames were fused into it.
    """

    clearance: np.ndarray
    observed: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    n_frames: int

    def sample(self, world_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clearance and observed-flag at world XY points.

        Args:
            world_xy: ``(N, 2)`` world coordinates.

        Returns:
            ``(clearance_m, observed)``, both ``(N,)``. Points outside the grid
            are reported as unobserved with zero clearance.
        """
        h, w = self.clearance.shape
        gx = np.floor((world_xy[:, 0] - self.origin_x) / self.resolution).astype(int)
        gy = np.floor((world_xy[:, 1] - self.origin_y) / self.resolution).astype(int)
        inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        gxc, gyc = np.clip(gx, 0, w - 1), np.clip(gy, 0, h - 1)
        clear = np.where(inside, self.clearance[gyc, gxc], 0.0)
        seen = inside & self.observed[gyc, gxc]
        return clear.astype(np.float32), seen


def yaw_of(world_T_cam: np.ndarray) -> float:
    """Heading of the camera's forward axis, projected onto the world XY plane.

    ``world_T_cam`` is in the OpenCV optical convention, so its third column is
    the viewing direction. Using only yaw (rather than the full rotation) keeps
    the body frame horizontal, matching how the occupancy grid and the NavDP
    trajectory both treat the ground plane.
    """
    fwd = world_T_cam[:3, 2]
    return float(np.arctan2(fwd[1], fwd[0]))


def body_to_world(traj_body: np.ndarray, world_T_cam: np.ndarray) -> np.ndarray:
    """Body-FLU ``(N, 2)`` ``[forward, left]`` waypoints -> world ``(N, 2)`` XY."""
    yaw = yaw_of(world_T_cam)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return traj_body[:, :2] @ rot.T + world_T_cam[:2, 3]


def build_judge_field(rec_dir: Path, K: np.ndarray, frames: np.ndarray,
                      poses: np.ndarray, centre: int,
                      config: JudgeMapConfig = JudgeMapConfig()) -> JudgeField:
    """Fuse the frames around ``centre`` into one clearance field.

    Args:
        rec_dir: recording directory holding ``depth/NNNNNN.npy``.
        K: ``(3, 3)`` camera intrinsics.
        frames: ``(M,)`` depth-frame indices that have poses.
        poses: ``(M, 4, 4)`` matching ``world_T_cam`` matrices.
        centre: position *within* ``frames`` of the frame being scored.
        config: fusion settings.

    Returns:
        The fused :class:`JudgeField`.
    """
    lo = max(0, centre - config.window)
    hi = min(len(frames), centre + config.window + 1)
    grid = LogOddsGridCostmap(LogOddsGridConfig(
        resolution_m=config.resolution_m, size_m=config.size_m))

    for i in range(lo, hi):
        depth = np.load(rec_dir / "depth" / f"{int(frames[i]):06d}.npy").astype(np.float32)
        update_grid_from_depth(
            grid, depth, K, poses[i],
            z_min_world=config.z_band[0], z_max_world=config.z_band[1],
            depth_min_m=config.depth_range_m[0], depth_max_m=config.depth_range_m[1],
            downsample=config.downsample, raytrace=True,
        )

    prob = grid.get_prob(unknown=np.nan)
    observed = np.isfinite(prob)
    occupied = observed & (prob > config.occ_prob)
    clearance = compute_clearance_field(
        occupied.astype(np.uint8), resolution=config.resolution_m)
    return JudgeField(clearance=clearance, observed=observed,
                      resolution=config.resolution_m,
                      origin_x=float(grid.origin_x), origin_y=float(grid.origin_y),
                      n_frames=hi - lo)
