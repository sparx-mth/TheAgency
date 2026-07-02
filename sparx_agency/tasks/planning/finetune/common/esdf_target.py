"""Per-frame potential-field / ESDF target-trajectory generator.

Turns a single RGB-D observation into the two artifacts the fine-tune loss needs:

1. a **signed ESDF grid** (meters, ``>0`` free, ``<0`` inside a wall) -- the fixed
   lookup the *differentiable* obstacle penalty samples at training time; and
2. a **corrected target trajectory** (``Path2D`` in the body FLU frame) -- a seed
   route (the drone's flown-future path, or a straight shot to the goal) pushed off
   the walls by the repo's live path correctors. This is the behavior-cloning
   label: "what the network *should* have output near obstacles".

Both come from the same single-frame occupancy map (see :mod:`.frames`) so the
label and the penalty are geometrically consistent. Everything here is numpy and
reuses the ROS-free core layers (``compute_sdf``, ``EsdfLayer``, the correctors),
so it runs in the plain ``.venv`` with no torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.mapping.costmap.sdf import compute_sdf, SDFParams
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D
from sparx_agency.core.planning.safety.path_correction import (
    EsdfCorrectorConfig,
    EsdfPathCorrector,
    PotentialFieldCorrectorConfig,
    PotentialFieldPathCorrector,
)

from .frames import (
    LocalMapConfig,
    cloud_to_occupancy_grid,
    depth_to_body_cloud,
    occupancy_binary,
)


@dataclass(frozen=True)
class EsdfTargetConfig:
    """Configuration for :func:`generate_target`.

    Attributes:
        local_map: Single-frame occupancy-map geometry.
        corrector: Which corrector produces the target path -- ``"potential_field"``
            (Gaussian repulsion re-centering) or ``"esdf"`` (distance-field ascent).
        target_clearance_m: Desired clearance the corrector aims for (the ESDF
            "safe distance" / PF ``min_clearance_m``). Also the natural ``margin``
            for the differentiable penalty. Defaults to 0.35 m (between the 0.25 m
            tube radius and the NavDP 0.5 m ``d_safe``).
        max_total_shift_m: Cap on how far a waypoint may be pushed off the seed.
        n_seed_points: Number of points on the straight seed path (origin -> goal)
            when no flown-future seed is supplied.
        sdf_clamp_m: Clamp the signed SDF to +/- this (keeps far-field gradients
            finite for the penalty). ``None`` disables clamping.
    """

    local_map: LocalMapConfig = field(default_factory=LocalMapConfig)
    corrector: str = "potential_field"
    target_clearance_m: float = 0.35
    max_total_shift_m: float = 1.5
    n_seed_points: int = 24
    sdf_clamp_m: Optional[float] = 4.0


@dataclass(frozen=True)
class PerFrameTarget:
    """Output of :func:`generate_target`.

    Attributes:
        corrected_path: The wall-avoiding target trajectory (body FLU ``Path2D``).
        seed_path: The uncorrected seed (for diagnostics / visualization).
        occupancy: The single-frame occupancy grid it was built on.
        sdf_m: ``(H, W)`` float32 signed ESDF in meters (``>0`` free), aligned to
            ``occupancy`` (indexed ``[gy, gx]``).
        num_moved: How many waypoints the corrector actually shifted.
    """

    corrected_path: Path2D
    seed_path: Path2D
    occupancy: OccupancyGrid2D
    sdf_m: np.ndarray
    num_moved: int


def _straight_seed(goal_body: Tuple[float, float], n: int) -> np.ndarray:
    """Origin -> goal straight polyline, ``(n, 2)`` ``[fwd, left]`` (incl. origin)."""
    gx, gy = float(goal_body[0]), float(goal_body[1])
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([ts * gx, ts * gy], axis=1).astype(np.float32)


def _polyline_to_path(points: np.ndarray, frame_id: str = "body") -> Path2D:
    """Wrap an ``(M>=2, 2)`` ``[fwd, left]`` polyline as a ``Path2D`` (x=fwd, y=left)."""
    pts = tuple(Pose2D(float(x), float(y)) for x, y in points)
    return Path2D(points=pts, frame_id=frame_id)


def signed_sdf(occupancy: OccupancyGrid2D, clamp_m: Optional[float]) -> np.ndarray:
    """Signed ESDF (meters) over an occupancy grid.

    Guards the all-free case (no obstacles) where the transform is degenerate, by
    returning a uniform large positive clearance.
    """
    binary = occupancy_binary(occupancy)
    if not binary.any():
        fill = clamp_m if clamp_m is not None else 1e3
        return np.full(binary.shape, float(fill), dtype=np.float32)
    return compute_sdf(binary, occupancy.resolution, SDFParams(clamp_m=clamp_m))


def generate_target(
    depth_m: np.ndarray,
    intrinsics,
    goal_body: Tuple[float, float],
    config: Optional[EsdfTargetConfig] = None,
    seed_path: Optional[np.ndarray] = None,
) -> PerFrameTarget:
    """Build a per-frame PF/ESDF-corrected target trajectory + signed ESDF.

    Args:
        depth_m: ``(H, W)`` float32 metric depth (meters).
        intrinsics: :class:`Intrinsics` matching ``depth_m``.
        goal_body: ``(forward, left)`` goal in the body frame (meters).
        config: :class:`EsdfTargetConfig` (defaults if ``None``).
        seed_path: Optional ``(K, 2)`` ``[fwd, left]`` seed (e.g. the drone's flown
            future). If ``None``, a straight origin->goal seed is used.

    Returns:
        A :class:`PerFrameTarget` with the corrected path, occupancy, and SDF.
    """
    cfg = config or EsdfTargetConfig()

    cloud = depth_to_body_cloud(depth_m, intrinsics, cfg.local_map)
    occ = cloud_to_occupancy_grid(cloud, cfg.local_map)
    sdf_m = signed_sdf(occ, cfg.sdf_clamp_m)

    if seed_path is None:
        seed = _straight_seed(goal_body, cfg.n_seed_points)
    else:
        seed = np.asarray(seed_path, dtype=np.float32).reshape(-1, 2)
        if seed.shape[0] < 2:
            seed = _straight_seed(goal_body, cfg.n_seed_points)
    seed_path2d = _polyline_to_path(seed)

    if cfg.corrector == "esdf":
        corrector = EsdfPathCorrector(
            EsdfCorrectorConfig(
                target_clearance_m=cfg.target_clearance_m,
                max_total_shift_m=cfg.max_total_shift_m,
            )
        )
    elif cfg.corrector == "potential_field":
        corrector = PotentialFieldPathCorrector(
            PotentialFieldCorrectorConfig(
                min_clearance_m=cfg.target_clearance_m,
                max_total_shift_m=cfg.max_total_shift_m,
            )
        )
    else:
        raise ValueError(f"unknown corrector {cfg.corrector!r}")

    result = corrector.correct(seed_path2d, occ)
    return PerFrameTarget(
        corrected_path=result.path,
        seed_path=seed_path2d,
        occupancy=occ,
        sdf_m=sdf_m,
        num_moved=int(result.num_moved),
    )
