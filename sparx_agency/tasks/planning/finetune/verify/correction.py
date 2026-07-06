"""The 'connection': push a NavDP trajectory off walls with the same-frame field.

NavDP outputs a trajectory in the body FLU frame (x=forward, y=left, meters). The
instantaneous potential field / signed ESDF built from the *same* depth frame
lives in that identical frame (:mod:`..common.frames` puts the robot at the grid
origin, x=fwd, y=left). So connecting them needs no transform at all -- we simply
hand NavDP's waypoints to :func:`..common.esdf_target.generate_target` as the
**seed path** it corrects, instead of the default straight-to-goal seed.

The result is the pair the fine-tune wants to compare and, later, train on:

* ``seed_path``      -- what NavDP predicted (the network's raw output), and
* ``corrected_path`` -- what we'd like it to have predicted (lightly pushed off
  the walls), the behaviour-cloning target.

Both, plus the occupancy grid and the signed ESDF, come back in one bundle.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.tasks.planning.finetune.common.esdf_target import (
    EsdfTargetConfig,
    PerFrameTarget,
    generate_target,
)
from sparx_agency.tasks.planning.finetune.common.frames import LocalMapConfig


def make_config(
    corrector: str = "esdf",
    target_clearance_m: float = 0.5,
    max_total_shift_m: float = 0.8,
    pitch_deg: float = 0.0,
    camera_height_m: float = 1.0,
    smooth_strength: float = 0.5,
    z_band_m: Tuple[float, float] = (0.10, 2.0),
    resolution_m: float = 0.10,
    forward_extent_m: float = 8.0,
    half_width_m: float = 5.0,
) -> EsdfTargetConfig:
    """Assemble an :class:`EsdfTargetConfig` from the UI's live knobs.

    ``pitch_deg`` and ``camera_height_m`` shape the single-frame occupancy (the
    floor/obstacle split); ``corrector`` / ``target_clearance_m`` /
    ``max_total_shift_m`` control how hard the trajectory is pushed off walls;
    ``smooth_strength`` (0 disables) sets how hard the post-correction smoothing
    relaxes kinks/zigzags out of the pushed trajectory.
    """
    return EsdfTargetConfig(
        local_map=LocalMapConfig(
            resolution_m=resolution_m,
            forward_extent_m=forward_extent_m,
            half_width_m=half_width_m,
            z_band_m=z_band_m,
            camera_height_m=camera_height_m,
            pitch_deg=pitch_deg,
        ),
        corrector=corrector,
        target_clearance_m=target_clearance_m,
        max_total_shift_m=max_total_shift_m,
        smooth=smooth_strength > 0.0,
        smooth_strength=smooth_strength,
    )


def correct_navdp_trajectory(
    navdp_xy: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    goal_body: Tuple[float, float],
    config: EsdfTargetConfig,
) -> PerFrameTarget:
    """Push a NavDP trajectory off the single-frame walls.

    Args:
        navdp_xy: ``(K, 2)`` NavDP ``[fwd, left]`` waypoints (meters, body FLU).
        depth_m: ``(H, W)`` float32 depth (meters) -- the same frame NavDP saw.
        intrinsics: pinhole intrinsics for ``depth_m``.
        goal_body: ``(forward, left)`` goal (only used if the seed is degenerate).
        config: :class:`EsdfTargetConfig` (see :func:`make_config`).

    Returns:
        A :class:`PerFrameTarget` with ``seed_path`` = NavDP (origin-prepended),
        ``corrected_path`` = the pushed target, plus ``occupancy``, ``sdf_m`` and
        ``num_moved``.
    """
    seed = np.asarray(navdp_xy, np.float32).reshape(-1, 2)
    # Prepend the robot origin so the seed starts under the drone; the corrector
    # pins the first (and, by default, last) waypoint.
    if not (seed.shape[0] and np.allclose(seed[0], 0.0, atol=1e-3)):
        seed = np.vstack([[0.0, 0.0], seed]).astype(np.float32)
    return generate_target(depth_m, intrinsics, goal_body, config, seed_path=seed)
