"""Differentiable ESDF obstacle penalty on a network's own output trajectory.

This is the *second*, distinct use of the per-frame ESDF (the first is the
non-differentiable BC label built in :mod:`.esdf_target`). Here the signed ESDF
grid is a **fixed lookup** (built once per frame from depth, stop-gradient w.r.t.
the pixels); we differentiate only w.r.t. the network's predicted waypoints. Under
the hood the numpy ``PotentialFieldSampler._bilinear`` becomes
``torch.nn.functional.grid_sample``, which is autograd-differentiable to the query
coordinates -- so a hinge on ``ESDF < margin`` pushes waypoints out of walls.

Signed SDF (``>0`` free, ``<0`` inside a wall) is required so a waypoint driven
*inside* an obstacle keeps a monotone push-out gradient; the unsigned distance
field saturates to a flat gradient inside walls.

Torch only -- runs in the navdp / flownav_trt conda env, not the plain ``.venv``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EsdfPenaltyConfig:
    """Configuration for :class:`EsdfHingePenalty`.

    Attributes:
        margin_m: Desired clearance; waypoints closer than this to a wall are
            penalized. Default 0.35 m (matches the target-generator clearance).
        weight: Multiplier applied to the mean hinge (``lambda_esdf``).
        clamp_grad_m: If set, the sampled SDF is clamped to ``[-clamp, clamp]``
            before the hinge, bounding the penalty magnitude.
    """

    margin_m: float = 0.35
    weight: float = 0.5
    clamp_grad_m: Optional[float] = 4.0


def sample_sdf(
    sdf_grid: torch.Tensor,
    waypoints_body: torch.Tensor,
    resolution_m: float,
    origin_x: float,
    origin_y: float,
) -> torch.Tensor:
    """Bilinearly sample a signed ESDF grid at body-frame waypoints.

    Args:
        sdf_grid: ``(B, 1, H, W)`` signed SDF (meters), indexed ``[gy, gx]``, in the
            body FLU frame with the given origin/resolution.
        waypoints_body: ``(B, N, 2)`` waypoints ``(x=forward, y=left)`` meters.
        resolution_m: Grid cell size (meters).
        origin_x: World x of grid cell (0, 0) (forward origin).
        origin_y: World y of grid cell (0, 0) (left origin).

    Returns:
        ``(B, N)`` sampled signed distance (meters); out-of-grid queries return the
        border value (mask them with :func:`in_bounds_mask` if needed).
    """
    if sdf_grid.dim() != 4 or sdf_grid.shape[1] != 1:
        raise ValueError(f"sdf_grid must be (B,1,H,W), got {tuple(sdf_grid.shape)}")
    b, _, h, w = sdf_grid.shape
    x = waypoints_body[..., 0]
    y = waypoints_body[..., 1]

    # Continuous cell indices (grid_to_world uses (g+0.5)*res+origin).
    col = (x - origin_x) / resolution_m - 0.5   # -> gx axis (width)
    row = (y - origin_y) / resolution_m - 0.5   # -> gy axis (height)

    # Normalize to [-1, 1] for grid_sample(align_corners=True).
    gx_n = 2.0 * col / max(w - 1, 1) - 1.0
    gy_n = 2.0 * row / max(h - 1, 1) - 1.0

    grid = torch.stack([gx_n, gy_n], dim=-1).unsqueeze(2)  # (B, N, 1, 2)
    sampled = F.grid_sample(
        sdf_grid, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # (B, 1, N, 1)
    return sampled[:, 0, :, 0]


def in_bounds_mask(
    waypoints_body: torch.Tensor,
    grid_hw: tuple,
    resolution_m: float,
    origin_x: float,
    origin_y: float,
) -> torch.Tensor:
    """``(B, N)`` bool mask: True where the waypoint lies inside the grid."""
    h, w = grid_hw
    x = waypoints_body[..., 0]
    y = waypoints_body[..., 1]
    gx = (x - origin_x) / resolution_m
    gy = (y - origin_y) / resolution_m
    return (gx >= 0) & (gx <= (w - 1)) & (gy >= 0) & (gy <= (h - 1))


class EsdfHingePenalty(torch.nn.Module):
    """Hinge-on-clearance penalty: ``mean( relu(margin - SDF(waypoint))^2 )``.

    Differentiable w.r.t. the waypoints (the SDF grid is a fixed buffer). Use it as
    ``lambda_esdf * penalty`` added to the behavior-cloning loss.
    """

    def __init__(self, config: Optional[EsdfPenaltyConfig] = None) -> None:
        super().__init__()
        self.config = config or EsdfPenaltyConfig()

    def forward(
        self,
        waypoints_body: torch.Tensor,
        sdf_grid: torch.Tensor,
        resolution_m: float,
        origin_x: float,
        origin_y: float,
    ) -> torch.Tensor:
        """Return the scalar penalty for a batch of predicted trajectories.

        Args:
            waypoints_body: ``(B, N, 2)`` predicted body-frame waypoints (meters).
            sdf_grid: ``(B, 1, H, W)`` signed ESDF (meters), same body frame.
            resolution_m / origin_x / origin_y: grid geometry.
        """
        cfg = self.config
        d = sample_sdf(sdf_grid, waypoints_body, resolution_m, origin_x, origin_y)
        if cfg.clamp_grad_m is not None:
            d = d.clamp(min=-cfg.clamp_grad_m, max=cfg.clamp_grad_m)
        hinge = F.relu(cfg.margin_m - d) ** 2

        mask = in_bounds_mask(
            waypoints_body, sdf_grid.shape[-2:], resolution_m, origin_x, origin_y
        ).to(hinge.dtype)
        denom = mask.sum().clamp(min=1.0)
        return cfg.weight * (hinge * mask).sum() / denom
