"""FlowNav fine-tuning loss: flow-matching velocity MSE + distance + ESDF penalty.

Reproduces FlowNav's shipped objective (``flownav/training/train.py``):

    flow_loss = action_reduce( MSE(v_theta(x_t, t, cond), u_t), action_mask )
    dist_loss = MSE(dist_pred, distance)   (goal-masked)
    bc        = alpha * dist_loss + (1 - alpha) * flow_loss      # alpha = 1e-4

with ``u_t = x1 - x0`` and ``x_t = (1-t) x0 + t x1`` (rectified / OT-CFM, sigma=0).
The *target* ``x1`` = the (optionally ESDF-corrected) normalized action label, so
this same loss trains toward the wall-avoiding trajectory.

On top, an optional **differentiable ESDF hinge** penalizes the model's *decoded*
trajectory for entering walls (see :mod:`..common.esdf_penalty`).

Two shipped bugs are fixed here (documented, opt-out via flags): the distance-loss
reduction order (which made masking a near-no-op) and the note about the double
LR-scheduler step lives in the train loop, not here.

Torch only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.esdf_penalty import EsdfHingePenalty, EsdfPenaltyConfig


def action_reduce(unreduced: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """FlowNav's masked per-sample mean (``training/utils.py:action_reduce``)."""
    while unreduced.dim() > 1:
        unreduced = unreduced.mean(dim=-1)
    return (unreduced * mask).mean() / (mask.mean() + 1e-2)


@dataclass(frozen=True)
class FlowNavLossConfig:
    """Weights / flags for :class:`FlowNavLoss`.

    Attributes:
        alpha: Distance-loss weight (FlowNav default 1e-4; set 0 to drop distance).
        fix_dist_mask: Fix the shipped reduction-order bug so the distance loss is
            averaged only over goal-present samples.
        esdf: ESDF hinge configuration (``None`` disables the penalty).
    """

    alpha: float = 1e-4
    fix_dist_mask: bool = True
    esdf: Optional[EsdfPenaltyConfig] = field(default_factory=EsdfPenaltyConfig)


class FlowNavLoss(nn.Module):
    """FlowNav behavior-cloning loss (+ optional ESDF trajectory penalty)."""

    def __init__(self, config: Optional[FlowNavLossConfig] = None) -> None:
        super().__init__()
        self.config = config or FlowNavLossConfig()
        self.esdf = EsdfHingePenalty(self.config.esdf) if self.config.esdf else None

    def bc_loss(
        self,
        vt: torch.Tensor,
        ut: torch.Tensor,
        action_mask: torch.Tensor,
        dist_pred: Optional[torch.Tensor] = None,
        distance: Optional[torch.Tensor] = None,
        goal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Flow-matching + distance behavior-cloning loss.

        Args:
            vt: ``(B, T, 2)`` predicted velocity ``v_theta(x_t, t, cond)``.
            ut: ``(B, T, 2)`` target velocity ``x1 - x0``.
            action_mask: ``(B,)`` per-sample 0/1 weight.
            dist_pred / distance: ``(B,)`` predicted / target temporal distance.
            goal_mask: ``(B,)`` 1 = goal masked (unconditional), 0 = goal used.

        Returns:
            Dict with ``flow``, ``dist``, and ``bc`` (the weighted total).
        """
        flow = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)

        dist = torch.zeros((), device=vt.device, dtype=vt.dtype)
        if dist_pred is not None and distance is not None and self.config.alpha > 0:
            if self.config.fix_dist_mask and goal_mask is not None:
                w = (1.0 - goal_mask.float())
                per = F.mse_loss(dist_pred, distance.float(), reduction="none")
                dist = (per * w).sum() / (w.sum() + 1e-2)
            else:
                dist = F.mse_loss(dist_pred, distance.float())

        a = self.config.alpha
        bc = a * dist + (1.0 - a) * flow
        return {"flow": flow, "dist": dist, "bc": bc}

    def esdf_loss(
        self,
        waypoints_body: torch.Tensor,
        sdf_grid: torch.Tensor,
        resolution_m: float,
        origin_x: float,
        origin_y: float,
    ) -> torch.Tensor:
        """ESDF hinge on the model's decoded trajectory (meters, body frame)."""
        if self.esdf is None:
            return torch.zeros((), device=waypoints_body.device)
        return self.esdf(waypoints_body, sdf_grid, resolution_m, origin_x, origin_y)
