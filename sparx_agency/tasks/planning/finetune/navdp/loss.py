"""NavDP fine-tuning loss: diffusion epsilon-MSE + critic + ESDF penalty.

Reconstructs NavDP's training objective (no train code ships upstream; recipe from
arXiv 2505.08712 reconciled with the shipped ``DDPMScheduler`` and forwards):

    L_act    = MSE( eps_theta(x_k, k, goal, rgbd), eps )          # DDPM eps-pred BC
    L_critic = MSE( V_theta(x0_aug, rgbd), V(tau) )               # privileged value
      with V(tau) = -sum 1[d^k < d_safe] + alpha * sum (d^{k+1} - d^k),
      d_safe = 0.5 m, alpha = 0.1, d^k = ESDF clearance at waypoint k.

The BC target ``x0`` is the (optionally ESDF-corrected) 4x-scaled per-step action.
On top, an optional **differentiable ESDF hinge** penalizes the model's decoded
trajectory (``cumsum(x0_hat / 4)``) for entering walls.

Keep the scheduler identical to inference (10 steps, squaredcos_cap_v2, epsilon,
clip_sample) so train and deploy match -- do not re-tune the betas on small data.

Torch only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.esdf_penalty import EsdfHingePenalty, EsdfPenaltyConfig, sample_sdf


@dataclass(frozen=True)
class NavDPLossConfig:
    """Weights / constants for :class:`NavDPLoss`.

    Attributes:
        critic_weight: ``lambda`` on the critic loss (paper implies ~1.0).
        use_critic: Whether to train the critic head at all.
        d_safe_m: Collision-clearance threshold for the critic target (0.5 m).
        progress_alpha: Progress/clearance-gain weight in ``V`` (0.1).
        action_scale: The x4 factor (``trajectory = cumsum(action / 4)``).
        esdf: ESDF hinge configuration (``None`` disables the penalty).
    """

    critic_weight: float = 1.0
    use_critic: bool = True
    d_safe_m: float = 0.5
    progress_alpha: float = 0.1
    action_scale: float = 4.0
    esdf: Optional[EsdfPenaltyConfig] = field(default_factory=EsdfPenaltyConfig)


class NavDPLoss(nn.Module):
    """NavDP diffusion BC + critic loss (+ optional ESDF trajectory penalty)."""

    def __init__(self, config: Optional[NavDPLossConfig] = None) -> None:
        super().__init__()
        self.config = config or NavDPLossConfig()
        self.esdf = EsdfHingePenalty(self.config.esdf) if self.config.esdf else None

    def diffusion_loss(self, pred_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Epsilon-prediction MSE (the behavior-cloning term)."""
        return F.mse_loss(pred_noise, noise)

    def critic_loss(self, critic_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        """Critic regression to the privileged value target."""
        return self.config.critic_weight * F.mse_loss(
            critic_pred.reshape(-1), v_target.reshape(-1)
        )

    @torch.no_grad()
    def critic_target_from_sdf(
        self,
        action_x0: torch.Tensor,
        sdf_grid: torch.Tensor,
        resolution_m: float,
        origin_x: float,
        origin_y: float,
    ) -> torch.Tensor:
        """Privileged value ``V(tau)`` from the ESDF along a trajectory.

        Args:
            action_x0: ``(B, 24, 3)`` per-step action (4x deltas); the trajectory is
                ``cumsum(action_x0[..., :2] / action_scale)``.
            sdf_grid: ``(B, 1, H, W)`` signed ESDF (meters), body frame.
            resolution_m / origin_x / origin_y: grid geometry.

        Returns:
            ``(B,)`` value target (non-differentiable; the critic regresses to it).
        """
        cfg = self.config
        wp = torch.cumsum(action_x0[..., :2] / cfg.action_scale, dim=1)  # (B,24,2)
        d = sample_sdf(sdf_grid, wp, resolution_m, origin_x, origin_y)   # (B,24)
        collide = (d < cfg.d_safe_m).float().sum(dim=1)
        progress = (d[:, 1:] - d[:, :-1]).sum(dim=1)
        return -collide + cfg.progress_alpha * progress

    def esdf_loss(
        self,
        action_x0_hat: torch.Tensor,
        sdf_grid: torch.Tensor,
        resolution_m: float,
        origin_x: float,
        origin_y: float,
    ) -> torch.Tensor:
        """ESDF hinge on the model's decoded trajectory ``cumsum(x0_hat / 4)``."""
        if self.esdf is None:
            return torch.zeros((), device=action_x0_hat.device)
        wp = torch.cumsum(action_x0_hat[..., :2] / self.config.action_scale, dim=1)
        return self.esdf(wp, sdf_grid, resolution_m, origin_x, origin_y)

    def total(
        self,
        parts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sum the provided named loss terms (``act`` + ``critic`` + ``esdf``)."""
        return sum(parts.values())
