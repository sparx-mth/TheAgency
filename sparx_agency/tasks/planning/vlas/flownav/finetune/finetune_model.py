"""FlowNav fine-tuning model: freeze policy + grad-enabled forwards + optional FiLM.

Unlike NavDP, FlowNav ships a trainer, so its ``NoMaD`` forwards are already
grad-enabled -- we call the real string-dispatched model directly (preserving the
goal-mask dropout that the inference export wrappers bake away):

    obsgoal_cond = model("vision_encoder", obs_img, goal_img, input_goal_mask=mask)
    vt           = model("noise_pred_net", sample=x_t, timestep=t, global_cond=cond)
    dist         = model("dist_pred_net", obsgoal_cond=cond)

Default fine-tune policy: freeze both EfficientNets and the (already frozen) DINOv2
depth prior; train the ConditionalUnet1D velocity field + the self-attention fusion
+ the distance head (+ the compress projections). An optional **viewpoint-FiLM**
adapter on ``obsgoal_cond`` is the "add a small layer, freeze the rest" fallback.

Torch + the external FlowNav repo -- runs in the ``flownav_trt`` conda env.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from sparx_agency.tasks.planning.vlas.flownav.trt.export.build_model import (
    build_flownav_model,
    load_action_stats,
)

# EfficientNet + DINOv2 backbones -> frozen. Everything else -> trained.
FROZEN_PREFIXES = (
    "vision_encoder.obs_encoder",
    "vision_encoder.goal_encoder",
    "vision_encoder.depth_encoder",
)


class ViewpointFiLM(nn.Module):
    """Tiny FiLM adapter on the pooled conditioning: ``h -> gamma * h + beta``.

    Initialized to identity (gamma=1, beta=0) so it is a no-op at the start of
    fine-tuning. This is the smallest "add a trainable layer, freeze the base"
    option; it recalibrates the frozen vision features to the drone viewpoint.
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.gamma * cond + self.beta


@dataclass(frozen=True)
class FlowNavFinetuneConfig:
    """Freeze / adapter configuration.

    Attributes:
        train_compress: Also train the compress_{obs,goal,depth}_enc projections.
        use_film: Add the :class:`ViewpointFiLM` adapter on ``obsgoal_cond``.
        metric_waypoint_spacing: Meters per waypoint-unit for the drone dataset
            (used to decode trajectories to meters for the ESDF penalty).
        lr: Learning rate (FlowNav default 1e-4).
    """

    train_compress: bool = True
    use_film: bool = False
    metric_waypoint_spacing: float = 0.25
    lr: float = 1e-4


class FlowNavFinetune(nn.Module):
    """Wraps FlowNav ``NoMaD`` with a freeze policy and trajectory decoding."""

    def __init__(self, ckpt_path: str, flownav_repo: Optional[str] = None,
                 device: str = "cuda",
                 config: Optional[FlowNavFinetuneConfig] = None) -> None:
        super().__init__()
        self.config = config or FlowNavFinetuneConfig()
        self.model = build_flownav_model(ckpt_path, flownav_repo, device=device)
        amin, amax = load_action_stats(flownav_repo)
        self.register_buffer("action_min", torch.as_tensor(np.asarray(amin), dtype=torch.float32))
        self.register_buffer("action_max", torch.as_tensor(np.asarray(amax), dtype=torch.float32))
        self.film = ViewpointFiLM(256) if self.config.use_film else None
        self._set_trainable()

    def _set_trainable(self) -> None:
        keep_compress = self.config.train_compress
        for name, p in self.model.named_parameters():
            frozen = name.startswith(FROZEN_PREFIXES)
            if frozen and keep_compress and "compress" in name:
                frozen = False
            p.requires_grad_(not frozen)

    def trainable_parameters(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.film is not None:
            params += list(self.film.parameters())
        return params

    # ---------------------------------------------------------------- forwards
    def encode(self, obs_img: torch.Tensor, goal_img: torch.Tensor,
               goal_mask: torch.Tensor) -> torch.Tensor:
        """``obs_img (B,12,96,96)``, ``goal_img (B,3,96,96)`` -> ``obsgoal_cond (B,256)``."""
        cond = self.model("vision_encoder", obs_img=obs_img, goal_img=goal_img,
                          input_goal_mask=goal_mask)
        if self.film is not None:
            cond = self.film(cond)
        return cond

    def vfield(self, sample: torch.Tensor, timestep: torch.Tensor,
               cond: torch.Tensor) -> torch.Tensor:
        """Velocity field ``v_theta(x_t, t, cond)`` -> ``(N,8,2)``."""
        return self.model("noise_pred_net", sample=sample, timestep=timestep,
                          global_cond=cond)

    def distance(self, cond: torch.Tensor) -> torch.Tensor:
        """Temporal-distance head ``(B,256) -> (B,)``."""
        return self.model("dist_pred_net", obsgoal_cond=cond).squeeze(-1)

    # ------------------------------------------------------- trajectory decode
    def _unnormalize(self, ndeltas: torch.Tensor) -> torch.Tensor:
        amin = self.action_min.to(ndeltas.device)
        amax = self.action_max.to(ndeltas.device)
        return (ndeltas + 1.0) / 2.0 * (amax - amin) + amin

    def rollout_waypoints(self, cond: torch.Tensor, horizon: int = 8,
                          num_steps: int = 8,
                          noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Integrate the velocity field to decoded body-frame waypoints (meters).

        Deterministic Euler over ``linspace(0,1,num_steps)`` (differentiable), then
        unnormalize + cumsum + metric spacing. Use the result for the ESDF penalty.

        Args:
            cond: ``(B,256)`` conditioning.
            horizon: Waypoint count (8).
            num_steps: Euler steps K (keep consistent with deployment).
            noise: Optional fixed ``(B,horizon,2)`` start; else standard normal.

        Returns:
            ``(B, horizon, 2)`` waypoints ``(x=fwd, y=left)`` in meters.
        """
        b = cond.shape[0]
        device = cond.device
        x = noise if noise is not None else torch.randn(b, horizon, 2, device=device)
        ts = torch.linspace(0.0, 1.0, num_steps, device=device)
        for i in range(num_steps - 1):
            t = ts[i].expand(b)
            v = self.vfield(x, t, cond)
            x = x + (ts[i + 1] - ts[i]) * v
        deltas = self._unnormalize(x)                       # waypoint-unit deltas
        wp_units = torch.cumsum(deltas, dim=1)              # absolute waypoint-units
        return wp_units * self.config.metric_waypoint_spacing  # -> meters
