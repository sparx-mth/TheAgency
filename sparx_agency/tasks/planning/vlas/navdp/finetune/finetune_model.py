"""NavDP fine-tuning model: grad-enabled forwards + freeze/PEFT configuration.

The shipped NavDP forwards run under ``torch.no_grad()`` (``policy_backbone.py:60``,
``policy_network.predict_*``), so gradients never reach the encoder/decoder. Rather
than monkey-patch that guard, we reuse the **export wrappers** (``EncoderWrapper``,
``DenoiseStepWrapper``, ``CriticWrapper``) which already re-implement the exact
forwards *with* the autograd graph (they were built for ONNX tracing). They wrap the
policy's real submodules, so optimizing them optimizes the real weights.

Default fine-tune policy (the report's recommendation for a small drone dataset):
freeze the RGB DINOv2 trunk, train the fusion decoder + heads + Q-Former, and
optionally adapt the *depth* DINOv2 (the viewpoint-sensitive path). LoRA/DoRA on the
depth-ViT attention is the recommended next step; the target module names are
exposed via :meth:`lora_target_modules` for ``peft``.

Torch + the external NavDP repo -- runs in the ``navdp`` conda env, not ``.venv``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from sparx_agency.tasks.planning.vlas.navdp.trt.export.build_policy import build_navdp_policy
from sparx_agency.tasks.planning.vlas.navdp.trt.export.wrappers import (
    CriticWrapper,
    DenoiseStepWrapper,
    EncoderWrapper,
)

# Parameter-name prefixes that make up the trainable "policy head" (everything
# except the frozen RGB DINOv2 trunk). Mirrors the export REQUIRED_PREFIXES minus
# rgb_model.
# ``decoder.layers`` and not ``decoder`` -- ``NavDP_Policy`` keeps the prototype
# ``self.decoder_layer`` it handed to ``nn.TransformerDecoder``, which deep-copies
# it. Those 2.4 M parameters are in the checkpoint and are never used in any
# forward, so a bare "decoder" prefix put dead weight into the optimizer, the EMA
# shadow, the L2-SP reference and every saved checkpoint. Excluding it changes no
# result (a parameter with no gradient never moves) and costs less of all four.
HEAD_PREFIXES = (
    "point_encoder",
    "decoder.layers",
    "input_embed",
    "cond_pos_embed",
    "out_pos_embed",
    "action_head",
    "critic_head",
    "layernorm",
    "rgbd_encoder.former_net",
    "rgbd_encoder.former_pe",
    "rgbd_encoder.former_query",
    "rgbd_encoder.project_layer",
)
RGB_TRUNK_PREFIX = "rgbd_encoder.rgb_model"
DEPTH_TRUNK_PREFIX = "rgbd_encoder.depth_model"


@dataclass(frozen=True)
class NavDPFinetuneConfig:
    """Freeze / learning-rate configuration.

    Attributes:
        train_depth_encoder: Unfreeze the depth DINOv2 (stage-2 / PEFT). The RGB
            trunk is *always* frozen.
        lr_head: LR for the fusion decoder + heads + Q-Former.
        lr_backbone: LR for the depth encoder when trained (much smaller).
    """

    train_depth_encoder: bool = False
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5


class NavDPFinetune(nn.Module):
    """Wraps ``NavDP_Policy`` with grad-enabled forwards and a freeze policy."""

    def __init__(self, ckpt_path: str, navdp_repo: Optional[str] = None,
                 device: str = "cuda",
                 config: Optional[NavDPFinetuneConfig] = None) -> None:
        super().__init__()
        self.config = config or NavDPFinetuneConfig()
        policy = build_navdp_policy(ckpt_path, navdp_repo, device=device)
        self.policy = policy
        # Grad-enabled forwards over the policy's real submodules.
        self.encoder = EncoderWrapper(policy.rgbd_encoder)
        self.denoise = DenoiseStepWrapper(policy)
        self.critic = CriticWrapper(policy)
        self.point_encoder = policy.point_encoder
        self.time_emb = policy.time_emb
        self.scheduler = policy.noise_scheduler
        self.predict_size = policy.predict_size
        self._set_trainable()

    # ------------------------------------------------------------------ freeze
    def _set_trainable(self) -> None:
        cfg = self.config
        for name, p in self.policy.named_parameters():
            if name.startswith(RGB_TRUNK_PREFIX):
                p.requires_grad_(False)
            elif name.startswith(DEPTH_TRUNK_PREFIX):
                p.requires_grad_(cfg.train_depth_encoder)
            elif name.startswith(HEAD_PREFIXES):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    def param_groups(self) -> List[dict]:
        """Discriminative-LR param groups (head fast, depth backbone slow)."""
        head, backbone = [], []
        for name, p in self.policy.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if name.startswith(DEPTH_TRUNK_PREFIX) else head).append(p)
        groups = [{"params": head, "lr": self.config.lr_head}]
        if backbone:
            groups.append({"params": backbone, "lr": self.config.lr_backbone})
        return groups

    def lora_target_modules(self) -> List[str]:
        """Recommended DoRA/LoRA targets (depth-ViT attention + Q-Former), for peft."""
        return [
            "rgbd_encoder.depth_model.blocks.*.attn.qkv",
            "rgbd_encoder.depth_model.blocks.*.attn.proj",
            "rgbd_encoder.former_net.layers.*.self_attn",
            "rgbd_encoder.former_net.layers.*.multihead_attn",
        ]

    # ---------------------------------------------------------------- forwards
    def encode(self, images: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """``images (B,8,3,224,224)``, ``depths (B,1,224,224)`` -> ``rgbd_embed (B,128,384)``."""
        return self.encoder(images, depths)

    def goal_embed(self, goal_body: torch.Tensor) -> torch.Tensor:
        """``goal_body (B,3)`` -> ``(B,1,384)`` point-goal token."""
        return self.point_encoder(goal_body).unsqueeze(1)

    def predict_noise(self, x_k: torch.Tensor, k: torch.Tensor,
                      goal_embed: torch.Tensor, rgbd_embed: torch.Tensor) -> torch.Tensor:
        """Predict the diffusion noise for a noised action chunk (grad-enabled)."""
        time_token = self.time_emb(k).unsqueeze(1)     # (B,1,384)
        return self.denoise(x_k, time_token, goal_embed, rgbd_embed)

    def predict_critic(self, action: torch.Tensor, rgbd_embed: torch.Tensor) -> torch.Tensor:
        """Critic value for a trajectory action ``(B,24,3)`` -> ``(B,1)``."""
        return self.critic(action, rgbd_embed)

    def x0_from_eps(self, x_k: torch.Tensor, k: torch.Tensor,
                    eps: torch.Tensor) -> torch.Tensor:
        """One-step denoised action estimate ``x0_hat`` (for the ESDF penalty).

        ``x0 = (x_k - sqrt(1 - abar_k) * eps) / sqrt(abar_k)``. Cheaper than a full
        10-step rollout and still differentiable w.r.t. the network output ``eps``.
        """
        abar = self.scheduler.alphas_cumprod.to(x_k.device)[k]       # (B,)
        abar = abar.view(-1, *([1] * (x_k.dim() - 1)))
        return (x_k - torch.sqrt(1.0 - abar) * eps) / torch.sqrt(abar)
