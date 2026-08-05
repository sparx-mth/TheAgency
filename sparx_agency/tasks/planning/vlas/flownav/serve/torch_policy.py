"""Eager-torch FlowNav policy -- the UNOPTIMIZED backend (FlowNavTRTPolicy twin).

Runs the FlowNav model in plain PyTorch (no TensorRT), exposing the SAME
``predict(obs_img, goal_img) -> (actions, distance)`` interface (and the same
``num_samples`` / ``num_steps`` / ``precision`` attributes) as the core
``FlowNavTRTPolicy``, so the host server's ``--backend torch`` is a drop-in swap:
the node, client, window and launch are all unchanged -- only the three forward
passes run on the GPU eagerly instead of through the engines.

Use it to A/B the model with vs without TensorRT through the identical FALCON
path (it is ~3-5x slower end-to-end; see the benchmark). The numpy Euler loop and
action de-normalization are the SAME core code the TRT runtime uses, so the only
difference vs the TRT path is FP16-engine precision.

This module imports torch + the FlowNav model, so it lives under ``tasks/`` and
is never imported by ``core``.
"""
from __future__ import annotations

import numpy as np
import torch

from sparx_agency.core.planning.vlas.flownav.trt.postprocess import get_action
from sparx_agency.core.planning.vlas.flownav.trt.scheduler import FlowMatchEulerScheduler
from sparx_agency.tasks.planning.vlas.flownav.trt.export.build_model import (
    build_flownav_model, load_action_stats,
)
from sparx_agency.tasks.planning.vlas.flownav.trt.export.wrappers import (
    DistWrapper, EncoderWrapper, VFieldWrapper,
)


class FlowNavTorchPolicy:
    """Eager-torch FlowNav inference with the ``FlowNavTRTPolicy`` interface.

    Args:
        ckpt: path to ``flownav_weights.pth``.
        flownav_repo: external FlowNav repo path (else ``FLOWNAV_REPO``).
        num_samples: trajectory sample count N.
        num_steps: flow-matching Euler step count K.
        horizon, action_dim: trajectory shape (8, 2 for FlowNav).
        device: torch device (default: cuda if available, else cpu).
    """

    def __init__(self, ckpt, flownav_repo=None, num_samples=8, num_steps=4,
                 horizon=8, action_dim=2, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = build_flownav_model(ckpt, flownav_repo=flownav_repo, device=self.device)
        self.enc = EncoderWrapper(model.vision_encoder).to(self.device).eval()
        self.vf = VFieldWrapper(model.noise_pred_net).to(self.device).eval()
        self.dist_net = DistWrapper(model.dist_pred_net).to(self.device).eval()
        self.action_min, self.action_max = load_action_stats(flownav_repo)
        self.num_samples = int(num_samples)
        self.scheduler = FlowMatchEulerScheduler(int(num_steps))
        self.horizon, self.action_dim = int(horizon), int(action_dim)
        self.precision = "torch"

    @property
    def num_steps(self):
        """The flow-matching step count K."""
        return self.scheduler.num_steps

    def _t(self, a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(self.device)

    @torch.no_grad()
    def predict(self, obs_img, goal_img, init_noise=None):
        """Encode, score distance, and integrate the velocity field K-1 Euler steps.

        Mirrors ``FlowNavTRTPolicy.predict`` exactly (same numpy scheduler + action
        de-normalization); only the three forwards differ (eager torch vs engines).

        Returns:
            ``(actions (N, horizon, action_dim), distance float)``.
        """
        cond = self.enc(self._t(obs_img), self._t(goal_img))         # (1, 256)
        cond_n = cond.repeat_interleave(self.num_samples, dim=0)     # (N, 256)
        dist = float(self.dist_net(cond).cpu().numpy().reshape(-1)[0])
        shape = (self.num_samples, self.horizon, self.action_dim)
        x = (np.random.randn(*shape).astype(np.float32)
             if init_noise is None else np.asarray(init_noise, np.float32))
        for i in range(self.scheduler.num_field_evals):
            t = self._t(np.array([self.scheduler.timesteps[i]], np.float32))
            vfield = self.vf(self._t(x), t, cond_n).cpu().numpy()
            x = self.scheduler.step(vfield, i, x)
        return get_action(x, self.action_min, self.action_max), dist
