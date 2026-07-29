"""Run NavDP exactly as it is deployed: 16 samples, 10 denoise steps, critic picks.

Evaluation has to measure the thing that actually flies, and what flies is not
the mean of the diffusion head -- it is one of sixteen stochastic samples,
selected by the critic. A fine-tune that improved the mean trajectory while
leaving the critic ranking untouched would look good on a regression metric and
change nothing in the air.

So this reproduces ``NavDP_Policy.predict_pointgoal_action`` step for step,
including the two easily-missed details in its tail: the division by 4 happens
*before* the cumulative sum, and any sample whose final XY displacement is under
0.5 m has its XY zeroed (NavDP's "stop" convention) before the critic ranking is
applied.

The one addition is a **seed**. Both arms of a comparison draw the same initial
noise and the same per-step variance noise, so a difference between them is the
weights and nothing else. Without that, a 16-sample stochastic policy compared
against itself shows a spread that swamps most real effects.

Torch, no_grad throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

ACTION_SCALE = 4.0
STOP_THRESHOLD_M = 0.5


@dataclass
class InferenceResult:
    """One batch of point-goal inferences."""

    trajectory: torch.Tensor      # (B, T, 2) the executed path, body FLU, metres
    all_trajectory: torch.Tensor  # (B, N, T, 3) every sample
    critic: torch.Tensor          # (B, N) critic value per sample
    chosen: torch.Tensor          # (B,) index of the executed sample


class NavDPRunner:
    """Deterministic-on-demand NavDP point-goal inference."""

    def __init__(self, model, sample_num: int = 16,
                 action_scale: float = ACTION_SCALE) -> None:
        self.model = model
        self.sample_num = int(sample_num)
        self.action_scale = float(action_scale)

    @torch.no_grad()
    def run(self, rgbd: torch.Tensor, goal: torch.Tensor,
            seed: Optional[int] = None) -> InferenceResult:
        """Sample, denoise, rank.

        Args:
            rgbd: ``(B, 128, 384)`` scene embedding.
            goal: ``(B, 3)`` body-frame point goal ``(forward, left, 0)``.
            seed: Fixes both the initial noise and the scheduler's variance
                noise. Pass the same value to every arm being compared.

        Returns:
            An :class:`InferenceResult`.
        """
        model = self.model
        device = rgbd.device
        batch, samples = rgbd.shape[0], self.sample_num
        horizon = model.predict_size

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))

        rgbd_n = rgbd.repeat_interleave(samples, dim=0)
        goal_n = model.goal_embed(goal).repeat_interleave(samples, dim=0)
        action = torch.randn((batch * samples, horizon, 3), device=device,
                             dtype=rgbd.dtype, generator=generator)

        scheduler = model.scheduler
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        for timestep in scheduler.timesteps:
            step = timestep.to(device).expand(batch * samples)
            noise = model.predict_noise(action, step, goal_n, rgbd_n)
            action = scheduler.step(noise, timestep, action,
                                    generator=generator).prev_sample

        critic = model.predict_critic(action, rgbd_n).reshape(batch, samples)
        trajectory = torch.cumsum(action / self.action_scale, dim=1)
        trajectory = trajectory.reshape(batch, samples, horizon, 3)

        # NavDP's stop convention: a sample that goes nowhere keeps only its yaw.
        short = trajectory[:, :, -1, 0:2].norm(dim=-1) < STOP_THRESHOLD_M
        keep_yaw = trajectory.new_tensor([0.0, 0.0, 1.0])
        trajectory[short] = trajectory[short] * keep_yaw

        chosen = critic.argmax(dim=1)
        rows = torch.arange(batch, device=device)
        return InferenceResult(
            trajectory=trajectory[rows, chosen][:, :, :2],
            all_trajectory=trajectory, critic=critic, chosen=chosen)
