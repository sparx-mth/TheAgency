"""Pure-numpy DDIM sampler for NavDP's point-goal diffusion head (few-step).

DDPM (``scheduler.py``) runs all ``num_train_timesteps`` (=10) steps and is the
bit-faithful trained default. DDIM lets the SAME denoise network run in FEWER
steps by (a) subsampling the trained timesteps and (b) using the deterministic
(``eta=0``) reverse update, which needs no per-step variance noise. This is the
Tier-1 runtime speed lever: fewer denoise-engine calls, no re-export, no retrain.

Because the exported time-embedding table has one row per *trained* timestep
(``0..T-1``), the inference timesteps are chosen as a subset of those integers
(``linspace(0, T-1, K)`` rounded, high->low), so every step reuses an existing
time embedding -- no new export needed. ``K = T`` reproduces the full trained
timestep set (but deterministic, unlike DDPM).

DDIM(``eta=0``) update for ``prediction_type='epsilon'`` (matches diffusers
``DDIMScheduler.step`` with ``eta=0``); the ``clip_sample`` clamp is applied to
the predicted ``x0`` BEFORE the direction term, exactly like the DDPM scheduler::

    x0        = (x_t - sqrt(1 - abar_t) * eps) / sqrt(abar_t)
    x0        = clip(x0, -1, 1)                         # clip_sample
    x_{prev}  = sqrt(abar_prev) * x0 + sqrt(1 - abar_prev) * eps

with ``abar_prev = alphas_cumprod[t_prev]`` (or ``1.0`` when ``t_prev < 0``), where
``t_prev`` is the NEXT (lower) timestep in the subsampled schedule -- not ``t-1``.
At the final step ``t_prev = -1`` so ``abar_prev = 1`` and ``x_prev = clip(x0)``,
matching the DDPM scheduler's ``t=0`` endpoint. Deterministic: ``step`` takes NO
variance noise (the kwarg is accepted and ignored so the policy denoise loop is
sampler-agnostic).
"""
from __future__ import annotations

import numpy as np

_F32 = np.float32
_ONE = np.float32(1.0)


class NumpyDDIMScheduler:
    """Float32 numpy DDIM sampler over a subset of the trained timesteps.

    Args:
        alphas_cumprod: 1-D cumulative-alpha array from the trained scheduler
            (length == ``num_train_timesteps``), taken verbatim from the export.
        num_inference_steps: number of denoise steps ``K`` (``1 <= K <= T``).
        clip_sample: clamp the predicted ``x0`` (NavDP uses ``True``).
        clip_sample_range: symmetric clamp bound (diffusers default 1.0).
    """

    def __init__(self, alphas_cumprod, num_inference_steps, clip_sample=True,
                 clip_sample_range=1.0):
        self.alphas_cumprod = np.asarray(alphas_cumprod, dtype=_F32).reshape(-1)
        if self.alphas_cumprod.ndim != 1 or self.alphas_cumprod.size == 0:
            raise ValueError("alphas_cumprod must be a non-empty 1-D array")
        self.num_train_timesteps = int(self.alphas_cumprod.shape[0])
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)
        self.set_inference_steps(num_inference_steps)

    def set_inference_steps(self, num_inference_steps):
        """Pick ``K`` timesteps evenly across ``0..T-1`` (incl. endpoints), high->low."""
        k = int(num_inference_steps)
        t = self.num_train_timesteps
        if k < 1 or k > t:
            raise ValueError("num_inference_steps must be in [1, %d], got %d" % (t, k))
        ts = np.unique(np.rint(np.linspace(0, t - 1, k)).astype(np.int64))  # dedupe collisions
        self.timesteps = ts[::-1].copy()                 # e.g. [9,6,3,0] for T=10, K=4
        prev = np.empty_like(self.timesteps)
        prev[:-1] = self.timesteps[1:]                   # next (lower) timestep in the schedule
        prev[-1] = -1                                    # last step lands on clean x0
        self._prev = {int(a): int(b) for a, b in zip(self.timesteps, prev)}

    def _abar(self, t):
        """alpha_cumprod at ``t``, or 1.0 when ``t < 0`` (the final endpoint)."""
        return self.alphas_cumprod[t] if t >= 0 else _ONE

    def step(self, model_output, timestep, sample, variance_noise=None):
        """One deterministic DDIM (eta=0) reverse step. ``variance_noise`` ignored.

        Args:
            model_output: predicted epsilon ``(N, predict_size, 3)``.
            timestep: current timestep ``t`` (must be one of ``self.timesteps``).
            sample: current noisy action ``x_t``.
            variance_noise: accepted for a sampler-agnostic loop; unused (DDIM is
                deterministic).

        Returns:
            ``x_{prev}`` as float32, same shape as ``sample``.
        """
        t = int(timestep)
        prev_t = self._prev[t]
        eps = np.asarray(model_output, dtype=_F32)
        x = np.asarray(sample, dtype=_F32)
        abar_t = self.alphas_cumprod[t]
        abar_prev = self._abar(prev_t)

        pred_x0 = (x - np.sqrt(_ONE - abar_t) * eps) / np.sqrt(abar_t)
        if self.clip_sample:
            pred_x0 = np.clip(pred_x0, -self.clip_sample_range, self.clip_sample_range)
        x_prev = np.sqrt(abar_prev) * pred_x0 + np.sqrt(_ONE - abar_prev) * eps
        return x_prev.astype(_F32, copy=False)
