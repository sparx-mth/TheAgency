"""Pure-numpy DDPM sampler, bit-faithful to NavDP's diffusers scheduler.

NavDP's ``predict_pointgoal_action`` denoises a trajectory with a
``diffusers.DDPMScheduler`` configured as::

    DDPMScheduler(num_train_timesteps=10, beta_schedule='squaredcos_cap_v2',
                  clip_sample=True, prediction_type='epsilon')   # variance_type='fixed_small'

and runs the full ``num_train_timesteps`` inference steps (timesteps 9..0). The
TensorRT runtime only swaps the heavy transformer forward passes (encoder /
denoiser / critic) for engines; the scheduler math stays here in Python so the
behaviour is identical to the PyTorch reference.

This module re-implements only that scheduler, in float32 numpy, so that the
ROS-free ``core`` runtime needs neither ``torch`` nor ``diffusers`` at runtime
(the FALCON Noetic adapter imports ``core`` under Python 3.8). The cosine betas
are NOT recomputed here: the trained scheduler's ``alphas_cumprod`` is extracted
once at export time and injected, removing any drift from re-deriving betas.

Parity-critical details replicated from diffusers 0.33.1 ``DDPMScheduler.step``
(``prediction_type='epsilon'``, ``variance_type='fixed_small'``):
  * ``prev_t = t - 1`` (inference steps == train steps); ``alpha_prod_t_prev = 1``
    when ``prev_t < 0``.
  * epsilon -> x0, then ``clip_sample`` clamp to ``[-1, 1]`` is applied to the
    predicted ``x0`` BEFORE the posterior coefficients (load-bearing: ``abar``
    is ~2.4e-5 at the last step, a ~200x error amplifier).
  * posterior mean ``pred_orig_coeff * x0 + current_sample_coeff * x_t``.
  * variance ``(1-abar_prev)/(1-abar_t) * current_beta_t``, clamped to >= 1e-20,
    std = sqrt(variance); variance noise added only for ``t > 0``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_F32 = np.float32
_ONE = np.float32(1.0)


class NumpyDDPMScheduler:
    """Float32 numpy DDPM sampler driven by a precomputed ``alphas_cumprod``.

    Args:
        alphas_cumprod: 1-D array of cumulative products of ``alpha`` for the
            scheduler's training timesteps (length == ``num_train_timesteps``),
            taken verbatim from the trained ``DDPMScheduler``.
        clip_sample: whether to clamp the predicted ``x0`` (NavDP uses ``True``).
        clip_sample_range: symmetric clamp bound for ``x0`` (diffusers default 1.0).
    """

    def __init__(self, alphas_cumprod, clip_sample=True, clip_sample_range=1.0):
        self.alphas_cumprod = np.asarray(alphas_cumprod, dtype=_F32).reshape(-1)
        if self.alphas_cumprod.ndim != 1 or self.alphas_cumprod.size == 0:
            raise ValueError("alphas_cumprod must be a non-empty 1-D array")
        self.num_train_timesteps = int(self.alphas_cumprod.shape[0])
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)
        # NavDP calls set_timesteps(num_train_timesteps): inference == train steps,
        # iterated from the noisiest (T-1) down to 0.
        self.timesteps = np.arange(self.num_train_timesteps - 1, -1, -1, dtype=np.int64)

    def _alpha_prev(self, prev_t):
        """alpha_cumprod at the previous timestep, or 1.0 when prev_t < 0."""
        return self.alphas_cumprod[prev_t] if prev_t >= 0 else _ONE

    def variance_std(self, timestep):
        """Standard deviation of the posterior noise added at ``timestep``."""
        t = int(timestep)
        prev_t = t - 1
        a_t = self.alphas_cumprod[t]
        a_prev = self._alpha_prev(prev_t)
        current_beta_t = _ONE - a_t / a_prev
        variance = (_ONE - a_prev) / (_ONE - a_t) * current_beta_t
        variance = _F32(max(float(variance), 1e-20))
        return _F32(np.sqrt(variance))

    def step(self, model_output, timestep, sample, variance_noise=None):
        """One reverse diffusion step: x_t -> x_{t-1}.

        Args:
            model_output: predicted epsilon, shape ``(N, predict_size, 3)`` float.
            timestep: integer scheduler timestep ``t``.
            sample: current noisy action ``x_t``, same shape as ``model_output``.
            variance_noise: optional pre-sampled standard-normal noise with the
                same shape as ``model_output``. When ``None`` and ``t > 0`` a
                fresh ``np.random.randn`` draw is used (production). The harness
                injects identical noise into both this and the torch reference so
                the accuracy gate is meaningful.

        Returns:
            ``x_{t-1}`` as float32, same shape as ``sample``.
        """
        t = int(timestep)
        prev_t = t - 1
        mo = np.asarray(model_output, dtype=_F32)
        x = np.asarray(sample, dtype=_F32)

        a_t = self.alphas_cumprod[t]
        a_prev = self._alpha_prev(prev_t)
        beta_t = _ONE - a_t
        beta_prev = _ONE - a_prev
        current_alpha_t = a_t / a_prev
        current_beta_t = _ONE - current_alpha_t

        # epsilon -> predicted x0, then clip BEFORE the posterior coefficients.
        pred_x0 = (x - np.sqrt(beta_t) * mo) / np.sqrt(a_t)
        if self.clip_sample:
            pred_x0 = np.clip(pred_x0, -self.clip_sample_range, self.clip_sample_range)

        pred_orig_coeff = np.sqrt(a_prev) * current_beta_t / beta_t
        current_sample_coeff = np.sqrt(current_alpha_t) * beta_prev / beta_t
        prev_sample = pred_orig_coeff * pred_x0 + current_sample_coeff * x

        if t > 0:
            if variance_noise is None:
                variance_noise = np.random.randn(*mo.shape).astype(_F32)
            prev_sample = prev_sample + self.variance_std(t) * np.asarray(
                variance_noise, dtype=_F32)
        return prev_sample.astype(_F32, copy=False)
