"""Deterministic flow-matching Euler integrator (pure numpy).

FlowNav produces an action trajectory by integrating a learned velocity field
``v(t, x)`` (the ``noise_pred_net`` / ``vfield`` engine) from ``t=0`` to ``t=1``
with an explicit fixed-step Euler ODE solver. The reference deployment uses::

    torchdiffeq.odeint(lambda t, x: net(x, t, cond),
                        x0, torch.linspace(0, 1, k_steps), method="euler")

``torchdiffeq``'s fixed-grid ``euler`` evaluates the field at the *left* endpoint
of each interval and takes ``x <- x + dt * v(t_i, x_i)``; the returned trajectory
is sampled at every grid point and only ``traj[-1]`` is used. This module
reproduces exactly that update in float32 numpy so the ROS-free ``core`` runtime
needs neither ``torch`` nor ``torchdiffeq`` at runtime.

The integration is fully **deterministic**: the only randomness is the initial
state ``x0 ~ N(0, I)`` drawn once before the loop (and injectable for the
accuracy gate). There is no per-step noise (unlike NavDP's DDPM). ``num_steps``
is the "K" -- flow matching stays accurate at low K (a handful of steps), which
is its main throughput advantage over diffusion, so K is the dominant speed lever.
"""
from __future__ import annotations

import numpy as np

_F32 = np.float32


class FlowMatchEulerScheduler:
    """Fixed-step explicit-Euler integrator over a uniform ``[0, 1]`` time grid.

    Args:
        num_steps: number of grid points ``K`` in ``linspace(0, 1, K)``. The
            velocity field is evaluated ``K - 1`` times (one per Euler step).
            Must be >= 2.

    Attributes:
        timesteps: ``(K,)`` float32 array of grid times; ``timesteps[i]`` is the
            time fed to the field at Euler step ``i`` (``i`` in ``range(K - 1)``).
    """

    def __init__(self, num_steps):
        self.num_steps = int(num_steps)
        if self.num_steps < 2:
            raise ValueError("num_steps (K) must be >= 2, got %d" % self.num_steps)
        self.timesteps = np.linspace(0.0, 1.0, self.num_steps, dtype=_F32)

    @property
    def num_field_evals(self):
        """Number of velocity-field evaluations per trajectory (``K - 1``)."""
        return self.num_steps - 1

    def step(self, vfield, index, sample):
        """One Euler step from ``timesteps[index]`` to ``timesteps[index + 1]``.

        Args:
            vfield: velocity ``v(t_i, x_i)`` from the field engine, shape
                ``(N, horizon, action_dim)`` float.
            index: Euler step index ``i`` in ``range(num_steps - 1)``.
            sample: current state ``x_i``, same shape as ``vfield``.

        Returns:
            ``x_{i+1} = x_i + (t_{i+1} - t_i) * vfield`` as float32.
        """
        i = int(index)
        if i < 0 or i >= self.num_steps - 1:
            raise ValueError("Euler step index %d out of range [0, %d)"
                             % (i, self.num_steps - 1))
        dt = _F32(self.timesteps[i + 1] - self.timesteps[i])
        x = np.asarray(sample, dtype=_F32)
        v = np.asarray(vfield, dtype=_F32)
        return (x + dt * v).astype(_F32, copy=False)
