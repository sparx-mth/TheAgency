"""Numpy re-implementation of NavDP's point-goal encoder.

NavDP's ``point_encoder`` is a single ``nn.Linear(3, token_dim)`` that maps a
body-frame point goal ``(forward, left, 0)`` to a conditioning token. It is far
too small to be worth a TensorRT engine of its own, so the runtime keeps it in
numpy and feeds its output straight into the denoiser engine. The weight and
bias are exported once (``dump_head_params``) alongside the engines.

The upstream ``process_pointgoal`` clipping (clip to [-10, 10], forward to
[0, 10]) is applied on the host by ``NavDP_Agent.step_pointgoal`` before the goal
reaches the policy, so this module only performs the affine map.
"""
from __future__ import annotations

import numpy as np

_F32 = np.float32


class NavDPPointEncoder:
    """Affine map ``goal (B, 3) -> token (B, token_dim)`` (NavDP ``point_encoder``).

    Args:
        weight: linear weight of shape ``(token_dim, 3)``.
        bias: linear bias of shape ``(token_dim,)``.
    """

    def __init__(self, weight, bias):
        self.weight = np.asarray(weight, dtype=_F32)
        self.bias = np.asarray(bias, dtype=_F32)
        if self.weight.ndim != 2 or self.weight.shape[1] != 3:
            raise ValueError(
                "point_encoder weight must be (token_dim, 3), got %r"
                % (self.weight.shape,))
        if self.bias.shape != (self.weight.shape[0],):
            raise ValueError(
                "point_encoder bias must be (token_dim,), got %r"
                % (self.bias.shape,))

    @property
    def token_dim(self):
        """Output embedding dimension."""
        return int(self.weight.shape[0])

    def __call__(self, goal_point):
        """Encode goals.

        Args:
            goal_point: array of shape ``(B, 3)`` (forward, left, 0), already
                range-clipped by the host.

        Returns:
            ``(B, token_dim)`` float32 conditioning embedding.
        """
        g = np.asarray(goal_point, dtype=_F32)
        if g.ndim != 2 or g.shape[1] != 3:
            raise ValueError("goal_point must be (B, 3), got %r" % (g.shape,))
        return g @ self.weight.T + self.bias
