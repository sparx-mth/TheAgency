"""Trajectory assembly + critic ranking for NavDP point-goal (numpy parity).

This is the tail of ``NavDP_Policy.predict_pointgoal_action`` after the diffusion
loop and critic, kept in numpy so it is identical to the PyTorch reference and
stays out of the TensorRT engines (the engines return only raw ``epsilon`` and
critic scalars). The exact, order-sensitive steps replicated:

  1. ``all_trajectory = cumsum(naction / 4.0, axis=step)`` -- the divide by 4 is
     applied BEFORE the cumulative sum along the predict-horizon axis.
  2. reshape to ``(B, sample_num, predict_size, 3)``.
  3. zero the X/Y of any sample whose final-waypoint XY norm < 0.5 m, keeping Z
     (``traj *= [0, 0, 1]``) -- a "stay roughly in place" shaping.
  4. rank the ``sample_num`` candidates by critic value: the top-2 highest are
     the positive trajectories (index 0 is the executed one), the bottom-2 the
     negative trajectories.

Returns the same 4-tuple as the reference so ``NavDP_Agent.step_pointgoal`` is
unchanged: ``(all_trajectory, critic_values, positive_trajectory,
negative_trajectory)``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

_F32 = np.float32
# Zero X/Y but keep Z for "in place" samples; matches the reference [[[0,0,1.0]]].
_KEEP_Z = np.array([0.0, 0.0, 1.0], dtype=_F32)


def finalize_trajectories(naction, critic_values, batch_size, sample_num):
    # type: (np.ndarray, np.ndarray, int, int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """Integrate, shape, zero-short, and rank the denoised actions.

    Args:
        naction: denoised actions ``(batch_size * sample_num, predict_size, 3)``.
        critic_values: critic scores ``(batch_size * sample_num,)`` or
            ``(batch_size, sample_num)``.
        batch_size: number of goals/drones (1 for a single drone).
        sample_num: number of diffusion samples per goal.

    Returns:
        Tuple ``(all_trajectory, critic_values, positive_trajectory,
        negative_trajectory)`` where ``all_trajectory`` is
        ``(B, sample_num, predict_size, 3)``, ``critic_values`` is
        ``(B, sample_num)``, and the positive/negative trajectories are
        ``(B, 2, predict_size, 3)`` (best/worst two by critic value).
    """
    naction = np.asarray(naction, dtype=_F32)
    predict_size = naction.shape[1]

    all_traj = np.cumsum(naction / _F32(4.0), axis=1)
    all_traj = all_traj.reshape(batch_size, sample_num, predict_size, 3)

    final_xy = all_traj[:, :, -1, 0:2]
    length = np.linalg.norm(final_xy, axis=-1)          # (B, sample_num)
    short = length < 0.5
    all_traj[short] = all_traj[short] * _KEEP_Z          # broadcast (k,P,3)*(3,)

    crit = np.asarray(critic_values, dtype=_F32).reshape(batch_size, sample_num)
    positive, negative = rank_by_critic(all_traj, crit)
    return all_traj, crit, positive, negative


def rank_by_critic(all_trajectory, critic_values):
    # type: (np.ndarray, np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    """Select the best-2 and worst-2 trajectories per batch by critic value.

    Mirrors the reference ``(-critic).argsort()[:, :2]`` for positives and
    ``(critic).argsort()[:, :2]`` for negatives. Index 0 of the positives is the
    executed trajectory.

    Args:
        all_trajectory: ``(B, sample_num, predict_size, 3)``.
        critic_values: ``(B, sample_num)``.

    Returns:
        ``(positive (B, 2, P, 3), negative (B, 2, P, 3))``.
    """
    batch = critic_values.shape[0]
    pos_idx = np.argsort(-critic_values, axis=1)[:, 0:2]
    neg_idx = np.argsort(critic_values, axis=1)[:, 0:2]
    batch_idx = np.arange(batch)[:, None]
    positive = all_trajectory[batch_idx, pos_idx]
    negative = all_trajectory[batch_idx, neg_idx]
    return positive, negative
