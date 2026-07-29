"""Trajectory assembly + critic ranking parity (torch-free).

Re-implements the reference tail of ``predict_pointgoal_action`` independently
with plain numpy and asserts ``finalize_trajectories`` matches it, plus targeted
checks for the two order-sensitive behaviours (cumsum-after-/4 and the <0.5 XY
zeroing) and the critic ranking.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.vlas.navdp.trt.postprocess import (
    finalize_trajectories, rank_by_critic,
)


def _reference(naction, critic, sample_num):
    """Independent reference, mirroring policy_network.predict_pointgoal_action."""
    all_traj = np.cumsum(naction / 4.0, axis=1).reshape(1, sample_num, naction.shape[1], 3)
    length = np.linalg.norm(all_traj[:, :, -1, 0:2], axis=-1)
    keep = np.array([0, 0, 1.0], np.float32)
    for s in range(sample_num):
        if length[0, s] < 0.5:
            all_traj[0, s] = all_traj[0, s] * keep
    crit = critic.reshape(1, sample_num)
    pos = all_traj[[[0]], np.argsort(-crit, axis=1)[:, :2]]
    neg = all_traj[[[0]], np.argsort(crit, axis=1)[:, :2]]
    return all_traj, crit, pos, neg


def test_matches_reference_tail():
    rng = np.random.RandomState(3)
    n = 16
    naction = rng.randn(n, 24, 3).astype(np.float32)
    critic = rng.randn(n).astype(np.float32)

    a_traj, a_crit, a_pos, a_neg = finalize_trajectories(naction, critic, 1, n)
    r_traj, r_crit, r_pos, r_neg = _reference(naction.copy(), critic.copy(), n)

    np.testing.assert_allclose(a_traj, r_traj, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a_crit, r_crit, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a_pos, r_pos, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a_neg, r_neg, rtol=1e-6, atol=1e-6)


def test_cumsum_is_after_divide_by_four():
    naction = np.ones((2, 24, 3), np.float32)
    crit = np.zeros(2, np.float32)
    traj, _, _, _ = finalize_trajectories(naction, crit, 1, 2)
    # Each step contributes 1/4; the i-th cumulative point along Z is (i+1)/4.
    np.testing.assert_allclose(traj[0, 0, :, 2], (np.arange(24) + 1) / 4.0,
                               rtol=1e-6, atol=1e-6)


def test_short_trajectories_zero_xy_keep_z():
    # Construct one long and one short trajectory; only the short one's XY zeroes.
    naction = np.zeros((2, 24, 3), np.float32)
    naction[0, :, 0] = 1.0    # long: final X cumsum = 24/4 = 6 m  (> 0.5)
    naction[1, :, 0] = 0.01   # short: final X cumsum = 0.06 m     (< 0.5)
    naction[:, :, 2] = 0.4    # both have Z content
    crit = np.array([1.0, 0.0], np.float32)
    traj, _, _, _ = finalize_trajectories(naction, crit, 1, 2)
    assert np.all(traj[0, 0, :, 0] != 0.0)            # long X kept
    np.testing.assert_array_equal(traj[0, 1, :, 0:2], 0.0)   # short XY zeroed
    assert np.all(traj[0, 1, :, 2] != 0.0)            # short Z kept


def test_ranking_orders_by_critic():
    all_traj = np.zeros((1, 4, 2, 3), np.float32)
    for s in range(4):
        all_traj[0, s] = s                # tag each sample by its index value
    crit = np.array([[0.1, 0.9, 0.5, 0.2]], np.float32)
    pos, neg = rank_by_critic(all_traj, crit)
    assert pos[0, 0, 0, 0] == 1          # highest critic -> sample 1 first
    assert neg[0, 0, 0, 0] == 0          # lowest critic  -> sample 0 first
