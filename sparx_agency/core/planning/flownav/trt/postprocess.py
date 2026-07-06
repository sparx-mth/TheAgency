"""Action de-normalization for FlowNav (numpy parity with the reference).

This is the tail of FlowNav inference after the Euler flow-matching loop. The
``vfield`` engine integrates to a normalized **delta** trajectory in ``[-1, 1]``;
``get_action`` reverses the training normalization and integrates the deltas into
absolute robot-frame waypoints. It mirrors ``flownav.training.utils`` exactly::

    ndata  = (naction + 1) / 2
    deltas = ndata * (max - min) + min          # unnormalize_data
    actions = np.cumsum(deltas, axis=1)          # get_action

Unlike NavDP there is **no critic ranking** of the sampled trajectories: FlowNav
executes a single sample (index 0 by default) and the ``dist_pred_net`` head is
used only for topomap-node localization, not for choosing among action samples.
The ``num_samples`` (N) candidates differ only by their initial noise.
"""
from __future__ import annotations

import numpy as np

_F32 = np.float32


def get_action(naction, action_min, action_max):
    """De-normalize and integrate normalized action deltas into waypoints.

    Args:
        naction: ``(N, horizon, action_dim)`` normalized deltas in ``[-1, 1]``
            (the integrated Euler output of the velocity-field engine).
        action_min: per-dim minimum used at training time, shape ``(action_dim,)``.
        action_max: per-dim maximum used at training time, shape ``(action_dim,)``.

    Returns:
        ``(N, horizon, action_dim)`` absolute waypoints (cumulative sum of the
        de-normalized deltas) as float32.
    """
    nd = np.asarray(naction, dtype=_F32)
    lo = np.asarray(action_min, dtype=_F32).reshape(-1)
    hi = np.asarray(action_max, dtype=_F32).reshape(-1)
    ndata = (nd + _F32(1.0)) / _F32(2.0)
    deltas = ndata * (hi - lo) + lo
    return np.cumsum(deltas, axis=1).astype(_F32, copy=False)


def chosen_waypoint(actions, waypoint_index, sample_index=0):
    """Select the executed waypoint, mirroring the reference navigate loop.

    The reference takes ``naction[0][args.waypoint]`` -- sample 0, a fixed
    waypoint index along the horizon.

    Args:
        actions: ``(N, horizon, action_dim)`` absolute waypoints from
            :func:`get_action`.
        waypoint_index: index along the predicted horizon (reference default 2).
        sample_index: which of the N samples to execute (reference uses 0).

    Returns:
        ``(action_dim,)`` float32 waypoint.
    """
    a = np.asarray(actions, dtype=_F32)
    return a[int(sample_index), int(waypoint_index)].astype(_F32, copy=False)
