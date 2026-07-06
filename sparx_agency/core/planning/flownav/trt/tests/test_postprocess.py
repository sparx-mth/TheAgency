"""Parity tests for FlowNav action de-normalization.

``get_action`` must match ``flownav.training.utils`` exactly: de-normalize the
``[-1, 1]`` deltas with ``(d + 1) / 2 * (max - min) + min`` and cumulatively sum
them along the horizon. A small independent reference recomputes the expected
result.
"""
import numpy as np

from sparx_agency.core.planning.flownav.trt.postprocess import chosen_waypoint, get_action

# FlowNav data_config.yaml action_stats.
ACTION_MIN = np.array([-2.5, -4.0], dtype=np.float32)
ACTION_MAX = np.array([5.0, 4.0], dtype=np.float32)


def _reference(naction, lo, hi):
    ndata = (naction + 1.0) / 2.0
    deltas = ndata * (hi - lo) + lo
    return np.cumsum(deltas, axis=1)


def test_get_action_matches_reference():
    rng = np.random.RandomState(0)
    naction = rng.uniform(-1, 1, size=(8, 8, 2)).astype(np.float32)
    got = get_action(naction, ACTION_MIN, ACTION_MAX)
    exp = _reference(naction, ACTION_MIN, ACTION_MAX)
    np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-6)
    assert got.shape == (8, 8, 2)


def test_zero_delta_maps_to_midpoint_cumsum():
    # delta == 0 (normalized) -> de-normalized midpoint (lo+hi)/2 per step;
    # cumsum gives k * midpoint at waypoint k.
    naction = np.zeros((1, 3, 2), dtype=np.float32)
    got = get_action(naction, ACTION_MIN, ACTION_MAX)
    mid = (ACTION_MIN + ACTION_MAX) / 2.0
    np.testing.assert_allclose(got[0, 0], mid, rtol=1e-6)
    np.testing.assert_allclose(got[0, 2], 3 * mid, rtol=1e-6)


def test_chosen_waypoint_selects_sample_zero():
    actions = np.arange(2 * 8 * 2, dtype=np.float32).reshape(2, 8, 2)
    wp = chosen_waypoint(actions, waypoint_index=2, sample_index=0)
    np.testing.assert_array_equal(wp, actions[0, 2])
