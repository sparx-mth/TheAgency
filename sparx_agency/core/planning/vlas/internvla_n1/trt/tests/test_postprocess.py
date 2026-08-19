"""The decision function: candidate trajectories in, executed actions out."""
import numpy as np
import pytest

from sparx_agency.core.planning.vlas.internvla_n1.trt import postprocess


def _deltas(dx, dy, samples=32, steps=32):
    """Constant per-step deltas for every candidate."""
    out = np.zeros((samples, steps, 3), dtype=np.float32)
    out[:, :, 0] = dx
    out[:, :, 1] = dy
    return out


def test_delta_scale_is_applied_before_integration():
    """dx/dy are divided by 4; dropping that scales every distance by 4."""
    path = postprocess.mean_path(_deltas(0.4, 0.0, samples=1, steps=3))
    assert np.allclose(path[:, 0], [0.0, 0.1, 0.2, 0.3])


def test_candidates_are_averaged_after_integration_not_before():
    """The mean of the paths, not the path of the mean deltas.

    They coincide for linear integration, so the discriminating case is a
    candidate set whose per-step signs differ: averaging deltas first cancels
    them, averaging paths keeps each candidate's excursion in the mean.
    """
    deltas = np.zeros((2, 2, 3), dtype=np.float32)
    deltas[0, :, 0] = [0.4, -0.4]
    deltas[1, :, 0] = [-0.4, 0.4]
    path = postprocess.mean_path(deltas)
    assert np.allclose(path[:, 0], [0.0, 0.0, 0.0])
    assert path.shape == (3, 2)


def test_mean_path_starts_at_the_origin_and_is_one_longer():
    path = postprocess.mean_path(_deltas(0.3, 0.1, steps=32))
    assert path.shape == (33, 2)
    assert np.allclose(path[0], [0.0, 0.0])


def test_mean_path_rejects_a_two_dimensional_input():
    with pytest.raises(ValueError, match=r"\(B, T, >=2\)"):
        postprocess.mean_path(np.zeros((32, 3)))


def test_straight_ahead_is_all_forward():
    assert postprocess.action_queue(_deltas(0.4, 0.0)) == [1, 1, 1, 1]


def test_veering_left_turns_left_before_advancing():
    """A path angled left must produce TURN_LEFT (2) before FORWARD (1)."""
    actions = postprocess.action_queue(_deltas(0.3, 0.3))
    assert actions[0] == postprocess.TURN_LEFT
    assert postprocess.FORWARD in actions


def test_veering_right_turns_right():
    actions = postprocess.action_queue(_deltas(0.3, -0.3))
    assert actions[0] == postprocess.TURN_RIGHT


def test_queue_is_capped_at_four_like_the_agent():
    """``S1Output(idx=action_list[:4])`` -- the agent never queues more."""
    assert len(postprocess.action_queue(_deltas(0.4, 0.0))) <= postprocess.ACTIONS_KEPT


def test_a_stationary_prediction_yields_no_actions():
    """Zero deltas put the goal at the start, inside the tolerance."""
    assert postprocess.action_queue(np.zeros((32, 32, 3), dtype=np.float32)) == []


def test_the_walk_is_bounded_even_when_the_path_never_approaches_its_end():
    """Upstream's while loop has no cap; a degenerate path must not hang."""
    rng = np.random.default_rng(0)
    deltas = rng.standard_normal((32, 32, 3)).astype(np.float32) * 8.0
    actions = postprocess.discrete_actions(postprocess.mean_path(deltas),
                                           max_actions=16)
    assert len(actions) <= 16
