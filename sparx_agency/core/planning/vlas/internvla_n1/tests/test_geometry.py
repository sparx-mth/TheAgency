"""Body-frame trajectory shaping from InternVLA-N1's output."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.vlas.internvla_n1 import geometry


def test_heading_column_points_each_vertex_along_its_next_step():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    out = geometry.heading_column(xy)
    assert out.shape == (3, 3)
    # first step runs +x (yaw 0), second runs +y (yaw +pi/2)
    assert out[0, 2] == pytest.approx(0.0)
    assert out[1, 2] == pytest.approx(np.pi / 2)
    # the last vertex has no next step, so it holds the previous heading
    assert out[2, 2] == pytest.approx(out[1, 2])


def test_heading_column_carries_heading_through_a_degenerate_step():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # third == second
    out = geometry.heading_column(xy)
    # the zero-length middle->last step must not snap the heading east
    assert out[1, 2] == pytest.approx(0.0)


def test_heading_column_rejects_bad_shape():
    with pytest.raises(ValueError):
        geometry.heading_column(np.zeros((3,)))


def test_trajectory_from_deltas_starts_at_the_origin_and_carries_yaw():
    deltas = np.random.default_rng(1).standard_normal((32, 32, 3)).astype("float32")
    traj = geometry.trajectory_from_deltas(deltas)
    assert traj.shape == (33, 3)
    assert traj[0, 0] == pytest.approx(0.0)
    assert traj[0, 1] == pytest.approx(0.0)


def test_trajectory_from_action_forward_advances_straight():
    traj = geometry.trajectory_from_action(geometry.FORWARD, step_m=0.25)
    assert traj.shape == (2, 3)
    assert traj[-1, 0] == pytest.approx(0.25)
    assert traj[-1, 1] == pytest.approx(0.0)
    assert traj[-1, 2] == pytest.approx(0.0)


def test_trajectory_from_action_turns_bend_left_positive():
    left = geometry.trajectory_from_action(geometry.TURN_LEFT, step_m=0.25, turn_deg=15.0)
    right = geometry.trajectory_from_action(geometry.TURN_RIGHT, step_m=0.25, turn_deg=15.0)
    assert left[-1, 1] > 0.0 and left[-1, 2] == pytest.approx(np.deg2rad(15.0))
    assert right[-1, 1] < 0.0 and right[-1, 2] == pytest.approx(-np.deg2rad(15.0))


def test_trajectory_from_action_stop_is_none():
    assert geometry.trajectory_from_action(geometry.STOP) is None
    assert geometry.trajectory_from_action(99) is None


def test_trajectory_from_path_takes_the_first_batch_and_keeps_a_yaw_column():
    batched = np.array([[[0.0, 0.0, 0.1], [1.0, 0.0, 0.2]]])  # (1, 2, 3)
    out = geometry.trajectory_from_path(batched)
    assert out.shape == (2, 3)
    assert out[0, 2] == pytest.approx(0.1)  # explicit yaw preserved


def test_trajectory_from_response_prefers_the_continuous_curve():
    raw = {"action": [{"action": [1]}],
           "trajectory": [[[0.0, 0.0], [0.25, 0.0], [0.5, 0.2]]]}
    out = geometry.trajectory_from_response(raw)
    assert out is not None and out.shape == (3, 3)


def test_trajectory_from_response_is_none_when_only_an_action_is_present():
    assert geometry.trajectory_from_response({"action": [{"action": [2]}]}) is None
    assert geometry.trajectory_from_response("not a dict") is None


def test_trajectory_from_response_reads_the_patched_internnav_server_shape():
    # The exact response the trajectory-patched InternNav agent/server returns:
    # the continuous curve rides inside action[0], next to the discrete action
    # and the pixel goal.
    raw = {
        "action": [{"action": [1], "ideal_flag": True,
                    "pixel_goal": [240, 320],
                    "trajectory": [[0.0, 0.0], [0.24, 0.03], [0.48, 0.09], [0.71, 0.20]]}],
        "pixel_goal": [240, 320],
        "pixel_goal_step": 12,
    }
    out = geometry.trajectory_from_response(raw)
    assert out is not None and out.shape == (4, 3)
    assert out[0, 0] == pytest.approx(0.0) and out[-1, 0] == pytest.approx(0.71)


def test_trajectory_from_response_none_trajectory_field_falls_back():
    # A pure-S2 discrete step: trajectory is null, the client must fall back.
    raw = {"action": [{"action": [2], "trajectory": None}], "pixel_goal": None}
    assert geometry.trajectory_from_response(raw) is None


