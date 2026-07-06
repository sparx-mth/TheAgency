"""Tests for the server-side image preprocessing (transform_images parity)."""
import numpy as np

from sparx_agency.tasks.planning.flownav.server import preprocess


def test_preprocess_frame_shape_and_normalization():
    rgb = np.full((20, 30, 3), 255, dtype=np.uint8)      # white -> (1-mean)/std
    chw = preprocess.preprocess_frame(rgb, 96)
    assert chw.shape == (3, 96, 96)
    assert chw.dtype == np.float32
    expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(chw[:, 0, 0], expected, rtol=1e-4)


def test_build_obs_stack_shape_and_order():
    # 4 frames -> 12 channels; current (last) frame occupies the last 3 channels.
    frames = [np.full((48, 48, 3), v, np.uint8) for v in (10, 20, 30, 40)]
    obs = preprocess.build_obs_stack(frames, 96, 4)
    assert obs.shape == (1, 12, 96, 96)
    # last frame (value 40) is last in the stack
    last3 = obs[0, 9:12]
    ref = preprocess.preprocess_frame(frames[-1], 96)
    np.testing.assert_allclose(last3, ref, rtol=1e-5)


def test_build_obs_stack_left_pads_when_short():
    one = np.full((48, 48, 3), 77, np.uint8)
    obs = preprocess.build_obs_stack([one], 96, 4)        # pad oldest to fill
    assert obs.shape == (1, 12, 96, 96)
    # all four padded slots equal the single frame
    ref = preprocess.preprocess_frame(one, 96)
    for k in range(4):
        np.testing.assert_allclose(obs[0, 3 * k:3 * k + 3], ref, rtol=1e-5)


def test_build_goal_shape():
    goal = np.zeros((64, 64, 3), np.uint8)
    g = preprocess.build_goal(goal, 96)
    assert g.shape == (1, 3, 96, 96)
