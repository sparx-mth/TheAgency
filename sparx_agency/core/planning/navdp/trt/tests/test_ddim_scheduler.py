"""Unit tests for the numpy DDIM sampler (timestep subsampling + reverse step)."""
import numpy as np
import pytest

from sparx_agency.core.planning.navdp.trt.ddim_scheduler import NumpyDDIMScheduler

# Synthetic decreasing alphas_cumprod (index 0 = least noise, T-1 = most).
ACP = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], np.float32)


def test_timesteps_subsample_high_to_low():
    assert list(NumpyDDIMScheduler(ACP, 4).timesteps) == [9, 6, 3, 0]


def test_full_steps_equal_all_trained_timesteps():
    assert list(NumpyDDIMScheduler(ACP, 10).timesteps) == list(range(9, -1, -1))


def test_prev_map_last_lands_on_clean_x0():
    assert NumpyDDIMScheduler(ACP, 4)._prev == {9: 6, 6: 3, 3: 0, 0: -1}


def test_out_of_range_steps_raise():
    with pytest.raises(ValueError):
        NumpyDDIMScheduler(ACP, 0)
    with pytest.raises(ValueError):
        NumpyDDIMScheduler(ACP, 11)


def test_step_preserves_shape_and_dtype():
    out = NumpyDDIMScheduler(ACP, 4).step(
        np.zeros((16, 24, 3), np.float32), 9, np.zeros((16, 24, 3), np.float32))
    assert out.shape == (16, 24, 3) and out.dtype == np.float32


def test_step_matches_hand_computed_mid_step():
    # t=6 (prev=3): abar_t=0.3, abar_prev=0.6; x=0.4, eps=0.1
    # pred_x0=(0.4 - sqrt(0.7)*0.1)/sqrt(0.3)=0.577547
    # x_prev=sqrt(0.6)*0.577547 + sqrt(0.4)*0.1 = 0.510597
    out = NumpyDDIMScheduler(ACP, 4).step(
        np.full((2, 1, 3), 0.1, np.float32), 6, np.full((2, 1, 3), 0.4, np.float32))
    assert np.allclose(out, 0.510597, atol=1e-4)


def test_final_step_returns_clipped_x0():
    # t=0 (prev=-1, abar_prev=1), eps=0 -> x_prev = clip(x / sqrt(abar_0))
    out = NumpyDDIMScheduler(ACP, 4).step(
        np.zeros((2, 1, 3), np.float32), 0, np.full((2, 1, 3), 0.4, np.float32))
    assert np.allclose(out, 0.4 / np.sqrt(0.9), atol=1e-4)


def test_clip_sample_bounds_x0():
    # large x -> pred_x0 > 1 -> clipped to clip_sample_range at the final step
    out = NumpyDDIMScheduler(ACP, 4, clip_sample=True, clip_sample_range=1.0).step(
        np.zeros((1, 1, 3), np.float32), 0, np.full((1, 1, 3), 5.0, np.float32))
    assert np.allclose(out, 1.0, atol=1e-5)
