"""Unit tests for the ROS-free depth noise model."""
import numpy as np

from sparx_agency.core.mapping.depth_noise import DepthNoiseParams, add_depth_noise


def test_disabled_is_passthrough_as_float32():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    out = add_depth_noise(depth, DepthNoiseParams())
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, depth.astype(np.float32))


def test_additive_noise_changes_valid_pixels_only():
    depth = np.array([[2.0, 0.0], [np.inf, 5.0]], dtype=np.float32)
    out = add_depth_noise(depth, DepthNoiseParams(std=0.1),
                          np.random.RandomState(0))
    # Invalid pixels (0, inf) are untouched.
    assert out[0, 1] == 0.0
    assert np.isinf(out[1, 0])
    # The two valid pixels moved.
    assert out[0, 0] != 2.0 and out[1, 1] != 5.0


def test_result_is_non_negative():
    depth = np.full((8, 8), 0.05, dtype=np.float32)
    out = add_depth_noise(depth, DepthNoiseParams(std=10.0),
                          np.random.RandomState(1))
    assert (out >= 0.0).all()


def test_proportional_scales_with_depth():
    # With a fixed seed the same scale factor hits every pixel; deep pixels
    # move more in absolute terms than shallow ones.
    depth = np.array([[1.0, 100.0]], dtype=np.float32)
    out = add_depth_noise(depth, DepthNoiseParams(proportional=0.05),
                          np.random.RandomState(2))
    assert abs(out[0, 1] - 100.0) > abs(out[0, 0] - 1.0)


def test_seeded_runs_are_deterministic():
    depth = np.linspace(0.1, 9.0, 64, dtype=np.float32).reshape(8, 8)
    p = DepthNoiseParams(std=0.05, proportional=0.02)
    a = add_depth_noise(depth, p, np.random.RandomState(7))
    b = add_depth_noise(depth, p, np.random.RandomState(7))
    np.testing.assert_array_equal(a, b)


def test_does_not_mutate_input():
    depth = np.ones((4, 4), dtype=np.float32)
    add_depth_noise(depth, DepthNoiseParams(std=1.0), np.random.RandomState(0))
    np.testing.assert_array_equal(depth, np.ones((4, 4), dtype=np.float32))
