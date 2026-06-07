"""Unit tests for the ROS-free dead-reckoning localization noise model."""
import math

import numpy as np

from sparx_agency.core.localization import se3
from sparx_agency.core.localization.dead_reckoning_noise import (
    AXES,
    DeadReckoningNoiseModel,
    DeadReckoningNoiseParams,
)


def _pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    """World->body 4x4 from a planar (x, y, z, yaw)."""
    q = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    return se3.make_transform((x, y, z), q)


def test_params_enabled_flag():
    assert DeadReckoningNoiseParams().enabled() is False
    p = DeadReckoningNoiseParams()
    p.drift_mean_per_motion["x"] = 0.1
    assert p.enabled() is True


def test_clean_run_is_exact_passthrough():
    model = DeadReckoningNoiseModel(DeadReckoningNoiseParams(),
                                    np.random.RandomState(0))
    # First call returns truth.
    out = model.step(_pose(0, 0), 0.1)
    np.testing.assert_allclose(out, _pose(0, 0), atol=1e-12)
    # Subsequent ticks with zero noise stay glued to truth.
    for i in range(1, 20):
        gt = _pose(0.3 * i, 0.0, 0.0, 0.05 * i)
        out = model.step(gt, 0.1)
        np.testing.assert_allclose(out, gt, atol=1e-9)


def test_scale_factor_drift_overshoots_forward_motion():
    # +10% forward scale error, no randomness -> belief runs ahead of truth.
    p = DeadReckoningNoiseParams()
    p.drift_mean_per_motion["x"] = 0.1
    model = DeadReckoningNoiseModel(p, np.random.RandomState(0))
    model.step(_pose(0, 0), 0.1)
    out = model.step(_pose(1.0, 0.0), 0.1)   # moved +1 m forward in body x
    assert out[0, 3] > 1.0                    # overshot
    assert abs(out[0, 3] - 1.1) < 1e-9        # exactly +10%


def test_yaw_drift_bends_subsequent_forward_motion():
    # A yaw scale error makes the belief heading wrong, so a later straight
    # leg lands off the true +x axis (the classic dead-reckoning failure).
    p = DeadReckoningNoiseParams()
    p.drift_mean_per_motion["yaw"] = 0.2     # 20% extra yaw per rad turned
    model = DeadReckoningNoiseModel(p, np.random.RandomState(0))
    model.step(_pose(0, 0, 0.0, 0.0), 0.1)
    model.step(_pose(0, 0, 0.0, math.radians(90)), 0.1)   # yaw in place 90deg
    out = model.step(_pose(1.0, 0.0, 0.0, math.radians(90)), 0.1)  # +1 m fwd
    # Truth is back at x=1,y=0; the wrong heading pushes the belief off-axis.
    assert abs(out[1, 3]) > 0.05


def test_seeded_runs_are_deterministic():
    def run():
        p = DeadReckoningNoiseParams()
        p.drift_std_per_motion["x"] = 0.05
        p.bias_per_s_std["yaw"] = 0.01
        m = DeadReckoningNoiseModel(p, np.random.RandomState(42))
        m.step(_pose(0, 0), 0.1)
        outs = [m.step(_pose(0.2 * i, 0.0, 0.0, 0.0), 0.1) for i in range(1, 10)]
        return np.array(outs)
    np.testing.assert_array_equal(run(), run())


def test_time_bias_wanders_while_stationary():
    # Always-on bias makes the belief move even though truth never changes.
    p = DeadReckoningNoiseParams()
    p.bias_per_s_mean["x"] = 0.1             # 0.1 m/s constant bias
    model = DeadReckoningNoiseModel(p, np.random.RandomState(0))
    model.step(_pose(0, 0), 0.1)
    for _ in range(10):
        out = model.step(_pose(0, 0), 0.1)   # hovering perfectly still
    assert out[0, 3] > 0.05                   # belief drifted forward


def test_reset_reanchors_to_truth():
    p = DeadReckoningNoiseParams()
    p.drift_mean_per_motion["x"] = 0.1
    model = DeadReckoningNoiseModel(p, np.random.RandomState(0))
    model.step(_pose(0, 0), 0.1)
    model.step(_pose(1.0, 0.0), 0.1)
    model.reset()
    out = model.step(_pose(5.0, 0.0), 0.1)    # first post-reset == truth
    np.testing.assert_allclose(out, _pose(5.0, 0.0), atol=1e-9)


def test_axes_constant():
    assert AXES == ("x", "y", "z", "yaw")
