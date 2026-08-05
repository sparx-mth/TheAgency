"""Parity tests for the deterministic flow-matching Euler scheduler.

The numpy integrator must reproduce ``torchdiffeq.odeint(..., method="euler")``
on a uniform ``linspace(0, 1, K)`` grid: each step evaluates the field at the
left endpoint and takes ``x <- x + dt * v``. These tests use closed-form fields
(no torch needed) where the exact Euler result is known.
"""
import numpy as np
import pytest

from sparx_agency.core.planning.vlas.flownav.trt.scheduler import FlowMatchEulerScheduler


def test_grid_and_eval_count():
    s = FlowMatchEulerScheduler(num_steps=4)
    assert s.num_steps == 4
    assert s.num_field_evals == 3
    np.testing.assert_allclose(s.timesteps, [0.0, 1.0 / 3, 2.0 / 3, 1.0], rtol=1e-6)


def test_requires_at_least_two_steps():
    with pytest.raises(ValueError):
        FlowMatchEulerScheduler(num_steps=1)


def test_constant_field_integrates_exactly():
    # dx/dt = c (constant) -> x(1) = x0 + c. Euler is exact for a constant field
    # regardless of K.
    s = FlowMatchEulerScheduler(num_steps=5)
    x = np.zeros((2, 3, 2), dtype=np.float32)
    c = np.full_like(x, 0.7)
    for i in range(s.num_field_evals):
        x = s.step(c, i, x)
    np.testing.assert_allclose(x, 0.7, rtol=1e-5)


def test_matches_manual_euler_for_time_dependent_field():
    # dx/dt = t  -> manual explicit Euler on the same grid (NOT the analytic 0.5,
    # since Euler under-integrates) must match step-for-step.
    s = FlowMatchEulerScheduler(num_steps=4)
    x = np.zeros((1, 1, 1), dtype=np.float32)
    manual = 0.0
    t = s.timesteps
    for i in range(s.num_field_evals):
        v = np.full_like(x, t[i])           # field value v(t_i) = t_i
        x = s.step(v, i, x)
        manual += (t[i + 1] - t[i]) * t[i]
    np.testing.assert_allclose(x.reshape(()), manual, rtol=1e-5)


def test_step_index_out_of_range():
    s = FlowMatchEulerScheduler(num_steps=3)
    x = np.zeros((1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        s.step(x, 2, x)            # only indices 0,1 are valid (K-1 == 2 steps)
