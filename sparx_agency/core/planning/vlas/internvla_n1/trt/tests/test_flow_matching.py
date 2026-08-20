"""The numpy flow-matching schedule must equal the diffusers one it replaces."""
import numpy as np
import pytest

from sparx_agency.core.planning.vlas.internvla_n1.trt import flow_matching


def test_sigmas_match_upstream_linspace():
    """``generate_traj`` builds linspace(1.0, 1/steps, steps) plus a terminal 0."""
    for steps in (4, 10, 20):
        expected = list(np.linspace(1.0, 1.0 / steps, steps)) + [0.0]
        assert np.allclose(flow_matching.sigmas(steps), expected, atol=1e-12)


def test_ten_step_ladder_is_the_deployed_one():
    """The shipped configuration is 10 steps; pin its exact values."""
    assert np.allclose(
        flow_matching.sigmas(10),
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], atol=1e-12)


def test_timesteps_are_sigma_times_num_train_timesteps():
    """The network is conditioned on sigma * 1000, not on the step index."""
    steps = [t for _, _, t in flow_matching.schedule(10)]
    assert np.allclose(steps, [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100])


def test_schedule_yields_one_triple_per_step_and_ends_at_zero():
    triples = list(flow_matching.schedule(7))
    assert len(triples) == 7
    assert triples[-1][1] == 0.0
    for (_, next_sigma, _), (sigma, _, _) in zip(triples, triples[1:]):
        assert next_sigma == sigma


def test_euler_step_is_the_diffusers_update():
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((4, 32, 3)).astype(np.float32)
    velocity = rng.standard_normal((4, 32, 3)).astype(np.float32)
    got = flow_matching.euler_step(sample, velocity, 0.9, 0.8)
    assert np.allclose(got, sample + np.float32(-0.1) * velocity, atol=0)


def test_euler_step_returns_float32_from_any_input_dtype():
    """diffusers upcasts before the update; the dtype must not depend on input."""
    sample = np.zeros((2, 32, 3), dtype=np.float64)
    velocity = np.ones((2, 32, 3), dtype=np.float16)
    assert flow_matching.euler_step(sample, velocity, 1.0, 0.9).dtype == np.float32


def test_euler_step_rejects_a_shape_mismatch():
    """Broadcasting here would silently produce a wrong trajectory."""
    with pytest.raises(ValueError, match="must match"):
        flow_matching.euler_step(np.zeros((32, 32, 3)), np.zeros((32, 3)), 1.0, 0.9)


@pytest.mark.parametrize("steps", [0, 1, -3])
def test_fewer_than_two_steps_is_an_error(steps):
    with pytest.raises(ValueError, match="at least 2 steps"):
        flow_matching.sigmas(steps)
