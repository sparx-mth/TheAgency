"""The thrust model, and the estimator that keeps it honest as the battery sags."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.control.constants import GRAVITY_MPS2
from sparx_agency.core.control.thrust_model import (
    ThrustModel, ThrustModelParams, specific_force_along,
)

UP = (0.0, 0.0, 1.0)


def _fly(model, true_hover_throttle, seconds, dt=0.02, throttle=None):
    """Run the estimator against an airframe with a known thrust curve.

    Args:
        model: The model under test.
        true_hover_throttle: The airframe's real hover throttle.
        seconds: How long to run.
        dt: Tick length.
        throttle: Throttle to command; the model's own hover guess by default,
            which is the closed-loop case that actually happens in flight.
    """
    true_scale = GRAVITY_MPS2 / true_hover_throttle
    for _ in range(int(seconds / dt)):
        command = model.hover_throttle if throttle is None else throttle
        # The airframe's honest response: thrust up, gravity down.
        acceleration = (0.0, 0.0, command * true_scale - GRAVITY_MPS2)
        model.observe(command, acceleration, UP, dt)


def test_specific_force_of_a_hovering_aircraft_is_one_g():
    """Not accelerating means the rotors are exactly carrying the weight."""
    assert specific_force_along((0.0, 0.0, 0.0), UP) == pytest.approx(GRAVITY_MPS2)


def test_specific_force_uses_the_thrust_axis_not_the_vertical():
    """A tilted aircraft holding altitude is working harder than a level one."""
    tilt = math.radians(30.0)
    axis = (math.sin(tilt), 0.0, math.cos(tilt))
    # Level flight in a turn: no vertical acceleration, horizontal from the tilt.
    horizontal = GRAVITY_MPS2 * math.tan(tilt)
    force = specific_force_along((horizontal, 0.0, 0.0), axis)
    assert force == pytest.approx(GRAVITY_MPS2 / math.cos(tilt), abs=1e-9)
    assert force > GRAVITY_MPS2


def test_the_seed_hover_throttle_is_what_a_hover_costs():
    """Before learning anything, asking for 1 g asks for the seed throttle."""
    model = ThrustModel(ThrustModelParams(hover_throttle=0.55))
    assert model.normalized(GRAVITY_MPS2) == pytest.approx(0.55)
    assert model.hover_throttle == pytest.approx(0.55)


def test_the_estimate_converges_on_an_airframe_that_is_not_the_seed():
    """Seeded wrong at 0.50, flying an airframe that hovers at 0.65."""
    model = ThrustModel(ThrustModelParams(hover_throttle=0.50, learn_tau_s=2.0))
    _fly(model, true_hover_throttle=0.65, seconds=30.0)
    assert model.hover_throttle == pytest.approx(0.65, abs=0.01)
    assert model.observations > 0


def test_it_tracks_a_battery_sagging_over_a_flight():
    """The thrust curve moving is the reason this is estimated rather than set."""
    model = ThrustModel(ThrustModelParams(hover_throttle=0.55, learn_tau_s=2.0))
    _fly(model, true_hover_throttle=0.55, seconds=20.0)
    assert model.hover_throttle == pytest.approx(0.55, abs=0.01)
    _fly(model, true_hover_throttle=0.70, seconds=40.0)
    assert model.hover_throttle == pytest.approx(0.70, abs=0.02)


def test_a_near_zero_throttle_is_not_an_observation():
    """Dividing an acceleration by almost no throttle invents an enormous scale."""
    model = ThrustModel()
    before = model.hover_throttle
    assert not model.observe(0.01, (0.0, 0.0, -GRAVITY_MPS2), UP, 0.02)
    assert model.hover_throttle == before


def test_an_implausible_observation_is_rejected():
    """A propeller strike accelerates the airframe and says nothing about thrust."""
    model = ThrustModel(ThrustModelParams(hover_throttle=0.5, max_observation_ratio=1.5))
    before = model.hover_throttle
    assert not model.observe(0.5, (0.0, 0.0, 200.0), UP, 0.02)
    assert model.hover_throttle == before


def test_free_fall_is_rejected_rather_than_learned():
    """Negative measured thrust is not physical; it is a bad input."""
    model = ThrustModel()
    assert not model.observe(0.5, (0.0, 0.0, -2.0 * GRAVITY_MPS2), UP, 0.02)


def test_the_estimate_stays_inside_its_bounds():
    """An airframe outside the believable range clamps rather than running away."""
    params = ThrustModelParams(hover_throttle=0.5, min_hover_throttle=0.3,
                               max_hover_throttle=0.7, learn_tau_s=0.5,
                               max_observation_ratio=100.0)
    model = ThrustModel(params)
    _fly(model, true_hover_throttle=0.95, seconds=30.0)
    assert model.hover_throttle <= 0.7 + 1e-9
    model.reset()
    _fly(model, true_hover_throttle=0.05, seconds=30.0)
    assert model.hover_throttle >= 0.3 - 1e-9


def test_output_throttle_is_clamped_with_headroom_at_both_ends():
    """Never zero -- spun-down rotors cannot hold attitude either -- and never full."""
    model = ThrustModel(ThrustModelParams(min_throttle=0.06, max_throttle=0.9))
    assert model.normalized(0.0) == pytest.approx(0.06)
    assert model.normalized(1000.0) == pytest.approx(0.9)


def test_reset_returns_to_the_seed():
    """A scale learned in ground effect must not open the next flight."""
    model = ThrustModel(ThrustModelParams(hover_throttle=0.5, learn_tau_s=1.0))
    _fly(model, true_hover_throttle=0.7, seconds=20.0)
    assert model.hover_throttle > 0.6
    model.reset()
    assert model.hover_throttle == pytest.approx(0.5)
    assert model.observations == 0


def test_a_non_positive_timestep_is_refused():
    """The filter weight is a function of dt; a zero dt is a caller bug."""
    with pytest.raises(ValueError, match="dt must be > 0"):
        ThrustModel().observe(0.5, (0.0, 0.0, 0.0), UP, 0.0)


def test_bad_parameters_are_refused():
    """The bounds have to be a range, and the seed has to be inside it."""
    with pytest.raises(ValueError, match="hover throttle bounds"):
        ThrustModelParams(min_hover_throttle=0.8, max_hover_throttle=0.2)
    with pytest.raises(ValueError, match="outside its own bounds"):
        ThrustModelParams(hover_throttle=0.9, max_hover_throttle=0.8)
    with pytest.raises(ValueError, match="learn_tau_s"):
        ThrustModelParams(learn_tau_s=0.0)


def test_it_acquires_fast_and_then_filters_slowly():
    """The two regimes, and why both are needed.

    A long time constant is right for *tracking* a battery that sags over
    minutes and wrong for *acquiring* a scale that is unknown at takeoff. The
    running-mean warm-up gets the estimate to the truth in a second or two; the
    exponential filter then takes over so the settled estimate does not chase
    vibration.
    """
    model = ThrustModel(ThrustModelParams(hover_throttle=0.50, learn_tau_s=8.0))
    _fly(model, true_hover_throttle=0.65, seconds=2.0)
    assert model.hover_throttle == pytest.approx(0.65, abs=0.01)

    # Settled -- the running mean has fallen below the filter and handed over.
    # A single outlier inside the plausibility band now barely moves the estimate.
    _fly(model, true_hover_throttle=0.65, seconds=20.0)
    settled = model.hover_throttle
    model.observe(0.5, (0.0, 0.0, 4.0), UP, 0.004)
    assert model.hover_throttle == pytest.approx(settled, abs=0.005)


def test_the_warm_up_restarts_after_a_reset():
    """A reset means the scale is unknown again, not merely stale."""
    model = ThrustModel(ThrustModelParams(hover_throttle=0.50, learn_tau_s=8.0))
    _fly(model, true_hover_throttle=0.65, seconds=2.0)
    model.reset()
    assert model.observations == 0
    _fly(model, true_hover_throttle=0.45, seconds=2.0)
    assert model.hover_throttle == pytest.approx(0.45, abs=0.01)


# ── non-finite input ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_acceleration_is_rejected(bad):
    """The guards are comparisons, and NaN is false against all of them.

    Left unchecked a single NaN reaches `max(low, min(high, nan))`, where
    Python's min returns `high` -- pinning the scale to its maximum -- and
    every honest observation afterwards fails the ratio test against it.
    """
    model = ThrustModel(ThrustModelParams(hover_throttle=0.62))
    before = model.full_scale_mps2
    assert model.observe(0.62, (0.0, 0.0, bad), (0.0, 0.0, 1.0), 0.01) is False
    assert model.full_scale_mps2 == before
    assert model.observations == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_throttle_is_rejected(bad):
    model = ThrustModel(ThrustModelParams(hover_throttle=0.62))
    before = model.full_scale_mps2
    assert model.observe(bad, (0.0, 0.0, 9.81), (0.0, 0.0, 1.0), 0.01) is False
    assert model.full_scale_mps2 == before


def test_a_non_finite_dt_raises():
    model = ThrustModel(ThrustModelParams(hover_throttle=0.62))
    with pytest.raises(ValueError, match="finite"):
        model.observe(0.62, (0.0, 0.0, 9.81), (0.0, 0.0, 1.0), float("nan"))


def test_one_nan_does_not_lock_the_estimator_out_of_the_rest_of_the_flight():
    """The consequence, end to end: a NaN must cost nothing but its own sample."""
    model = ThrustModel(ThrustModelParams(hover_throttle=0.62))
    truth = 9.81 / 0.55                      # the airframe really hovers at 0.55

    model.observe(0.62, (0.0, 0.0, 0.62 * truth - 9.81), (0.0, 0.0, 1.0), 0.01)
    model.observe(0.62, (0.0, 0.0, float("nan")), (0.0, 0.0, 1.0), 0.01)
    for _ in range(200):
        model.observe(0.62, (0.0, 0.0, 0.62 * truth - 9.81), (0.0, 0.0, 1.0), 0.01)

    assert model.full_scale_mps2 == pytest.approx(truth, rel=0.02)
    assert model.hover_throttle == pytest.approx(0.55, abs=0.02)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_request_returns_hover_not_full_throttle(bad):
    """The command path had the same NaN trap the learning path was hardened against.

    `max(min_throttle, min(max_throttle, nan))` is `max_throttle`: a single
    non-finite request would have commanded full collective, with the attitude
    stage reporting tilt 0.0 deg because acos(clamp(nan)) is 0.
    """
    model = ThrustModel(ThrustModelParams(hover_throttle=0.62))
    assert model.normalized(bad) == pytest.approx(model.hover_throttle)
    assert model.normalized(bad) < model.params.max_throttle
