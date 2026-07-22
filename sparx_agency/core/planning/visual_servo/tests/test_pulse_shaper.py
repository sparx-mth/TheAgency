"""Unit tests for :class:`PulseShaper` (minimum-burst + coast-brake actuation)."""
from __future__ import annotations

import pytest

from sparx_agency.core.common.types import ControlCommand
from sparx_agency.core.planning.visual_servo.force_shaping import AxisForceProfile
from sparx_agency.core.planning.visual_servo.pulse_shaper import PulseShaper

_P = AxisForceProfile(min_magnitude=0.14, max_magnitude=0.6, mode="fixed")


def _shaper(min_burst=2, brake=0):
    return PulseShaper(_P, _P, _P, min_burst_ticks=min_burst, brake_ticks=brake)


def _yaw(s, w):
    return round(s.shape(ControlCommand.velocity(0.0, 0.0, 0.0, w)).yaw_rate, 4)


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        PulseShaper(_P, _P, _P, min_burst_ticks=0)
    with pytest.raises(ValueError):
        PulseShaper(_P, _P, _P, brake_ticks=-1)


def test_single_tick_request_becomes_a_min_burst():
    # The servo wants yaw for ONE tick then stops; the platform would ignore a lone
    # tick, so it is latched into a full min_burst_ticks (=2) burst.
    s = _shaper(min_burst=2)
    assert [_yaw(s, w) for w in (0.5, 0.0, 0.0)] == [0.14, 0.14, 0.0]


def test_min_burst_three_ticks():
    s = _shaper(min_burst=3)
    assert [_yaw(s, w) for w in (0.5, 0.0, 0.0, 0.0)] == [0.14, 0.14, 0.14, 0.0]


def test_sustained_motion_passes_through_at_fixed_level():
    s = _shaper(min_burst=2)
    assert [_yaw(s, w) for w in (0.5, 0.5, 0.5, 0.5)] == [0.14, 0.14, 0.14, 0.14]


def test_brake_emits_one_opposite_pulse_after_a_real_burst():
    s = _shaper(min_burst=2, brake=1)
    assert [_yaw(s, w) for w in (0.5, 0.5, 0.0, 0.0)] == [0.14, 0.14, -0.14, 0.0]


def test_no_brake_by_default():
    s = _shaper(min_burst=2, brake=0)
    assert [_yaw(s, w) for w in (0.5, 0.5, 0.0)] == [0.14, 0.14, 0.0]


def test_direction_change_starts_a_fresh_burst():
    s = _shaper(min_burst=2)
    # left burst (positive), then the servo flips to right: a new >=2-tick burst.
    out = [_yaw(s, w) for w in (0.5, 0.5, -0.5, 0.0, 0.0)]
    assert out == [0.14, 0.14, -0.14, -0.14, 0.0]


def test_below_deadband_is_zero():
    s = _shaper(min_burst=2)
    assert _yaw(s, 0.0005) == 0.0


def test_reset_clears_burst_state():
    s = _shaper(min_burst=2)
    _yaw(s, 0.5)                      # start a burst (mid-minimum-burst)
    s.reset()
    # After reset a stop command is a clean zero (no leftover burst to finish).
    assert _yaw(s, 0.0) == 0.0


def test_axes_are_independent():
    s = _shaper(min_burst=2)
    c = s.shape(ControlCommand.velocity(0.5, 0.0, 0.0, 0.0))   # only vx requested
    assert round(c.x, 4) == 0.14 and c.y == 0.0 and c.yaw_rate == 0.0
