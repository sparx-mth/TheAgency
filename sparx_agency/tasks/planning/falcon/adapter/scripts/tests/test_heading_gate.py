"""Tests for the turn-in-place gate.

ROS-free: the gate is a pure state machine over the body-frame demand, which is
the whole reason it lives in its own module rather than inside the ROS node.
The adapter module sits next to the ROS nodes in ``scripts/``, so the sibling
dir goes on the path before importing it -- same pattern as
``test_pure_pursuit_follower.py``.

The cases are the failures from run ``nav_debug_20260906_235809``: frames 984,
993 and 1122 all published a large negative body ``vx`` while the demand sat
behind the nose.
"""
import math
import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from heading_gate import (  # noqa: E402
    TRACKING,
    TURNING,
    HeadingGate,
    HeadingGateParams,
)


def _demand(speed, angle_deg):
    """A body-frame velocity of ``speed`` pointing ``angle_deg`` off the nose."""
    angle = math.radians(angle_deg)
    return speed * math.cos(angle), speed * math.sin(angle)


def test_a_demand_ahead_is_flown_straight_at():
    gate = HeadingGate()
    out = gate.update(*_demand(0.8, 10.0))
    assert out.state == TRACKING and out.allow_translation


def test_a_demand_behind_the_nose_stops_translation():
    """Frame 1122: body vx -0.90, vy +0.42 -- 155 deg off the nose."""
    gate = HeadingGate()
    out = gate.update(-0.90, 0.42)
    assert out.state == TURNING
    assert not out.allow_translation
    assert abs(math.degrees(out.demand_angle_rad)) > 90.0


def test_the_turn_holds_until_well_inside_the_engage_angle():
    """Hysteresis: a single threshold chatters, measured 56-196 times a run."""
    gate = HeadingGate(HeadingGateParams(engage_deg=90.0, resume_deg=60.0))
    assert gate.update(*_demand(0.8, 120.0)).turning
    assert gate.update(*_demand(0.8, 85.0)).turning     # inside engage, still turning
    assert gate.update(*_demand(0.8, 70.0)).turning
    assert not gate.update(*_demand(0.8, 55.0)).turning  # released


def test_the_gate_does_not_re_engage_on_its_own_overshoot():
    gate = HeadingGate(HeadingGateParams(engage_deg=90.0, resume_deg=60.0))
    gate.update(*_demand(0.8, 120.0))
    gate.update(*_demand(0.8, 40.0))                     # released
    for angle in (45.0, 55.0, 70.0, 85.0):               # ordinary mid-turn values
        assert not gate.update(*_demand(0.8, angle)).turning


def test_a_slow_demand_is_not_gated_on_its_own_direction_noise():
    """The direction of a small vector is ill-conditioned -- B31's mechanism."""
    gate = HeadingGate(HeadingGateParams(min_speed_mps=0.25))
    out = gate.update(*_demand(0.05, 179.0))
    assert out.state == TRACKING and out.allow_translation


def test_a_turn_is_abandoned_rather_than_latched_when_the_demand_dies():
    gate = HeadingGate(HeadingGateParams(min_speed_mps=0.25))
    assert gate.update(*_demand(0.8, 150.0)).turning
    assert not gate.update(*_demand(0.01, 150.0)).turning


def test_engaged_now_fires_once_per_turn():
    gate = HeadingGate()
    assert gate.update(*_demand(0.8, 150.0)).engaged_now
    assert not gate.update(*_demand(0.8, 150.0)).engaged_now


def test_the_gate_is_not_conditioned_on_anything_but_the_demand():
    """The structural defect it replaces: the old gate sat inside
    ``not use_lateral`` and a lateral-enabled flight skipped it entirely.

    A large lateral demand is exactly the case that used to slip through, so it
    must gate on the ANGLE, not on which axis carries the demand.
    """
    gate = HeadingGate()
    # Pure lateral: no backward component at all, but 90 deg off the nose.
    assert not gate.update(0.0, 0.8).turning       # exactly at the threshold
    assert gate.update(-0.01, 0.8).turning         # just past it


def test_reverse_is_clamped_at_the_publish_boundary():
    gate = HeadingGate()
    assert gate.limit_reverse(-0.9) == pytest.approx(0.0)
    assert gate.limit_reverse(0.5) == pytest.approx(0.5)


def test_the_escape_reflex_may_still_reverse():
    """Backing out of a nose-in contact is the one known-clear direction."""
    gate = HeadingGate()
    assert gate.limit_reverse(-0.3, allow_reverse=True) == pytest.approx(-0.3)


def test_a_reverse_allowance_can_be_configured():
    gate = HeadingGate(HeadingGateParams(max_reverse_mps=0.2))
    assert gate.limit_reverse(-0.9) == pytest.approx(-0.2)


def test_disabling_the_gate_restores_the_old_behaviour():
    gate = HeadingGate(HeadingGateParams(enabled=False))
    out = gate.update(-0.9, 0.42)
    assert out.state == TRACKING and out.allow_translation


@pytest.mark.parametrize("kwargs", [
    {"engage_deg": 0.0},
    {"engage_deg": 181.0},
    {"resume_deg": 90.0},          # equal to engage: no hysteresis band
    {"resume_deg": 120.0},         # above engage
    {"min_speed_mps": -0.1},
    {"max_reverse_mps": -0.1},
])
def test_invalid_params_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HeadingGateParams(**kwargs)
