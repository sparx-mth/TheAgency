"""The converter's turn and tilt choices, one fact at a time.

Every turn and LOOK the converter emits reduces to one of these functions, so
each is pinned here against hand-computed values: which way the target lies,
and the dead bands that keep a discrete agent from dithering or asking for a
LOOK the simulator refuses.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.objnav.action_converter.action_choice import (
    PITCH_LIMIT_EPS_RAD,
    heading_error,
    pitch_action,
    pitch_within_limits,
    turn_action,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction

TURN = math.radians(30.0)
TILT = math.radians(30.0)
LIMIT = math.radians(30.0)


# -- heading ----------------------------------------------------------------

def test_heading_error_is_positive_to_the_left():
    """Positive means TURN_LEFT: counter-clockwise, as yaw is."""
    assert heading_error(0.0, 0.0, 0.0, 0.0, 1.0) == pytest.approx(math.pi / 2)
    assert heading_error(0.0, 0.0, 0.0, 0.0, -1.0) == pytest.approx(
        -math.pi / 2)


def test_a_target_straight_behind_reads_as_plus_pi_and_turns_left():
    """Exactly behind must pick one side every time, not flip between two."""
    assert heading_error(0.0, 0.0, 0.0, -1.0, 0.0) == math.pi
    assert heading_error(0.0, 0.0, 0.0, -1.0, -0.0) == math.pi
    assert turn_action(math.pi, TURN) == DiscreteAction.TURN_LEFT


def test_heading_error_wraps_an_unnormalised_yaw():
    """AgentPose.yaw is not normalised; the error must be, or it turns the long way."""
    assert heading_error(0.0, 0.0, 2 * math.pi + 0.1, 1.0, 0.0) == \
        pytest.approx(-0.1)


# -- turn and tilt choices ---------------------------------------------------

@pytest.mark.parametrize("error_deg, expected", [
    (0.0, None),
    (15.0, None),
    (-15.0, None),
    (15.001, DiscreteAction.TURN_LEFT),
    (-15.001, DiscreteAction.TURN_RIGHT),
    (179.0, DiscreteAction.TURN_LEFT),
])
def test_the_turn_dead_band_is_half_a_turn(error_deg, expected):
    """Narrower would dither; wider would leave the agent visibly off course."""
    assert turn_action(math.radians(error_deg), TURN) == expected


@pytest.mark.parametrize("error_deg", [15.001, 30.0, 44.999])
def test_one_turn_from_just_outside_the_dead_band_lands_inside_it(error_deg):
    """This is what makes a left turn never need a right turn to undo it."""
    error = math.radians(error_deg)
    assert turn_action(error, TURN) == DiscreteAction.TURN_LEFT
    assert turn_action(error - TURN, TURN) is None


def test_an_unwrapped_heading_error_is_refused():
    """1.5 pi is -0.5 pi the short way; turning left would take nine turns, not three."""
    with pytest.raises(ObjNavError):
        turn_action(1.5 * math.pi, TURN)


def test_a_non_positive_turn_angle_is_refused():
    """A zero turn would make every error look outside the dead band forever."""
    with pytest.raises(ObjNavError):
        turn_action(0.5, 0.0)


def test_pitch_action_looks_down_toward_a_downward_pitch():
    """REP-103: a positive pitch looks down, and LOOK_DOWN increases it."""
    assert pitch_action(0.0, math.radians(30.0), TILT) == \
        DiscreteAction.LOOK_DOWN
    assert pitch_action(math.radians(30.0), math.radians(30.0), TILT) is None


def test_pitch_action_looks_up_toward_an_upward_pitch():
    """The other sign: a negative pitch looks up."""
    assert pitch_action(0.0, math.radians(-30.0), TILT) == \
        DiscreteAction.LOOK_UP


def test_a_desired_pitch_past_the_limit_is_aimed_at_the_limit():
    """Asking for all the way down is legal; at the limit there is nothing left to do."""
    assert pitch_action(0.0, math.radians(60.0), TILT, -LIMIT, LIMIT) == \
        DiscreteAction.LOOK_DOWN
    assert pitch_action(LIMIT, math.radians(60.0), TILT, -LIMIT, LIMIT) is None
    assert pitch_action(-LIMIT, math.radians(-60.0), TILT, -LIMIT, LIMIT) \
        is None


def test_a_look_that_would_leave_the_limits_is_never_asked_for():
    """AI2-THOR fails such a LOOK: a wasted step that changes nothing."""
    assert pitch_action(math.radians(10.0), math.radians(45.0), TILT,
                        -LIMIT, LIMIT) is None


def test_exactly_the_limit_is_within_the_limits():
    """A LOOK from 0 to exactly +30 degrees is legal on AI2-THOR."""
    assert pitch_within_limits(LIMIT, -LIMIT, LIMIT)
    assert not pitch_within_limits(LIMIT + 2.0 * PITCH_LIMIT_EPS_RAD, -LIMIT,
                                   LIMIT)
    assert pitch_within_limits(math.radians(80.0))


@pytest.mark.parametrize("noise_deg", [-1e-7, -1e-6, -1e-5])
def test_a_horizon_a_hair_past_the_lattice_still_gets_its_legal_look(noise_deg):
    """Unity reports the horizon with float32 noise; the LOOK_UP to -30 degrees is legal, and Unity executes it."""
    current = math.radians(noise_deg)
    assert pitch_action(current, -LIMIT, TILT, -LIMIT, LIMIT) == \
        DiscreteAction.LOOK_UP
    assert pitch_within_limits(current - TILT, -LIMIT, LIMIT)


def test_the_pitch_limit_slack_absorbs_float_noise_and_nothing_near_a_tilt():
    """A slack grown toward a tilt would ask for LOOKs AI2-THOR really refuses."""
    assert pitch_within_limits(LIMIT + 0.5 * PITCH_LIMIT_EPS_RAD, -LIMIT, LIMIT)
    assert not pitch_within_limits(-LIMIT - 2.0 * PITCH_LIMIT_EPS_RAD, -LIMIT,
                                   LIMIT)
    assert PITCH_LIMIT_EPS_RAD < TILT / 1000.0


def test_an_inverted_pitch_range_is_refused():
    """With min above max every pitch is out of range, and every LOOK is refused."""
    with pytest.raises(ObjNavError):
        pitch_action(0.0, 0.1, TILT, 0.5, -0.5)
    with pytest.raises(ObjNavError):
        pitch_within_limits(0.0, 0.5, -0.5)


@pytest.mark.parametrize("call", [
    lambda: heading_error(0.0, float("nan"), 0.0, 1.0, 1.0),
    lambda: pitch_action(0.0, float("nan"), TILT),
    lambda: pitch_action(0.0, 0.1, 0.0),
])
def test_invalid_numbers_are_refused(call):
    """A NaN compares false with everything and would pick an action silently."""
    with pytest.raises(ObjNavError):
        call()
