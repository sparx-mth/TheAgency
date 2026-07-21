"""Tests for the cautious altitude hold: climb is the guarded direction."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from altitude_hold import AltitudeHold, AltitudeHoldParams  # noqa: E402


def _hold(**kw):
    return AltitudeHold(AltitudeHoldParams(**kw))


def _up(h, z, conf=0.6):
    return h.update(z=z, confidence=conf, coasting=False,
                    pose_valid=True, flying=True)


def test_holds_inside_the_deadband():
    cmd = _up(_hold(target_z=1.0, deadband_m=0.10), 0.95)
    assert cmd.vz == 0.0 and cmd.translation_scale == 1.0


def test_small_sag_trims_up_while_still_flying():
    # target 1.0, deadband 0.10, trigger 0.20 -> z=0.85 is the TRIM band.
    cmd = _up(_hold(target_z=1.0, climb_max=0.20, pulse_trigger_m=0.20), 0.85)
    assert 0.0 < cmd.vz <= 0.20
    assert cmd.translation_scale == 1.0          # not a pulse -- keeps flying


def test_big_sag_pulses_climb_and_yields_translation():
    h = _hold(target_z=1.0, kp=1.0, climb_max=0.20, pulse_trigger_m=0.20,
              pulse_translation_scale=0.2)
    cmd = _up(h, 0.70)                            # 0.30 below -> pulse
    assert cmd.vz == 0.20 and cmd.translation_scale == 0.2 and h.pulsing


def test_pulse_climb_tapers_near_the_target_never_flat_out():
    """The platform climbs far harder than nominal -- a flat-out pulse coasted
    to 1.5-1.6 m on the logs. The demand must shrink as the target nears."""
    h = _hold(target_z=1.0, kp=0.5, climb_max=0.20, pulse_trigger_m=0.20)
    far = _up(h, 0.55).vz                          # 0.45 below
    near = _up(h, 0.87).vz                         # 0.13 below, still pulsing
    assert far == 0.20                             # capped when far
    assert 0.0 < near < far                        # tapered when near


def test_pulse_releases_at_half_trigger_and_finishes_as_a_trim():
    h = _hold(target_z=1.0, deadband_m=0.10, kp=1.0, climb_max=0.20,
              pulse_trigger_m=0.20, pulse_translation_scale=0.2)
    assert _up(h, 0.70).translation_scale == 0.2   # pulse starts (0.30 sag)
    assert _up(h, 0.85).translation_scale == 0.2   # 0.15 sag > half-trigger: pulsing
    eased = _up(h, 0.92)                            # 0.08 sag <= half-trigger
    assert not h.pulsing and eased.translation_scale == 1.0
    assert eased.vz == 0.0                          # inside deadband: on altitude
    assert _up(h, 0.85).translation_scale == 1.0   # a fresh small sag only trims


def test_never_climbs_on_a_vague_or_coasted_pose():
    h = _hold(target_z=1.0, conf_min_climb=0.35)
    assert _up(h, 0.70, conf=0.20).vz == 0.0
    coasted = h.update(z=0.70, confidence=0.60, coasting=True,
                       pose_valid=True, flying=True)
    assert coasted.vz == 0.0 and coasted.translation_scale == 1.0


def test_never_climbs_at_the_ceiling_but_descends_above_target():
    h = _hold(target_z=1.0, ceiling_m=1.2, descend_max=0.10)
    assert -0.10 <= _up(h, 1.25).vz < 0.0        # overshoot -> pulls down
    weird = AltitudeHold(AltitudeHoldParams(target_z=1.0, ceiling_m=1.2))
    weird.params.target_z = 1.5                  # simulate bad runtime state
    assert _up(weird, 1.21, conf=0.9).vz == 0.0  # above ceiling, refuses to climb


def test_a_climb_pulse_never_suppresses_translation_below_a_believed_descent():
    # descending does not steal lift, so it keeps full translation authority.
    h = _hold(target_z=1.0, conf_min_climb=0.35, conf_min_descend=0.10)
    down = _up(h, 1.30, conf=0.15)
    assert down.vz < 0.0 and down.translation_scale == 1.0
    assert _up(h, 1.30, conf=0.05).vz == 0.0     # garbage pose -> no descent


def test_hands_off_on_the_ground_when_held_or_without_a_pose():
    h = _hold(target_z=1.0, min_z_m=0.2)
    for cmd in (_up(h, 0.05),
                h.update(z=0.70, confidence=0.9, coasting=False,
                         pose_valid=True, flying=False),
                h.update(z=None, confidence=0.9, coasting=False,
                         pose_valid=True, flying=True),
                h.update(z=0.70, confidence=0.9, coasting=False,
                         pose_valid=False, flying=True)):
        assert cmd.vz == 0.0 and cmd.translation_scale == 1.0


def test_params_validate_the_operator_red_lines():
    with pytest.raises(ValueError):
        AltitudeHoldParams(target_z=1.0, deadband_m=0.10, ceiling_m=1.05)
    with pytest.raises(ValueError):
        AltitudeHoldParams(conf_min_climb=0.2, conf_min_descend=0.5)
    with pytest.raises(ValueError):                # pulse must exceed deadband
        AltitudeHoldParams(deadband_m=0.10, pulse_trigger_m=0.10)
