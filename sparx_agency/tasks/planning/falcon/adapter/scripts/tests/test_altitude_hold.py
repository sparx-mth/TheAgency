"""Tests for the cautious altitude hold: climb is the guarded direction."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from altitude_hold import AltitudeHold, AltitudeHoldParams  # noqa: E402


def _hold(**kw):
    return AltitudeHold(AltitudeHoldParams(**kw))


def test_holds_inside_the_deadband():
    h = _hold(target_z=1.0, deadband_m=0.10)
    assert h.update(z=0.95, confidence=0.6, coasting=False,
                    pose_valid=True, flying=True) == 0.0


def test_climbs_slowly_when_low_and_sure():
    h = _hold(target_z=1.0, climb_max=0.06)
    vz = h.update(z=0.70, confidence=0.6, coasting=False,
                  pose_valid=True, flying=True)
    assert 0.0 < vz <= 0.06                      # capped tiny, and positive


def test_never_climbs_on_a_vague_or_coasted_pose():
    h = _hold(target_z=1.0, conf_min_climb=0.35)
    vague = h.update(z=0.70, confidence=0.20, coasting=False,
                     pose_valid=True, flying=True)
    coasted = h.update(z=0.70, confidence=0.60, coasting=True,
                       pose_valid=True, flying=True)
    assert vague == 0.0 and coasted == 0.0


def test_never_climbs_at_the_ceiling_but_descends_above_target():
    h = _hold(target_z=1.0, ceiling_m=1.2, descend_max=0.10)
    # Above target: pulls DOWN (run 152711 overshot to 1.25 m).
    vz = h.update(z=1.25, confidence=0.6, coasting=False,
                  pose_valid=True, flying=True)
    assert -0.10 <= vz < 0.0
    # Mis-configured target above the ceiling must still refuse to climb.
    weird = AltitudeHold(AltitudeHoldParams(target_z=1.0, ceiling_m=1.2))
    object.__setattr__(weird.params, "target_z", 1.5)   # simulate bad runtime state
    assert weird.update(z=1.21, confidence=0.9, coasting=False,
                        pose_valid=True, flying=True) == 0.0


def test_hands_off_on_the_ground_when_held_or_without_a_pose():
    h = _hold(target_z=1.0, min_z_m=0.2)
    grounded = h.update(z=0.05, confidence=0.9, coasting=False,
                        pose_valid=True, flying=True)
    held = h.update(z=0.70, confidence=0.9, coasting=False,
                    pose_valid=True, flying=False)
    no_pose = h.update(z=None, confidence=0.9, coasting=False,
                       pose_valid=True, flying=True)
    stale = h.update(z=0.70, confidence=0.9, coasting=False,
                     pose_valid=False, flying=True)
    assert grounded == held == no_pose == stale == 0.0


def test_descend_needs_less_trust_than_climb_but_not_none():
    h = _hold(target_z=1.0, conf_min_climb=0.35, conf_min_descend=0.10)
    ok = h.update(z=1.30, confidence=0.15, coasting=False,
                  pose_valid=True, flying=True)
    assert ok < 0.0                              # descends on a modest pose
    garbage = h.update(z=1.30, confidence=0.05, coasting=False,
                       pose_valid=True, flying=True)
    assert garbage == 0.0                        # but not on a garbage one


def test_params_validate_the_operator_red_lines():
    with pytest.raises(ValueError):
        AltitudeHoldParams(target_z=1.0, deadband_m=0.10, ceiling_m=1.05)
    with pytest.raises(ValueError):
        AltitudeHoldParams(conf_min_climb=0.2, conf_min_descend=0.5)
