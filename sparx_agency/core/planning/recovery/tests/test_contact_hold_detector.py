"""Tests for :mod:`...recovery.contact_hold_detector`.

The behaviours worth pinning down are the ones that separate a held airframe
from the two things it must never be confused with: ordinary aggressive flight
(tilted, but moving) and a genuine free capsize (tilted, but falling).
"""
import pytest

from sparx_agency.core.planning.recovery.contact_hold_detector import (
    ContactHoldDetector, ContactHoldParams, ContactVerdict)


def _hold(det, t0, n, dt=0.05, tilt=55.0, speed=0.0, climb=0.0, ratio=None):
    """Feed n ticks of one signature and return the last verdict."""
    v = ContactVerdict()
    for i in range(n):
        v = det.update(t0 + i * dt, tilt, speed, climb, ratio)
    return v


def test_a_held_airframe_is_confirmed_after_confirm_s():
    det = ContactHoldDetector(ContactHoldParams(confirm_s=2.0))
    assert not _hold(det, 0.0, 10).held        # 0.5 s -- not yet
    assert _hold(det, 0.5, 60).held            # past 2.0 s


def test_tilted_but_moving_is_flight_not_contact():
    """Aggressive flight tilts. It is only a hold if it is also not going anywhere."""
    det = ContactHoldDetector()
    assert not _hold(det, 0.0, 200, tilt=60.0, speed=0.9).held


def test_a_falling_airframe_is_a_capsize_and_must_not_be_reported():
    """The discriminator that keeps the existing capsize reflex in charge."""
    det = ContactHoldDetector()
    assert not _hold(det, 0.0, 200, tilt=70.0, speed=0.0, climb=-1.4).held


def test_level_hover_never_confirms():
    det = ContactHoldDetector()
    assert not _hold(det, 0.0, 400, tilt=3.0).held


def test_since_s_reports_the_cost_of_the_episode():
    det = ContactHoldDetector(ContactHoldParams(confirm_s=1.0))
    v = _hold(det, 0.0, 201)      # 10.0 s of signature
    assert v.held
    assert v.since_s == pytest.approx(10.0, abs=0.2)


def test_a_single_clean_sample_does_not_end_an_episode():
    """Pose noise must not chop one long episode into many short ones."""
    det = ContactHoldDetector(ContactHoldParams(confirm_s=1.0, clear_s=0.5))
    assert _hold(det, 0.0, 60).held
    det.update(3.05, 10.0, 0.0, 0.0)          # one stray level sample
    assert det.update(3.10, 55.0, 0.0, 0.0).held


def test_sustained_flight_clears_a_standing_hold():
    det = ContactHoldDetector(ContactHoldParams(confirm_s=1.0, clear_s=0.5))
    assert _hold(det, 0.0, 60).held
    assert not _hold(det, 3.05, 40, tilt=5.0, speed=0.8).held


def test_support_ratio_corroborates_when_it_matches_the_tilt():
    """ranger/height should read 1/cos(tilt); 1/cos(55 deg) = 1.74."""
    det = ContactHoldDetector(ContactHoldParams(confirm_s=1.0))
    v = _hold(det, 0.0, 60, tilt=55.0, ratio=1.74)
    assert v.held and v.corroborated


def test_a_disagreeing_support_ratio_still_reports_but_uncorroborated():
    det = ContactHoldDetector(ContactHoldParams(confirm_s=1.0))
    v = _hold(det, 0.0, 60, tilt=55.0, ratio=1.0)
    assert v.held and not v.corroborated


def test_disabled_never_confirms():
    det = ContactHoldDetector(ContactHoldParams(enabled=False))
    assert not _hold(det, 0.0, 400).held


@pytest.mark.parametrize("kwargs", [
    {"tilt_deg": 0.0}, {"tilt_deg": 90.0}, {"max_speed_mps": 0.0},
    {"confirm_s": -1.0}, {"clear_s": -0.1},
])
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        ContactHoldParams(**kwargs)
