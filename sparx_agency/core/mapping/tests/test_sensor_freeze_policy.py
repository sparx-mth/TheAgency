"""Tests for the mode-authoritative map-freeze decision."""
from sparx_agency.core.mapping.sensor_freeze_policy import (
    SRC_EXPLICIT_FALLBACK,
    SRC_EXPLICIT_ONLY,
    SRC_MODE_AUTH,
    SensorFreezePolicy,
    freeze_mode_names,
)


def test_default_is_unfrozen_via_explicit_fallback():
    # No mode message yet: fall back to the explicit request (which is False).
    assert SensorFreezePolicy().decide() == (False, SRC_EXPLICIT_FALLBACK)


def test_explicit_request_used_before_first_mode_message():
    p = SensorFreezePolicy()
    p.note_explicit(True)
    assert p.decide() == (True, SRC_EXPLICIT_FALLBACK)


def test_mode_becomes_authoritative_once_it_speaks():
    p = SensorFreezePolicy()
    p.note_explicit(True)            # would freeze under fallback
    p.note_mode(False)               # but the mode says we are not turning
    assert p.decide() == (False, SRC_MODE_AUTH)
    p.note_mode(True)
    assert p.decide() == (True, SRC_MODE_AUTH)


def test_stale_explicit_freeze_cannot_stick_once_mode_clears():
    # The motivating bug: controller froze, mode later leaves turning before the
    # controller unfreezes. Mode wins, so we do not stay frozen forever.
    p = SensorFreezePolicy()
    p.note_explicit(True)
    p.note_mode(True)
    assert p.decide()[0] is True
    p.note_mode(False)               # turn ended; explicit is still True
    assert p.decide() == (False, SRC_MODE_AUTH)


def test_explicit_only_ignores_mode():
    p = SensorFreezePolicy(freeze_on_turning_mode=False)
    p.note_mode(True)                # ignored entirely
    assert p.decide() == (False, SRC_EXPLICIT_ONLY)
    p.note_explicit(True)
    assert p.decide() == (True, SRC_EXPLICIT_ONLY)


def test_reset_mode_freeze_clears_until_next_turning_message():
    p = SensorFreezePolicy()
    p.note_mode(True)
    assert p.decide()[0] is True
    p.reset_mode_freeze()
    assert p.decide() == (False, SRC_MODE_AUTH)
    p.note_mode(True)                # a genuine new turning message re-sets it
    assert p.decide()[0] is True


# ── Which mode names mean "freeze" ──────────────────────────────────
def test_turning_always_freezes():
    assert "turning" in freeze_mode_names("turning")


def test_recovery_freezes_by_default():
    """The lost-localization recovery flies blind: its frames must never fuse."""
    assert "recovery" in freeze_mode_names("turning")


def test_names_are_matched_as_the_topic_sends_them():
    assert freeze_mode_names("  TURNING  ") == freeze_mode_names("turning")


def test_extra_modes_can_be_overridden():
    names = freeze_mode_names("turning", "visual_servoing, recovery")
    assert names == {"turning", "visual_servoing", "recovery"}


def test_empty_extra_restores_turning_only_behaviour():
    """The escape hatch: '' opts out of every non-turning freeze."""
    assert freeze_mode_names("turning", "") == {"turning"}
