"""Tests for the rotate-with-stops scan-at-goal policy."""
import pytest

from sparx_agency.core.planning.visual_servo.scan_search import (
    PAUSE,
    RELOCATE,
    ROTATE,
    ScanSearchConfig,
    ScanSearchPolicy,
)


def test_starts_paused_then_rotates():
    pol = ScanSearchPolicy(ScanSearchConfig(yaw_rate=0.4, rotate_s=1.0, pause_s=1.0))
    c = pol.command(0.0)
    assert pol.phase == PAUSE
    assert (c.x, c.y, c.yaw_rate) == (0.0, 0.0, 0.0)      # stop first
    pol.command(0.6)
    assert pol.phase == PAUSE                             # still within pause_s
    c = pol.command(0.6)                                  # t=1.2 >= pause_s -> rotate
    assert pol.phase == ROTATE
    assert c.yaw_rate == pytest.approx(0.4)
    assert c.metadata["source"] == "scan"
    assert c.metadata["phase"] == ROTATE


def test_rotate_returns_to_pause_when_forward_disabled():
    pol = ScanSearchPolicy(ScanSearchConfig(rotate_s=1.0, pause_s=1.0))
    pol.command(1.0)      # -> ROTATE
    assert pol.phase == ROTATE
    pol.command(1.0)      # rotate_s elapsed -> PAUSE (no relocate; forward disabled)
    assert pol.phase == PAUSE


def test_direction_sign():
    pol = ScanSearchPolicy(ScanSearchConfig(yaw_rate=0.5, rotate_s=1.0, pause_s=0.5,
                                            direction=-1.0))
    pol.command(0.5)      # -> ROTATE
    c = pol.command(0.0)
    assert c.yaw_rate == pytest.approx(-0.5)


def test_forward_relocate_after_bursts():
    cfg = ScanSearchConfig(rotate_s=1.0, pause_s=0.5, forward_speed=0.1,
                           forward_s=1.0, bursts_before_move=2)
    pol = ScanSearchPolicy(cfg)
    seen = set()
    # Drive several full pause/rotate cycles; the 2nd rotate burst should relocate.
    for _ in range(40):
        c = pol.command(0.5)
        seen.add(pol.phase)
        if pol.phase == RELOCATE:
            assert c.x == pytest.approx(0.1)
            assert c.yaw_rate == 0.0
    assert {PAUSE, ROTATE, RELOCATE} <= seen


def test_reset_returns_to_pause():
    pol = ScanSearchPolicy(ScanSearchConfig(rotate_s=1.0, pause_s=1.0))
    pol.command(1.0)      # -> ROTATE
    pol.reset()
    assert pol.phase == PAUSE


def test_validation():
    with pytest.raises(ValueError):
        ScanSearchConfig(yaw_rate=0.0)
    with pytest.raises(ValueError):
        ScanSearchConfig(direction=2.0)
