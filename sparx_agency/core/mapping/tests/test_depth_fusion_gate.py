"""Tests for the rotation-aware depth-fusion gate.

Exercises the full handshake the gate enforces: fuse while flying straight,
freeze every frame while turning, and on resume reject the stale in-flight turn
frames until a frame captured strictly after the turn ends arrives.
"""
from sparx_agency.core.mapping.depth_fusion_gate import (
    FROZEN_ROTATING,
    FUSE,
    STALE_AFTER_ROTATION,
    DepthFusionGate,
)
from sparx_agency.core.mapping.sensor_freeze_policy import SensorFreezePolicy


def _gate(**kw):
    return DepthFusionGate(policy=SensorFreezePolicy(), **kw)


def test_flying_straight_fuses_every_frame():
    g = _gate()
    g.note_mode(False, now=0.0)      # confirmed fly-straight
    assert g.is_passing() is True
    assert g.should_fuse(1.0) == (True, FUSE)
    assert g.should_fuse(2.0) == (True, FUSE)


def test_turning_freezes_every_frame():
    g = _gate()
    g.note_mode(True, now=1.0)       # turning confirmed
    assert g.frozen is True
    assert g.is_passing() is False
    assert g.should_fuse(1.1) == (False, FROZEN_ROTATING)
    assert g.should_fuse(1.2) == (False, FROZEN_ROTATING)


def test_resume_rejects_in_flight_turn_frames_then_fuses_fresh():
    g = _gate()
    g.note_mode(False, now=0.0)
    g.should_fuse(1.0)               # a normal pre-turn frame
    g.note_mode(True, now=2.0)       # start turning
    g.should_fuse(2.5)               # captured during the turn (frozen, tracked)
    g.should_fuse(3.0)               # newest frame seen during the turn

    g.note_mode(False, now=3.0)      # turn ends at t=3.0 -> watermark = 3.0
    assert g.awaiting_fresh_frame is True
    assert g.is_passing() is False
    # A late frame captured during the turn (stamp <= watermark) is rejected.
    assert g.should_fuse(2.9) == (False, STALE_AFTER_ROTATION)
    assert g.should_fuse(3.0) == (False, STALE_AFTER_ROTATION)
    # The first genuinely fresh frame retires the guard and is fused.
    assert g.should_fuse(3.1) == (True, FUSE)
    assert g.is_passing() is True
    assert g.should_fuse(3.2) == (True, FUSE)


def test_watermark_uses_wall_clock_when_mode_signal_overtakes_frames():
    # The mode "stop turning" can arrive before the last in-flight turn frames.
    # The newest frame seen is only at t=10, but the turn really ended at t=12,
    # so a frame captured at t=11 must still be rejected.
    g = _gate()
    g.note_mode(True, now=5.0)
    g.should_fuse(10.0)              # newest frame seen during the turn
    g.note_mode(False, now=12.0)     # mode flip arrives "late" on the wall clock
    assert g.should_fuse(11.0) == (False, STALE_AFTER_ROTATION)
    assert g.should_fuse(12.0) == (False, STALE_AFTER_ROTATION)
    assert g.should_fuse(12.5) == (True, FUSE)


def test_resume_settle_margin_extends_the_watermark():
    g = _gate(resume_settle_sec=0.5)
    g.note_mode(True, now=1.0)
    g.should_fuse(2.0)
    g.note_mode(False, now=2.0)      # watermark = 2.0 + 0.5 = 2.5
    assert g.should_fuse(2.4) == (False, STALE_AFTER_ROTATION)
    assert g.should_fuse(2.5) == (False, STALE_AFTER_ROTATION)
    assert g.should_fuse(2.6) == (True, FUSE)


def test_resume_without_clock_falls_back_to_newest_seen_stamp():
    g = _gate()
    g.note_mode(True)
    g.should_fuse(4.0)               # newest seen during the turn
    g.note_mode(False)               # no clock -> watermark = 4.0
    assert g.should_fuse(4.0) == (False, STALE_AFTER_ROTATION)
    assert g.should_fuse(4.5) == (True, FUSE)


def test_explicit_freeze_drives_the_same_resume_guard():
    g = _gate()
    g.note_explicit(True, now=0.0)   # explicit freeze (no mode topic)
    assert g.frozen is True
    g.should_fuse(1.0)
    g.note_explicit(False, now=1.0)  # explicit unfreeze arms the resume guard
    assert g.awaiting_fresh_frame is True
    assert g.should_fuse(0.9) == (False, STALE_AFTER_ROTATION)
    assert g.should_fuse(1.1) == (True, FUSE)


def test_reset_freeze_arms_resume_guard():
    g = _gate()
    g.note_mode(True, now=1.0)
    g.should_fuse(2.0)
    g.reset_freeze(now=2.0)          # manual recovery also ends the freeze
    assert g.frozen is False
    assert g.awaiting_fresh_frame is True
    assert g.should_fuse(2.5) == (True, FUSE)


def test_reentering_freeze_discards_pending_resume_guard():
    g = _gate()
    g.note_mode(True, now=1.0)
    g.should_fuse(2.0)
    g.note_mode(False, now=2.0)      # resume guard armed
    assert g.awaiting_fresh_frame is True
    g.note_mode(True, now=2.1)       # turning again before any fresh frame
    assert g.awaiting_fresh_frame is False
    assert g.frozen is True
    assert g.should_fuse(2.2) == (False, FROZEN_ROTATING)
