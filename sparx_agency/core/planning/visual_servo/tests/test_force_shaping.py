"""Tests for per-axis closure force shaping (none / snap / fixed)."""
import math

import pytest

from sparx_agency.core.common.types import ControlCommand
from sparx_agency.core.planning.trackers.multi_axis_follower.allocation import shape_axis
from sparx_agency.core.planning.visual_servo.force_shaping import (
    AxisForceProfile,
    CommandForceShaper,
    shape_axis_force,
)


# ── none ──────────────────────────────────────────────────────────────────────
def test_none_passes_through_and_caps_at_max():
    p = AxisForceProfile(min_magnitude=0.06, max_magnitude=0.4, mode="none")
    assert shape_axis_force(0.01, p) == pytest.approx(0.01)   # no floor
    assert shape_axis_force(0.9, p) == pytest.approx(0.4)     # capped
    assert shape_axis_force(-0.9, p) == pytest.approx(-0.4)


def test_none_without_max_is_identity():
    p = AxisForceProfile(min_magnitude=0.06, mode="none")
    assert shape_axis_force(0.123, p) == pytest.approx(0.123)


# ── snap (multi-axis parity) ──────────────────────────────────────────────────
def test_snap_matches_multi_axis_shape_axis():
    p = AxisForceProfile(min_magnitude=0.06, release_frac=0.5, zero_eps=1e-3, mode="snap")
    for v in (-0.5, -0.06, -0.04, -0.02, 0.0, 0.01, 0.03, 0.05, 0.06, 0.2):
        assert shape_axis_force(v, p) == pytest.approx(
            shape_axis(v, 0.06, 0.5, 1e-3))


def test_snap_regions():
    p = AxisForceProfile(min_magnitude=0.06, release_frac=0.5, mode="snap")
    assert shape_axis_force(0.02, p) == 0.0          # below release_frac*min (0.03)
    assert shape_axis_force(0.05, p) == pytest.approx(0.06)   # snapped up to min
    assert shape_axis_force(0.2, p) == pytest.approx(0.2)     # above min: passthrough


def test_snap_caps_at_max():
    p = AxisForceProfile(min_magnitude=0.06, max_magnitude=0.3, mode="snap")
    assert shape_axis_force(0.9, p) == pytest.approx(0.3)


# ── fixed (bang-bang) ─────────────────────────────────────────────────────────
def test_fixed_is_bang_bang_at_min():
    p = AxisForceProfile(min_magnitude=0.06, release_frac=0.5, mode="fixed")
    assert shape_axis_force(0.02, p) == 0.0                   # below deadband
    assert shape_axis_force(0.05, p) == pytest.approx(0.06)   # -> +fixed
    assert shape_axis_force(0.9, p) == pytest.approx(0.06)    # still +fixed (not analog)
    assert shape_axis_force(-0.9, p) == pytest.approx(-0.06)  # sign preserved


def test_fixed_uses_explicit_fixed_magnitude():
    p = AxisForceProfile(min_magnitude=0.06, fixed_magnitude=0.15, mode="fixed")
    assert shape_axis_force(0.5, p) == pytest.approx(0.15)


def test_fixed_level_capped_by_max():
    p = AxisForceProfile(min_magnitude=0.06, fixed_magnitude=0.5, max_magnitude=0.2,
                         mode="fixed")
    assert shape_axis_force(0.9, p) == pytest.approx(0.2)


# ── CommandForceShaper ────────────────────────────────────────────────────────
def _cmd(vx, vy, vz, wz, **meta):
    return ControlCommand.velocity(vx, vy, vz, wz, **meta)


def test_command_shaper_shapes_all_axes_fixed():
    shaper = CommandForceShaper(
        vx=AxisForceProfile(min_magnitude=0.06, mode="fixed"),
        vy=AxisForceProfile(min_magnitude=0.06, mode="fixed"),
        wz=AxisForceProfile(min_magnitude=math.radians(8), mode="fixed"),
    )
    out = shaper.shape(_cmd(0.4, -0.01, 0.0, 0.5))
    assert out.x == pytest.approx(0.06)          # forward -> +fixed
    assert out.y == 0.0                          # tiny lateral -> deadband to 0
    assert out.z == 0.0                          # vz passthrough (no profile)
    assert out.yaw_rate == pytest.approx(math.radians(8))


def test_command_shaper_preserves_metadata_and_mode():
    shaper = CommandForceShaper(
        vx=AxisForceProfile(min_magnitude=0.06, mode="none"),
        vy=AxisForceProfile(min_magnitude=0.06, mode="none"),
        wz=AxisForceProfile(min_magnitude=0.1, mode="none"),
    )
    out = shaper.shape(_cmd(0.1, 0.0, 0.0, 0.0, source="servo", phase="APPROACH"))
    assert out.metadata["source"] == "servo"
    assert out.metadata["phase"] == "APPROACH"
    assert out.mode.value == "velocity"


def test_command_shaper_vz_shaped_when_profile_given():
    shaper = CommandForceShaper(
        vx=AxisForceProfile(min_magnitude=0.06, mode="fixed"),
        vy=AxisForceProfile(min_magnitude=0.06, mode="fixed"),
        wz=AxisForceProfile(min_magnitude=0.1, mode="fixed"),
        vz=AxisForceProfile(min_magnitude=0.05, mode="fixed"),
    )
    out = shaper.shape(_cmd(0.0, 0.0, 0.3, 0.0))
    assert out.z == pytest.approx(0.05)


# ── validation ────────────────────────────────────────────────────────────────
def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        AxisForceProfile(min_magnitude=0.06, mode="bang")


def test_invalid_release_frac_rejected():
    with pytest.raises(ValueError):
        AxisForceProfile(min_magnitude=0.06, release_frac=1.5)
