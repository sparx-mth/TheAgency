"""Unit tests for the pure visual-servo control-law functions.

Covers the sign conventions and saturations of the stateless helpers in
:mod:`sparx_agency.core.planning.visual_servo.algorithm` and the validation
logic of :class:`sparx_agency.core.planning.visual_servo.params.VisualServoParams`.

Sign conventions under test (REP-103 body frame, image origin top-left):
    ``+vx`` forward, ``+vy`` left, ``+vz`` up, ``+yaw_rate`` CCW; image ``+x``
    right, ``+y`` down. A target right of centre (``ox > 0``) is centred by
    yawing CW (``yaw_rate < 0``) / crabbing right (``vy < 0``); a target above
    centre (``oy < 0``) by climbing (``vz > 0``).
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.visual_servo import algorithm as alg
from sparx_agency.core.planning.visual_servo.params import VisualServoParams


# --------------------------------------------------------------------------- #
# saturate / clamp01
# --------------------------------------------------------------------------- #
class TestSaturate:
    def test_within_limits_passthrough(self):
        assert alg.saturate(0.3, 1.0) == pytest.approx(0.3)
        assert alg.saturate(-0.3, 1.0) == pytest.approx(-0.3)
        assert alg.saturate(0.0, 1.0) == 0.0

    def test_clamps_upper(self):
        assert alg.saturate(5.0, 1.0) == pytest.approx(1.0)

    def test_clamps_lower(self):
        assert alg.saturate(-5.0, 1.0) == pytest.approx(-1.0)

    def test_exactly_at_limit(self):
        assert alg.saturate(1.0, 1.0) == pytest.approx(1.0)
        assert alg.saturate(-1.0, 1.0) == pytest.approx(-1.0)

    def test_returns_float(self):
        assert isinstance(alg.saturate(2, 1.0), float)


class TestClamp01:
    def test_within(self):
        assert alg.clamp01(0.5) == pytest.approx(0.5)

    def test_below_zero(self):
        assert alg.clamp01(-0.7) == 0.0

    def test_above_one(self):
        assert alg.clamp01(3.2) == 1.0

    def test_boundaries(self):
        assert alg.clamp01(0.0) == 0.0
        assert alg.clamp01(1.0) == 1.0

    def test_returns_float(self):
        assert isinstance(alg.clamp01(1), float)


# --------------------------------------------------------------------------- #
# yaw_command
# --------------------------------------------------------------------------- #
class TestYawCommand:
    KP = 1.2
    MAX = 0.6
    DB = 0.03

    def test_right_of_centre_yields_negative_yaw(self):
        # ox > 0 -> yaw CW -> negative.
        y = alg.yaw_command(0.2, self.KP, self.MAX, self.DB)
        assert y < 0.0

    def test_left_of_centre_yields_positive_yaw(self):
        y = alg.yaw_command(-0.2, self.KP, self.MAX, self.DB)
        assert y > 0.0

    def test_inside_deadband_is_zero(self):
        assert alg.yaw_command(0.02, self.KP, self.MAX, self.DB) == 0.0
        assert alg.yaw_command(-0.02, self.KP, self.MAX, self.DB) == 0.0

    def test_exactly_at_deadband_is_zero(self):
        assert alg.yaw_command(self.DB, self.KP, self.MAX, self.DB) == 0.0
        assert alg.yaw_command(-self.DB, self.KP, self.MAX, self.DB) == 0.0

    def test_just_outside_deadband_is_nonzero(self):
        assert alg.yaw_command(self.DB + 1e-6, self.KP, self.MAX, self.DB) != 0.0

    def test_saturates_at_max(self):
        # ox large: -kp*ox << -max -> clamps to -max.
        assert alg.yaw_command(10.0, self.KP, self.MAX, self.DB) == pytest.approx(-self.MAX)
        assert alg.yaw_command(-10.0, self.KP, self.MAX, self.DB) == pytest.approx(self.MAX)

    def test_proportional_before_saturation(self):
        # 0.1 offset -> -0.12, within the 0.6 saturation.
        assert alg.yaw_command(0.1, self.KP, self.MAX, self.DB) == pytest.approx(-0.12)


# --------------------------------------------------------------------------- #
# lateral_command
# --------------------------------------------------------------------------- #
class TestLateralCommand:
    KP = 0.25
    MAX = 0.25
    DB = 0.03

    def test_right_of_centre_yields_negative_vy(self):
        # ox > 0 -> crab right -> vy < 0.
        assert alg.lateral_command(0.2, self.KP, self.MAX, self.DB) < 0.0

    def test_left_of_centre_yields_positive_vy(self):
        assert alg.lateral_command(-0.2, self.KP, self.MAX, self.DB) > 0.0

    def test_inside_deadband_zero(self):
        assert alg.lateral_command(0.01, self.KP, self.MAX, self.DB) == 0.0

    def test_saturates_at_max(self):
        assert alg.lateral_command(10.0, self.KP, self.MAX, self.DB) == pytest.approx(-self.MAX)
        assert alg.lateral_command(-10.0, self.KP, self.MAX, self.DB) == pytest.approx(self.MAX)


# --------------------------------------------------------------------------- #
# vertical_command
# --------------------------------------------------------------------------- #
class TestVerticalCommand:
    KP = 0.25
    MAX = 0.2
    DB = 0.03

    def test_above_centre_climbs(self):
        # oy < 0 (image +y is down => above centre) -> climb -> vz > 0.
        assert alg.vertical_command(-0.2, self.KP, self.MAX, self.DB) > 0.0

    def test_below_centre_descends(self):
        assert alg.vertical_command(0.2, self.KP, self.MAX, self.DB) < 0.0

    def test_inside_deadband_zero(self):
        assert alg.vertical_command(0.01, self.KP, self.MAX, self.DB) == 0.0
        assert alg.vertical_command(-0.01, self.KP, self.MAX, self.DB) == 0.0

    def test_saturates_at_max(self):
        assert alg.vertical_command(-10.0, self.KP, self.MAX, self.DB) == pytest.approx(self.MAX)
        assert alg.vertical_command(10.0, self.KP, self.MAX, self.DB) == pytest.approx(-self.MAX)


# --------------------------------------------------------------------------- #
# centering_gain
# --------------------------------------------------------------------------- #
class TestCenteringGain:
    MAX = 0.35

    def test_one_when_centred(self):
        assert alg.centering_gain(0.0, self.MAX) == pytest.approx(1.0)

    def test_zero_at_and_beyond_max(self):
        assert alg.centering_gain(self.MAX, self.MAX) == pytest.approx(0.0)
        assert alg.centering_gain(self.MAX + 0.1, self.MAX) == pytest.approx(0.0)
        assert alg.centering_gain(-(self.MAX + 0.5), self.MAX) == pytest.approx(0.0)

    def test_symmetric_in_sign(self):
        assert alg.centering_gain(0.1, self.MAX) == pytest.approx(
            alg.centering_gain(-0.1, self.MAX)
        )

    def test_linear_midpoint(self):
        # |ox| = half of max -> gain 0.5.
        assert alg.centering_gain(self.MAX / 2.0, self.MAX) == pytest.approx(0.5)

    def test_monotone_decreasing(self):
        xs = np.linspace(0.0, self.MAX, 50)
        gains = [alg.centering_gain(float(x), self.MAX) for x in xs]
        for a, b in zip(gains, gains[1:]):
            assert b <= a + 1e-12
        # Strictly decreasing across the full span (endpoints differ).
        assert gains[0] > gains[-1]

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(20240708)
        for x in rng.uniform(-1.0, 1.0, size=100):
            g = alg.centering_gain(float(x), self.MAX)
            assert 0.0 <= g <= 1.0

    def test_nonpositive_max_degenerate(self):
        # advance_offset_max <= 0: 1 only when perfectly centred, else 0.
        assert alg.centering_gain(0.0, 0.0) == pytest.approx(1.0)
        assert alg.centering_gain(0.01, 0.0) == pytest.approx(0.0)
        assert alg.centering_gain(0.0, -1.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# forward_from_range
# --------------------------------------------------------------------------- #
class TestForwardFromRange:
    TARGET = 0.8
    SLOWDOWN = 2.0
    VXMAX = 0.35
    KP = 0.6

    def _f(self, r):
        return alg.forward_from_range(r, self.TARGET, self.SLOWDOWN, self.VXMAX, self.KP)

    def test_full_speed_beyond_slowdown(self):
        assert self._f(self.SLOWDOWN) == pytest.approx(self.VXMAX)
        assert self._f(self.SLOWDOWN + 1.0) == pytest.approx(self.VXMAX)

    def test_zero_at_and_within_target(self):
        assert self._f(self.TARGET) == 0.0
        assert self._f(self.TARGET - 0.2) == 0.0

    def test_ramps_between(self):
        mid = self._f((self.TARGET + self.SLOWDOWN) / 2.0)
        assert 0.0 < mid <= self.VXMAX

    def test_ramp_value(self):
        # range 1.0: clamp01(0.6 * (1.0 - 0.8)) * 0.35 = 0.12 * 0.35.
        assert self._f(1.0) == pytest.approx(0.12 * self.VXMAX)

    def test_never_negative_and_monotone_nondecreasing(self):
        xs = np.linspace(0.0, self.SLOWDOWN + 1.0, 200)
        vals = [self._f(float(x)) for x in xs]
        assert all(v >= 0.0 for v in vals)
        for a, b in zip(vals, vals[1:]):
            assert b >= a - 1e-12

    def test_never_exceeds_vxmax(self):
        xs = np.linspace(0.0, self.SLOWDOWN + 5.0, 100)
        assert all(self._f(float(x)) <= self.VXMAX + 1e-12 for x in xs)


# --------------------------------------------------------------------------- #
# forward_from_area
# --------------------------------------------------------------------------- #
class TestForwardFromArea:
    TARGET = 0.12
    SLOWDOWN = 0.03
    VXMAX = 0.35

    def _f(self, a):
        return alg.forward_from_area(a, self.TARGET, self.SLOWDOWN, self.VXMAX)

    def test_full_speed_below_slowdown(self):
        assert self._f(0.0) == pytest.approx(self.VXMAX)
        assert self._f(self.SLOWDOWN - 0.005) == pytest.approx(self.VXMAX)

    def test_zero_at_and_above_target(self):
        assert self._f(self.TARGET) == 0.0
        assert self._f(self.TARGET + 0.1) == 0.0

    def test_ramps_between(self):
        mid = self._f((self.TARGET + self.SLOWDOWN) / 2.0)
        assert 0.0 < mid < self.VXMAX

    def test_ramp_value_midpoint(self):
        # area at midpoint of [slowdown, target] -> 0.5 * vx_max.
        mid_area = (self.TARGET + self.SLOWDOWN) / 2.0
        assert self._f(mid_area) == pytest.approx(0.5 * self.VXMAX)

    def test_never_negative_and_monotone_nonincreasing(self):
        xs = np.linspace(0.0, self.TARGET + 0.05, 200)
        vals = [self._f(float(x)) for x in xs]
        assert all(v >= 0.0 for v in vals)
        for a, b in zip(vals, vals[1:]):
            assert b <= a + 1e-12

    def test_never_exceeds_vxmax(self):
        xs = np.linspace(0.0, self.TARGET + 0.05, 100)
        assert all(self._f(float(x)) <= self.VXMAX + 1e-12 for x in xs)


# --------------------------------------------------------------------------- #
# ema
# --------------------------------------------------------------------------- #
class TestEma:
    def test_half_blend(self):
        assert alg.ema(0.0, 1.0, 0.5) == pytest.approx(0.5)

    def test_blend_one_is_target(self):
        assert alg.ema(3.0, 9.0, 1.0) == pytest.approx(9.0)

    def test_blend_zero_is_prev(self):
        assert alg.ema(3.0, 9.0, 0.0) == pytest.approx(3.0)

    def test_convex_combination(self):
        rng = np.random.default_rng(1234)
        for _ in range(100):
            prev, target = rng.uniform(-5.0, 5.0, size=2)
            blend = rng.uniform(0.0, 1.0)
            out = alg.ema(float(prev), float(target), float(blend))
            lo, hi = sorted((prev, target))
            assert lo - 1e-9 <= out <= hi + 1e-9
            assert out == pytest.approx(prev + blend * (target - prev))

    def test_returns_float(self):
        assert isinstance(alg.ema(0, 1, 0.5), float)


# --------------------------------------------------------------------------- #
# VisualServoParams.__post_init__
# --------------------------------------------------------------------------- #
class TestVisualServoParamsValidation:
    def test_defaults_valid(self):
        p = VisualServoParams()
        assert p.mode == "holonomic"

    def test_both_modes_accepted(self):
        assert VisualServoParams(mode="holonomic").mode == "holonomic"
        assert VisualServoParams(mode="yaw_forward_xor").mode == "yaw_forward_xor"

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError):
            VisualServoParams(mode="sideways")

    def test_nonpositive_advance_offset_rejected(self):
        with pytest.raises(ValueError):
            VisualServoParams(advance_offset_max=0.0)

    def test_slowdown_range_not_greater_than_target_rejected(self):
        with pytest.raises(ValueError):
            VisualServoParams(target_range_m=0.8, slowdown_range_m=0.8)  # equal
        with pytest.raises(ValueError):
            VisualServoParams(target_range_m=1.0, slowdown_range_m=0.5)  # less

    def test_target_area_not_greater_than_slowdown_rejected(self):
        with pytest.raises(ValueError):
            VisualServoParams(target_area_frac=0.03, slowdown_area_frac=0.03)  # equal
        with pytest.raises(ValueError):
            VisualServoParams(target_area_frac=0.02, slowdown_area_frac=0.05)  # less

    def test_yaw_deadband_exit_not_less_than_enter_rejected(self):
        with pytest.raises(ValueError):
            VisualServoParams(yaw_deadband_exit=0.20, yaw_deadband_enter=0.20)  # equal
        with pytest.raises(ValueError):
            VisualServoParams(yaw_deadband_exit=0.30, yaw_deadband_enter=0.20)  # greater

    @pytest.mark.parametrize("field", ["speed_smoothing", "yaw_smoothing"])
    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_smoothing_out_of_range_rejected(self, field, bad):
        with pytest.raises(ValueError):
            VisualServoParams(**{field: bad})

    @pytest.mark.parametrize("field", ["speed_smoothing", "yaw_smoothing"])
    def test_smoothing_boundaries_accepted(self, field):
        # (0, 1] -> 1.0 valid, small positive valid.
        VisualServoParams(**{field: 1.0})
        VisualServoParams(**{field: 1e-6})
