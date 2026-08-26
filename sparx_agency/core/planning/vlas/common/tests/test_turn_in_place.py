"""A discrete turn has to end pointing somewhere new, and has to end at all."""
import math

import pytest

from sparx_agency.core.planning.vlas.common.turn_in_place import (
    IDLE,
    SETTLING,
    TURNING,
    TurnInPlace,
    TurnSpec,
    describe,
    turn_spec_from_config,
)

DT = 0.05


def fly(turn, delta_deg, ticks=1200, blocked=False, dt=DT):
    """Run a manoeuvre against a perfect yaw integrator; return (yaw, seconds, cmd)."""
    yaw, now = 0.0, 0.0
    turn.start(yaw, math.radians(delta_deg), now)
    last = None
    for _ in range(ticks):
        last = turn.update(yaw, now)
        if not blocked:
            yaw += last.yaw_rate * dt
        now += dt
        if last.done:
            break
    return yaw, now, last


class TestArrival:
    @pytest.mark.parametrize("delta_deg", [15.0, -15.0, 30.0, -45.0, 90.0])
    def test_ends_within_tolerance_of_the_commanded_turn(self, delta_deg):
        turn = TurnInPlace(TurnSpec(timeout_s=30.0))
        yaw, _, cmd = fly(turn, delta_deg)
        assert cmd.done and not cmd.timed_out
        assert abs(math.degrees(yaw) - delta_deg) <= math.degrees(turn.spec.tolerance_rad)

    def test_reports_done_exactly_once_then_goes_idle(self):
        turn = TurnInPlace()
        _, now, cmd = fly(turn, 15.0)
        assert cmd.done
        after = turn.update(0.26, now + DT)
        assert not after.done and not after.active and after.state == IDLE

    def test_the_turn_is_a_rotation_and_commands_no_translation(self):
        # The whole point: a TurnCommand carries a yaw rate and nothing else, so
        # a follower physically cannot crab through the turn the way a bent
        # waypoint let it.
        turn = TurnInPlace()
        cmd = turn.update(0.0, 0.0) if turn.active else None
        assert cmd is None or not cmd.active
        turn.start(0.0, math.radians(15.0), 0.0)
        cmd = turn.update(0.0, 0.0)
        assert cmd.active and cmd.yaw_rate > 0.0
        assert set(cmd.__dataclass_fields__) == {
            "yaw_rate", "active", "done", "timed_out", "remaining_rad", "state"}


class TestSettling:
    def test_a_coasting_aircraft_is_not_settled(self):
        # Inside tolerance but still turning: the observation at the end of the
        # manoeuvre is the point of it, and it must not be taken mid-coast.
        turn = TurnInPlace(TurnSpec(settle_s=0.4, settle_rate_rad_s=0.05))
        turn.start(0.0, 0.0, 0.0)          # already at the target
        now = 0.0
        for _ in range(40):
            cmd = turn.update(0.0, now, measured_yaw_rate=0.5)
            now += DT
            assert not cmd.done
            assert cmd.state == SETTLING and cmd.yaw_rate == 0.0

    def test_settles_once_the_yaw_rate_falls(self):
        turn = TurnInPlace(TurnSpec(settle_s=0.4))
        turn.start(0.0, 0.0, 0.0)
        now = 0.0
        for _ in range(40):
            cmd = turn.update(0.0, now, measured_yaw_rate=0.0)
            now += DT
            if cmd.done:
                break
        assert cmd.done and not cmd.timed_out
        assert now == pytest.approx(0.45, abs=0.06)

    def test_drifting_back_out_of_tolerance_resumes_turning(self):
        turn = TurnInPlace(TurnSpec(settle_s=1.0))
        turn.start(0.0, 0.0, 0.0)
        assert turn.update(0.0, 0.0, measured_yaw_rate=0.0).state == SETTLING
        blown = turn.update(-0.5, 0.05, measured_yaw_rate=0.0)
        assert blown.state == TURNING and blown.yaw_rate > 0.0


class TestItNeverWedgesTheFlight:
    def test_a_blocked_rotation_times_out_rather_than_hanging(self):
        turn = TurnInPlace(TurnSpec(timeout_s=2.0))
        _, now, cmd = fly(turn, 15.0, blocked=True)
        assert cmd.done and cmd.timed_out
        assert now == pytest.approx(2.05, abs=0.1)
        assert abs(math.degrees(cmd.remaining_rad)) == pytest.approx(15.0, abs=0.1)

    def test_idle_updates_are_inert(self):
        turn = TurnInPlace()
        cmd = turn.update(1.0, 0.0)
        assert not cmd.active and not cmd.done and cmd.yaw_rate == 0.0

    def test_cancel_stops_without_reporting_done(self):
        turn = TurnInPlace()
        turn.start(0.0, 1.0, 0.0)
        turn.cancel()
        assert not turn.active and turn.target is None
        assert not turn.update(0.0, 0.1).done


class TestWrap:
    def test_a_turn_across_pi_takes_the_short_way(self):
        turn = TurnInPlace(TurnSpec(timeout_s=30.0))
        yaw = math.radians(175.0)
        turn.start(yaw, math.radians(20.0), 0.0)
        assert turn.target == pytest.approx(math.radians(-165.0), abs=1e-6)
        cmd = turn.update(yaw, 0.0)
        assert cmd.yaw_rate > 0.0          # CCW, the 20 deg way, not 340 the other


class TestSpec:
    @pytest.mark.parametrize("kwargs", [
        {"yaw_rate": 0.0}, {"min_yaw_rate": -0.1}, {"tolerance_rad": 0.0},
        {"timeout_s": 0.0}, {"slow_down_rad": 0.0}, {"settle_s": -1.0},
        {"min_yaw_rate": 1.0, "yaw_rate": 0.3},
    ])
    def test_a_spec_that_cannot_converge_is_refused(self, kwargs):
        with pytest.raises(ValueError):
            TurnSpec(**kwargs)

    def test_config_reads_degrees_where_a_human_writes_degrees(self):
        spec = turn_spec_from_config({
            "turn_yaw_rate_deg_s": 30.0, "turn_tolerance_deg": 3.0,
            "turn_settle_s": 0.7}, prefix="turn_")
        assert spec.yaw_rate == pytest.approx(math.radians(30.0))
        assert spec.tolerance_rad == pytest.approx(math.radians(3.0))
        assert spec.settle_s == pytest.approx(0.7)
        assert spec.timeout_s == TurnSpec().timeout_s     # untouched knobs default

    def test_describe_names_the_manoeuvre_in_degrees(self):
        assert "deg/s" in describe(TurnSpec())


def test_module_imports_without_numpy():
    """The Noetic container needs this; numpy 1.17 is there but nothing heavier."""
    import sparx_agency.core.planning.vlas.common.turn_in_place as mod
    src = open(mod.__file__).read()
    assert "import numpy" not in src and "import cv2" not in src
