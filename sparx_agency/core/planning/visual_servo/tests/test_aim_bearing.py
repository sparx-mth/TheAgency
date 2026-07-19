"""Tests for the turn-to-a-bearing "aim and look" policy."""
import math

import pytest

from sparx_agency.core.planning.visual_servo.aim_bearing import (
    DONE,
    LOOK,
    SETTLE,
    TURN,
    AimBearingConfig,
    AimBearingPolicy,
)

# Small settle/look windows keep the tick loops in these tests short.
CFG = AimBearingConfig(yaw_rate=0.7, yaw_coast_rad=0.2, tolerance_rad=0.15,
                       min_burst_s=0.2, max_burst_s=0.6, settle_s=0.4, look_s=1.0,
                       timeout_s=10.0)


def _run(pol, error, ticks, dt=0.1):
    """Drive the policy for ``ticks`` at a constant heading error; return the last."""
    dec = None
    for _ in range(ticks):
        dec = pol.update(error, dt)
    return dec


def test_starts_stopped_to_arrest_the_arrival_motion():
    pol = AimBearingPolicy(CFG)
    dec = pol.update(1.0, 0.0)
    assert dec.phase == SETTLE
    assert (dec.command.x, dec.command.y, dec.command.yaw_rate) == (0.0, 0.0, 0.0)
    assert not dec.on_bearing and not dec.finished


def test_settle_then_turns_toward_a_left_bearing():
    pol = AimBearingPolicy(CFG)
    dec = _run(pol, 1.0, 4)                      # t = 0.4 >= settle_s
    assert dec.phase == TURN
    assert dec.command.yaw_rate == pytest.approx(0.7)    # + = CCW = left
    assert dec.command.metadata["source"] == "aim"


def test_turns_the_other_way_for_a_right_bearing():
    pol = AimBearingPolicy(CFG)
    dec = _run(pol, -1.0, 4)
    assert dec.phase == TURN
    assert dec.command.yaw_rate == pytest.approx(-0.7)


def test_burst_is_aimed_short_by_the_coast():
    """A 0.9 rad error with a 0.2 rad coast commands 0.7 rad => exactly 1.0 s,
    which the max_burst_s cap then clips to 0.6 s."""
    pol = AimBearingPolicy(CFG)
    _run(pol, 0.9, 4)                            # -> TURN
    assert pol.phase == TURN
    _run(pol, 0.9, 5)                            # 0.5 s of burst: still turning
    assert pol.phase == TURN
    _run(pol, 0.9, 1)                            # 0.6 s = max_burst_s -> settle
    assert pol.phase == SETTLE


def test_small_error_uses_the_minimum_burst_not_a_vanishing_one():
    """Just outside tolerance: the burst is floored at min_burst_s so the platform's
    yaw deadband cannot swallow it."""
    cfg = AimBearingConfig(yaw_rate=0.7, yaw_coast_rad=0.2, tolerance_rad=0.15,
                           min_burst_s=0.3, max_burst_s=0.6, settle_s=0.2, look_s=1.0)
    pol = AimBearingPolicy(cfg)
    _run(pol, 0.16, 2)                           # -> TURN (0.16 > tolerance)
    assert pol.phase == TURN
    _run(pol, 0.16, 2)                           # 0.2 s < min_burst_s: still turning
    assert pol.phase == TURN
    _run(pol, 0.16, 1)                           # 0.3 s = min_burst_s -> settle
    assert pol.phase == SETTLE


def test_within_tolerance_goes_straight_to_look_and_holds_still():
    pol = AimBearingPolicy(CFG)
    dec = _run(pol, 0.05, 4)                     # already on the bearing
    assert dec.phase == LOOK
    assert dec.command.yaw_rate == 0.0           # hold still: no motion blur
    assert dec.on_bearing and not dec.finished


def test_look_expires_into_a_finished_done():
    pol = AimBearingPolicy(CFG)
    _run(pol, 0.05, 4)                           # -> LOOK
    dec = _run(pol, 0.05, 11)                    # look_s = 1.0 s (float drift: 11 ticks)
    assert dec.phase == DONE
    assert dec.finished and dec.on_bearing and not dec.timed_out
    assert dec.command.yaw_rate == 0.0


def test_done_is_terminal_and_keeps_commanding_a_stop():
    pol = AimBearingPolicy(CFG)
    _run(pol, 0.05, 4)
    _run(pol, 0.05, 11)                          # -> DONE
    dec = _run(pol, 1.5, 20)                     # a huge error cannot restart it
    assert dec.phase == DONE
    assert dec.finished
    assert dec.command.yaw_rate == 0.0


def test_converging_heading_reaches_look_across_several_bursts():
    """The realistic loop, against a plant that actually turns: each burst removes
    error (a +yaw_rate shrinks a + error) until it is inside tolerance -> LOOK."""
    pol = AimBearingPolicy(CFG)
    dt, error = 0.05, 1.2
    for _ in range(400):
        dec = pol.update(error, dt)
        error -= dec.command.yaw_rate * dt       # the platform obeys the command
        if dec.phase == LOOK:
            break
    assert pol.phase == LOOK
    assert abs(error) <= CFG.tolerance_rad


def test_timeout_finishes_when_the_drone_never_turns():
    """A platform that will not yaw (the error never shrinks) must still terminate,
    and must report that it never made the bearing."""
    cfg = AimBearingConfig(yaw_rate=0.7, yaw_coast_rad=0.2, tolerance_rad=0.15,
                           min_burst_s=0.2, max_burst_s=0.6, settle_s=0.2,
                           look_s=1.0, timeout_s=2.0)
    pol = AimBearingPolicy(cfg)
    dec = _run(pol, 1.5, 30, dt=0.1)             # 3.0 s > timeout_s, error never moves
    assert dec.phase == DONE
    assert dec.finished and dec.timed_out
    assert not dec.on_bearing                    # never actually pointed at it


def test_timeout_does_not_cut_a_look_short():
    """Once looking, the look window owns the ending: the detector's chance is not
    truncated by the episode cap."""
    cfg = AimBearingConfig(yaw_rate=0.7, yaw_coast_rad=0.2, tolerance_rad=0.15,
                           min_burst_s=0.2, max_burst_s=0.6, settle_s=0.2,
                           look_s=3.0, timeout_s=1.0)
    pol = AimBearingPolicy(cfg)
    dec = _run(pol, 0.05, 12, dt=0.1)            # 1.2 s: past timeout, mid-LOOK
    assert dec.phase == LOOK
    assert not dec.finished


def test_reset_restarts_the_manoeuvre():
    pol = AimBearingPolicy(CFG)
    _run(pol, 0.05, 4)
    _run(pol, 0.05, 11)                          # -> DONE
    pol.reset()
    dec = pol.update(0.05, 0.0)
    assert dec.phase == SETTLE
    assert not dec.finished and not dec.timed_out


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        AimBearingConfig(yaw_rate=0.0)
    with pytest.raises(ValueError):
        AimBearingConfig(tolerance_rad=0.0)
    with pytest.raises(ValueError):
        AimBearingConfig(min_burst_s=0.5, max_burst_s=0.2)
    with pytest.raises(ValueError):
        AimBearingConfig(timeout_s=0.0)
