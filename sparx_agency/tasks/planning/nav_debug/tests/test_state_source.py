"""Differentiating a pose spine into a velocity, without inventing numbers.

This is the fallback that makes the reference-vs-actual comparison work on runs
already on disk: the ROS1 half never recorded a measured velocity (telemetry's
``vx``/``vy``/``vz``/``wz`` are the *command*), so for every flight without a
ROS2 half the only measurement available is the pose itself. The properties
worth pinning down are therefore the ones that decide whether a derived number
is trustworthy: it must be centred, it must fold heading wraps, and it must
refuse rather than guess at the ends of a run and across a dropout.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.tasks.planning.nav_debug import state_source
from sparx_agency.tasks.planning.nav_debug.state_source import (
    StateSource, derive_velocities,
)

_DT = 0.05


def _rows(n, *, vx=1.0, yaw_rate=0.0, t0=0.0, gap_at=None):
    """A straight-line spine at ``vx`` m/s, optionally with a dropout."""
    rows, t = [], t0
    for i in range(n):
        if gap_at is not None and i == gap_at:
            t += 5.0                      # a recording dropout, not a sample gap
        rows.append({"t": t, "x": vx * t, "y": 2.0, "z": 1.5,
                     "yaw": _wrap(yaw_rate * t)})
        t += _DT
    return rows


def _wrap(radians):
    return math.atan2(math.sin(radians), math.cos(radians))


def test_a_steady_run_recovers_its_own_speed():
    derived = derive_velocities(_rows(60, vx=1.3))
    middle = derived[30]
    assert middle["vx"] == pytest.approx(1.3, rel=1e-6)
    assert middle["vy"] == pytest.approx(0.0, abs=1e-9)


def test_the_ends_of_a_run_are_unknown_not_zero():
    """There is no window to centre on, and 0.0 would read as "standing still"."""
    derived = derive_velocities(_rows(60))
    assert derived[0] is None
    assert derived[-1] is None


def test_a_dropout_is_not_differenced_across():
    """Two poses either side of a 5 s gap say nothing about the speed between them."""
    derived = derive_velocities(_rows(60, gap_at=30))
    assert derived[30] is None
    assert derived[29] is None            # its forward reach crosses the gap too
    assert derived[10] is not None        # the rest of the run is unaffected


def test_a_heading_wrap_is_not_a_spike():
    """Crossing +/-pi must fold, or one frame reports a ~125 rad/s turn rate."""
    derived = derive_velocities(_rows(60, vx=0.0, yaw_rate=1.0))
    rates = [d["wz"] for d in derived if d is not None]
    assert rates
    assert max(abs(r) for r in rates) == pytest.approx(1.0, rel=1e-6)


def test_the_window_is_centred_not_trailing():
    """A trailing difference would lag the frame by half a window on every row."""
    rows = _rows(80, vx=1.0)
    i = 40
    derived = derive_velocities(rows)[i]
    assert derived is not None
    # Symmetric about the frame: the same result forwards and backwards in time.
    mirrored = derive_velocities(list(reversed(
        [dict(r, t=rows[-1]["t"] - r["t"] + rows[0]["t"], x=-r["x"])
         for r in rows])))[len(rows) - 1 - i]
    assert derived["vx"] == pytest.approx(mirrored["vx"], rel=1e-6)


# ── source precedence ────────────────────────────────────────────────────────
def test_odom_wins_over_ground_truth():
    """The state the controller reacted to beats a second process's measurement."""
    source = StateSource(_rows(20))
    state = source.at(
        10, {"state": {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.4,
                       "vx": 0.7, "vy": 0.0, "vz": 0.0, "wz": 0.1}},
        {"vx": 9.9, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0})
    assert state.source == state_source.ODOM
    assert state.vx == pytest.approx(0.7)


def test_a_state_section_without_velocity_is_not_a_velocity_source():
    """Falling through beats reporting an all-dash "measurement" from a real source."""
    source = StateSource(_rows(20))
    state = source.at(10, {"state": {"x": 1.0, "y": 2.0}},
                      {"vx": 0.4, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0})
    assert state.source == state_source.TRUTH
    assert state.vx == pytest.approx(0.4)


def test_ground_truth_reports_its_own_pose():
    """Sphera knows where it is; the sphera sub-dict must not be dropped."""
    source = StateSource(_rows(20))
    state = source.at(10, None, {"vx": 0.4, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.2,
                                 "sphera": {"x": -59.1, "y": 18.4, "z": 0.12,
                                            "yaw": 0.08}})
    assert state.source == state_source.TRUTH
    assert state.x == pytest.approx(-59.1)
    assert state.wz == pytest.approx(0.2)


def test_the_pose_spine_is_the_last_resort():
    source = StateSource(_rows(20))
    state = source.at(10, None, None)
    assert state.source == state_source.POSE_DIFF
    assert state.vx == pytest.approx(1.0, rel=1e-6)


def test_an_underivable_frame_still_reports_its_pose():
    """Position and heading are measured even where velocity is not derivable."""
    state = StateSource(_rows(20)).at(0, None, None)
    assert state.source == state_source.POSE_ONLY
    assert state.y == pytest.approx(2.0)
    assert state.vx is None and state.speed is None
