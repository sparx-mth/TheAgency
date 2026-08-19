"""The exploration progress monitor must fire on stuck, and only on stuck.

Every test here is a mission shape that really happened in the SJTU Gazebo
campaign, or the legitimate mission it is most easily confused with. The pairs
matter more than the individual cases: a watchdog that fires on a doorway loop
is worthless if it also fires on a thorough sweep of one crowded room.
"""
import math

import pytest

from sparx_agency.core.planning.exploration.progress_monitor import (
    ABORT_CONFINED,
    ABORT_NO_GROWTH,
    ABORT_NO_MOVEMENT,
    ABORT_TIME_CAP,
    ExplorationProgressConfig,
    ExplorationProgressMonitor,
    NUDGE_CONFINED,
    RUNNING,
)


def _config(**overrides):
    """A short-window config so a test mission is a few hundred samples."""
    base = dict(time_cap_s=1000.0, grace_s=20.0, window_s=60.0,
                confine_radius_m=3.0, growth_m3_per_min=3.0,
                confine_cap_s=120.0, nudge_every_s=30.0, barren_cap_s=150.0,
                no_move_m=0.6, no_move_cap_s=60.0, min_growth_m3=0.5)
    base.update(overrides)
    return ExplorationProgressConfig(**base)


def _fly(monitor, path, dt=1.0, t0=0.0):
    """Feed ``(x, y, z, coverage)`` samples one per ``dt`` and keep the verdicts."""
    verdicts = []
    t = t0
    for x, y, z, cov in path:
        verdicts.append(monitor.update(t, (x, y, z), cov))
        t += dt
    return verdicts


def _sweep(n, dt=1.0, speed=0.25, growth_per_s=0.6):
    """A healthy mission: flying away in a straight line, mapping as it goes."""
    return [(speed * dt * i, 0.0, 1.2, growth_per_s * dt * i) for i in range(n)]


def _orbit(n, radius=1.2, period_s=20.0, dt=1.0, coverage=200.0,
           growth_per_s=0.0, centre=(0.0, 0.0)):
    """The failure: circling one spot, learning nothing."""
    out = []
    for i in range(n):
        a = 2.0 * math.pi * (i * dt) / period_s
        out.append((centre[0] + radius * math.cos(a),
                    centre[1] + radius * math.sin(a),
                    1.2, coverage + growth_per_s * dt * i))
    return out


# ── the healthy missions, which must never be aborted ────────────────────

def test_a_straight_productive_flight_is_never_faulted():
    monitor = ExplorationProgressMonitor(_config())
    verdicts = _fly(monitor, _sweep(600))
    assert all(v.state == RUNNING for v in verdicts)


def test_a_slow_thorough_sweep_of_one_room_is_not_confinement():
    """Confined, but the map is growing: a crowded room legitimately takes time.

    This is the case that rules out a radius-only watchdog. The aircraft stays
    inside 2.5 m for the whole mission and is doing exactly what it should.
    """
    monitor = ExplorationProgressMonitor(_config())
    path = _orbit(600, radius=2.0, period_s=40.0, growth_per_s=0.2)
    verdicts = _fly(monitor, path)
    assert all(v.state == RUNNING for v in verdicts), \
        [v.reason for v in verdicts if v.state != RUNNING][:1]


def test_a_long_transit_through_mapped_space_is_not_barrenness():
    """Not growing, but going somewhere: the flight to a far frontier.

    This is the case that rules out a growth-only watchdog over short windows.
    Coverage is flat for 100 s while the aircraft crosses a corridor it has
    already mapped, and that is normal.
    """
    monitor = ExplorationProgressMonitor(_config())
    transit = [(0.25 * i, 0.0, 1.2, 300.0) for i in range(100)]
    arrival = [(25.0 + 0.25 * i, 0.0, 1.2, 300.0 + 0.6 * i) for i in range(100)]
    verdicts = _fly(monitor, transit + arrival)
    assert all(v.state == RUNNING for v in verdicts)


def test_the_grace_period_covers_takeoff_and_the_survey_turn():
    """Parked on the spot with flat coverage is normal for the first seconds."""
    monitor = ExplorationProgressMonitor(_config(grace_s=60.0))
    verdicts = _fly(monitor, [(1.0, 1.0, 1.0, 0.0)] * 55)
    assert all(v.state == RUNNING for v in verdicts)


# ── the failures, which must be caught ───────────────────────────────────

def test_a_doorway_orbit_nudges_before_it_aborts():
    """The real hospital failure: circling in front of a door it cannot thread.

    The escalation is the point. A confined mission is first asked to try
    something else, several times, and only killed once that has not worked --
    a watchdog that can only kill wastes every run it fires on.
    """
    monitor = ExplorationProgressMonitor(_config())
    verdicts = _fly(monitor, _orbit(400))
    states = [v.state for v in verdicts]
    assert NUDGE_CONFINED in states
    assert ABORT_CONFINED in states
    assert states.index(NUDGE_CONFINED) < states.index(ABORT_CONFINED)
    assert monitor.nudges >= 2


def test_confinement_aborts_only_after_the_cap_has_actually_elapsed():
    monitor = ExplorationProgressMonitor(_config(confine_cap_s=120.0))
    verdicts = _fly(monitor, _orbit(400))
    abort = next(v for v in verdicts if v.state == ABORT_CONFINED)
    assert abort.confined_for_s > 120.0
    assert abort.confinement_radius_m < 3.0
    assert abort.growth_m3_per_min < 3.0


def test_leaving_the_confining_region_clears_the_confinement_clock():
    """A mission that gets out on its own must not be killed for having been stuck."""
    monitor = ExplorationProgressMonitor(_config())
    stuck = _orbit(100)
    escape = [(20.0 + 0.25 * i, 0.0, 1.2, 200.0 + 0.6 * i) for i in range(300)]
    verdicts = _fly(monitor, stuck + escape)
    assert not any(v.is_abort for v in verdicts)
    assert verdicts[-1].state == RUNNING


def test_a_wide_tour_of_already_mapped_space_still_aborts_on_barrenness():
    """Not confined -- and still worthless. The backstop confinement misses.

    The aircraft flies a 40 m circuit of corridors it has already mapped
    because the planner cannot reach any real frontier. The confinement test
    passes it; the barren cap is what catches it.
    """
    monitor = ExplorationProgressMonitor(_config())
    path = _orbit(400, radius=20.0, period_s=200.0, growth_per_s=0.0)
    verdicts = _fly(monitor, path)
    assert any(v.state == ABORT_NO_GROWTH for v in verdicts)


def test_a_pinned_aircraft_aborts_on_movement_before_anything_else():
    """Wedged against a wall: the fastest failure, and it must be the first verdict."""
    monitor = ExplorationProgressMonitor(_config())
    verdicts = _fly(monitor, [(1.0, 1.0, 1.2, 100.0)] * 300)
    first = next(v for v in verdicts if v.is_abort)
    assert first.state == ABORT_NO_MOVEMENT


def test_jitter_under_the_movement_threshold_still_counts_as_pinned():
    """A wedged aircraft twitches; that is not travel.

    ``no_move_m`` is a net-displacement bar precisely so grinding on a wall,
    which produces continuous small motion, does not read as progress.
    """
    monitor = ExplorationProgressMonitor(_config())
    path = [(1.0 + 0.2 * math.sin(i), 1.0 + 0.2 * math.cos(i), 1.2, 100.0)
            for i in range(300)]
    verdicts = _fly(monitor, path)
    assert any(v.state == ABORT_NO_MOVEMENT for v in verdicts)


def test_the_time_cap_ends_a_mission_that_is_otherwise_healthy():
    monitor = ExplorationProgressMonitor(_config(time_cap_s=120.0))
    verdicts = _fly(monitor, _sweep(300))
    assert verdicts[-1].state == ABORT_TIME_CAP
    assert "time cap" in verdicts[-1].reason


# ── the arithmetic the verdicts are made of ──────────────────────────────

def test_no_verdict_is_reached_before_the_window_is_full():
    """Metrics stay None rather than being computed from a partial window."""
    monitor = ExplorationProgressMonitor(_config(grace_s=0.0, window_s=60.0))
    verdicts = _fly(monitor, _orbit(30))
    assert all(v.confinement_radius_m is None for v in verdicts)
    assert all(v.state == RUNNING for v in verdicts)


def test_confinement_radius_is_measured_from_the_window_centroid():
    """A 1.2 m orbit reports ~1.2 m of radius, not the 2.4 m diameter."""
    monitor = ExplorationProgressMonitor(_config(grace_s=0.0))
    verdicts = _fly(monitor, _orbit(200, radius=1.2, period_s=20.0))
    settled = verdicts[-1]
    assert settled.confinement_radius_m == pytest.approx(1.2, abs=0.15)


def test_growth_is_reported_per_minute_over_the_window():
    monitor = ExplorationProgressMonitor(_config(grace_s=0.0))
    verdicts = _fly(monitor, _sweep(200, growth_per_s=0.5))
    assert verdicts[-1].growth_m3_per_min == pytest.approx(30.0, abs=1.0)


def test_a_coverage_dither_does_not_re_arm_the_barren_clock():
    """A mapper whose volume wobbles must not hold a dead mission open.

    Keyed off the mission's best coverage rather than the previous sample: an
    up-tick that merely recovers ground already counted is not progress.
    """
    monitor = ExplorationProgressMonitor(_config())
    path = [(1.0, 1.0, 1.2, 200.0 + 0.2 * math.sin(i * 0.5)) for i in range(400)]
    verdicts = _fly(monitor, path)
    assert any(v.is_abort for v in verdicts)


def test_a_simulator_restart_restarts_the_mission_rather_than_corrupting_it():
    """Time going backwards is a new mission, not a negative window."""
    monitor = ExplorationProgressMonitor(_config())
    _fly(monitor, _orbit(100))
    after = monitor.update(0.0, (0.0, 0.0, 1.0), 0.0)
    assert after.state == RUNNING
    assert after.elapsed_s == 0.0
    assert monitor.nudges == 0


def test_the_verdict_serialises_for_the_run_trace():
    monitor = ExplorationProgressMonitor(_config())
    verdict = _fly(monitor, _orbit(400))[-1]
    record = verdict.as_dict()
    assert set(record) == {"state", "reason", "elapsed_s", "confinement_radius_m",
                           "growth_m3_per_min", "coverage_m3", "confined_for_s",
                           "barren_for_s"}
    assert isinstance(record["state"], str)
