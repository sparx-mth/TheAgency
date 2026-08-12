"""Choosing where to retreat to when the aircraft is wedged.

The safety of the whole manoeuvre rests on one property: the retreat target is
a place the aircraft *has already been*, seconds ago, so the corridor back is
known clear. These tests pin that down, plus the two ways it can be asked for
something it cannot provide.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.planning.falcon_pegasus.isaac.mission import (
    BLOCKAGE_MEMORY_S, UNWEDGE_RETREAT_M, ExplorationMission,
)


def _mission(breadcrumbs):
    """A mission with nothing but a flown path -- no Isaac, no PX4."""
    mission = ExplorationMission.__new__(ExplorationMission)
    mission._breadcrumbs = [(float(i) * 0.5, p) for i, p in enumerate(breadcrumbs)]
    return mission


def test_the_retreat_target_is_a_place_the_aircraft_has_been():
    trail = [(x, 0.0, 1.4) for x in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)]
    target = _mission(trail)._retreat_target((5.0, 0.0, 1.4))
    assert target in trail


def test_it_retreats_far_enough_to_matter_but_no_further():
    """The NEAREST point at least a retreat away, not the oldest.

    Walking the trail from the oldest end would fly the aircraft back across
    the building to escape something it could step off in two metres.
    """
    trail = [(x, 0.0, 1.4) for x in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)]
    target = _mission(trail)._retreat_target((5.0, 0.0, 1.4))
    assert target == (2.0, 0.0, 1.4)          # 3.0 back: the first >= UNWEDGE_RETREAT_M
    assert target != (0.0, 0.0, 1.4)          # not the oldest crumb on the trail


def test_a_trail_that_never_gets_far_enough_yields_nothing():
    """An aircraft that wedged on the spot has nowhere known-clear to go."""
    trail = [(0.01 * i, 0.0, 1.4) for i in range(20)]     # 0.19 m of travel
    assert _mission(trail)._retreat_target((0.19, 0.0, 1.4)) is None


def test_an_empty_trail_yields_nothing():
    assert _mission([])._retreat_target((0.0, 0.0, 1.4)) is None


def test_the_target_is_always_at_least_the_retreat_distance_away():
    trail = [(0.0, 0.0, 1.4), (0.5, 0.0, 1.4), (4.0, 0.0, 1.4), (4.2, 0.0, 1.4)]
    here = (4.2, 0.0, 1.4)
    target = _mission(trail)._retreat_target(here)
    assert target is not None
    gap = sum((a - b) ** 2 for a, b in zip(here, target)) ** 0.5
    assert gap >= UNWEDGE_RETREAT_M


# ── the contact reflex ────────────────────────────────────────────────────

class _Vehicle:
    def __init__(self, vx, vy):
        self.state = type("S", (), {"linear_velocity": (vx, vy, 0.0)})()


class _Adapter:
    def __init__(self, vx, vy):
        self.vehicle = _Vehicle(vx, vy)


def _flying(samples):
    """Feed a velocity sequence through the reflex; return whether it fired."""
    mission = ExplorationMission.__new__(ExplorationMission)
    mission._recent_velocity = []
    mission.loop = type("L", (), {"sim_time": 0.0})()
    fired = False
    for t, vx, vy in samples:
        mission.loop.sim_time = t
        mission.adapter = _Adapter(vx, vy)
        fired = mission._touched() or fired
    return fired


def test_steady_flight_does_not_trip_the_reflex():
    assert not _flying([(i * 0.004, 1.0, 0.0) for i in range(200)])


def test_a_normal_turn_does_not_trip_the_reflex():
    """A cornering aircraft swings its course, but nothing like this fast."""
    import math as _m
    samples = []
    for i in range(200):                       # 90 deg over 2 s
        a = (_m.pi / 2) * (i / 200.0)
        samples.append((i * 0.01, _m.cos(a), _m.sin(a)))
    assert not _flying(samples)


def test_a_bounce_trips_the_reflex():
    """Velocity reversing inside the window is a contact, not a manoeuvre."""
    samples = [(i * 0.01, 1.2, 0.0) for i in range(30)]
    samples += [(0.30 + i * 0.01, -1.1, 0.0) for i in range(10)]
    assert _flying(samples)


def test_being_stopped_dead_trips_the_reflex():
    """A square-on strike leaves no course to reverse -- only a deceleration."""
    samples = [(i * 0.01, 2.5, 0.0) for i in range(30)]
    samples += [(0.30 + i * 0.01, 0.2, 0.0) for i in range(10)]
    assert _flying(samples)


def test_gentle_braking_does_not_trip_the_reflex():
    """The controller can brake hard; it cannot brake like a wall."""
    samples = [(i * 0.02, max(0.0, 1.5 - 3.0 * i * 0.02), 0.0) for i in range(60)]
    assert not _flying(samples)


def test_pruning_keeps_the_route_out_but_drops_the_obstacle():
    """Consecutive retreats need history; the wedge point must not be a target.

    Clearing the whole trail after a retreat left the next one with nothing to
    aim at -- on a real flight the third contact in a row reported "no flown
    path to retreat along" and held, and the flight diverged from there.
    """
    wedge = (0.0, 0.0, 1.4)
    trail = [(x, 0.0, 1.4) for x in (-8.0, -6.0, -4.0, -1.0, -0.2, 0.0)]
    kept = [p for p in trail if
            sum((a - b) ** 2 for a, b in zip(p, wedge)) ** 0.5 > UNWEDGE_RETREAT_M]
    assert kept, "the way in must survive pruning"
    assert wedge not in kept
    assert (-0.2, 0.0, 1.4) not in kept          # too close to the obstacle
    assert (-8.0, 0.0, 1.4) in kept              # the route out is still there


def test_the_retreat_has_a_fallback_when_the_trail_is_gone():
    """The no-breadcrumbs case is the one that matters, not an edge case.

    After 60 s pinned, every surviving crumb is within the retreat radius, so
    _retreat_target returns None -- and the recovery used to become a hold at
    exactly the moment it was needed. Backing out along -body-x is sound
    because the aircraft is pinned nose-first and flew in forwards.
    """
    import math as _m
    # facing +x, pinned; the escape must be behind it, at cruise height
    yaw, here, cruise = 0.0, (10.0, 5.0, 1.2), 1.4
    target = (here[0] - UNWEDGE_RETREAT_M * _m.cos(yaw),
              here[1] - UNWEDGE_RETREAT_M * _m.sin(yaw), cruise)
    assert target[0] < here[0], "must retreat backwards, not forwards"
    assert target[2] == cruise, "and return to cruise height"
    assert _m.dist(target[:2], here[:2]) == pytest.approx(UNWEDGE_RETREAT_M)


# ── the pin detector ──────────────────────────────────────────────────────

class _Cmd:
    def __init__(self, tilt_deg, holding=False):
        import math as _m
        self.attitude = type("A", (), {"tilt_rad": _m.radians(tilt_deg)})()
        self.holding = holding


def _pin_run(samples):
    """Feed (t, tilt_deg, speed) through the detector; return whether it fired."""
    m = ExplorationMission.__new__(ExplorationMission)
    m._pinned_since = None
    m.loop = type("L", (), {"sim_time": 0.0})()
    fired = False
    for t, tilt, speed in samples:
        m.loop.sim_time = t
        m.adapter = _Adapter(speed, 0.0)
        fired = m._pinned(_Cmd(tilt)) or fired
    return fired


def test_pushing_hard_and_going_nowhere_is_a_pin():
    """The case _touched cannot see: 10-15 deg commanded, 0.16 m/s achieved."""
    assert _pin_run([(i * 0.1, 13.0, 0.16) for i in range(60)])


def test_normal_cornering_is_not_a_pin():
    """Tilt with motion is just flying."""
    assert not _pin_run([(i * 0.1, 15.0, 0.9) for i in range(60)])


def test_hovering_level_is_not_a_pin():
    """No tilt demand means nothing is being asked for."""
    assert not _pin_run([(i * 0.1, 0.5, 0.02) for i in range(60)])


def test_a_brief_slow_moment_is_not_a_pin():
    """Turning on the spot dips the speed; it must not trip in under PINNED_HOLD_S."""
    assert not _pin_run([(i * 0.1, 12.0, 0.05) for i in range(15)])


def test_a_deliberate_hold_is_not_a_pin():
    m = ExplorationMission.__new__(ExplorationMission)
    m._pinned_since = None
    m.loop = type("L", (), {"sim_time": 0.0})()
    fired = False
    for i in range(60):
        m.loop.sim_time = i * 0.1
        m.adapter = _Adapter(0.0, 0.0)
        fired = m._pinned(_Cmd(12.0, holding=True)) or fired
    assert not fired


# ── blockage memory ───────────────────────────────────────────────────────

def _with_memory(struck, now=100.0):
    m = ExplorationMission.__new__(ExplorationMission)
    m.loop = type("L", (), {"sim_time": now})()
    m._struck = [(now, p) for p in struck]
    return m


class _Curve:
    """A straight trajectory along +x at 1 m/s, sampled by position_at."""
    duration = 10.0

    def __init__(self, y=0.0):
        self.y = y

    def elapsed(self, _now):
        return 0.0

    def position_at(self, t):
        return (t * 1.0, self.y, 1.4)


def test_a_route_back_to_a_struck_place_is_refused():
    m = _with_memory([(2.0, 0.0, 1.4)])
    assert m._leads_into_blockage(_Curve())


def test_a_route_that_stays_clear_is_followed():
    """Proximity, not direction: a curve passing well wide is fine."""
    m = _with_memory([(2.0, 0.0, 1.4)])
    assert not m._leads_into_blockage(_Curve(y=5.0))


def test_a_struck_place_is_forgotten_after_the_memory_expires():
    """A permanent blacklist would carve the building up and strand frontiers."""
    m = _with_memory([(2.0, 0.0, 1.4)], now=100.0)
    m.loop.sim_time = 100.0 + BLOCKAGE_MEMORY_S + 1.0
    assert not m._leads_into_blockage(_Curve())


def test_nothing_is_refused_before_anything_has_been_struck():
    m = _with_memory([])
    assert not m._leads_into_blockage(_Curve())
