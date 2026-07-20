"""Behavioural tests for the command prior's earned-trust contract.

The one property that must never break: a drone that is COMMANDED to move but
measurably is NOT moving (pressed against a wall, snagged on an obstacle) must
stop being believed — quickly, and without the pose drifting meanwhile. Every
test here is a variation on that promise.
"""
import math

import pytest

from sparx_agency.core.localization.command_motion_model import (
    CommandMotionModel,
    CommandMotionParams,
)


def make(**kw):
    return CommandMotionModel(CommandMotionParams(**kw))


def drive(model, vx, wz, t0, n, dt=0.1):
    """Send n commands at dt spacing starting at t0; returns the end time."""
    t = t0
    for _ in range(n):
        model.set_command(vx, 0.0, wz, t)
        t += dt
    return t


def test_integrates_commanded_forward_motion():
    m = make(trust_max=1.0, eff_initial=1.0)
    t = drive(m, vx=0.25, wz=0.0, t0=0.0, n=10)          # 1 s at 0.25 m/s
    dx, dy, dyaw, rdx, rdy, rdyaw = m.consume(t, yaw=0.0)
    assert rdx == pytest.approx(0.25, abs=0.01)
    assert rdy == pytest.approx(0.0, abs=1e-9)
    # capped: one step may not exceed max_step_m even at full trust
    assert dx == pytest.approx(CommandMotionParams().max_step_m)


def test_rotation_maps_body_to_world():
    m = make(trust_max=1.0, eff_initial=1.0, max_step_m=10.0)
    t = drive(m, vx=0.2, wz=0.0, t0=0.0, n=5)            # 0.1 m commanded
    dx, dy, _, _, _, _ = m.consume(t, yaw=math.pi / 2)   # facing world +Y
    assert dx == pytest.approx(0.0, abs=1e-6)
    assert dy == pytest.approx(0.1, abs=0.01)


def test_stuck_drone_loses_trust_within_a_second():
    """Commanded 0.25 m/s, camera sees zero motion -> effectiveness collapses."""
    m = make()
    t = 0.0
    for _ in range(10):                                   # ~1 s of 10 Hz fixes
        t = drive(m, vx=0.25, wz=0.0, t0=t, n=1)
        _, _, _, rdx, rdy, rdyaw = m.consume(t, 0.0)
        m.observe(0.0, 0.0, 0.0, rdx, rdy, rdyaw, confidence=0.9)
    assert m.effectiveness_lin < 0.1


def test_moving_drone_gains_trust():
    m = make()
    t = 0.0
    for _ in range(15):
        t = drive(m, vx=0.25, wz=0.0, t0=t, n=1)
        _, _, _, rdx, rdy, rdyaw = m.consume(t, 0.0)
        m.observe(rdx, rdy, 0.0, rdx, rdy, rdyaw, confidence=0.9)  # achieved fully
    assert m.effectiveness_lin > 0.85


def test_low_confidence_fixes_teach_nothing():
    """A garbage pose must not be able to declare the drone stuck."""
    m = make()
    e0 = m.effectiveness_lin
    t = drive(m, vx=0.25, wz=0.0, t0=0.0, n=1)
    _, _, _, rdx, rdy, rdyaw = m.consume(t, 0.0)
    m.observe(0.0, 0.0, 0.0, rdx, rdy, rdyaw, confidence=0.1)  # below floor
    assert m.effectiveness_lin == e0


def test_hover_teaches_nothing():
    """No commanded motion -> the achieved/commanded ratio is meaningless."""
    m = make()
    e0 = m.effectiveness_lin
    m.observe(0.02, -0.01, 0.0, 0.0, 0.0, 0.0, confidence=0.9)  # noise while hovering
    assert m.effectiveness_lin == e0


def test_stale_command_stops_integrating():
    """One command then silence: only cmd_timeout_s worth may accumulate."""
    m = make(trust_max=1.0, eff_initial=1.0, max_step_m=10.0)
    m.set_command(0.3, 0.0, 0.0, 0.0)
    _, _, _, rdx, _, _ = m.consume(10.0, 0.0)             # 10 s later
    assert rdx == pytest.approx(0.3 * CommandMotionParams().cmd_timeout_s)


def test_yaw_and_translation_trust_are_independent():
    """Wedged against a wall the drone often still turns: yaw trust must survive
    translation trust dying."""
    m = make()
    t = 0.0
    for _ in range(10):
        t = drive(m, vx=0.25, wz=0.5, t0=t, n=1)
        _, _, _, rdx, rdy, rdyaw = m.consume(t, 0.0)
        m.observe(0.0, 0.0, rdyaw, rdx, rdy, rdyaw, confidence=0.9)  # turns, no advance
    assert m.effectiveness_lin < 0.1
    assert m.effectiveness_yaw > 0.8


def test_prediction_step_caps_bound_a_dropout():
    """Even at full trust, one consume over a long gap is bounded in m and rad."""
    m = make(trust_max=1.0, eff_initial=1.0, cmd_timeout_s=60.0)
    t = drive(m, vx=1.0, wz=2.0, t0=0.0, n=50, dt=0.1)    # 5 m / 10 rad commanded
    dx, dy, dyaw, _, _, _ = m.consume(t, 0.0)
    p = CommandMotionParams()
    assert math.hypot(dx, dy) <= p.max_step_m + 1e-9
    assert abs(dyaw) <= p.max_step_rad + 1e-9


def test_trust_zero_disables():
    m = make(trust_max=0.0)
    assert not m.enabled
    t = drive(m, vx=0.3, wz=0.0, t0=0.0, n=5)
    dx, dy, dyaw, _, _, _ = m.consume(t, 0.0)
    assert dx == dy == dyaw == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
