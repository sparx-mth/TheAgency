"""The world -> PX4 frame transform, which is where two flights were lost.

PX4's local frame is both *shifted* (it anchors where PX4 booted) and *rotated*
(its heading reference is a magnetometer, not the simulator's grid). Getting
either wrong sends the aircraft somewhere else, and getting the rotation wrong
sends it somewhere else by an amount that grows with distance.

These exercise the geometry alone -- no MAVLink connection is opened.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.tasks.planning.sim_flight_recording.px4_offboard import (
    PX4Offboard, enu_to_ned,
)


def _offboard() -> PX4Offboard:
    """A PX4Offboard with no socket, for testing the pure geometry on it."""
    client = PX4Offboard.__new__(PX4Offboard)
    client.local_ned = None
    client.attitude_ned = None
    client.frame_offset = (0.0, 0.0, 0.0)
    client.heading_bias = 0.0
    client._origin_world = (0.0, 0.0, 0.0)
    client._origin_px4 = (0.0, 0.0, 0.0)
    return client


def _latched(true_xy, true_yaw, px4_ned, px4_yaw_ned) -> PX4Offboard:
    """A client latched against a given ground truth and PX4 report."""
    client = _offboard()
    client.local_ned = px4_ned
    client.attitude_ned = (0.0, 0.0, px4_yaw_ned)
    client.latch_frame((true_xy[0], true_xy[1], 1.5), true_yaw)
    return client


def test_enu_to_ned_swaps_the_axes_and_flips_the_heading():
    north, east, down, yaw = enu_to_ned(1.0, 2.0, 3.0, 0.0)
    assert (north, east, down) == (2.0, 1.0, -3.0)
    assert yaw == pytest.approx(math.pi / 2)


def test_an_aligned_frame_leaves_a_target_alone():
    """PX4 booted at the origin, agreeing about north: the transform is identity."""
    client = _latched((0.0, 0.0), 0.0, px4_ned=(0.0, 0.0, -1.5), px4_yaw_ned=math.pi / 2)

    assert client.heading_bias == pytest.approx(0.0, abs=1e-9)
    assert client.world_to_px4(5.0, 3.0, 1.5) == pytest.approx((5.0, 3.0, 1.5))


def test_a_shifted_origin_is_subtracted():
    """PX4 anchored where it booted, which was not the world origin."""
    client = _latched((-4.6, 4.4), 0.0, px4_ned=(0.0, 0.0, -1.5), px4_yaw_ned=math.pi / 2)

    local = client.world_to_px4(-4.0, 3.5, 1.5)
    assert local == pytest.approx((0.6, -0.9, 1.5))


def test_a_rotated_frame_rotates_the_displacement():
    """The bug: a 90 deg heading bias turns 'go north' into 'go east'."""
    # Truly facing +y (world yaw +90 deg), while PX4 reports yaw_ned = +90 deg,
    # i.e. it believes it is facing world +x. Its frame is rotated 90 deg.
    client = _latched((0.0, 0.0), math.pi / 2, px4_ned=(0.0, 0.0, -1.5),
                      px4_yaw_ned=math.pi / 2)

    assert client.heading_bias == pytest.approx(math.pi / 2)
    # 10 m along world +x must be commanded as 10 m along PX4's -y.
    local = client.world_to_px4(10.0, 0.0, 1.5)
    assert local[0] == pytest.approx(0.0, abs=1e-6)
    assert local[1] == pytest.approx(-10.0)


def test_the_rotation_error_grows_with_distance():
    """Why an uncorrected bias is survivable at 1 m and fatal at 14 m."""
    bias = math.radians(20.0)
    client = _latched((0.0, 0.0), bias, px4_ned=(0.0, 0.0, -1.5), px4_yaw_ned=math.pi / 2)

    def error_at(distance):
        local = client.world_to_px4(distance, 0.0, 1.5)
        return math.hypot(local[0] - distance, local[1])

    assert error_at(1.0) < 0.4
    assert error_at(14.0) > 4.0
    assert error_at(14.0) == pytest.approx(14.0 * error_at(1.0), rel=1e-6)


def test_the_transform_is_invertible_at_the_latch_point():
    client = _latched((3.0, -2.0), 1.1, px4_ned=(7.0, -1.0, -1.5), px4_yaw_ned=0.3)
    assert client.world_to_px4(3.0, -2.0, 1.5) == pytest.approx((-1.0, 7.0, 1.5))


def test_a_target_at_the_latch_point_needs_no_setpoint_motion():
    """Whatever the bias, commanding 'stay here' must not command a move."""
    for bias_deg in (0.0, 17.0, -95.0, 179.0):
        client = _latched((5.0, -3.0), math.radians(bias_deg),
                          px4_ned=(2.0, 1.0, -1.5), px4_yaw_ned=math.pi / 2)
        assert client.world_to_px4(5.0, -3.0, 1.5) == pytest.approx((1.0, 2.0, 1.5))


def test_frame_drift_is_zero_when_the_estimate_still_matches():
    client = _latched((0.0, 0.0), 0.0, px4_ned=(0.0, 0.0, -1.5), px4_yaw_ned=math.pi / 2)
    client.local_ned = (4.0, 6.0, -1.5)        # PX4 now reports (east 6, north 4)
    assert client.frame_drift((6.0, 4.0, 1.5)) == pytest.approx(0.0, abs=1e-6)


def test_frame_drift_reports_a_diverging_estimate():
    client = _latched((0.0, 0.0), 0.0, px4_ned=(0.0, 0.0, -1.5), px4_yaw_ned=math.pi / 2)
    client.local_ned = (4.0, 6.0, -1.5)
    assert client.frame_drift((6.0, 7.0, 1.5)) == pytest.approx(3.0, abs=1e-6)


def test_latching_without_an_attitude_message_keeps_the_previous_bias():
    """No ATTITUDE yet is not a reason to assume the frames are aligned."""
    client = _offboard()
    client.heading_bias = 0.5
    client.local_ned = (0.0, 0.0, 0.0)
    client.latch_frame((0.0, 0.0, 0.0), 1.0)
    assert client.heading_bias == 0.5


def test_latching_without_a_position_message_falls_back_to_ground_truth():
    client = _offboard()
    client.latch_frame((2.0, 3.0, 1.0), 0.0)
    assert client.frame_offset == pytest.approx((0.0, 0.0, 0.0))
    assert client.world_to_px4(2.0, 3.0, 1.0) == pytest.approx((2.0, 3.0, 1.0))


class _RecordingLink:
    """Captures the one setpoint PX4Offboard would have sent."""

    def __init__(self):
        self.sent = None
        self.target_system = 1
        self.target_component = 1
        self.mav = self

    def set_position_target_local_ned_send(self, _time, _sys, _comp, _frame, _mask,
                                           north, east, down, *rest):
        self.sent = (east, north, -down)   # back to an ENU triple


def _servo_client(heading_bias: float, px4_ned) -> PX4Offboard:
    """A latched client wired to a fake link, for exercising the closed loop.

    No pymavlink: ``send_setpoint`` only touches the connection's ``mav``, so a
    stand-in is enough and the geometry stays testable in the repo venv.
    """
    client = _offboard()
    client.heading_bias = heading_bias
    client.local_ned = px4_ned
    client._conn = _RecordingLink()
    return client


def test_the_setpoint_servos_on_ground_truth_not_on_px4s_origin():
    """PX4's estimate can be anywhere; only the world-frame error is commanded."""
    # PX4 believes it is at ENU (100, 200, 5) while it is truly at (1, 2, 1.5).
    client = _servo_client(0.0, px4_ned=(200.0, 100.0, -5.0))

    client.send_setpoint_world(4.0, 6.0, 1.5, 0.0, vehicle_enu=(1.0, 2.0, 1.5))

    # 3 m east, 4 m north, 0 up of error, applied to where PX4 thinks it is.
    assert client._conn.sent == pytest.approx((103.0, 204.0, 5.0))


def test_the_commanded_error_is_rotated_into_px4s_frame():
    client = _servo_client(math.pi / 2, px4_ned=(0.0, 0.0, 0.0))

    client.send_setpoint_world(10.0, 0.0, 0.0, 0.0, vehicle_enu=(0.0, 0.0, 0.0))

    east, north, _up = client._conn.sent
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(-10.0)


def test_arriving_at_the_target_commands_no_further_motion():
    client = _servo_client(0.4, px4_ned=(9.0, -3.0, -2.0))
    client.send_setpoint_world(5.0, 5.0, 1.5, 0.0, vehicle_enu=(5.0, 5.0, 1.5))
    assert client._conn.sent == pytest.approx((-3.0, 9.0, 2.0))


def test_without_ground_truth_it_falls_back_to_the_latched_transform():
    client = _servo_client(0.0, px4_ned=(0.0, 0.0, 0.0))
    client._origin_world = (10.0, 10.0, 0.0)
    client._origin_px4 = (0.0, 0.0, 0.0)

    client.send_setpoint_world(12.0, 13.0, 1.0, 0.0)

    assert client._conn.sent == pytest.approx((2.0, 3.0, 1.0))
