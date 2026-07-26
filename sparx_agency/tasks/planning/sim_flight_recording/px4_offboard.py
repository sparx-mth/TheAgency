"""Non-blocking MAVLink offboard control of a PX4 SITL instance.

Every call here is non-blocking **by design**. PX4 SITL is built with
``ENABLE_LOCKSTEP_SCHEDULER``: its clock only advances when the simulator feeds
it sensor data, and the simulator only feeds it sensor data while the caller's
``world.step()`` loop is running. So any blocking pymavlink call
(``wait_heartbeat()``, ``motors_armed_wait()``, ...) is a guaranteed deadlock --
it stops the loop that produces the very messages it is waiting for. This class
is polled once per simulation step instead.

Frames: the simulator's world frame is ENU (x=East, y=North, z=Up, the Pegasus
convention) while MAVLink local setpoints are NED. :func:`enu_to_ned` is the
only place that conversion happens.
"""
from __future__ import annotations

import math

# PX4's custom-mode encoding for MAV_CMD_DO_SET_MODE (see PX4's commander/px4_custom_mode.h)
_PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
_MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1

# SET_POSITION_TARGET_LOCAL_NED type_mask: use position + yaw, ignore velocity,
# acceleration and yaw-rate (bits 3-8 and 11).
_TYPE_MASK_POSITION_YAW = 0b100111111000

_MAV_FRAME_LOCAL_NED = 1

# PX4 ships tuned for open sky: ~12 m/s cruise, 45 deg of tilt, aggressive
# acceleration. In a furnished room that overshoots every waypoint into a wall
# -- a simple_room run reached the far wall at 4.35 m and finished the flight
# nose-down on the floor. These are the same limits an operator would set on a
# real indoor drone: slow, gentle, shallow.
INDOOR_LIMITS = {
    "MPC_XY_VEL_MAX": 1.5,      # m/s, horizontal speed ceiling
    "MPC_XY_CRUISE": 1.0,       # m/s, target speed between waypoints
    "MPC_ACC_HOR_MAX": 1.5,     # m/s^2
    "MPC_ACC_HOR": 1.0,         # m/s^2
    "MPC_JERK_AUTO": 2.0,       # m/s^3, smooths the corners
    "MPC_TILTMAX_AIR": 20.0,    # deg, shallow tilt keeps the camera useful
    "MPC_Z_VEL_MAX_UP": 1.0,    # m/s
    "MPC_Z_VEL_MAX_DN": 0.7,    # m/s
    "MPC_YAWRAUTO_MAX": 45.0,   # deg/s, so the camera pans rather than whips
    # Takeoff is the least stable moment: MPC_TILTMAX_AIR does not govern the
    # takeoff ramp, and snapping to full climb thrust while still in ground
    # contact has tipped the airframe over on its back (roll -150 deg two
    # seconds after arming). Ramp the thrust in and climb slowly instead.
    "MPC_TKO_RAMP_T": 3.0,      # s, thrust ramp-up time
    "MPC_TKO_SPEED": 0.5,       # m/s, initial climb rate
    "MPC_LAND_SPEED": 0.4,      # m/s, touchdown rate
}


def enu_to_ned(x_east: float, y_north: float, z_up: float, yaw_enu: float):
    """Convert a simulator-frame (ENU) pose to the MAVLink local NED frame.

    Args:
        x_east: East coordinate, metres.
        y_north: North coordinate, metres.
        z_up: Altitude above the origin, metres.
        yaw_enu: Heading, radians CCW from East.

    Returns:
        ``(north, east, down, yaw_ned)`` with ``yaw_ned`` in radians CW from
        North.
    """
    return y_north, x_east, -z_up, math.pi / 2.0 - yaw_enu


class PX4Offboard:
    """A polled MAVLink client for arming and position-controlling PX4 SITL.

    Args:
        port: UDP port PX4 SITL instance 0 opens for its companion-computer
            link. See ``PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/
            px4-rc.mavlink``.
    """

    def __init__(self, port: int = 14540):
        from pymavlink import mavutil

        self._mavutil = mavutil
        self._conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{port}")
        self.heartbeat_seen = False
        self.armed = False
        self.mode = None
        self.local_ned = None

    def poll(self) -> None:
        """Drain pending MAVLink messages and refresh heartbeat/arm/position state.

        Call once per simulation step. Never blocks.
        """
        while True:
            msg = self._conn.recv_match(blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            if kind == "HEARTBEAT":
                self.heartbeat_seen = True
                self.armed = bool(msg.base_mode & self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = msg.custom_mode
            elif kind == "LOCAL_POSITION_NED":
                self.local_ned = (msg.x, msg.y, msg.z)

    def frame_offset(self, vehicle_enu):
        """Offset between the simulator's world frame and PX4's local frame.

        PX4's local NED frame is anchored where its estimator initialised --
        that is, wherever the vehicle happened to be sitting when PX4 booted --
        **not** at the simulator's world origin. Sending world coordinates as
        local setpoints therefore flies the drone off by exactly the spawn
        offset (measured: commanded ``(-4.0, 3.5)``, arrived ``(-8.2, 7.8)``,
        from a spawn at ``(-4.6, 4.4)``).

        Comparing PX4's own reported local position against the simulator's
        ground truth recovers that offset continuously, so it also tracks any
        later estimator drift.

        Args:
            vehicle_enu: The vehicle's true world-frame ``(x, y, z)``.

        Returns:
            ``(dx, dy, dz)`` to subtract from a world-frame target before
            sending it as a setpoint. Zero until PX4 reports a position.
        """
        if self.local_ned is None:
            return (0.0, 0.0, 0.0)
        north, east, down = self.local_ned
        estimated_enu = (east, north, -down)
        return tuple(vehicle_enu[i] - estimated_enu[i] for i in range(3))

    def send_setpoint_world(self, x: float, y: float, z: float, yaw: float, vehicle_enu) -> None:
        """Stream a setpoint given in the *simulator's* world frame.

        Converts through :meth:`frame_offset` into PX4's local frame first.
        """
        dx, dy, dz = self.frame_offset(vehicle_enu)
        self.send_setpoint(x - dx, y - dy, z - dz, yaw)

    def set_params(self, params: dict) -> None:
        """Push parameters to PX4.

        Call after the first heartbeat (PX4 must be booted to accept
        parameters) and before arming.

        The value's **Python type selects the MAVLink parameter type**: pass an
        ``int`` for PX4's INT32 parameters and a ``float`` for its REAL32 ones.
        PX4 rejects a mismatch outright -- ``ERROR [mavlink] param types
        mismatch param: EKF2_GPS_CTRL`` -- and the only sign in flight is that
        the setting silently did not apply.

        Args:
            params: Parameter name to value.
        """
        for name, value in params.items():
            param_type = (self._mavutil.mavlink.MAV_PARAM_TYPE_INT32
                          if isinstance(value, int) and not isinstance(value, bool)
                          else self._mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            self._conn.mav.param_set_send(
                self._conn.target_system, self._conn.target_component,
                name.encode(), float(value), param_type,
            )

    def set_indoor_limits(self, limits: dict = None) -> None:
        """Push the conservative indoor speed/tilt limits to PX4.

        Args:
            limits: Parameter name to value. Defaults to :data:`INDOOR_LIMITS`.
        """
        self.set_params(limits or INDOOR_LIMITS)

    def _command_long(self, command: int, *params) -> None:
        args = list(params) + [0.0] * (7 - len(params))
        self._conn.mav.command_long_send(
            self._conn.target_system, self._conn.target_component, command, 0, *args,
        )

    def send_setpoint(self, x_east: float, y_north: float, z_up: float, yaw_enu: float = 0.0) -> None:
        """Stream one offboard position setpoint, given in the simulator's ENU frame.

        PX4 drops out of offboard mode if setpoints stop arriving at >2 Hz, and
        refuses to *enter* it until a stream is already flowing -- so this must
        be called every step, both before and after :meth:`set_offboard_mode`.
        """
        north, east, down, yaw_ned = enu_to_ned(x_east, y_north, z_up, yaw_enu)
        self._conn.mav.set_position_target_local_ned_send(
            0, self._conn.target_system, self._conn.target_component,
            _MAV_FRAME_LOCAL_NED, _TYPE_MASK_POSITION_YAW,
            north, east, down,
            0.0, 0.0, 0.0,   # velocity (ignored by the type mask)
            0.0, 0.0, 0.0,   # acceleration (ignored)
            yaw_ned, 0.0,    # yaw, yaw rate (rate ignored)
        )

    def send_vision_pose(self, north: float, east: float, down: float,
                         roll: float, pitch: float, yaw: float, usec: int) -> None:
        """Send one ``VISION_POSITION_ESTIMATE``, already in NED/FRD.

        This must go over the **companion link**, not the simulator link.
        Pegasus has its own ``send_vision_msgs`` on its HIL connection, but
        PX4's ``simulator_mavlink`` module only consumes ``HIL_*`` messages
        there -- a vision pose sent down it is silently dropped, and the EKF,
        once told to depend on vision, then refuses to arm with "Preflight
        Fail: ekf2 missing data".

        Args:
            north, east, down: Position in PX4's local NED frame, metres.
            roll, pitch, yaw: Attitude in NED/FRD, radians.
            usec: Timestamp on the same clock as the sensor stream.
        """
        self._conn.mav.vision_position_estimate_send(
            usec, north, east, down, roll, pitch, yaw,
        )

    def set_offboard_mode(self) -> None:
        """Request OFFBOARD mode. Requires a setpoint stream to already be flowing."""
        self._command_long(
            self._mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            _MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, _PX4_CUSTOM_MAIN_MODE_OFFBOARD, 0,
        )

    def arm(self) -> None:
        """Request arming. Idempotent -- PX4 ignores a redundant arm command."""
        self._command_long(self._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)

    def land(self) -> None:
        """Switch to PX4's autonomous land mode."""
        self._command_long(self._mavutil.mavlink.MAV_CMD_NAV_LAND)

    def close(self) -> None:
        """Release the UDP port so the next run can bind it."""
        self._conn.close()
