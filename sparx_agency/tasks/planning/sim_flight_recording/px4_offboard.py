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

# PX4's custom-mode encoding for MAV_CMD_DO_SET_MODE (see PX4's commander/px4_custom_mode.h).
# HEARTBEAT.custom_mode packs the main mode into bits 16-23 and the sub mode into
# 24-31, which is how a caller can tell whether a mode request was actually
# honoured -- MAV_CMD_DO_SET_MODE is fire-and-forget and PX4 declines silently.
_PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
_MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
_CUSTOM_MAIN_MODE_SHIFT = 16
_CUSTOM_MAIN_MODE_MASK = 0xFF

# SET_POSITION_TARGET_LOCAL_NED type_mask: use position + yaw, ignore velocity,
# acceleration and yaw-rate (bits 3-8 and 11).
_TYPE_MASK_POSITION_YAW = 0b100111111000
# ... and its counterpart: use velocity + yaw, ignore position (bits 0-2),
# acceleration (6-8) and yaw-rate (11).
_TYPE_MASK_VELOCITY_YAW = 0b100111000111

_MAV_FRAME_LOCAL_NED = 1

# MAV_LANDED_STATE, from EXTENDED_SYS_STATE. PX4's land detector is the only
# authority on whether the aircraft is actually down; inferring it from altitude
# gets a touchdown on a desk wrong, and inferring it from disarm misses a
# landing that PX4 chose not to disarm after.
LANDED_STATE_ON_GROUND = 1
LANDED_STATE_IN_AIR = 2


def _normalize(angle: float) -> float:
    """Wrap an angle into ``(-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


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
        instance: PX4 SITL instance id. Selects the companion-computer UDP port
            (``14540 + instance``), which is what makes several aircraft
            controllable from one machine. See :mod:`px4_launch` for the rest of
            the per-instance identity.
    """

    def __init__(self, instance: int = 0):
        from pymavlink import mavutil

        from sparx_agency.tasks.planning.sim_flight_recording.px4_launch import offboard_port

        self._mavutil = mavutil
        self.instance = instance
        self._conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{offboard_port(instance)}")
        self.heartbeat_seen = False
        self.armed = False
        self.mode = None
        self.local_ned = None
        self.attitude_ned = None
        self.landed_state = None
        self.acknowledged_params = set()
        self.status_texts = []
        # World-frame -> PX4-local-frame transform. Latched, never live; see
        # latch_frame for why that distinction is load-bearing.
        self.frame_offset = (0.0, 0.0, 0.0)
        self.heading_bias = 0.0
        # The last setpoint actually put on the wire, in PX4's own NED frame.
        # Nothing reads it in normal operation; it is here because when the
        # aircraft does not go where it was sent, "what did we literally send,
        # and where does PX4 think it is" is the question that settles it.
        self.last_setpoint_ned = None
        self._origin_world = (0.0, 0.0, 0.0)
        self._origin_px4 = (0.0, 0.0, 0.0)

    def poll(self) -> None:
        """Drain pending MAVLink messages and refresh the cached vehicle state.

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
            elif kind == "ATTITUDE":
                self.attitude_ned = (msg.roll, msg.pitch, msg.yaw)
            elif kind == "EXTENDED_SYS_STATE":
                self.landed_state = msg.landed_state
            elif kind == "PARAM_VALUE":
                # PX4 echoes every parameter it accepts. A parameter it rejected
                # (a type mismatch, a name that does not exist) is never echoed,
                # and the only other sign is a line in PX4's own console -- so
                # this is how a caller can tell a setting actually applied.
                self.acknowledged_params.add(
                    msg.param_id.decode() if isinstance(msg.param_id, bytes) else msg.param_id
                )
            elif kind == "STATUSTEXT":
                text = msg.text.decode() if isinstance(msg.text, bytes) else msg.text
                self.status_texts.append(text)

    @property
    def on_ground(self) -> bool:
        """PX4's own land detector says the aircraft is down.

        None until an ``EXTENDED_SYS_STATE`` has arrived, which reads as False.
        """
        return self.landed_state == LANDED_STATE_ON_GROUND

    @property
    def main_mode(self):
        """PX4's current main flight mode, or None before the first heartbeat."""
        if self.mode is None:
            return None
        return (int(self.mode) >> _CUSTOM_MAIN_MODE_SHIFT) & _CUSTOM_MAIN_MODE_MASK

    @property
    def in_offboard(self) -> bool:
        """True while PX4 is actually flying the setpoints being streamed at it.

        Worth checking every step, not just once. ``MAV_CMD_DO_SET_MODE`` is
        fire-and-forget -- PX4 declines it silently if no setpoint stream is
        flowing yet -- and PX4 also *leaves* offboard on its own whenever a
        failsafe fires. An armed aircraft that is not in offboard ignores every
        setpoint sent to it, which looks exactly like a drone that will not fly:
        one campaign recorded 1210 frames of a stationary aircraft and called it
        a successful flight because nothing was checking this.
        """
        return self.main_mode == _PX4_CUSTOM_MAIN_MODE_OFFBOARD

    def drain_status_texts(self) -> list:
        """Return and clear the ``STATUSTEXT`` messages seen since the last call.

        This is where PX4 says *why* it refused to arm. Surfacing it turns a
        campaign's "waypoint timed out" into "Preflight Fail: ..." without
        having to go and read a separate console log.
        """
        texts, self.status_texts = self.status_texts, []
        return texts

    def measure_frame_offset(self, vehicle_enu):
        """The offset between the simulator's world frame and PX4's local frame, right now.

        PX4's local NED frame is anchored where its estimator initialised --
        that is, wherever the vehicle happened to be sitting when PX4 booted --
        **not** at the simulator's world origin. Sending world coordinates as
        local setpoints therefore flies the drone off by exactly the spawn
        offset (measured: commanded ``(-4.0, 3.5)``, arrived ``(-8.2, 7.8)``,
        from a spawn at ``(-4.6, 4.4)``).

        Comparing PX4's own reported local position against the simulator's
        ground truth recovers that offset.

        Args:
            vehicle_enu: The vehicle's true world-frame ``(x, y, z)``.

        Returns:
            ``(dx, dy, dz)``. Zero until PX4 reports a position.
        """
        if self.local_ned is None:
            return (0.0, 0.0, 0.0)
        north, east, down = self.local_ned
        estimated_enu = (east, north, -down)
        return tuple(vehicle_enu[i] - estimated_enu[i] for i in range(3))

    @property
    def estimated_yaw_enu(self):
        """PX4's estimated heading, in the simulator's ENU convention.

        ``ATTITUDE`` reports yaw as radians CW from *its* north; this is the
        same angle expressed as radians CCW from +X, so it can be compared with
        the simulator's ground-truth heading directly. None before the first
        ``ATTITUDE`` message.
        """
        if self.attitude_ned is None:
            return None
        return _normalize(math.pi / 2.0 - self.attitude_ned[2])

    def latch_frame(self, vehicle_enu, vehicle_yaw_enu: float) -> None:
        """Freeze the world-to-PX4 transform, and use it until re-latched.

        **Two things have to be captured, and both have to be constants.**

        *The translation* is where PX4 booted. *The rotation* is the angle
        between PX4's idea of north and the simulator's +y, which is not zero:
        PX4 takes its heading reference from a magnetometer, and there is no
        reason for a simulated magnetic north to line up with a world grid.
        Sending a world-frame displacement without rotating it into PX4's frame
        commands the aircraft off by that angle -- which grows with distance and
        put one 14 m flight 7 m from its goal.

        And they must be *latched*, not measured live. Recomputing on every
        setpoint closes a feedback loop: the commanded point becomes
        ``target - truth + estimate(truth)``, so a position-dependent estimate
        error moves the setpoint as the aircraft moves toward it. With a
        rotational error that displacement is perpendicular to the motion, and
        the aircraft flies a circle around its waypoint instead of arriving --
        observed as a stable 1.1 m orbit held for 100 seconds through three
        waypoint timeouts, with PX4 reporting a healthy offboard mode
        throughout.

        Latch while the aircraft is stationary on the ground, where a live
        measurement is safe. :meth:`frame_drift` exists to notice the latched
        value going stale.

        Args:
            vehicle_enu: The vehicle's true world-frame ``(x, y, z)``.
            vehicle_yaw_enu: The vehicle's true heading, radians CCW from +X.
        """
        self._origin_world = tuple(float(v) for v in vehicle_enu[:3])
        self._origin_px4 = self._estimated_enu() or self._origin_world
        estimated_yaw = self.estimated_yaw_enu
        if estimated_yaw is not None:
            self.heading_bias = _normalize(float(vehicle_yaw_enu) - estimated_yaw)
        self.frame_offset = tuple(
            self._origin_world[i] - self._origin_px4[i] for i in range(3))

    def _estimated_enu(self):
        """PX4's reported position, converted from local NED to an ENU triple."""
        if self.local_ned is None:
            return None
        north, east, down = self.local_ned
        return (east, north, -down)

    def frame_drift(self, vehicle_enu) -> float:
        """How far PX4's estimate has moved relative to truth since latching, metres.

        A healthy flight keeps this within a few tens of centimetres. A large
        value means the latched transform is stale and the aircraft is being
        commanded to the wrong place -- worth recording alongside a flight that
        went wrong.
        """
        estimated = self._estimated_enu()
        if estimated is None:
            return 0.0
        expected = self.world_to_px4(vehicle_enu[0], vehicle_enu[1], vehicle_enu[2])
        return math.sqrt(sum((estimated[i] - expected[i]) ** 2 for i in range(3)))

    def world_to_px4(self, x: float, y: float, z: float) -> tuple:
        """Convert a world-frame (ENU) point into PX4's local frame.

        Rotates the displacement from the latch point by the latched heading
        bias, then re-anchors it on where PX4 thought it was at that moment.
        """
        rotated = self._rotate_into_px4(x - self._origin_world[0],
                                        y - self._origin_world[1])
        return (self._origin_px4[0] + rotated[0],
                self._origin_px4[1] + rotated[1],
                self._origin_px4[2] + (z - self._origin_world[2]))

    def send_setpoint_world(self, x: float, y: float, z: float, yaw: float,
                            vehicle_enu=None) -> None:
        """Stream a setpoint given in the *simulator's* world frame.

        When the caller supplies ground truth, the setpoint is closed around it:
        the *world-frame error* is rotated into PX4's frame and added to where
        PX4 currently believes it is. That is a position servo on the true
        position, and it is exact whatever PX4's estimate is doing -- a wrong
        origin, a wrong heading reference and accumulated drift all cancel,
        because only the *displacement* is ever taken from PX4.

        It is also stable rather than the feedback trap a naive live correction
        would be. While PX4's estimate and the truth differ by a fixed rotation
        and translation, moving the aircraft by delta moves both terms by the
        same amount, so the commanded point stays put; it moves only when that
        relationship is violated, which is exactly when it should.

        Without ground truth it falls back to the latched transform, which is
        open-loop and only as good as the latch.

        Args:
            x, y, z: Target in the simulator's world (ENU) frame, metres.
            yaw: Heading to hold, radians CCW from +X in the world frame.
            vehicle_enu: The vehicle's true world-frame ``(x, y, z)``.
        """
        yaw_local = _normalize(yaw - self.heading_bias)
        estimated = self._estimated_enu()
        if vehicle_enu is None or estimated is None:
            local = self.world_to_px4(x, y, z)
            self.send_setpoint(local[0], local[1], local[2], yaw_local)
            return

        error = self._rotate_into_px4(x - vehicle_enu[0], y - vehicle_enu[1])
        self.send_setpoint(estimated[0] + error[0], estimated[1] + error[1],
                           estimated[2] + (z - vehicle_enu[2]), yaw_local)

    def _rotate_into_px4(self, dx: float, dy: float) -> tuple:
        """Rotate a world-frame horizontal displacement into PX4's local frame."""
        cos_bias, sin_bias = math.cos(-self.heading_bias), math.sin(-self.heading_bias)
        return (dx * cos_bias - dy * sin_bias, dx * sin_bias + dy * cos_bias)

    def send_velocity_world(self, vx: float, vy: float, vz: float, yaw: float) -> None:
        """Stream a **velocity** setpoint given in the simulator's world frame.

        This is how the aircraft is actually flown, and position setpoints are
        kept only for holding still before takeoff. PX4's offboard *position*
        path did not work here: given a setpoint a metre away, in a healthy
        offboard mode, with no failsafe and its own estimate tracking ground
        truth to 30 cm, it closed the gap at one centimetre per second and the
        flight timed out. Velocity setpoints go almost straight to the velocity
        controller, so there is no trajectory smoother, no position-error
        clamping and no local-frame *origin* to get wrong -- only the heading
        bias, which is measured.

        It also matches how everything else in this repo flies a drone: the
        FALCON stack's followers all emit ``/cmd_vel``.

        Args:
            vx, vy: World-frame horizontal velocity, m/s (x = East, y = North).
            vz: World-frame vertical velocity, m/s, positive up.
            yaw: Heading to hold, radians CCW from +X in the world frame.
        """
        east, north = self._rotate_into_px4(vx, vy)
        yaw_ned = math.pi / 2.0 - _normalize(yaw - self.heading_bias)
        self.last_setpoint_ned = (north, east, -vz)
        self._conn.mav.set_position_target_local_ned_send(
            0, self._conn.target_system, self._conn.target_component,
            _MAV_FRAME_LOCAL_NED, _TYPE_MASK_VELOCITY_YAW,
            0.0, 0.0, 0.0,          # position (ignored by the type mask)
            north, east, -vz,       # velocity, NED
            0.0, 0.0, 0.0,          # acceleration (ignored)
            yaw_ned, 0.0,           # yaw, yaw rate (rate ignored)
        )

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

    def request_data_streams(self, rate_hz: float = 5.0) -> None:
        """Ask PX4 to stream the messages this class reads.

        ``EXTENDED_SYS_STATE`` (the land detector) is not in PX4's default
        onboard stream set, so without this :attr:`on_ground` never becomes
        True and a mission waits out its landing timeout every time.

        Args:
            rate_hz: Stream rate. A land detection does not need to be prompt.
        """
        interval_us = int(1e6 / max(rate_hz, 0.1))
        for message_id in (self._mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE,
                           self._mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
                           self._mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED):
            self._command_long(
                self._mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                float(message_id), float(interval_us),
            )

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
        self.last_setpoint_ned = (north, east, down)
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

    def disarm(self, force: bool = False) -> None:
        """Request disarming.

        Args:
            force: Disarm even in flight. PX4 refuses a normal disarm request
                while it believes it is airborne; the magic ``21196`` is its
                documented override. Only for aborting a stuck episode.
        """
        self._command_long(self._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                           0, 21196.0 if force else 0.0)

    def land(self) -> None:
        """Switch to PX4's autonomous land mode."""
        self._command_long(self._mavutil.mavlink.MAV_CMD_NAV_LAND)

    def close(self) -> None:
        """Release the UDP port so the next run can bind it."""
        self._conn.close()
