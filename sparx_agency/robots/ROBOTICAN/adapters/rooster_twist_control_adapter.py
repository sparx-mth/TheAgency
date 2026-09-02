#!/usr/bin/env python3
"""rooster_twist_control_adapter.py

Translates a planner's geometry_msgs/Twist (FALCON's waypoint_follower_node.py
publishes on /cmd_vel, bridged ROS1->ROS2) into cmd_nav "move" JSON commands
for rooster_command_unit.py.

This does NOT publish ManualControl directly. rooster_command_unit.py's
RoosterUnit is the single owner of /<rooster_id>/manual_control and
/<rooster_id>/keep_alive (see that module's docstring: "there is exactly one
place per drone that talks to the FCU, regardless of who issued the
command"). A second, independent ManualControl publisher would fight it for
the same topic - in particular it would zero the z axis (throttle/altitude-
hold) on every Twist that doesn't set linear.z, dropping the drone out of the
sky the moment a planner's Twist arrived. Routing through cmd_nav's "move"
action (which only ever touches x/y/r, never z) is what keeps that guarantee.

Twist mapping:
  linear.x   forward/backward -> axis x
  linear.y   lateral          -> axis y, NEGATED (see below)
  angular.z  yaw rate         -> axis r, NEGATED (see below)
  linear.z is ignored - altitude is rooster_command_unit.py's job alone.

angular.z AND linear.y are negated: REP103 has positive angular.z = turn left
and positive linear.y = move left, but this drone's FCU axis convention has
positive r = right and positive y = right (rooster_command_unit.py's
live-validated turn_left/turn_right and left/right sign maps). The y negation
was latent until 2026-08-31 -- lateral was introduced disabled and the unproven
un-negated mapping never flew; with it, every lateral correction would push the
wrong way. See LESSONS.md.

max_linear_x/max_linear_y/max_yaw_rate are "real-world rate produced at full
axis deflection (1000)" - the scale factor a planner's Twist is normalized
against before becoming an axis value. max_yaw_rate was recalibrated
2026-07-30 from 0.5 to 1.8 rad/s: a logged manual flight (command+pose,
see docs/progress/entries/007-rooster-velocity-controller.md) showed
axis r=500 (turn_right) produced ~55 deg/s (~0.96 rad/s) over 8 isolated
turn segments, i.e. axis 1000 -> ~1.9 rad/s - the old 0.5 rad/s default was
never live-validated and was ~4x too low, meaning any planner asking for
even a modest yaw rate was actually commanding a much faster real turn than
intended. See LESSONS.md for the full derivation. max_linear_x/max_linear_y
were left unchanged: the same flight's forward/lateral segments were too
short and interleaved with turns (leftover momentum contaminates each
segment) to extract a trustworthy number - a dedicated calibration flight
(isolated single-axis moves, no interleaving) is needed before touching
those with the same confidence.

Yaw axis (r) is also slew-rate-limited (max_yaw_axis_step_per_sec): PX4's
own yaw-rate loop has zero derivative gain (MC_YAWRATE_D=0.0, P/I only), so
an instantaneous step change in commanded rate excites it into oscillation.
Ramping the output here smooths that without touching the PX4 param.
max_yaw_axis_step_per_sec=2500 is a first, conservative guess -- live-test
and retune. See LESSONS.md.
"""

from __future__ import annotations

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from std_msgs.msg import String

from sparx_agency.core.control.axis_velocity_servo import (
    AxisVelocityServo, feedforward_axis,
)
from sparx_agency.robots.ROBOTICAN.rooster_axis_curve import (
    ROOSTER_HORIZONTAL_CURVE,
)
from sparx_agency.robots.common.math_utils import clamp_axis, slew


def velocity_to_axis(v_mps: float, deadzone: float, v_full: float,
                     min_command_mps: float) -> float:
    """Convert a requested body velocity (m/s) into a ManualControl axis.

    The horizontal axes are NOT linear from zero: they are dead up to a
    breakaway and then ramp roughly linearly. Measured 2026-08-18 by
    ``tools/rooster_axis_calibration.py`` over a per-sign sweep of standing
    starts: forward is dead to ~466 counts and reaches ~1.31 m/s at 1000;
    lateral to ~406 counts and ~0.97 m/s. An earlier single hover-step test put
    the forward dead band at 620, but it never probed below it -- the sweep's
    lowest value, 550, already moved the aircraft.

    This shape is why the old ``v / max_v * 1000`` scale failed: it put every
    normal FALCON request right on the dead-band edge, where a few counts flip
    the real speed between nothing and a lurch. That was the long-standing
    jerky tracking. See LESSONS.md and the calibration run's calibration.md.

    Args:
        v_mps: Requested velocity along this axis, m/s (signed).
        deadzone: Axis counts below which the platform does not move at all.
        v_full: Velocity produced at axis 1000, m/s.
        min_command_mps: Requests smaller than this are emitted as 0 rather
            than as a deadzone-edge command, because speeds below roughly
            ``v_full * (1 - deadzone/1000)``'s first step are not continuously
            achievable on this platform at all.

    Returns:
        Axis value in [-1000, 1000].
    """
    if v_full <= 0.0 or abs(v_mps) < min_command_mps:
        return 0.0
    span = max(1.0, 1000.0 - deadzone)
    magnitude = deadzone + min(1.0, abs(v_mps) / v_full) * span
    return clamp_axis(magnitude if v_mps > 0.0 else -magnitude)


class RoosterTwistControlNode(Node):
    def __init__(
        self,
        rooster_id: str = "R1",
        cmd_vel_topic: str | None = None,
        max_linear_x: float = 0.25,
        max_linear_y: float = 0.25,
        # Superseded by the measured r curve below; kept only for
        # use_measured_axis_curve=False. See r_deadzone / r_v_full.
        max_yaw_rate: float = 1.8,
        # MEASURED 2026-08-19, calibration block (iii): the yaw axis is dead
        # below ~102 counts and reaches ~2.59 rad/s at full stick, and it is
        # symmetric (r+ 104/2.580 over 7 points, r- 101/2.597 over 8) -- which
        # retires the 2026-07-30 claim of a 1.9-vs-1.5 rad/s left/right split.
        #
        # Yaw had never had a calibrated inverse: it went through
        # `wz / max_yaw_rate * 1000`, a through-origin scale with NO dead band,
        # which is exactly the form velocity_to_axis's docstring says cannot work
        # for a dead-banded axis. The consequences either side of the crossover:
        # at the follower's own snap floor of 0.14 rad/s it asked for axis 78,
        # below the dead band, so small heading corrections produced NO yaw at
        # all; above ~0.8 rad/s it over-commanded by 15-25%. Both matter during
        # exploration, where the follower turns constantly and heading error
        # feeds straight back into forward speed through cos(heading_err).
        r_deadzone: float = 102.0,
        r_v_full: float = 2.589,
        max_yaw_axis_step_per_sec: float = 2500.0,
        command_hz: float = 20.0,
        cmd_timeout_sec: float = 0.4,
        # The horizontal feedforward generation. False (default): both axes fly
        # the measured expo curve (rooster_axis_curve.ROOSTER_HORIZONTAL_CURVE,
        # 2026-08-25..31 manual calibration -- no dead band, one law for every
        # regime) through their own velocity servos, and lateral is live. True:
        # the pre-2026-08-31 behaviour, kept verbatim as the A/B baseline --
        # dead-band+linear standing/moving pair on x, lateral hard-zeroed.
        legacy_feedforward: bool = False,
        # Ground-truth-measured axis response (2026-08-17). See
        # velocity_to_axis() above and LESSONS.md; these replace the
        # never-validated linear max_linear_x/y scaling, which is now only
        # used when use_measured_axis_curve=False.
        use_measured_axis_curve: bool = True,
        # REVERTED to the in-flight values after the measured standing-start
        # curve (466 / 1.313, from calibration block (i)) was flown and was much
        # worse: distance 256.8 -> 125.5 m, stops 1.3 -> 8.7 per minute, time
        # below stop speed 5% -> 42%. A standing start with a 4 s settle is a
        # different plant from an aircraft mid-flight, and the lower dead band
        # put ordinary requests back under the threshold at which this platform
        # actually moves. Do not re-apply a standing-start fit to flight until
        # block (ii) has measured the MOVING regime -- that gap is the whole
        # point of block (ii). The block (i) numbers are kept in the run's
        # calibration.md, not thrown away.
        x_deadzone: float = 620.0,
        x_v_full: float = 1.25,
        # Full-scale speed once ALREADY MOVING; <=0 disables the second regime.
        #
        # REVERTED to 4.0 after 1.847 -- the block (ii) measurement -- flew worse
        # over two runs: stops 0.4-1.1 -> 2.5-3.9 per minute, time at zero speed
        # 2-5% -> 12-30%, escapes 2 -> 7, and peak speed 1.5-2.1 -> 3.1-3.5 m/s.
        #
        # The measurement was not wrong; PAIRING it was. Block (ii) measured the
        # moving regime as (dead band 412, 1.847 m/s at full stick) -- a curve.
        # Feeding its full-scale into a feedforward that still uses the STANDING
        # dead band of 620 describes no real regime at all, and over-commands: at
        # a 0.6 m/s request it asks 743 counts, which on the real moving curve is
        # ((743-412) * 1.847/588) = 1.04 m/s. Hence the overspeed.
        #
        # The fix is deadzone_moving below, so the pair stays consistent. 4.0 is
        # an empirical value that happens to compensate the mismatch; it should
        # be retired once the consistent (412, 1.847) pair is proven in flight.
        x_v_full_moving: float = 1.847,
        # Dead band once already moving. Measured by calibration block (ii) as
        # 412 counts, and it is the OTHER HALF of x_v_full_moving=1.847 above --
        # the two are one curve and are set together or not at all. Enabled
        # 2026-08-19 as a single coherent change, after 1.847 paired with the
        # standing 620 over-commanded and had to be reverted. At a 0.6 m/s
        # request this asks ~603 counts while moving, LESS than the 677 the
        # empirical 4.0 asked, so the risk here is under-delivery, not overspeed.
        x_deadzone_moving: float = 412.0,
        move_eps_mps: float = 0.10,
        # Lateral is capped off by max_lateral_axis below, so these are inert
        # today. The sweep fitted 406 / 0.969 from 3 points (y- only; every y+
        # segment aborted on tilt or never settled) -- too thin to adopt, and
        # from the same standing-start regime the forward revert above distrusts.
        y_deadzone: float = 700.0,
        y_v_full: float = 1.02,
        # None resolves by mode: 0.05 on the curve (no dead band, so slow
        # commands are genuinely executable), 0.15 legacy (the old first-step
        # floor). Explicit values win. Crossing the curve-mode floor is still a
        # 0 -> ~320-count step in the servo's TARGET, but the slew spreads it
        # over ~0.25 s and 320 counts is only 0.05 m/s of plant response --
        # nothing like the old 620-count dead-band lurch. Kept nonzero so
        # near-zero demands read as "stop" instead of dithering the stick.
        min_command_mps: float | None = None,
        # Ceiling on the lateral axis, counts; <=0 disables lateral (the
        # default). The axis WORKS -- the 2026-08-31 calibration measured it
        # identical to forward at every level; the old "dead until ~1000, then
        # 27-36 deg roll" finding was wall contact -- but it stays opt-in
        # because this adapter serves every /cmd_vel producer (click-to-fly,
        # waypoint followers, demos), and only the exploration follower's
        # lateral use has been validated against the calibration. Enable with
        # 900 (the curve ceiling); the campaign does so per controller variant.
        max_lateral_axis: float = 0.0,
        # Close the forward axis on truth-derived velocity instead of trusting
        # the measured curve open loop. Measured 2026-08-18 over a full FALCON
        # exploration leg: commanded 0.30 m/s mean, achieved 0.11 m/s -- the
        # curve is a steady-state fit and the aircraft is almost never settled.
        # kp/ki are in axis counts per (m/s) and per (m/s*s) of error; the
        # correction is capped well under full deflection so a stale velocity
        # estimate can bias the stick but never own it. See
        # core/control/axis_velocity_servo.py.
        use_velocity_servo: bool = True,
        velocity_topic: str | None = None,
        pose_topic: str | None = None,
        # Retuned 2026-08-18 after the first closed-loop flight: kp=220 tracked
        # to 76% of demand but the velocity error changed sign 2.3 times per
        # second -- a ~1.15Hz limit cycle. The plant lags by a few hundred ms and
        # the feedback is a differentiated position, so a high kp mostly amplifies
        # noise into the stick. Most of the authority now sits in the integrator,
        # which is what the steady 25% shortfall actually needs.
        servo_kp: float = 90.0,
        servo_ki: float = 220.0,
        servo_max_correction: float = 350.0,
        # Counts per second the forward axis may move. The yaw axis has had this
        # since 2026-07-30 for the same reason (PX4's rate loop has no derivative
        # gain); the forward axis needs it too, now that a closed loop can demand
        # a large step. 1200 crosses the whole usable span (~380 counts above the
        # dead band) in ~0.3s. <=0 disables.
        # Ceiling on the forward axis, counts. Saturation is where this
        # platform misbehaves: measured over 34k samples of normal flight the
        # forward axis is benign to ~900 counts (vertical disturbance p90
        # 0.016-0.032 m/s, pitch p90 6-13 deg) and turns nasty beyond it (vz p90
        # 0.111 m/s, pitch p90 22.6 deg) -- the same condition as the full-stick
        # lock-up, where the integrator winds up, the airframe pitches 20-35 deg
        # and translation stops entirely.
        #
        # Capping the CORRECTION does not prevent this and was tried and undone:
        # in the standing regime the feedforward alone is 802 counts at 0.6 m/s
        # (dead band 620, full scale 1.25), so any correction saturates. Only a
        # ceiling on the total holds the axis inside the benign band.
        max_forward_axis: float = 900.0,
        forward_axis_step_per_sec: float = 1200.0,
        # The lateral axis gets its own, much slower ramp (2026-08-31). With
        # the shared 1200/s a full 600-count crab swing developed in 0.5 s at
        # 9-16 sign flips a minute -- measured roll rate p90 ~11 deg/s, and
        # the operator read it as aggressive rolling. 400/s spreads a swing
        # over 1.5 s (steady crab roll is only ~2 deg, so rate was the whole
        # problem) and doubles as a low-pass: a sub-second flip never develops
        # amplitude. Release is faster so escapes shed lateral within ~1 s,
        # and stop_motion stays instant.
        lateral_axis_step_per_sec: float = 400.0,
        lateral_axis_release_per_sec: float = 600.0,
        # Releases were instantaneous, which on this platform is not a coast:
        # PX4 flies Position mode, so a dropped stick is an active brake. That
        # is what made every cutoff feel like a hard stop. Ramp them too, just
        # faster than the acceleration ramp. stop_motion() stays instant -- the
        # 0.4s command-timeout path really is an emergency.
        forward_axis_release_per_sec: float = 3000.0,
        # Feedback older than this is not feedback. The servo holds
        # feed-forward-only rather than integrating against a frozen number.
        velocity_timeout_sec: float = 0.5,
        # ── Altitude, without ever taking the throttle axis ────────────────
        # linear.z used to be discarded outright, because RoosterUnit owns
        # /manual_control's z and a second publisher would drop the aircraft out
        # of the sky (see this module's docstring). The cost was that the drone
        # mapped a single horizontal band at whatever height it took off to.
        #
        # So the demand is integrated into distance and spent as occasional
        # nudges of RoosterUnit's OWN hold setpoint (cmd_nav up/down ->
        # nudge_altitude_target). Nothing else touches the throttle: the hold
        # loop that was tuned to +/-0.04m keeps flying, its target just moves.
        # Deliberately slow and bounded -- altitude stability on this platform was
        # the hardest-won behaviour in the stack and is not worth risking for a
        # faster climb.
        follow_altitude: bool = True,
        # 0.3 -> 0.15 and the band 1.0 -> 0.3 on 2026-08-19. The nudge authority
        # was far too coarse for the ~1 m operating window: FALCON's vertical
        # demand simply saturated it, and the live target RAILED -- at the old
        # setpoint it pinned to the 1.35 m ceiling, and after the setpoint was
        # lowered it walked 0.90 -> 0.60 and sat on the floor while the aircraft
        # held 1.21 m. A 0.6 m standing error then drove the hold loop hard
        # (z sd 42 -> 114, ranger sd 0.065 -> 0.204, escapes 3-7 -> 11).
        # FALCON should be able to bias the cruise height, not relocate it.
        altitude_nudge_m: float = 0.15,
        altitude_nudge_interval_sec: float = 3.0,
        # Furthest the accumulated nudges may take the aircraft from the height
        # it was holding when tracking began, metres. A runaway vz demand can
        # then bias the cruise height but never fly it into the ceiling or floor.
        altitude_band_m: float = 0.3,
    ):
        super().__init__(f"{rooster_id.lower()}_twist_control")

        self.rooster_id = rooster_id
        self.max_linear_x = float(max_linear_x)
        self.max_linear_y = float(max_linear_y)
        self.max_yaw_rate = float(max_yaw_rate)
        self.r_deadzone = float(r_deadzone)
        self.r_v_full = float(r_v_full)
        self.legacy_feedforward = bool(legacy_feedforward)
        self.use_measured_axis_curve = bool(use_measured_axis_curve)
        self.x_deadzone = float(x_deadzone)
        self.x_v_full = float(x_v_full)
        self.x_v_full_moving = float(x_v_full_moving)
        self.x_deadzone_moving = float(x_deadzone_moving)
        self.move_eps_mps = float(move_eps_mps)
        self.y_deadzone = float(y_deadzone)
        self.y_v_full = float(y_v_full)
        self.min_command_mps = float(
            (0.15 if self.legacy_feedforward else 0.05)
            if min_command_mps is None else min_command_mps)
        self.max_lateral_axis = float(max_lateral_axis)
        self.max_yaw_axis_step_per_sec = float(max_yaw_axis_step_per_sec)
        self.command_hz = float(command_hz)

        self.cmd_timeout = Duration(seconds=float(cmd_timeout_sec))
        self.last_cmd_time = self.get_clock().now()
        self.current_twist = Twist()
        # Slew-limited state for the r axis only (see module docstring) --
        # x/y aren't implicated in the reported oscillation and stay as a
        # direct pass-through.
        self._r_axis = 0.0

        # FALCON's real_drone.launch/sphera_drone.launch default drone_ns to
        # "" for Rooster, so waypoint_follower publishes plain /cmd_vel, not
        # /R1/cmd_vel - matches that unless explicitly overridden.
        self.cmd_vel_topic = cmd_vel_topic if cmd_vel_topic is not None else "/cmd_vel"
        self.cmd_nav_topic = f"/{self.rooster_id}/cmd_nav"

        self.cmd_sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10)
        self.cmd_nav_pub = self.create_publisher(String, self.cmd_nav_topic, 10)

        self.use_velocity_servo = bool(use_velocity_servo)
        self.max_forward_axis = float(max_forward_axis)
        if not self.legacy_feedforward:
            # Never command past the last measured point of the curve.
            self.max_forward_axis = min(self.max_forward_axis,
                                        ROOSTER_HORIZONTAL_CURVE.max_counts)
            self.max_lateral_axis = min(self.max_lateral_axis,
                                        ROOSTER_HORIZONTAL_CURVE.max_counts)
        self.forward_axis_step_per_sec = float(forward_axis_step_per_sec)
        self.forward_axis_release_per_sec = float(forward_axis_release_per_sec)
        self.lateral_axis_step_per_sec = float(lateral_axis_step_per_sec)
        self.lateral_axis_release_per_sec = float(lateral_axis_release_per_sec)
        # Slew-limited output state per horizontal axis ("y" is REP103 body
        # lateral, leftward positive -- negated only at the cmd_nav boundary).
        self._axis_out = {"x": 0.0, "y": 0.0}
        self.follow_altitude = bool(follow_altitude)
        self.altitude_nudge_m = float(altitude_nudge_m)
        self.altitude_nudge_interval = Duration(
            seconds=float(altitude_nudge_interval_sec))
        self.altitude_band_m = float(altitude_band_m)
        self._pending_alt_m = 0.0       # demanded but not yet spent
        self._alt_offset_m = 0.0        # spent, relative to the initial hold
        self._last_nudge_time = None
        self._alt_tick = None
        self.velocity_timeout = Duration(seconds=float(velocity_timeout_sec))
        self._world_velocity = (0.0, 0.0)
        self._velocity_time = None
        self._yaw = 0.0
        self._servo_tick = None
        if self.legacy_feedforward:
            self._servos = {"x": AxisVelocityServo(
                deadzone=self.x_deadzone, v_full=self.x_v_full,
                kp=float(servo_kp), ki=float(servo_ki),
                max_correction=float(servo_max_correction),
                min_command_mps=self.min_command_mps,
                v_full_moving=self.x_v_full_moving,
                deadzone_moving=self.x_deadzone_moving,
                move_eps_mps=self.move_eps_mps,
                # The servo must know the ceiling this node clamps to, or its
                # anti-windup guards a limit that does not exist.
                output_limit=self.max_forward_axis)}
        else:
            # One measured curve, two identical axes, one servo each.
            self._servos = {
                axis: AxisVelocityServo(
                    0.0, 0.0, curve=ROOSTER_HORIZONTAL_CURVE,
                    kp=float(servo_kp), ki=float(servo_ki),
                    max_correction=float(servo_max_correction),
                    min_command_mps=self.min_command_mps,
                    output_limit=limit)
                for axis, limit in (("x", self.max_forward_axis),
                                    ("y", self.max_lateral_axis))
                if limit > 0.0}
        if self.use_velocity_servo:
            velocity_topic = (velocity_topic
                              or f"/{self.rooster_id}/velocity_truth")
            pose_topic = pose_topic or f"/{self.rooster_id}/localization"
            self.create_subscription(
                TwistStamped, velocity_topic, self._velocity_callback, 10)
            self.create_subscription(
                PoseStamped, pose_topic, self._pose_callback, 10)

        self.command_timer = self.create_timer(
            1.0 / float(command_hz), self.command_timer_callback)

        self.get_logger().info(
            f"RoosterTwistControlNode ready\n"
            f"  cmd_vel: {self.cmd_vel_topic}\n"
            f"  cmd_nav: {self.cmd_nav_topic} (action=move, x/y/r only)"
        )

    def cmd_vel_callback(self, msg: Twist) -> None:
        self.current_twist = msg
        self.last_cmd_time = self.get_clock().now()

    def _velocity_callback(self, msg: TwistStamped) -> None:
        self._world_velocity = (msg.twist.linear.x, msg.twist.linear.y)
        self._velocity_time = self.get_clock().now()

    def _pose_callback(self, msg: PoseStamped) -> None:
        # Planar contract: orientation is z=sin(yaw/2), w=cos(yaw/2).
        self._yaw = 2.0 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w)

    def measured_body_velocity(self) -> tuple | None:
        """Body-frame (forward, lateral) velocity from truth, or None if stale.

        Returns:
            ``(forward, lateral)`` in m/s, REP103 body frame (lateral positive
            = left), or ``None`` when no velocity has arrived within
            ``velocity_timeout_sec`` -- in which case the caller must fall
            back to feed-forward only.
        """
        if self._velocity_time is None:
            return None
        if (self.get_clock().now() - self._velocity_time) > self.velocity_timeout:
            return None
        vx, vy = self._world_velocity
        cos_y, sin_y = math.cos(self._yaw), math.sin(self._yaw)
        return (vx * cos_y + vy * sin_y, -vx * sin_y + vy * cos_y)

    def command_timer_callback(self) -> None:
        now = self.get_clock().now()
        if (now - self.last_cmd_time) > self.cmd_timeout:
            self.stop_motion()
            return
        self.publish_move(self.current_twist)

    def stop_motion(self) -> None:
        self.current_twist = Twist()
        self._r_axis = 0.0  # stop is immediate, never slew-limited
        for axis in self._axis_out:
            self._axis_out[axis] = 0.0
        for servo in self._servos.values():
            servo.reset()
        self._publish_cmd_nav("stop")

    def publish_move(self, twist: Twist) -> None:
        # Negated -- see module docstring. Through the measured dead-banded
        # curve, not a through-origin scale: see r_deadzone above.
        if self.use_measured_axis_curve and self.r_v_full > 0.0:
            target_r = feedforward_axis(-twist.angular.z, self.r_deadzone,
                                        self.r_v_full)
        else:
            target_r = (-twist.angular.z / self.max_yaw_rate * 1000.0
                        if self.max_yaw_rate else 0.0)
        max_step = self.max_yaw_axis_step_per_sec / self.command_hz
        self._r_axis = slew(target_r, self._r_axis, max_step)

        now = self.get_clock().now()
        dt = 0.0 if self._servo_tick is None else (
            (now - self._servo_tick).nanoseconds * 1e-9)
        self._servo_tick = now
        v_meas = self.measured_body_velocity()

        if not self.legacy_feedforward:
            # Measured-curve path: each horizontal axis gets its own servo and
            # slew state, all in REP103 body frame (lateral positive = left).
            ax_x = self._servo_axis("x", twist.linear.x,
                                    None if v_meas is None else v_meas[0], dt)
            ax_y_left = (self._servo_axis(
                "y", twist.linear.y,
                None if v_meas is None else v_meas[1], dt)
                if "y" in self._servos else 0.0)
            # Negated at the boundary -- FCU y positive is RIGHT (see module
            # docstring); everything upstream of this line is REP103.
            ax_y = -ax_y_left
        else:
            if self.use_velocity_servo:
                ax_x = self._servo_axis(
                    "x", twist.linear.x,
                    None if v_meas is None else v_meas[0], dt)
                ax_y = velocity_to_axis(twist.linear.y, self.y_deadzone,
                                        self.y_v_full, self.min_command_mps)
            elif self.use_measured_axis_curve:
                ax_x = velocity_to_axis(twist.linear.x, self.x_deadzone,
                                        self.x_v_full, self.min_command_mps)
                ax_y = velocity_to_axis(twist.linear.y, self.y_deadzone,
                                        self.y_v_full, self.min_command_mps)
            else:
                ax_x = clamp_axis(twist.linear.x / self.max_linear_x * 1000.0
                                  if self.max_linear_x else 0.0)
                ax_y = clamp_axis(twist.linear.y / self.max_linear_y * 1000.0
                                  if self.max_linear_y else 0.0)
        # Lateral ceiling; <=0 disables the axis (the legacy baseline).
        if self.max_lateral_axis <= 0.0:
            ax_y = 0.0
        else:
            ax_y = max(-self.max_lateral_axis, min(self.max_lateral_axis, ax_y))
        if self.follow_altitude:
            self._follow_altitude(twist.linear.z)
        axes = {
            "x": ax_x,
            "y": ax_y,
            "r": clamp_axis(self._r_axis),
        }
        self._publish_cmd_nav("move", axes=axes)

    def _follow_altitude(self, vz: float) -> None:
        """Spend a vertical velocity demand as nudges of the hold setpoint.

        Integrates ``vz`` into metres and, once a nudge's worth has accumulated
        and the interval has elapsed, asks rooster_command_unit to move its own
        altitude target. Never publishes a throttle axis.

        Args:
            vz: Demanded world-frame vertical velocity, m/s (positive up).
        """
        now = self.get_clock().now()
        dt = 0.0 if self._alt_tick is None else (
            (now - self._alt_tick).nanoseconds * 1e-9)
        self._alt_tick = now
        if dt <= 0.0 or dt > 1.0:      # first tick, or a gap worth distrusting
            return

        self._pending_alt_m += vz * dt
        if abs(self._pending_alt_m) < self.altitude_nudge_m:
            return
        if (self._last_nudge_time is not None
                and (now - self._last_nudge_time) < self.altitude_nudge_interval):
            return

        step = self.altitude_nudge_m if self._pending_alt_m > 0.0 else -self.altitude_nudge_m
        if abs(self._alt_offset_m + step) > self.altitude_band_m:
            # At the edge of the band: drop the demand rather than let it queue
            # up and fire the instant the band is re-entered.
            self._pending_alt_m = 0.0
            self.get_logger().warn(
                f"altitude demand held at band edge "
                f"({self._alt_offset_m:+.2f}m of +/-{self.altitude_band_m:.2f}m)",
                throttle_duration_sec=10.0)
            return

        self._pending_alt_m -= step
        self._alt_offset_m += step
        self._last_nudge_time = now
        # Pass the distance explicitly: the cmd_nav up/down actions default to
        # DEFAULT_ALTITUDE_NUDGE_M (0.5 m), and this node's altitude_nudge_m is
        # what the band accounting above is keeping track of.
        self._publish_cmd_nav("up" if step > 0.0 else "down",
                              value=abs(step))
        self.get_logger().info(
            f"altitude nudge {step:+.2f}m (now {self._alt_offset_m:+.2f}m from "
            f"the initial hold)")

    def _servo_axis(self, axis: str, v_cmd: float, v_meas: float | None,
                    dt: float) -> float:
        """One horizontal axis through its servo, degrading to feed-forward alone.

        Args:
            axis: ``"x"`` (forward) or ``"y"`` (REP103 lateral, left positive).
            v_cmd: Requested body velocity along the axis, m/s (signed).
            v_meas: Measured body velocity along the axis, m/s, or ``None``
                when the feedback is stale.
            dt: Seconds since the previous control tick (shared by both axes).

        Returns:
            Axis value, slew-limited and clamped to this axis's ceiling.
        """
        servo = self._servos[axis]
        limit = (self.max_forward_axis if axis == "x"
                 else self.max_lateral_axis)
        if v_meas is None or not self.use_velocity_servo:
            # No trustworthy feedback (or servo disabled): run open loop and
            # drop the integrator, so nothing accumulated against a dead
            # estimate survives its return.
            servo.reset()
            if self.use_velocity_servo:
                self.get_logger().warn(
                    "velocity feedback stale -- %s axis is feed-forward only"
                    % axis, throttle_duration_sec=5.0)
            if self.legacy_feedforward:
                target = velocity_to_axis(v_cmd, self.x_deadzone, self.x_v_full,
                                          self.min_command_mps)
            else:
                target = (0.0 if abs(v_cmd) < self.min_command_mps
                          else ROOSTER_HORIZONTAL_CURVE.axis_for(v_cmd))
        else:
            target = servo.update(v_cmd, v_meas, dt)
        target = max(-limit, min(limit, target))
        if axis == "y":
            step_rate = self.lateral_axis_step_per_sec
            release_rate = self.lateral_axis_release_per_sec
        else:
            step_rate = self.forward_axis_step_per_sec
            release_rate = self.forward_axis_release_per_sec
        if step_rate <= 0.0:
            self._axis_out[axis] = target
        else:
            # Warm-start across the dead band, legacy curve only. Ramping up
            # from zero spends ~0.5 s below the old 620-count dead band on
            # every restart of motion (measured 2026-08-18: 25% of a flight's
            # ticks). The measured curve has no dead band, so there is no
            # edge to jump to and the plain ramp is correct.
            if (self.legacy_feedforward
                    and self._axis_out[axis] == 0.0 and target != 0.0):
                edge = min(abs(target), self.x_deadzone)
                self._axis_out[axis] = edge if target > 0.0 else -edge
            rate = release_rate if target == 0.0 else step_rate
            self._axis_out[axis] = slew(target, self._axis_out[axis],
                                        rate / self.command_hz)
        return clamp_axis(self._axis_out[axis])

    def _publish_cmd_nav(self, action: str, **payload) -> None:
        msg = String()
        msg.data = json.dumps({"action": action, **payload})
        self.cmd_nav_pub.publish(msg)


def main(args=None):
    import argparse

    # Every value argument defaults to None and is passed through ONLY when the
    # operator actually set it, so the class defaults are the single source of
    # truth. The old pattern (argparse defaults mirroring the class defaults by
    # hand) silently overrode two tuned values for days when the copies drifted
    # -- same failure class as the roslaunch re-declaration lesson in
    # LESSONS.md, one layer down.
    parser = argparse.ArgumentParser(description="Rooster Twist -> cmd_nav control adapter")
    parser.add_argument("--rooster-id", default="R1")
    parser.add_argument("--cmd-vel-topic", default=None)
    for name in ("--max-linear-x", "--max-linear-y", "--max-yaw-rate",
                 "--max-yaw-axis-step-per-sec", "--command-hz",
                 "--cmd-timeout-sec", "--max-lateral-axis", "--servo-kp",
                 "--servo-ki", "--servo-max-correction",
                 "--forward-axis-step-per-sec", "--lateral-axis-step-per-sec",
                 "--lateral-axis-release-per-sec", "--altitude-nudge-m",
                 "--altitude-band-m", "--min-command-mps"):
        parser.add_argument(name, type=float, default=None)
    parser.add_argument("--legacy-feedforward", action="store_true",
                        help="fly the pre-2026-08-31 dead-band/two-regime "
                             "curve with lateral disabled (the A/B baseline)")
    parser.add_argument("--no-velocity-servo", action="store_true",
                        help="run the horizontal axes open loop off the "
                             "curve only (pre-2026-08-18 behaviour)")
    parser.add_argument("--no-follow-altitude", action="store_true",
                        help="discard linear.z instead of nudging the hold "
                             "setpoint (pre-2026-08-18 behaviour)")
    parsed, _ = parser.parse_known_args()

    kwargs = {name: value for name, value in vars(parsed).items()
              if value is not None and not name.startswith("no_")
              and name != "legacy_feedforward"}
    kwargs["legacy_feedforward"] = parsed.legacy_feedforward
    if parsed.no_velocity_servo:
        kwargs["use_velocity_servo"] = False
    if parsed.no_follow_altitude:
        kwargs["follow_altitude"] = False

    rclpy.init(args=args)
    node = RoosterTwistControlNode(**kwargs)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
