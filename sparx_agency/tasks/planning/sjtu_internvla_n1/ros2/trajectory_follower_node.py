#!/usr/bin/env python3
"""Follow N1's committed route and drive the SJTU drone's one control input.

The "tracking the trajectory" half of the stack, and the mirror of FALCON's
``waypoint_follower_node``: subscribe the world-frame ``nav_msgs/Path`` the N1
policy node commits, pursue it, and publish a body twist on
``/simple_drone/cmd_vel``. Nothing here is N1-specific -- it would fly a NavDP
path, an A* path or a hand-drawn one identically, which is the point of putting a
plain ``Path`` on the wire between the two.

The controller is the shared holonomic
:class:`~sparx_agency.core.planning.trackers.pure_pursuit.tracker.PurePursuitTracker3D`,
which emits a **world-frame** velocity aimed at a lookahead point. The SJTU plugin
reads ``cmd_vel`` in the **yaw-aligned body frame**, so this node performs the one
rotation world->body and clamps the result with the platform's own
:mod:`~sparx_agency.robots.SJTU.adapters.velocity_command` adapter -- the same
horizontal-pair clamp the airframe needs so a speed limit never becomes a
steering error.

Four disciplines this node owns, all of which are silent killers when missing:

* **The tracker is reset on every commitment.** ``PurePursuitTracker3D`` keeps
  ``progress_idx`` as a monotone high-water mark, which is right *within* one
  path and wrong *across* two. Every commitment is a fresh polyline with the
  aircraft at its start, so carrying progress over means the search window opens
  past the aircraft: a longer path reads as ``path_diverged`` and the drone
  stops forever, a shorter one indexes off the end and kills the timer.
* **Altitude is captured before the route is tracked.** The tracker folds
  altitude error into a *3D* cross-track, so an aircraft 1.2 m below its cruise
  band is already outside ``path_tolerance`` and gets a zero command -- it can
  never climb into tolerance, which reads as a dead follower rather than a
  deadlock. Until the aircraft is inside the band this node flies pure vertical
  and does not track at all.
* **Forward speed is braked on raw depth.** N1 plans where to go, not whether
  the way is clear, and this stack has no map. The reflex is
  :class:`~sparx_agency.core.planning.safety.depth_proximity_brake.DepthProximityBrake`
  -- the corridor minimum of the depth image against a stopping distance -- and
  it exists because the alternative is the next discipline.
* **Thinking and turning are stationary.** The policy node holds this follower
  (``/n1/hold``) while System 2 thinks, so the frame the model reasons about and
  the pose its route is anchored at are the same place the aircraft is when the
  answer arrives -- System 2 takes seconds, and an aircraft that keeps flying
  through them anchors its next route metres behind itself. It also asks for
  rotations outright (``/n1/yaw_goal``), because a discrete turn action *is* a
  rotation and flying it as a bent waypoint lets a holonomic tracker satisfy it
  by crabbing 0.25 m sideways without ever looking anywhere new. Both are flown
  with the altitude hold still live -- a zero twist would drop the aircraft.
* **A capsized airframe stops the flight.** The SJTU plugin thrusts along body
  z, so past ~35 deg of roll or pitch it cannot climb, translate or yaw -- while
  still reporting FLYING and a healthy 30 Hz of odometry. Every axis of every
  command is silently ignored and nothing in the topic set says so. Measured
  here: an aircraft that hit the reception desk lay at roll -83 deg for the rest
  of a 60 s recording while the policy cheerfully committed sixteen routes to
  it. This node reads attitude off odom and refuses to command a capsized
  aircraft, loudly.

CPU-only, no torch: the GPU is the network's.
"""
from __future__ import annotations

import os
import signal
import threading
import time
from math import asin, atan2, cos, degrees, radians, sin

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import rclpy
import yaml
from rclpy._rclpy_pybind11 import RCLError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int8

from sparx_agency.core.common.math.se3 import yaw_from_quaternion
from sparx_agency.core.common.types import (
    normalize_angle,
    KinematicLimits,
    Pose3D,
    State3D,
    Twist3D,
)
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest
from sparx_agency.core.planning.recovery.escape_maneuver import (
    EscapeManeuver,
    EscapeParams,
)
from sparx_agency.core.planning.recovery.stuck_detector import StuckVerdict
from sparx_agency.core.planning.safety.depth_proximity_brake import (
    DepthProximityBrake,
    DepthProximityBrakeConfig,
)
from sparx_agency.core.planning.trackers.pure_pursuit.params import PurePursuitParams3D
from sparx_agency.core.planning.trackers.pure_pursuit.tracker import PurePursuitTracker3D
from sparx_agency.core.planning.vlas.common.turn_in_place import (
    TurnInPlace,
    describe as describe_turn,
    turn_spec_from_config,
)
from sparx_agency.robots.SJTU.adapters.velocity_command import (
    BodyVelocityLimits,
    BodyTwistCommand,
    fill_twist,
    twist_fields,
    zero_twist_fields,
)
from sparx_agency.tasks.planning.sjtu_internvla_n1.path_trajectory import (
    trajectory_from_points,
)


def _yaw_from_quat(q):
    """Yaw (radians, CCW from +x) from a geometry_msgs quaternion."""
    return yaw_from_quaternion((q.x, q.y, q.z, q.w))


def _roll_pitch_from_quat(q):
    """Roll and pitch in radians, ZYX convention."""
    roll = atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    s = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
    return roll, asin(s)


def _depth_metres(msg):
    """A ``sensor_msgs/Image`` as an HxW float32 array of metres.

    Decoded here rather than through ``cv_bridge`` so this node keeps its only
    heavy dependency (numpy) and stays importable without a ROS perception
    stack. The two encodings this platform emits are handled; anything else is
    an error rather than a guess, because a depth image interpreted with the
    wrong scale brakes at the wrong distance in both directions.
    """
    if msg.encoding == "32FC1":
        dtype, scale = np.float32, 1.0
    elif msg.encoding == "16UC1":
        dtype, scale = np.uint16, 1e-3
    else:
        raise ValueError("unsupported depth encoding %r" % (msg.encoding,))
    # `step` is the row stride in BYTES and is not always width*itemsize -- a
    # padded row read as if it were tight shears the image diagonally, which
    # looks like a plausible depth map of a different scene.
    itemsize = np.dtype(dtype).itemsize
    row = msg.step // itemsize if msg.step else msg.width
    data = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row)
    return (data[:, :msg.width].astype(np.float32) * scale)


def _load_config(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


class TrajectoryFollowerNode(Node):
    """Pursue a world path and command the SJTU drone's body twist."""

    def __init__(self):
        super().__init__("trajectory_follower_node")
        self.declare_parameter("config_file", "")
        cfg = _load_config(self.get_parameter("config_file").value)

        topics = cfg.get("topics", {})
        foll = cfg.get("follower", {})

        self._path_topic = topics.get("trajectory", "/simple_drone/n1/trajectory")
        self._odom_topic = topics.get("odom", "/simple_drone/odom")
        self._cmd_topic = topics.get("cmd_vel", "/simple_drone/cmd_vel")
        self._state_topic = topics.get("state", "/simple_drone/state")
        self._alt_offset_topic = topics.get("altitude_offset",
                                            "/simple_drone/n1/altitude_offset")
        self._hold_topic = topics.get("hold", "/simple_drone/n1/hold")
        self._yaw_goal_topic = topics.get("yaw_goal", "/simple_drone/n1/yaw_goal")
        self._blocked_topic = topics.get("blocked", "/simple_drone/n1/blocked")

        self._cruise = float(foll.get("cruise_speed", 0.4))
        self._target_alt = float(foll.get("target_altitude_m", 1.2))
        self._control_rate = float(foll.get("control_rate_hz", 20.0))
        max_speed_xy = float(foll.get("max_speed_xy", 1.0))
        max_speed_z = float(foll.get("max_speed_z", 0.5))
        max_yaw_rate = float(foll.get("max_yaw_rate", 1.2))
        # The altitude band the route may be tracked inside, and the wider band
        # that hands control back to the climb. Two thresholds, not one: a single
        # one chatters between climbing and tracking on the noise.
        self._alt_capture_m = float(foll.get("altitude_capture_m", 0.25))
        self._alt_release_m = float(foll.get("altitude_release_m", 0.60))
        self._alt_kp = float(foll.get("altitude_kp", 1.2))
        self._odom_timeout_s = float(foll.get("odom_timeout_s", 1.0))
        self._alt_offset_limit = abs(float(foll.get("altitude_offset_limit_m", 0.8)))
        # The offset is RAMPED, not stepped. Stepping it 0.5 m instantly makes
        # the altitude error jump straight past `altitude_release_m`, so the
        # gate drops out of route tracking and the aircraft stops dead to go up
        # or down -- turning every look-down into a two-second hover at each
        # end. Ramped under the release band, the error never gets there and
        # the aircraft simply descends along the route it was already flying.
        self._alt_offset_rate = abs(float(foll.get("altitude_offset_rate_m_s", 0.35)))
        self._capsize_rad = radians(float(foll.get("capsize_deg", 35.0)))
        self._depth_timeout_s = float(foll.get("depth_timeout_s", 3.0))

        self._tracker = PurePursuitTracker3D(PurePursuitParams3D(
            cruise_speed=self._cruise,
            max_speed=max(self._cruise, float(foll.get("max_track_speed", self._cruise))),
            max_speed_z=max_speed_z,
            max_yaw_rate=max_yaw_rate,
            base_lookahead=float(foll.get("lookahead_m", 0.8)),
            slow_down_distance=float(foll.get("slow_down_distance_m", 0.15)),
            min_speed=float(foll.get("min_speed", 0.1)),
            goal_tolerance=float(foll.get("goal_tolerance_m", 0.10)),
            path_tolerance=float(foll.get("path_tolerance_m", 0.8)),
            yaw_lookahead=float(foll.get("yaw_lookahead_m", 1.5)),
            # Stop and rotate when the route heads far enough off the nose.
            # This is what makes N1's discrete TURN action a turn: the fallback
            # step it renders is only 0.25 m long, so flown as a translation the
            # aircraft creeps 0.25 m sideways and rotates a few degrees, and a
            # model that wants to look somewhere else never gets to.
            stop_turn_rad=float(foll.get("stop_turn_rad", 0.35)),
            resume_turn_rad=float(foll.get("resume_turn_rad", 0.12)),
        ))
        self._limits = KinematicLimits(max_speed_xy=max_speed_xy, max_speed_z=max_speed_z,
                                       max_yaw_rate=max_yaw_rate)
        self._body_limits = BodyVelocityLimits(max_speed_xy=max_speed_xy,
                                               max_speed_z=max_speed_z,
                                               max_yaw_rate=max_yaw_rate)

        self._lock = threading.Lock()
        self._path_xy = []      # list of (x, y) world
        self._state = None      # State3D
        self._state_stamp = 0.0  # monotonic seconds of the last odom
        self._path_epoch = 0     # bumped on every commitment
        self._tracked_epoch = -1  # the epoch the tracker was last reset for
        self._alt_captured = False
        self._flying = None      # None until /simple_drone/state is heard
        # Commanded departure from the cruise altitude, metres. The policy node
        # drives it negative to perform System 2's look-down: the model asks for
        # a lower view of the scene and computes its pixel goal in that frame,
        # and this airframe has no way to tilt its camera -- so it drops
        # instead, using the same forward camera from lower down.
        self._alt_offset = 0.0          # what is being flown, slewed
        self._alt_offset_target = 0.0   # what was asked for
        self._attitude = (0.0, 0.0)  # (roll, pitch) radians
        self._capsized = False
        self._depth = None
        self._depth_stamp = 0.0
        # Held by the policy node while it thinks. Translation and yaw stop; the
        # altitude hold does not, because this airframe sinks without it.
        self._hold = False
        # An absolute world heading the policy node has asked for, and the
        # manoeuvre that flies it. The turn is deliberately slow: the frame at
        # the end of it is the whole reason the model asked.
        self._yaw_goal = None
        self._turn = TurnInPlace(turn_spec_from_config(foll.get("turn", {}) or {}))
        self._blocked = False
        # BREAK CONTACT BEFORE LOOKING SOMEWHERE ELSE. Rotating on the spot does
        # not help an aircraft that is already inside its own stopping distance
        # of a wall: the wall stays inside the depth corridor across most of the
        # arc, so every heading is blocked and the policy is asked to choose
        # between them for ever. Measured in the hospital: pinned 0.45-0.70 m
        # from the office wall for a whole run, thirteen rotations, zero metres.
        # So a turn requested while hard-blocked backs off first.
        esc = foll.get("escape", {}) or {}
        self._escape_enabled = bool(esc.get("enabled", True))
        self._escape = EscapeManeuver(EscapeParams(
            brake_s=float(esc.get("brake_s", 0.4)),
            back_s=float(esc.get("back_s", 2.0)),
            back_speed=float(esc.get("back_speed", 0.25)),
            probe_s=float(esc.get("probe_s", 0.8)),
            probe_speed=float(esc.get("probe_speed", 0.15)),
            settle_s=float(esc.get("settle_s", 0.5)),
            # OFF, deliberately. The depth corridor only protects the aircraft
            # FORWARD; a sideways probe next to a wall slides toward a jamb
            # nothing is watching. The back-off retraces ground the aircraft was
            # occupying seconds ago, which is the one direction that is known
            # clear without a rear sensor.
            allow_lateral=bool(esc.get("allow_lateral", False)),
            max_attempts=int(esc.get("max_attempts", 3))))
        self._escaping = False

        camera = cfg.get("camera", {})
        brake_cfg = cfg.get("brake", {})
        self._brake_enabled = bool(brake_cfg.get("enabled", True))
        self._brake = DepthProximityBrake(DepthProximityBrakeConfig(
            fx=float(camera.get("fx", 390.642735)), fy=float(camera.get("fy", 390.642735)),
            cx=float(camera.get("cx", 300.5)), cy=float(camera.get("cy", 300.5)),
            corridor_halfwidth_m=float(brake_cfg.get("corridor_halfwidth_m", 0.35)),
            corridor_halfheight_m=float(brake_cfg.get("corridor_halfheight_m", 0.35)),
            nose_offset_m=float(brake_cfg.get("nose_offset_m", 0.10)),
            min_valid_m=float(brake_cfg.get("min_valid_m", 0.15)),
            brake_decel=float(brake_cfg.get("brake_decel", 0.8)),
            react_s=float(brake_cfg.get("react_s", 0.30)),
            margin_m=float(brake_cfg.get("margin_m", 0.45)),
            hard_block_d_m=float(brake_cfg.get("hard_block_d_m", 0.70)),
            stride=int(brake_cfg.get("stride", 4))))

        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Path, self._path_topic, self._on_path, latched)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, sensor_qos)
        self.create_subscription(Int8, self._state_topic, self._on_state, 1)
        self.create_subscription(Float32, self._alt_offset_topic, self._on_alt_offset, 1)
        self.create_subscription(Bool, self._hold_topic, self._on_hold, 1)
        self.create_subscription(Float32, self._yaw_goal_topic, self._on_yaw_goal, 1)
        if self._brake_enabled:
            self.create_subscription(Image, topics.get(
                "depth", "/simple_drone/front_depth/depth/image_raw"),
                self._on_depth, sensor_qos)
        self._cmd_pub = self.create_publisher(Twist, self._cmd_topic, 1)
        # Hard-blocked, published rather than only logged: the policy node has no
        # other way to learn that the route it keeps committing cannot be flown,
        # and a recording that shows a motionless aircraft with no explanation is
        # how seventy seconds of the last hospital run went unexplained.
        self._blocked_pub = self.create_publisher(Bool, self._blocked_topic, latched)

        self.create_timer(1.0 / max(1e-3, self._control_rate), self._control)
        self.get_logger().info(
            "trajectory_follower_node up: path=%s odom=%s -> cmd_vel=%s "
            "(cruise %.2f m/s, altitude %.2f m); %s"
            % (self._path_topic, self._odom_topic, self._cmd_topic,
               self._cruise, self._target_alt, describe_turn(self._turn.spec)))

    def _on_path(self, msg):
        xy = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        with self._lock:
            self._path_xy = xy
            self._path_epoch += 1

    def _on_alt_offset(self, msg):
        offset = float(msg.data)
        # Bounded on both sides: a runaway offset would fly the aircraft into
        # the floor or the ceiling, and neither is recoverable here.
        offset = max(-self._alt_offset_limit, min(self._alt_offset_limit, offset))
        with self._lock:
            self._alt_offset_target = offset

    def _on_hold(self, msg):
        with self._lock:
            self._hold = bool(msg.data)

    def _on_yaw_goal(self, msg):
        """Take an absolute world heading to rotate to.

        Stored, not started: the manoeuvre begins on the control step, where the
        measured yaw and the clock are already in hand. Republishing the heading
        already being flown is a no-op rather than a restart -- restarting would
        keep resetting the timeout and a genuinely blocked rotation would never
        report that it failed.
        """
        goal = float(msg.data)
        with self._lock:
            self._yaw_goal = goal

    def _on_state(self, msg):
        # The SJTU plugin's flight state: 0 landed, 1 flying. Tracking a route
        # while the aircraft is on the ground just spins the rotors against the
        # floor and moves the plan on without it.
        with self._lock:
            self._flying = int(msg.data) == 1

    def _on_odom(self, msg):
        p = msg.pose.pose
        t = msg.twist.twist
        state = State3D(
            pose=Pose3D(p.position.x, p.position.y, p.position.z, _yaw_from_quat(p.orientation)),
            twist=Twist3D(t.linear.x, t.linear.y, t.linear.z, t.angular.z))
        with self._lock:
            self._state = state
            self._state_stamp = time.monotonic()
            self._attitude = _roll_pitch_from_quat(p.orientation)

    def _on_depth(self, msg):
        try:
            depth = _depth_metres(msg)
        except ValueError as exc:
            self.get_logger().warn("depth frame ignored: %s" % (exc,))
            return
        with self._lock:
            self._depth = depth
            self._depth_stamp = time.monotonic()

    def _control(self):
        with self._lock:
            state = self._state
            stamp = self._state_stamp
            flying = self._flying
            attitude = self._attitude
            path_xy = list(self._path_xy)
            epoch = self._path_epoch
            hold = self._hold
            yaw_goal = self._yaw_goal
            self._yaw_goal = None

        self._slew_offset()
        if self._check_capsized(attitude):
            return
        if state is None or (time.monotonic() - stamp) > self._odom_timeout_s:
            # No fresh feedback is not a reason to keep the last command: the
            # plugin has no failsafe and would fly the stale twist indefinitely.
            self._publish(zero_twist_fields())
            return
        if not flying:
            # `None` -- never heard /simple_drone/state -- counts as "not
            # flying". Every other guard in this file fails closed and this one
            # should too: commanding an aircraft whose flight state is unknown
            # is exactly the case the state topic exists to prevent.
            self._publish(zero_twist_fields())
            return

        if not self._altitude_ready(state):
            return

        # A ROTATION BEFORE ANYTHING ELSE, including the hold. A rotation is
        # already a stationary manoeuvre, so it does not violate what a hold is
        # for -- whereas letting the hold win would silently swallow the turn
        # the policy asked for and leave the aircraft looking the wrong way at
        # the frame it is about to be shown.
        if self._rotate(state, yaw_goal):
            return

        if hold:
            # The policy node is thinking. Stop, but keep flying the altitude:
            # a zero twist here is a descent, not a hover.
            self._publish(self._hold_altitude(state))
            return

        if len(path_xy) < 2:
            self._publish(self._hold_altitude(state))
            return

        if epoch != self._tracked_epoch:
            # A new commitment is a new polyline with the aircraft at its start.
            self._tracker.reset()
            self._tracked_epoch = epoch

        # Hold the configured cruise altitude: every reference point sits at it,
        # so the 3D pursuit commands vz toward it while it tracks xy.
        points = [(x, y, self._commanded_altitude()) for (x, y) in path_xy]
        try:
            trajectory = trajectory_from_points(points, self._cruise)
        except ValueError:
            self._publish(self._hold_altitude(state))
            return

        result = self._tracker.step(TrackerRequest(
            state=state, trajectory=trajectory, t=0.0, limits=self._limits))

        meta = result.metadata or {}
        if meta.get("failed"):
            # Re-acquire rather than latch. The tracker only ever searches
            # forward of its high-water mark, so a divergence it does not clear
            # is permanent; resetting lets the next cycle find the whole path.
            self._tracker.reset()
            self.get_logger().warn(
                "pursuit lost the route (%s, cross-track %.2f m); re-acquiring"
                % (meta.get("reason", "unknown"),
                   float(meta.get("cross_track_error", 0.0))))
            self._publish(self._hold_altitude(state))
            return
        if meta.get("done"):
            # The commitment is flown. Hold station on the spot until the policy
            # node anchors the next one -- a zero body twist would also drop the
            # altitude hold.
            self._publish(self._hold_altitude(state))
            return

        cmd = result.command  # world-frame vx, vy, vz + yaw_rate

        body = self._world_to_body(cmd.x, cmd.y, cmd.z, state.pose.yaw)
        vx, vy = self._braked_horizontal(body[0], body[1])
        fields = twist_fields(
            BodyTwistCommand(vx=vx, vy=vy, vz=body[2], yaw_rate=cmd.yaw_rate),
            self._body_limits)
        self._publish(fields)

    def _rotate(self, state, yaw_goal):
        """Fly a requested rotation in place; return True while one is running.

        The control law and the "is it finished" test are both
        :class:`~sparx_agency.core.planning.vlas.common.turn_in_place.TurnInPlace`,
        which the policy node also runs against the same odometry -- one
        implementation, two readers, so the follower and the runner can never
        disagree about whether the aircraft has turned.
        """
        now = time.monotonic()
        if yaw_goal is not None:
            same = (self._turn.active and self._turn.target is not None
                    and abs(normalize_angle(yaw_goal - self._turn.target))
                    <= self._turn.spec.tolerance_rad)
            if not same:
                if (self._escape_enabled and self._blocked
                        and self._escape.trigger(
                            StuckVerdict(axis="forward", sign=1), prefer_left=True)):
                    self._escaping = True
                    self.get_logger().warn(
                        "hard blocked and asked to turn: backing off %.2f m to "
                        "break contact first, then rotating"
                        % (self._escape.params.back_s * self._escape.params.back_speed,))
                self._turn.start_to(yaw_goal, now)
                self.get_logger().info(
                    "turning %.1f deg to heading %.1f deg"
                    % (degrees(normalize_angle(self._turn.target - state.pose.yaw)),
                       degrees(self._turn.target)))
        # The back-off owns the aircraft until it is done; the rotation follows.
        if self._escaping:
            escape = self._escape.step(1.0 / max(1e-3, self._control_rate))
            if escape.active:
                fields = twist_fields(
                    BodyTwistCommand(vx=escape.vx, vy=escape.vy,
                                     vz=self._alt_kp * (self._commanded_altitude()
                                                        - state.pose.z),
                                     yaw_rate=0.0),
                    self._body_limits)
                self._publish(fields)
                self.get_logger().info("escape: %s" % (escape.reason,),
                                       throttle_duration_sec=1.0)
                return True
            self._escaping = False
            self.get_logger().info("contact broken; rotating")

        if not self._turn.active:
            return False
        cmd = self._turn.update(state.pose.yaw, now, state.twist.yaw_rate)
        if cmd.done:
            if cmd.timed_out:
                self.get_logger().warn(
                    "rotation timed out %.1f deg short of its heading -- the "
                    "aircraft did not turn (blocked, or not flying)"
                    % (degrees(cmd.remaining_rad),))
            else:
                self.get_logger().info("rotation complete at %.1f deg"
                                       % (degrees(state.pose.yaw),))
            return False
        self._publish(self._hold_altitude(state, yaw_rate=cmd.yaw_rate))
        return True

    # ── the two reflexes ─────────────────────────────────────────────

    def _check_capsized(self, attitude):
        """Stop commanding a capsized aircraft, and say so once.

        Returns:
            ``True`` when the aircraft is capsized -- a zero twist has already
            been published and the caller must not command anything else.
        """
        roll, pitch = attitude
        capsized = max(abs(roll), abs(pitch)) > self._capsize_rad
        if capsized and not self._capsized:
            self.get_logger().error(
                "CAPSIZED: roll %.0f deg pitch %.0f deg. The SJTU plugin thrusts "
                "along body z, so it can no longer climb, translate or yaw -- "
                "every command from here is ignored while odom stays healthy. "
                "Reset the drone (/simple_drone/reset) before flying again."
                % (degrees(roll), degrees(pitch)))
        elif self._capsized and not capsized:
            self.get_logger().info("attitude recovered; commanding again")
        self._capsized = capsized
        if capsized:
            self._publish(zero_twist_fields())
        return capsized

    def _braked_horizontal(self, vx, vy):
        """Scale the horizontal PAIR down to what the depth corridor allows.

        Scaled together, not clamped independently. The tracker is holonomic, so
        a commanded velocity toward an obstacle is generally split across both
        axes; clamping only ``vx`` leaves the aircraft sliding sideways into the
        same wall it has just been told to stop for -- and the corridor is only
        as wide as the airframe, so the jamb it slides into was never in view.
        Scaling preserves the direction the tracker asked for and reduces the
        magnitude, which is the same discipline the platform's own speed clamp
        uses.

        Returns:
            The braked ``(vx, vy)`` body-frame pair.
        """
        if not self._brake_enabled or vx <= 0.0:
            return vx, vy
        with self._lock:
            depth = self._depth
            fresh = (time.monotonic() - self._depth_stamp) <= self._depth_timeout_s
        if depth is None or not fresh:
            # No usable depth is not permission to fly at cruise into a corridor
            # nobody is watching. Half speed, and say so at most once a second.
            self.get_logger().warn("no fresh depth; horizontal speed halved",
                                   throttle_duration_sec=5.0)
            return 0.5 * vx, 0.5 * vy
        allowed, d_min = self._brake.allowed_forward_speed(depth)
        self._report_blocked(allowed <= 1e-3)
        if allowed >= vx:
            return vx, vy
        scale = max(0.0, allowed) / max(vx, 1e-6)
        self.get_logger().info(
            "brake: %.2f -> %.2f m/s (nearest %s)"
            % (vx, max(0.0, allowed),
               "none" if d_min is None else "%.2f m" % d_min),
            throttle_duration_sec=2.0)
        return vx * scale, vy * scale

    def _report_blocked(self, blocked):
        """Publish a change in the hard-block state, edge-triggered."""
        if bool(blocked) == self._blocked:
            return
        self._blocked = bool(blocked)
        msg = Bool()
        msg.data = self._blocked
        self._blocked_pub.publish(msg)
        if self._blocked:
            self.get_logger().warn(
                "HARD BLOCKED: the depth corridor allows no forward speed at "
                "all. Nothing this node can do about it -- the policy has to "
                "look somewhere else.")
        else:
            # Genuinely moving again, so the next blockage is a new episode and
            # gets its own attempts. Without this the second wall of a flight
            # would be met with a manoeuvre budget the first one had spent.
            self._escape.episode_over()
            self.get_logger().info("forward path clear again")

    def _commanded_altitude(self):
        """Cruise altitude plus the offset currently being flown."""
        with self._lock:
            return self._target_alt + self._alt_offset

    def _slew_offset(self):
        """Move the flown offset one control step toward the requested one."""
        step = self._alt_offset_rate / max(1e-3, self._control_rate)
        with self._lock:
            delta = self._alt_offset_target - self._alt_offset
            if abs(delta) <= step:
                self._alt_offset = self._alt_offset_target
            else:
                self._alt_offset += step if delta > 0 else -step

    def _altitude_ready(self, state):
        """Climb to the cruise band, and refuse to track until inside it.

        Returns:
            ``True`` when the route may be tracked this cycle. When it returns
            ``False`` it has already published a climb command.
        """
        error = self._commanded_altitude() - state.pose.z
        if self._alt_captured:
            if abs(error) <= self._alt_release_m:
                return True
            self._alt_captured = False
            self.get_logger().warn(
                "altitude lost (%.2f m off %.2f m); climbing before tracking again"
                % (error, self._commanded_altitude()))
        elif abs(error) <= self._alt_capture_m:
            self._alt_captured = True
            self.get_logger().info("cruise altitude captured at %.2f m" % state.pose.z)
            return True
        self._publish(self._hold_altitude(state))
        return False

    def _hold_altitude(self, state, yaw_rate=0.0):
        """A twist that corrects altitude and nothing else but the given yaw rate.

        Args:
            state: current :class:`State3D`.
            yaw_rate: rad/s to rotate at while holding position, for the
                stop-and-turn manoeuvre. Zero for a plain hover.
        """
        vz = self._alt_kp * (self._commanded_altitude() - state.pose.z)
        return twist_fields(
            BodyTwistCommand(vx=0.0, vy=0.0, vz=vz, yaw_rate=float(yaw_rate)),
            self._body_limits)

    @staticmethod
    def _world_to_body(vx, vy, vz, yaw):
        """Rotate a world-frame velocity into the yaw-aligned body frame."""
        c, s = cos(yaw), sin(yaw)
        return (c * vx + s * vy, -s * vx + c * vy, vz)

    def _publish(self, fields):
        self._cmd_pub.publish(fill_twist(Twist(), fields))

    def halt(self):
        """Publish a zero twist, several times, on the way out.

        THE AIRCRAFT DOES NOT STOP BY ITSELF. The SJTU plugin latches the last
        `cmd_vel` it received and re-applies it on every physics update; there
        is no watchdog and no expiry anywhere in it. This node is the sole
        publisher on that topic, so a follower that exits without zeroing leaves
        the drone flying its final command until something else happens to
        publish -- which, in the run script, is fifteen seconds later.
        Repeated because a single best-effort sample at shutdown is not a
        guarantee.
        """
        for _ in range(5):
            try:
                self._publish(zero_twist_fields())
            except Exception:  # noqa: BLE001 - the context may already be down
                return
            time.sleep(0.05)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryFollowerNode()
    def _stop_and_go(_signum, _frame):
        # SIGTERM does not raise in Python, so without this the process dies
        # holding the last twist it published.
        node.halt()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop_and_go)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        # An orderly stop, not a fault. Letting any of these escape exits 1,
        # which the launch file reads as "this node died" and turns into an
        # ERROR and an emergency shutdown of its siblings -- so a clean Ctrl-C
        # produces a log that looks like a crash, and the one alarm that exists
        # for a node dying at import becomes noise.
        #
        # RCLError belongs here because rclpy does NOT always raise the tidy
        # ExternalShutdownException: a shutdown that lands between spin
        # iterations surfaces as `failed to initialize wait set: the given
        # context is not valid`, and one that lands inside a callback as
        # `failed to publish: publisher's context is invalid`. Both are the
        # same event.
        pass
    finally:
        node.halt()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


