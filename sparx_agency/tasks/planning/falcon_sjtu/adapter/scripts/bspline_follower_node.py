#!/usr/bin/env python3
"""Fly FALCON's B-spline on the SJTU Gazebo drone.

The one node between the planner and the aircraft. It replaces
``cmd_to_vel.py`` from the previous stack, and the difference is not a tidy-up:

* **It reads the curve, not the samples.** The old node subscribed to
  ``/planning/pos_cmd``, the 100 Hz point stream ``traj_server`` produces by
  sampling the B-spline. That stream carries position, velocity, acceleration
  and yaw -- but it arrives 15-35 ms stale, it cannot be evaluated at any other
  instant, and it carries no jerk. Subscribing to ``/planning/bspline`` instead
  gives the whole curve, evaluated on this node's own clock at whatever instant
  the controller asks for, exactly.
* **It knows what it is flying.** The Gazebo plugin closes its own velocity loop
  with a measured 0.18 s of transport delay and a 0.51 s time constant. The old
  node ignored that and paid roughly 0.4 m of standing position error for it at
  a 0.6 m/s cruise. ``core.control.velocity_servo`` inverts it instead.
* **It closes the heading loop.** The old node fed the plan's yaw rate forward
  and corrected the heading with a P term computed from a stale unstamped pose.
  Heading matters more than it sounds here: FALCON chooses yaw to aim the depth
  camera at the frontier it means to observe next, so a heading that has drifted
  is a map built of the wrong wall.

Everything above the ROS boundary lives in ``core``: this file subscribes,
converts, calls one ``update()`` and publishes. That is deliberate -- the
control law is unit-tested against a simulated airframe in
``core/control/velocity_servo/tests/`` and must never need Gazebo to be
exercised.

Runs inside the ROS1 Noetic FALCON container, on **Python 3.8**.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Empty, Float32MultiArray, Int32, String
from trajectory.msg import Bspline

from sparx_agency.core.control.velocity_servo import (
    AxisPlant, VelocityLimits, VelocityPlant, VelocityServo, VelocityServoParams,
)
from sparx_agency.core.planning.safety.depth_proximity_brake import (
    DepthProximityBrake, DepthProximityBrakeConfig,
)
from sparx_agency.core.planning.safety.voxel_brake_gate import (
    VoxelBrakeGate, VoxelBrakeGateConfig,
)
from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory

REPLAN_TRAJECTORY_UNSAFE = 1
"""FALCON found the trajectory it is flying in collision. Hold, do not continue.

From ``exploration_fsm.cpp``. FALCON replans and says nothing else about it, so
a follower that ignores this keeps flying a curve the planner has already
condemned -- straight at whatever it just found.
"""

REPLAN_EXPLORATION_FINISHED = 2
"""The FSM reached FINISH. No further trajectory is coming."""


def _yaw_from_quaternion(orientation):
    # type: (object) -> float
    """Heading about world +z, radians, from a ROS quaternion."""
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class BsplineFollowerNode(object):
    """Subscribe the plan and the state, call the servo, publish a twist."""

    def __init__(self):
        # type: () -> None
        self._rate_hz = float(rospy.get_param("~ctrl_rate_hz", 50.0))
        self._odom_timeout_s = float(rospy.get_param("~odom_timeout_s", 0.5))
        self._takeoff_altitude_m = float(rospy.get_param("~takeoff_altitude_m", 1.0))
        # TWO scans, one knob each, because they answer different questions.
        #
        # survey_revs is the RECOVERY scan (_begin_resurvey): a respawned
        # exploration_node comes back with an EMPTY map and never
        # re-triggers itself, and a fresh turn is what gives its frontier
        # finder something to fire on. That failure is real; keep it.
        #
        # takeoff_survey_revs is the scan at the START, and it is now 0
        # because it was measured to buy nothing: FALCON published its
        # first plan 1.9 s after takeoff WITH the turn and 1.8 s WITHOUT
        # it -- the forward view alone is enough to seed the frontier
        # finder. What the turn did cost was a first trajectory planned
        # from a SPINNING aircraft: FALCON starts the curves clock at
        # the plan while the yaw is still slewing, so the follower opens
        # every mission unwinding a heading it never asked for.
        self._survey_revs = float(rospy.get_param("~survey_revs", 1.25))
        self._survey_remaining_rad = 2.0 * math.pi * float(
            rospy.get_param("~takeoff_survey_revs", 0.0))
        self._survey_started_at = None  # type: object  # set on the first tick
        # The scan is a fallback, not a ritual: it exists so FALCON has
        # something to plan out of when the map is empty. A PLAN IS PROOF THAT
        # IT DID -- so the turn is abandoned the moment one arrives, with no
        # minimum. Turning on past that point costs punctuality for nothing:
        # FALCON's clock starts with the plan while the aircraft is still
        # spinning, so following begins seconds late against a schedule that
        # has already run away.
        #
        # min_survey_rad is kept as a knob rather than deleted, because the
        # opposite failure is real: cutting off a scan that has barely started
        # leaves the cells beside the aircraft unobserved, and FALCON will
        # happily route through them (measured once as a pile strike ~40 s
        # later). Raise it to 2*pi to demand a full revolution again.
        self._survey_yaw_rate = float(rospy.get_param("~survey_yaw_rate", 0.9))
        self._min_survey_rad = float(rospy.get_param("~min_survey_rad", 0.0))
        self._survey_last_yaw = None
        # Re-survey when FALCON falls silent for this long after it has been
        # planning -- the signature of an exploration_node respawn (its LKH
        # solver segfaults, roslaunch restarts it, and it comes back with an
        # empty map at wherever the aircraft now is). A fresh 360 deg scan is
        # what lets it find frontiers again instead of sitting in "No path".
        self._resurvey_after_s = float(rospy.get_param("~resurvey_after_s", 12.0))
        # Below the re-survey window but above a normal replan gap: hold instead
        # of carrying the last curve on into an obstacle while FALCON is down (its
        # in-process LKH solver corrupts the heap and roslaunch is mid-respawn).
        # A velocity-commanded aircraft flown a stale plan blind is exactly what
        # tips into the one wall the depth camera never saw. 6 s clears FALCON's
        # ~3 s replan cadence so a normal gap is not mistaken for a crash.
        self._hold_silent_s = float(rospy.get_param("~hold_silent_s", 6.0))
        # Exploration stall: airborne and not moving while FALCON churns
        # plan-fails on a frontier whose only viewpoint is the current cell -- it
        # selects "here", cannot plan a zero-length trajectory to it, and loops
        # ("next_pos ... same as current pos and yaw") forever. A crash used to
        # break this by chance (the respawn wipes the map); a fresh 360 deg scan
        # does it deliberately, clearing or re-routing the stuck frontier.
        self._stall_window_s = float(rospy.get_param("~stall_window_s", 15.0))
        # Net displacement, not path length: a stuck aircraft still jitters ~0.5 m
        # as FALCON's degenerate plans nudge it, so the bar is a real leg, not a
        # twitch. 1.5 m in 15 s is well below any genuine exploration move.
        self._stall_move_m = float(rospy.get_param("~stall_move_m", 1.5))
        self._stall_track = []          # type: list  # (t_s, x, y, z)
        self._resurveys = 0
        self._last_bspline_at = None    # type: object
        self._servo = VelocityServo(self._params())

        # ── wedge/contact reflex ─────────────────────────────────────────
        # A velocity-commanded aircraft pinned against a wall keeps being told
        # to drive into it and never moves -- the map may not carry the wall
        # (the depth camera goes blind inside its ~0.95 m near clip), so nothing
        # in the plan chain notices. Two triggers: ground-truth contact from the
        # Gazebo bumper, and an inferred wedge (commanding motion, net-zero
        # travel). Recovery backs out far enough to bring the wall back inside
        # the near clip, so FALCON re-maps it and plans around it next.
        self._wedge_window_s = float(rospy.get_param("~wedge_window_s", 3.0))
        self._wedge_move_m = float(rospy.get_param("~wedge_move_m", 0.2))
        self._wedge_cmd_speed = float(rospy.get_param("~wedge_cmd_speed", 0.08))
        self._contact_hold_s = float(rospy.get_param("~contact_hold_s", 0.4))
        self._retreat_time_s = float(rospy.get_param("~retreat_time_s", 5.0))
        self._retreat_speed = float(rospy.get_param("~retreat_speed", 0.35))
        # Vertical escape, used only when the horizontal back-out is itself
        # walled. max_z must stay UNDER the flight box ceiling (warehouse
        # box_max_z is 1.8) or the aircraft climbs out of the volume FALCON is
        # allowed to plan in and no frontier is reachable from up there.
        # 0 disables the climb and restores the horizontal-only behaviour.
        self._escape_climb_s = float(rospy.get_param("~escape_climb_s", 4.0))
        self._escape_climb_speed = float(
            rospy.get_param("~escape_climb_speed", 0.4))
        self._escape_climb_max_z = float(
            rospy.get_param("~escape_climb_max_z", 1.65))
        # ── proximity speed governor ─────────────────────────────────────
        # cap = max(floor, slope * (d_near - stop)). It was hardcoded
        # max(0.08, 0.7 * (d_near - 0.30)), and measured attribution showed why
        # that matters: the governor bound 43-71% of ticks near obstacles at
        # roughly HALF the planned speed. The reason is that its knee was set
        # independently of the planner's own clearance. FALCON is asked for
        # safe_distance (0.55 m) of margin; at exactly that distance the old
        # curve allowed 0.7*(0.55-0.30) = 0.175 m/s against a 0.25 m/s plan, so
        # the aircraft was throttled to 70% while flying EXACTLY where the
        # planner intended. A governor must bite when the aircraft is closer
        # than planned, not when it is where it was told to be.
        #
        # stop = the drone radius: the distance at which speed must be zero,
        # which is a property of the airframe. slope 1.2 then clears a 0.25 m/s
        # plan at 0.46 m and the 0.55 m planned clearance with margin, while
        # still collapsing to the floor by 0.32 m. Tighten stop, not slope, if
        # this ever needs to be more conservative.
        self._prox_stop_m = float(rospy.get_param("~prox_stop_m", 0.25))
        self._prox_slope = float(rospy.get_param("~prox_slope", 1.2))
        self._prox_floor = float(rospy.get_param("~prox_floor", 0.08))
        self._retreat_clear_m = float(rospy.get_param("~retreat_clear_m", 1.3))
        # Turn-to-look: after backing out, face the struck point and stare so
        # the mapper fuses it. Caps, not guarantees -- see _retreat.
        self._face_time_s = float(rospy.get_param("~face_time_s", 6.0))
        self._look_dwell_s = float(rospy.get_param("~look_dwell_s", 1.5))
        self._contact_point = None      # type: object  # (x, y) world, strike spot
        self._retreat_phase = None      # type: object  # back | face | dwell

        # ── map brake gate ───────────────────────────────────────────────
        # The last line of defence: refuse to fly the commanded velocity into
        # voxels FALCON's own map says are occupied. Exists because FALCON's
        # runtime safety check demonstrably does not signal this follower (its
        # pre-publish check swaps in a fallback curve silently, and its
        # executing-trajectory check fired 0 times across 336 contacts), so
        # the map knowing an obstacle does not stop the aircraft -- this does.
        self._gate_enabled = bool(rospy.get_param("~map_gate", True))
        self._gate = VoxelBrakeGate(VoxelBrakeGateConfig(
            drone_radius_m=float(rospy.get_param("~gate_drone_radius_m", 0.30)),
            z_band=(float(rospy.get_param("~gate_z_lo", 0.4)),
                    float(rospy.get_param("~gate_z_hi", 2.0))),
        ))
        # A hard stop held this long is a physical fact the planner is not
        # resolving: treat it exactly like a contact and back out for a fresh
        # look, which also un-sticks a start cell swallowed by inflation.
        # 4 s: strictly more than FALCON's ~3 s replan cadence, so a block
        # always gives the planner one full chance to reroute before the
        # follower spends ~10 s on a physical back-out.
        self._gate_block_retreat_s = float(
            rospy.get_param("~gate_block_retreat_s", 4.0))
        self._gate_blocked_since = None     # type: object
        # Creep-through escalation. A brake that only ever says NO deadlocks
        # against a planner that keeps re-issuing the same legal route: the
        # aircraft holds, FALCON replans identically, forever. After repeated
        # blocks in the SAME place the follower concedes the planner has the
        # better global picture and creeps through at a speed where a touch is
        # a nudge, keeping the contact reflex as the true last resort.
        self._creep_speed = float(rospy.get_param("~creep_speed", 0.12))
        self._creep_time_s = float(rospy.get_param("~creep_time_s", 20.0))
        # Distance from the contested spot at which the concession is over.
        # 2.0 m is past the 1.5 m radius _note_block_episode groups episodes
        # by, so leaving cancels the creep without immediately re-arming it.
        self._creep_clear_m = float(rospy.get_param("~creep_clear_m", 2.0))
        self._creep_from = None         # type: object
        self._creep_until = None            # type: object
        self._block_episodes = []           # type: list  # (t_s, x, y)

        # ── depth visual bumper ──────────────────────────────────────────
        # The reflex under the map gate: the voxel map flaps thin obstacles
        # (raycast clearing erodes one-voxel silhouettes -- every remaining
        # strike of run 009 was one person, visible in raw depth the whole
        # time), so forward speed is ALSO clamped by the closest raw depth
        # return inside the flight corridor. Intrinsics come from the same
        # rosparams the launch feeds FALCON's mapper.
        self._depth_brake_enabled = bool(rospy.get_param("~depth_brake", True))
        self._depth_brake = DepthProximityBrake(DepthProximityBrakeConfig(
            fx=float(rospy.get_param(
                "/uav_model/sensing_parameters/camera_intrinsics/fx", 390.6427)),
            fy=float(rospy.get_param(
                "/uav_model/sensing_parameters/camera_intrinsics/fy", 390.6427)),
            cx=float(rospy.get_param(
                "/uav_model/sensing_parameters/camera_intrinsics/cx", 300.5)),
            cy=float(rospy.get_param(
                "/uav_model/sensing_parameters/camera_intrinsics/cy", 300.5)),
            # These were core defaults sized for a stack that planned 0.15 m
            # from obstacles. They are a VETO, not a brake: below
            # hard_block_d_m the node zeroes horizontal drive outright, and
            # because margin_m + nose_offset_m is the stopping distance, the
            # allowed speed is already 0 below 0.80 m. At the old 1.05/0.70
            # the aircraft was vetoed by anything within ~1 m of the nose
            # while FALCON was deliberately routing it 0.55 m from obstacles
            # -- the planner and the bumper asking for incompatible things.
            # 0.55/0.30 keeps a real stopping margin at this airframe's
            # 0.8 m/s^2 and 0.30 s latency without vetoing a planned pass.
            hard_block_d_m=float(rospy.get_param("~depth_block_d_m", 0.55)),
            margin_m=float(rospy.get_param("~depth_margin_m", 0.30)),
        ))
        self._depth_allow = None        # type: object  # (v_allow, d_min)
        self._depth_allow_at = None     # type: object
        self._depth_history = []        # type: list  # (t_s, v_allow) trailing window

        # ── altitude guard band ──────────────────────────────────────────
        # Derived from the MAP's own planning box: a start pose outside the
        # box is a pose FALCON can never plan from, so the guard keeps the
        # aircraft strictly inside it in every state (see _send). Slightly
        # inside the faces so plans that ride the box limits do not fight it.
        box_lo = float(rospy.get_param("/map_config/map_size/box_min_z", 0.6))
        box_hi = float(rospy.get_param("/map_config/map_size/box_max_z", 2.0))
        self._alt_min = box_lo + 0.10
        self._alt_max = box_hi - 0.10
        # Horizontal box for the same guard, with a small tolerance ring so
        # normal wall-adjacent flight is untouched.
        try:
            self._box_xy = (
                float(rospy.get_param("/map_config/map_size/box_min_x")) - 0.20,
                float(rospy.get_param("/map_config/map_size/box_max_x")) + 0.20,
                float(rospy.get_param("/map_config/map_size/box_min_y")) - 0.20,
                float(rospy.get_param("/map_config/map_size/box_max_y")) + 0.20)
        except KeyError:
            self._box_xy = None
        self._last_z = None             # type: object
        self._hold_point = None         # type: object  # (x,y,z) station latch
        # Takeoff must END inside the planning box or the first plan request
        # starts from an illegal altitude (maps may raise the floor above the
        # default takeoff height, e.g. the hospital's counter-clearing 1.25).
        self._takeoff_altitude_m = max(self._takeoff_altitude_m,
                                       self._alt_min + 0.05)
        self._takeoff_climb_speed = float(
            rospy.get_param("~takeoff_climb_speed", 0.4))
        # Attitude reflex: cut horizontal drive if roll or pitch crosses this, a
        # margin below the plugin's ~35 deg clamp past which the model thrusts
        # sideways and cannot recover. A near-clip contact can pitch the aircraft
        # up as it drives into the unseen obstacle; stopping the drive lets it
        # fall back level before it tips over.
        self._tilt_limit_rad = math.radians(
            float(rospy.get_param("~tilt_limit_deg", 22.0)))
        self._track = []                # type: list  # (t_s, x, y, z) history
        self._contact_seen_at = None    # type: object
        self._retreat_until = None      # type: object
        self._retreat_from = None       # type: object  # world pos at retreat start
        self._retreat_dir_world = None  # type: object
        self._retreats = 0

        self._odom = None               # type: object
        self._odom_at = None            # type: object
        self._airborne = False
        self._stopped = False
        self._finished = False

        self._cmd = rospy.Publisher(
            rospy.get_param("~cmd_topic", "/simple_drone/cmd_vel"), Twist, queue_size=1)
        self._takeoff = rospy.Publisher("/simple_drone/takeoff", Empty, queue_size=1)
        self._diagnostics = rospy.Publisher("~tracking", Float32MultiArray, queue_size=10)
        self._flying = rospy.Publisher("~following", Bool, queue_size=1, latch=True)
        # Live "who is limiting me right now", one word per tick.
        self._limiter_pub = rospy.Publisher("~limiter", String, queue_size=1)
        self._limiter_ticks = {}
        self._limiter_ratio = {}
        self._limiter_ratio_n = {}
        self._limiter_last_report = None
        self._limiter_report_s = float(
            rospy.get_param("~limiter_report_s", 15.0))

        rospy.Subscriber(rospy.get_param("~odom_topic", "/simple_drone/odom"),
                         Odometry, self._on_odom, queue_size=4)
        rospy.Subscriber("/planning/bspline", Bspline, self._on_bspline, queue_size=4)
        rospy.Subscriber("/planning/replan", Int32, self._on_replan, queue_size=10)
        rospy.Subscriber(rospy.get_param("~contact_topic", "/simple_drone/bumper_states"),
                         ContactsState, self._on_bumper, queue_size=4)
        if self._gate_enabled:
            # At 0.1 m resolution FALCON's publisher emits complete occupied
            # sweeps on this topic (its local-box variant only runs at
            # coarser resolutions), so every message REPLACES the gate's
            # world -- no free-cloud subscription, no ghost accumulation.
            # buff_size must exceed the biggest sweep: rospy's 64 KiB default
            # silently mangles multi-MB messages, which is how run 013's
            # gate lost every free update and livelocked on phantoms.
            rospy.Subscriber("/voxel_mapping/occupancy_grid_occupied",
                             PointCloud2, self._on_occupied_cloud,
                             queue_size=1, buff_size=2 ** 26)
        if self._depth_brake_enabled:
            rospy.Subscriber(rospy.get_param("~depth_topic",
                                             "/simple_drone/front_depth/depth/image_raw"),
                             Image, self._on_depth, queue_size=1, buff_size=2 ** 23)

        self._last_tick = None          # type: object
        rospy.Timer(rospy.Duration(1.0 / self._rate_hz), self._tick)
        rospy.loginfo("[follower] %.0f Hz, plant tau=%.3f s delay=%.3f s",
                      self._rate_hz,
                      self._servo.params.plant.horizontal.time_constant_s,
                      self._servo.params.plant.horizontal.delay_s)

    def _params(self):
        # type: () -> VelocityServoParams
        """Build the servo tuning from the launch parameters.

        The plant numbers are **measured**, not guessed -- see
        ``robots/SJTU/config/airframe.yaml`` for how, and re-measure them if the
        simulator's real-time factor or the plugin's gains change. A wrong time
        constant makes the lead term push the wrong amount and looks exactly
        like a mistuned position gain.
        """
        plant = VelocityPlant(
            horizontal=AxisPlant(
                dc_gain=float(rospy.get_param("~plant_xy_gain", 1.0)),
                time_constant_s=float(rospy.get_param("~plant_xy_tau", 0.51)),
                delay_s=float(rospy.get_param("~plant_xy_delay", 0.18))),
            vertical=AxisPlant(
                dc_gain=float(rospy.get_param("~plant_z_gain", 1.0)),
                time_constant_s=float(rospy.get_param("~plant_z_tau", 0.41)),
                delay_s=float(rospy.get_param("~plant_z_delay", 0.04))),
            yaw=AxisPlant(
                dc_gain=float(rospy.get_param("~plant_yaw_gain", 1.0)),
                time_constant_s=float(rospy.get_param("~plant_yaw_tau", 0.48)),
                delay_s=float(rospy.get_param("~plant_yaw_delay", 0.06))))
        limits = VelocityLimits(
            max_speed_xy=float(rospy.get_param("~max_speed_xy", 1.2)),
            max_speed_up=float(rospy.get_param("~max_speed_up", 0.8)),
            max_speed_down=float(rospy.get_param("~max_speed_down", 0.6)),
            max_accel_xy=float(rospy.get_param("~max_accel_xy", 2.0)),
            max_accel_z=float(rospy.get_param("~max_accel_z", 2.0)),
            max_yaw_rate=float(rospy.get_param("~max_yaw_rate", 1.4)),
            max_yaw_accel=float(rospy.get_param("~max_yaw_accel", 3.0)))
        # Vertical position loop. core defaults it to kp 1.8, barely above
        # the horizontal 1.2 -- but the two axes are not comparable. The
        # bound each gain sits under is 1 / (3 * delay): horizontal delay
        # 0.18 s gives 1.85, so 1.2 is close to its ceiling, while the
        # vertical axis answers in 0.04 s (thrust changes without waiting
        # for the airframe to rotate) and its bound is 8.3. At 1.8 the
        # climb loop was running at a FIFTH of the stiffness its own rule
        # allows, which is why altitude changes lag the reference while
        # the horizontal axes hold it. 3.6 doubles the authority and is
        # still less than half the delay bound.
        vertical_pid = dataclasses.replace(
            VelocityServoParams().vertical_pid,
            kp=float(rospy.get_param("~vertical_kp", 3.6)),
            ki=float(rospy.get_param("~vertical_ki", 0.4)))
        return VelocityServoParams(
            plant=plant, limits=limits, vertical_pid=vertical_pid,
            yaw_gain=float(rospy.get_param("~yaw_gain", 1.5)),
            use_feedforward_lead=bool(rospy.get_param("~use_feedforward_lead", True)),
            predict_reference=bool(rospy.get_param("~predict_reference", True)),
            max_overspeed=float(rospy.get_param("~max_overspeed", 0.25)),
            max_catchup_speed=float(rospy.get_param("~max_catchup_speed", 0.35)))

    # ── inputs ───────────────────────────────────────────────────────────

    def _on_odom(self, msg):
        # type: (Odometry) -> None
        """Latch the state. This is the only feedback the loop has."""
        self._odom = msg
        self._odom_at = rospy.Time.now()
        self._last_z = float(msg.pose.pose.position.z)

    def _on_bspline(self, msg):
        # type: (Bspline) -> None
        """Rebuild FALCON's curve and queue it.

        Rebuilt with FALCON's own construction rules so this node and
        ``traj_server`` evaluate the same polynomial. A disagreement between
        them would appear as a tracking error with no visible cause.

        The order is asserted rather than trusted: the publisher hardcodes 3
        while the real degree comes from a parameter, so a mismatch is a silent
        wrong-curve failure and this repo raises rather than guessing.
        """
        self._last_bspline_at = rospy.Time.now()
        if msg.order != 3:
            raise ValueError(
                "FALCON published a degree-%d position spline; this follower "
                "rebuilds degree 3, as traj_server does. Refusing to fly a "
                "curve it would evaluate incorrectly." % (msg.order,))
        trajectory = BsplineTrajectory.from_falcon(
            order=msg.order,
            knots=list(msg.knots),
            position_points=[(p.x, p.y, p.z) for p in msg.pos_pts],
            yaw_points=list(msg.yaw_pts),
            yaw_dt=msg.yaw_dt,
            start_time_s=msg.start_time.to_sec(),
            traj_id=msg.traj_id)
        if self._servo.set_trajectory(trajectory):
            # A new plan supersedes a stop: FALCON only replans after
            # condemning what it was flying, so the arrival of a curve is its
            # statement that it has found a way out.
            self._stopped = False

    def _on_replan(self, msg):
        # type: (Int32) -> None
        """React to FALCON's own verdict on the trajectory it is flying.

        Treated as a control input rather than telemetry. ``1`` means
        ``safetyCallback`` found the *executing* trajectory in collision, and a
        follower that keeps flying it carries the aircraft's momentum into the
        obstacle FALCON just found.
        """
        if msg.data == REPLAN_TRAJECTORY_UNSAFE:
            self._stopped = True
            rospy.logwarn("[follower] FALCON condemned the live trajectory; holding")
        elif msg.data == REPLAN_EXPLORATION_FINISHED:
            self._finished = True
            rospy.loginfo("[follower] exploration finished; holding station")

    # ── the loop ─────────────────────────────────────────────────────────

    def _note_limiter(self, name, ratio=None):
        # type: (str, object) -> None
        """Record which mechanism owned this tick, and what it cost.

        Five limiters can each slow or stop this aircraft, they are checked in
        sequence, and every one of them was added for a different incident.
        With no attribution, "the drone barely moves near obstacles" is
        unfalsifiable -- every mechanism looks equally guilty from the outside,
        and the log only ever shows the loudest one (a retreat) rather than the
        one that actually binds most of the time. This records exactly ONE
        binding limiter per tick plus the fraction of the planned speed that
        survived it, so a four-minute run yields a histogram that names the
        offender instead of a hunch.
        """
        self._limiter_ticks[name] = self._limiter_ticks.get(name, 0) + 1
        if ratio is not None:
            self._limiter_ratio[name] = (self._limiter_ratio.get(name, 0.0)
                                         + float(ratio))
            self._limiter_ratio_n[name] = self._limiter_ratio_n.get(name, 0) + 1
        if self._limiter_pub is not None:
            self._limiter_pub.publish(String(data=name))

        now_s = rospy.Time.now().to_sec()
        if self._limiter_report_s <= 0.0:
            return
        if self._limiter_last_report is None:
            self._limiter_last_report = now_s
            return
        if now_s - self._limiter_last_report < self._limiter_report_s:
            return
        self._limiter_last_report = now_s
        total = sum(self._limiter_ticks.values()) or 1
        parts = []
        for key in sorted(self._limiter_ticks,
                          key=lambda k: -self._limiter_ticks[k]):
            share = 100.0 * self._limiter_ticks[key] / total
            if share < 0.5:
                continue
            n = self._limiter_ratio_n.get(key, 0)
            if n:
                parts.append("%s %.0f%% (x%.2f)"
                             % (key, share, self._limiter_ratio[key] / n))
            else:
                parts.append("%s %.0f%%" % (key, share))
        rospy.loginfo("[follower] limiter share over %.0fs: %s",
                      self._limiter_report_s, ", ".join(parts))
        self._limiter_ticks = {}
        self._limiter_ratio = {}
        self._limiter_ratio_n = {}

    def _tick(self, _event):
        # type: (object) -> None
        """One control tick. Never raises into the timer thread."""
        now = rospy.Time.now()
        dt = self._elapsed(now)
        if dt is None:
            return
        if not self._state_is_fresh(now):
            # No state means no loop. Publishing the previous command would fly
            # the aircraft blind; publishing zero at least stops it.
            self._send(Twist())
            self._servo.reset()
            return

        position, velocity, yaw = self._state()
        if not self._airborne:
            self._climb(position)
            return

        # Attitude reflex, ahead of everything below (even the retreat, which
        # drives): a near-clip contact can tip the aircraft, and past the plugin's
        # ~35 deg clamp it cannot recover. If roll or pitch crosses the margin,
        # cut horizontal drive and hold so it settles back level rather than over.
        roll, pitch = self._roll_pitch()
        if abs(roll) > self._tilt_limit_rad or abs(pitch) > self._tilt_limit_rad:
            self._send(Twist())
            self._servo.reset()
            self._retreat_until = None
            self._retreat_phase = None
            self._contact_point = None
            self._gate_blocked_since = None
            rospy.logwarn_throttle(
                1.0, "[follower] tilt roll=%.0f pitch=%.0f deg; cutting drive to "
                "level before it capsizes", math.degrees(roll), math.degrees(pitch))
            return

        # Re-survey only after a crash: a respawned exploration_node comes back
        # with an EMPTY map and never re-triggers itself, and a fresh 360 deg
        # scan is what gives its frontier finder something to fire on. Position
        # stalls at an unreachable frontier are NOT handled here any more -- FALCON
        # itself now shadows those viewpoints (the dead-end guard in
        # exploration_manager), and yawing in place on every stall only thrashed
        # against that 20 s block and re-scanned a view that never changes.
        # Mission over: hold POSITION, not zero velocity. FALCON is silent
        # forever now, so without this the resurvey below would spin the
        # aircraft every 12 s for the rest of time, and a zero-velocity hold
        # open-loop drifts horizontally (the vertical twin of the drift the
        # _send guard exists for).
        if self._finished:
            self._note_limiter("finished")
            self._hold_station(position, yaw)
            return

        silent_s = self._falcon_silent_s(now)
        if (self._survey_remaining_rad <= 0.0 and self._retreat_until is None
                and silent_s > self._resurvey_after_s):
            self._begin_resurvey(now, "silent %.0fs (respawn?)" % silent_s)

        if self._survey_remaining_rad > 0.0:
            # The scan exists for ONE reason: FALCON cannot plan out of a cell
            # it has never observed. The moment it publishes a plan, that
            # reason is spent -- and turning on past it is actively harmful,
            # because FALCON's clock starts with the plan while the aircraft
            # is still spinning, so following begins seconds late against a
            # schedule that has already run away (measured: the reference
            # pinned to a curve's endpoint before the aircraft ever moved).
            turned = 2.0 * math.pi * self._survey_revs - self._survey_remaining_rad
            if (turned >= self._min_survey_rad
                    and self._last_bspline_at is not None
                    and self._survey_started_at is not None
                    and self._last_bspline_at > self._survey_started_at):
                self._survey_remaining_rad = 0.0
                self._survey_last_yaw = None
                self._send(Twist())
                self._servo.reset()
                rospy.loginfo("[follower] plan arrived mid-scan; cutting the "
                              "turn short and following it now")
            else:
                self._note_limiter("survey")
                self._survey(yaw)
                return

        # Contact recovery owns the aircraft while it is backing out of a wall:
        # the plan that put it there has been dropped, so nothing else may drive.
        if self._retreat_until is not None:
            self._note_limiter("retreat:%s" % (self._retreat_phase or "?"))
            self._retreat(now, position, yaw)
            return

        # FALCON briefly silent (crashed mid-plan, before the re-survey window):
        # hold STATION rather than carry the stale curve on blind into the wall
        # it could not see. Retreat above takes priority -- backing out of a real
        # contact must not be interrupted by a hold.
        if silent_s > self._hold_silent_s:
            self._note_limiter("falcon_silent")
            self._hold_station(position, yaw)
            self._servo.reset()
            return
        self._hold_point = None            # any other state releases the latch

        # Attribution for this tick: the planned speed before any limiter
        # touches it, and the worst single factor applied to it.
        binding = "following"
        worst = 1.0
        speed0 = None
        follow = not (self._stopped or self._finished)
        command = self._servo.update(position, velocity, yaw, dt, now.to_sec(),
                                     follow=follow)

        # Back out on a real contact or an inferred wedge. A wall inside the
        # depth near clip (~0.95 m) is invisible to FALCON, which then plans
        # straight through it, so the follower is the only thing that can notice.
        if follow and not command.holding and self._should_retreat(now, position, command):
            self._begin_retreat(position, command)
            return

        # Two brakes between the servo and the wire, sharing one escalation
        # timer: the DEPTH bumper clamps the forward axis on what the camera
        # sees right now (thin obstacles the voxel map flaps), and the MAP
        # gate vetoes any commanded direction into accumulated occupancy.
        # A sustained dead stop from either is a physical fact the planner is
        # not resolving: treat it exactly like a contact -- back out and look.
        if follow and not command.holding:
            # Personal-space bubble: closure from ANY bearing, not just the
            # motion direction -- wall-sliding grinds are invisible to both
            # directional brakes. A breach is a near-contact: retreat now.
            creeping = (self._creep_until is not None and now < self._creep_until)
            # Creep is a concession to ONE contested spot, not a cruise mode.
            # It was released on a 20 s timer alone, so once triggered the
            # aircraft crawled at creep_speed everywhere it flew next --
            # measured at 48-100% of ticks at 0.22x the planned speed, the
            # single largest cause of "it can barely move". Release it as soon
            # as the aircraft has actually left the spot; the timer stays as
            # the backstop for a concession that never gets anywhere.
            if creeping and self._creep_from is not None:
                gone = math.hypot(float(position[0]) - self._creep_from[0],
                                  float(position[1]) - self._creep_from[1])
                if gone > self._creep_clear_m:
                    self._creep_until = None
                    self._creep_from = None
                    creeping = False
                    rospy.loginfo("[follower] %.1f m clear of the contested "
                                  "spot; back to plan speed", gone)
            if (not creeping and self._gate_enabled and self._gate.bubble_blocked(
                    (float(position[0]), float(position[1]), float(position[2])),
                    float(rospy.get_param("~bubble_clearance_m", 0.28)))):
                self._note_limiter("bubble_breach")
                rospy.logwarn("[follower] bubble breach: occupied voxel inside "
                              "personal space; retreating")
                self._begin_retreat(position, command)
                return

            hard_blocked = False
            block_why = ""
            speed0 = math.hypot(command.vx, command.vy)

            if self._depth_brake_enabled and command.vx > 1e-3:
                v_allow = self._depth_forward_limit(now)
                if v_allow <= 0.05:
                    hard_blocked = True
                    d = self._depth_allow[1] if self._depth_allow else -1.0
                    block_why = "depth %.2f m in corridor" % (d if d else -1.0)
                elif command.vx > v_allow:
                    # one factor on every component, body AND world: the gate
                    # downstream sweeps its corridor from the world vector, so
                    # a body/world mismatch mis-aims it and understates speed
                    f = v_allow / command.vx
                    if f < worst:
                        worst, binding = f, "depth_brake"
                    command = dataclasses.replace(
                        command, vx=command.vx * f, vy=command.vy * f,
                        world_vx=command.world_vx * f,
                        world_vy=command.world_vy * f)

            if self._gate_enabled and not hard_blocked:
                scale, blocked = self._gate.command_scale(
                    (float(position[0]), float(position[1]), float(position[2])),
                    (command.world_vx, command.world_vy))
                if scale <= 0.0:
                    hard_blocked = True
                    block_why = "map voxel %.2f m ahead" % (
                        blocked if blocked is not None else -1.0)
                elif scale < 1.0:
                    if scale < worst:
                        worst, binding = scale, "map_gate"
                    # BodyTwistCommand is frozen: build the braked copy.
                    command = dataclasses.replace(
                        command,
                        vx=command.vx * scale, vy=command.vy * scale,
                        world_vx=command.world_vx * scale,
                        world_vy=command.world_vy * scale)

            # Proximity speed governor: room at ANY bearing bounds speed.
            # Each directional brake has a blind arc (nose-only depth,
            # commanded-direction corridor); a cruise-speed strike proved the
            # arcs can lose a race. Near anything mapped, be slow near it.
            if self._gate_enabled and not hard_blocked:
                d_near = self._gate.nearest_occupied(
                    (float(position[0]), float(position[1]), float(position[2])),
                    1.2)
                if d_near is not None:
                    cap = max(self._prox_floor,
                              self._prox_slope * (d_near - self._prox_stop_m))
                    speed = math.hypot(command.vx, command.vy)
                    if speed > cap:
                        f = cap / speed
                        if f < worst:
                            worst, binding = f, "proximity_cap"
                        command = dataclasses.replace(
                            command, vx=command.vx * f, vy=command.vy * f,
                            world_vx=command.world_vx * f,
                            world_vy=command.world_vy * f)

            if hard_blocked and creeping:
                speed = math.hypot(command.vx, command.vy)
                if speed > self._creep_speed:
                    f = self._creep_speed / speed
                    # Label it: creep silently produced ratios as low as
                    # 0.39 that the histogram was crediting to "following".
                    if f < worst:
                        worst, binding = f, "creep"
                    command = dataclasses.replace(
                        command, vx=command.vx * f, vy=command.vy * f,
                        world_vx=command.world_vx * f,
                        world_vy=command.world_vy * f)
                hard_blocked = False

            if hard_blocked:
                if self._gate_blocked_since is None:
                    self._gate_blocked_since = now
                    self._note_block_episode(now, position)
                    rospy.logwarn("[follower] brake: %s; holding", block_why)
                elif (now - self._gate_blocked_since).to_sec() > self._gate_block_retreat_s:
                    self._gate_blocked_since = None
                    self._begin_retreat(position, command)
                    return
                self._note_limiter(
                    "hard_block:%s" % ("depth" if block_why.startswith("depth")
                                       else "map_gate"), 0.0)
                stop = Twist()
                stop.linear.z = command.vz          # altitude hold stays live
                stop.angular.z = command.yaw_rate   # keep looking along the path
                self._send(stop)
                return
            self._gate_blocked_since = None

        if follow:
            # speed0 is None on a tick that never entered the limiter chain
            # (command.holding): nothing throttled it, so the ratio is 1.
            if speed0 is None:
                self._note_limiter("holding", 1.0)
            else:
                speed1 = math.hypot(command.vx, command.vy)
                ratio = (speed1 / speed0) if speed0 > 1e-6 else 1.0
                self._note_limiter(binding, ratio)
        self._publish(command)

    def _elapsed(self, now):
        # type: (object) -> object
        """Seconds since the previous tick, or None on the first one.

        Measured rather than assumed. The timer's nominal period is not what
        actually elapses once the simulator's real-time factor drops below one,
        and a control law integrating a fictitious dt learns a fictitious bias.
        """
        if self._last_tick is None:
            self._last_tick = now
            return None
        dt = (now - self._last_tick).to_sec()
        self._last_tick = now
        if dt <= 0.0:
            return None
        return dt

    def _state_is_fresh(self, now):
        # type: (object) -> bool
        """Whether the odometry is recent enough to close a loop on."""
        if self._odom is None or self._odom_at is None:
            return False
        return (now - self._odom_at).to_sec() <= self._odom_timeout_s

    def _state(self):
        # type: () -> tuple
        """Measured world position, world velocity and heading.

        The twist on ``/simple_drone/odom`` is expressed in the child frame, so
        it is rotated into the world here. The plugin's ``/simple_drone/gt_vel``
        looks like a shortcut and is not: it applies the rotation the wrong way
        round, which puts a yaw-dependent axis swap into any loop that uses it.
        """
        pose = self._odom.pose.pose
        twist = self._odom.twist.twist
        yaw = _yaw_from_quaternion(pose.orientation)
        cos, sin = math.cos(yaw), math.sin(yaw)
        world = np.array([cos * twist.linear.x - sin * twist.linear.y,
                          sin * twist.linear.x + cos * twist.linear.y,
                          twist.linear.z], dtype=float)
        position = np.array([pose.position.x, pose.position.y, pose.position.z],
                            dtype=float)
        return position, world, yaw

    def _roll_pitch(self):
        # type: () -> tuple
        """Roll and pitch, radians, from the latched odometry orientation."""
        o = self._odom.pose.pose.orientation
        sinr = 2.0 * (o.w * o.x + o.y * o.z)
        cosr = 1.0 - 2.0 * (o.x * o.x + o.y * o.y)
        roll = math.atan2(sinr, cosr)
        sinp = max(-1.0, min(1.0, 2.0 * (o.w * o.y - o.z * o.x)))
        pitch = math.asin(sinp)
        return roll, pitch

    def _survey(self, yaw):
        # type: (float) -> None
        """One full turn on the spot before FALCON is allowed to matter.

        The depth camera sees a 75 degree wedge. An aircraft that has only
        ever pointed one way hands FALCON a map that is one wedge of free
        space; its coverage tour then picks a viewpoint far outside it,
        A* cannot route through the unknown, and the FSM sits in PLAN_TRAJ
        reporting "No path to next viewpoint" forever -- measured on the
        Pegasus deployment before its survey turn existed, and the previous
        SJTU stack carried the same scan (mapping_scan_revs) for the same
        reason. 1.25 revolutions rather than 1.0 so the wedge edges overlap.

        Yaw progress is integrated from the measured heading, wrap-aware,
        rather than assumed from the commanded rate: the plugin's yaw loop
        is not instant and a timed turn undershoots.
        """
        if self._survey_last_yaw is not None:
            step = math.atan2(math.sin(yaw - self._survey_last_yaw),
                              math.cos(yaw - self._survey_last_yaw))
            self._survey_remaining_rad -= abs(step)
        self._survey_last_yaw = yaw
        command = Twist()
        if self._survey_remaining_rad <= 0.0:
            self._send(command)          # stop the turn cleanly
            # Start the stall clock fresh: the scan just built new map, so give
            # FALCON a full window to act on it before calling it stuck again --
            # otherwise the in-place turn itself reads as a stall and loops.
            self._stall_track = []
            # Restamp the silence clock at turn COMPLETION, not just at turn
            # start: the 1.25-rev turn takes ~16 s, longer than the 12 s
            # resurvey window, so stamping only at start meant FALCON got
            # ZERO post-scan planning time and the node spun back-to-back.
            self._last_bspline_at = rospy.Time.now()
            rospy.loginfo("[bspline_follower] survey turn complete -- "
                          "following FALCON")
            return
        command.angular.z = self._survey_yaw_rate
        self._send(command)

    def _note_block_episode(self, now, position):
        # type: (object, np.ndarray) -> None
        """Record a block; concede to the planner if this spot keeps repeating.

        Two blocks within 1.5 m inside 90 s means the disagreement is
        structural -- our clearance model versus the planner's -- not a
        transient. FALCON sees the whole map and keeps choosing this route, so
        the follower crosses it slowly instead of holding until the mission
        cap (the observed plan-hold-replan livelock).
        """
        now_s = now.to_sec()
        x, y = float(position[0]), float(position[1])
        self._block_episodes = [e for e in self._block_episodes
                                if now_s - e[0] < 90.0]
        repeats = sum(1 for e in self._block_episodes
                      if math.hypot(x - e[1], y - e[2]) < 1.5)
        self._block_episodes.append((now_s, x, y))
        if repeats >= 1:
            self._creep_until = now + rospy.Duration(self._creep_time_s)
            self._creep_from = (x, y)
            rospy.logwarn("[follower] blocked repeatedly at (%.1f, %.1f); "
                          "conceding to the planner, creeping at %.2f m/s "
                          "for %.0fs", x, y, self._creep_speed, self._creep_time_s)

    def _falcon_silent_s(self, now):
        # type: (object) -> float
        """Seconds since FALCON last published a plan; 0 before the first one.

        Zero until the first curve arrives so the startup climb and survey own
        that phase -- there is no plan to be stale yet, and a large value there
        would trigger a hold or a re-survey before exploration has even begun.
        """
        if self._last_bspline_at is None:
            return 0.0
        return (now - self._last_bspline_at).to_sec()

    def _track_and_check_stall(self, now, position):
        # type: (object, np.ndarray) -> bool
        """Whether the aircraft has gone nowhere for the whole stall window.

        Net displacement start-to-now, so a drone yawing on the spot at a
        viewpoint it cannot leave reads as stalled while one genuinely creeping
        along a corridor does not. Returns False until the window is full.
        """
        now_s = now.to_sec()
        self._stall_track.append((now_s, float(position[0]), float(position[1]),
                                  float(position[2])))
        cutoff = now_s - self._stall_window_s
        while len(self._stall_track) > 1 and self._stall_track[0][0] < cutoff:
            self._stall_track.pop(0)
        if self._stall_track[0][0] > cutoff:
            return False
        t0, x0, y0, z0 = self._stall_track[0]
        moved = math.sqrt((position[0] - x0) ** 2 + (position[1] - y0) ** 2
                          + (position[2] - z0) ** 2)
        return moved < self._stall_move_m

    def _begin_resurvey(self, now, reason):
        # type: (object, str) -> None
        """Restart the survey turn to rebuild the map and break a dead end.

        ``_last_bspline_at`` is pushed to ``now`` and the stall history cleared
        so the turn is armed once per dead end rather than every tick; the next
        real plan and the motion it produces reset both when exploration resumes.
        """
        self._survey_remaining_rad = 2.0 * math.pi * self._survey_revs
        self._survey_last_yaw = None
        self._survey_started_at = now
        self._last_bspline_at = now
        self._stall_track = []
        self._hold_point = None        # the next hold latches where IT begins
        self._resurveys += 1
        rospy.logwarn("[follower] FALCON %s (recovery #%d); re-surveying to "
                      "rebuild its map", reason, self._resurveys)

    def _climb(self, position):
        # type: (np.ndarray) -> None
        """Get off the ground before handing over to the servo.

        The servo's integrators are reset at the handover: a bias learned while
        the aircraft was on its skids belongs to a different flight regime, and
        carrying it across puts a step into the first command of the new one.
        """
        if position[2] >= self._takeoff_altitude_m:
            self._airborne = True
            self._survey_started_at = rospy.Time.now()
            self._servo.reset()
            self._flying.publish(Bool(data=True))
            rospy.loginfo("[follower] airborne at %.2f m; following the plan",
                          position[2])
            return
        # The trigger only switches the plugin into flying mode; its own climb
        # levels off around 1 m and holds. Any map whose flight band starts
        # higher (the warehouse floor is 1.55 m) would strand the aircraft
        # below the handover altitude FOREVER -- the mission simply never
        # starts. So fly the climb ourselves once the motors are running.
        self._takeoff.publish(Empty())
        twist = Twist()
        twist.linear.z = self._takeoff_climb_speed
        self._send(twist)

    @staticmethod
    def _cloud_xyz(msg):
        # type: (PointCloud2) -> np.ndarray
        """View a PointCloud2 with leading float32 x,y,z as an Nx3 array."""
        n = msg.width * msg.height
        if n == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.ndarray(shape=(n, 3), dtype="<f4", buffer=msg.data,
                          strides=(msg.point_step, 4))

    def _on_depth(self, msg):
        # type: (Image) -> None
        """Clamp forward speed by the closest return in the flight corridor."""
        if msg.encoding not in ("32FC1",):
            return                      # never guess a depth format
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.width)
        self._depth_allow = self._depth_brake.allowed_forward_speed(depth)
        self._depth_allow_at = rospy.Time.now()
        self._depth_history.append((self._depth_allow_at.to_sec(),
                                    self._depth_allow[0]))

    def _depth_forward_limit(self, now):
        # type: (object) -> float
        """Current forward speed ceiling from the visual bumper, inf if none.

        The TRAILING MINIMUM over a short window, not the latest frame: when
        an obstacle penetrates the near clip its pixels go invalid and the
        instantaneous corridor minimum jumps to the background -- the brake
        would release at exactly the closest range. Holding the window's
        minimum keeps the clamp on until the aircraft has actually backed off.
        A wholly stale stream (camera or bridge hiccup) lifts the clamp
        rather than grounding the mission -- the voxel gate stays underneath.
        """
        if (self._depth_allow is None or self._depth_allow_at is None
                or (now - self._depth_allow_at).to_sec() > 1.0):
            return float("inf")
        now_s = now.to_sec()
        while self._depth_history and self._depth_history[0][0] < now_s - 1.5:
            self._depth_history.pop(0)
        if not self._depth_history:
            return self._depth_allow[0]
        return min(v for _, v in self._depth_history)

    def _on_occupied_cloud(self, msg):
        # type: (PointCloud2) -> None
        self._gate.replace_occupied(self._cloud_xyz(msg))

    def _on_bumper(self, msg):
        # type: (ContactsState) -> None
        """Latch the time of the most recent ground-truth collision."""
        if len(msg.states) > 0:
            self._contact_seen_at = rospy.Time.now()

    def _should_retreat(self, now, position, command):
        # type: (object, np.ndarray, object) -> bool
        """Whether to back out: a fresh contact, or an inferred wedge."""
        return self._contact_recent(now) or self._is_wedged(now, position, command)

    def _contact_recent(self, now):
        # type: (object) -> bool
        """Whether the bumper reported a contact within the hold window."""
        if self._contact_seen_at is None:
            return False
        return (now - self._contact_seen_at).to_sec() <= self._contact_hold_s

    def _is_wedged(self, now, position, command):
        # type: (object, np.ndarray, object) -> bool
        """Whether the aircraft is pinned: commanded to move, going nowhere.

        Net travel over the window -- start point to now -- not the maximum
        excursion within it, because an aircraft grinding on a wall jitters back
        and forth by more than the threshold while making no actual progress, and
        a max-excursion test reads that jitter as movement and never fires.
        """
        wanted = math.hypot(command.world_vx, command.world_vy)
        now_s = now.to_sec()
        self._track.append((now_s, float(position[0]), float(position[1]),
                            float(position[2])))
        cutoff = now_s - self._wedge_window_s
        while len(self._track) > 1 and self._track[0][0] < cutoff:
            self._track.pop(0)
        if wanted < self._wedge_cmd_speed:
            return False
        if self._track[0][0] > cutoff:
            return False                # window not yet full
        t0, x0, y0, z0 = self._track[0]
        net = math.sqrt((position[0] - x0) ** 2 + (position[1] - y0) ** 2
                        + (position[2] - z0) ** 2)
        return net < self._wedge_move_m

    def _begin_retreat(self, position, command):
        # type: (np.ndarray, object) -> None
        """Start backing out of a wall, opposite the direction it drove in.

        The condemned curve is dropped (``reset``) so following resumes only when
        FALCON publishes a fresh plan. The retreat is three phases -- BACK out
        past the depth near clip, turn to FACE the point that was struck, DWELL
        looking at it -- because the baseline showed the two-phase version
        failing fatally: the drone backed out with the obstacle outside the
        camera's view, the map never gained it, and every replan drove straight
        back in until the airframe wedged and capsized. Facing the contact point
        while the mapper fuses a few frames is what turns a strike into map
        evidence FALCON can plan around.
        """
        self._retreats += 1
        speed = math.hypot(command.world_vx, command.world_vy)
        if speed > 1e-3:
            drive = (command.world_vx / speed, command.world_vy / speed)
            self._retreat_dir_world = (-drive[0], -drive[1])
        else:
            drive = None
            self._retreat_dir_world = None      # fall back to -body-x
        # Where the obstacle is: just ahead of where the strike happened, along
        # the direction the aircraft was driving. 0.5 m is "at or just past the
        # airframe's nose"; precision is not needed, the camera FOV is wide.
        if drive is not None:
            self._contact_point = (float(position[0]) + 0.5 * drive[0],
                                   float(position[1]) + 0.5 * drive[1])
        else:
            self._contact_point = None
        self._retreat_from = np.array(position, dtype=float)
        self._retreat_phase = "back"
        self._retreat_until = rospy.Time.now() + rospy.Duration(self._retreat_time_s)
        self._contact_seen_at = None
        self._gate_blocked_since = None     # a retreat consumes the block
        self._servo.reset()
        self._track = []
        self._send(Twist())
        rospy.logwarn("[follower] contact/wedge (retreat #%d); backing out %.1f m "
                      "then turning to look", self._retreats, self._retreat_clear_m)

    def _end_retreat(self, why):
        # type: (str) -> None
        """Leave the retreat state machine and hold for a fresh plan.

        The servo is reset a SECOND time here: a curve FALCON queued mid-
        retreat was planned from a mid-retreat pose before the dwell fused
        the obstacle into the map, and promoting it would fly straight back
        at the wall the whole maneuver existed to avoid. Following resumes
        on the next post-dwell plan, which is at most one cadence away.
        """
        self._retreat_until = None
        self._retreat_dir_world = None
        self._retreat_from = None
        self._contact_point = None
        self._retreat_phase = None
        self._gate_blocked_since = None
        self._servo.reset()
        self._send(Twist())
        rospy.loginfo("[follower] retreat done (%s); holding for a fresh plan", why)

    def _retreat(self, now, position, yaw):
        # type: (object, np.ndarray, float) -> None
        """BACK out of the wall, FACE what was struck, DWELL, then hold.

        Every phase carries a time cap (the deadline in ``_retreat_until``) so a
        blocked back-out or a fouled turn cannot deadlock the aircraft; the caps
        degrade the maneuver, never the mission.
        """
        if self._retreat_phase == "back":
            cleared = (float(np.linalg.norm(position - self._retreat_from))
                       >= self._retreat_clear_m)
            # A corner: the way back out is itself walled. Stop backing early
            # rather than grind the tail into the second wall (the run-3
            # capsize was exactly this geometry).
            back_blocked = False
            if self._gate_enabled and self._retreat_dir_world is not None:
                bd = self._gate.blocked_distance(
                    (float(position[0]), float(position[1]), float(position[2])),
                    self._retreat_dir_world, 0.7)
                back_blocked = bd is not None and bd < 0.45
            if cleared or back_blocked or now >= self._retreat_until:
                # A back-out that never moved is not a retreat, and reporting it
                # as one is what produced the observed deadlock: the gate vetoes
                # the way out on the first tick, the maneuver falls through to
                # FACE and DWELL, "retreat done" is logged, the aircraft is
                # still exactly where it was, and the breach re-fires forever
                # (measured: 47 cycles at (-1.13, 4.84), zero bumper contacts,
                # coverage frozen). Feed it to the SAME escalation the map gate
                # uses -- an aircraft that cannot extract itself is the
                # structural our-clearance-versus-the-planner's disagreement
                # _note_block_episode exists to concede, and conceding means
                # creeping across at creep_speed rather than holding until the
                # mission cap.
                moved = float(np.linalg.norm(position - self._retreat_from))
                if moved < 0.05 and not cleared:
                    rospy.logwarn("[follower] could not back out (%s); the way "
                                  "out is walled too -- conceding to the planner",
                                  "gate vetoed it" if back_blocked else "timed out")
                    self._note_block_episode(now, position)
                    # Pinned in the PLANE, but this aircraft is holonomic and
                    # the gate is layered in z (_layers_for_z returns only the
                    # layers the airframe can strike at its current altitude),
                    # so gaining altitude genuinely changes what blocks it --
                    # a 1.10 m clutter pile stops obstructing an aircraft at
                    # 1.6 m. Horizontal retreat is one axis of a three-axis
                    # escape and it was the only one being used, which is why
                    # a wedge took minutes of creep to resolve instead of
                    # seconds. Climb before conceding.
                    if (self._escape_climb_s > 0.0
                            and float(position[2]) < self._escape_climb_max_z):
                        self._retreat_phase = "climb"
                        self._retreat_until = now + rospy.Duration(
                            self._escape_climb_s)
                        rospy.logwarn("[follower] pinned at %.2f m; climbing to "
                                      "clear the layer", float(position[2]))
                        return
                if self._contact_point is None:
                    self._end_retreat("cleared" if cleared else "timed out")
                    return
                self._retreat_phase = "face"
                self._retreat_until = now + rospy.Duration(self._face_time_s)
                self._send(Twist())
                return
            twist = Twist()
            if self._retreat_dir_world is None:
                twist.linear.x = -self._retreat_speed     # straight back
            else:
                cos, sin = math.cos(yaw), math.sin(yaw)
                wx, wy = self._retreat_dir_world
                twist.linear.x = self._retreat_speed * (cos * wx + sin * wy)
                twist.linear.y = self._retreat_speed * (-sin * wx + cos * wy)
            self._send(twist)
            return

        if self._retreat_phase == "climb":
            # Straight up, no yaw and no translation: the point is to change
            # which z layers the gate is testing, and translating while pinned
            # is what put the aircraft here. Ends the moment the personal-space
            # bubble is clear at the new altitude, or on the time cap.
            clear = True
            if self._gate_enabled:
                clear = not self._gate.bubble_blocked(
                    (float(position[0]), float(position[1]), float(position[2])),
                    float(rospy.get_param("~bubble_clearance_m", 0.28)))
            if (clear or now >= self._retreat_until
                    or float(position[2]) >= self._escape_climb_max_z):
                self._send(Twist())
                self._end_retreat("climbed clear at %.2f m" % float(position[2])
                                  if clear else "climb capped")
                return
            twist = Twist()
            twist.linear.z = self._escape_climb_speed
            self._send(twist)
            return

        if self._retreat_phase == "face":
            target = math.atan2(self._contact_point[1] - float(position[1]),
                                self._contact_point[0] - float(position[0]))
            err = (target - yaw + math.pi) % (2.0 * math.pi) - math.pi
            if abs(err) < 0.15 or now >= self._retreat_until:
                self._retreat_phase = "dwell"
                self._retreat_until = now + rospy.Duration(self._look_dwell_s)
                self._send(Twist())
                rospy.loginfo("[follower] facing the obstacle (yaw err %.0f deg); "
                              "letting the mapper see it", math.degrees(err))
                return
            twist = Twist()
            twist.angular.z = max(-self._survey_yaw_rate,
                                  min(self._survey_yaw_rate, 1.5 * err))
            self._send(twist)
            return

        # dwell: stare at the strike point so a few depth frames fuse
        if now >= self._retreat_until:
            self._end_retreat("looked at the obstacle")
            return
        self._send(Twist())

    def _hold_station(self, position, yaw):
        # type: (np.ndarray, float) -> None
        """Closed-loop position hold at the point where the hold began.

        A velocity-commanded airframe parked on zero twists drifts (run 010:
        ~1.8 mm/s up, plus horizontal wander); this latches the entry point
        and P-controls back to it, capped gently so a hold can never become
        an attack.
        """
        if self._hold_point is None:
            self._hold_point = np.array(
                [float(position[0]), float(position[1]), float(position[2])])
        err = self._hold_point - position
        wx = max(-0.2, min(0.2, 0.6 * float(err[0])))
        wy = max(-0.2, min(0.2, 0.6 * float(err[1])))
        cos, sin = math.cos(yaw), math.sin(yaw)
        twist = Twist()
        twist.linear.x = cos * wx + sin * wy
        twist.linear.y = -sin * wx + cos * wy
        twist.linear.z = max(-0.2, min(0.2, 0.8 * float(err[2])))
        self._send(twist)

    def _send(self, twist):
        # type: (Twist) -> None
        """The one choke point to the actuator, with the altitude guard.

        A velocity-commanded airframe parked on zero drifts: run 010 measured
        ~1.8 mm/s of open-loop climb during long holds, which walked the
        aircraft out through the planning box ceiling (z 1.8 -> 4.2) -- and a
        planner whose start pose is outside the box can never plan again, so
        the hold became permanent. Every publish, in every state, therefore
        passes through this band clamp: above the ceiling forces descent,
        below the floor forces climb, inside the band the command is its own.
        """
        z = self._last_z
        if z is not None:
            if z > self._alt_max and twist.linear.z > -0.15:
                twist.linear.z = -0.15
            elif z < self._alt_min and twist.linear.z < 0.15:
                twist.linear.z = 0.15
        # The horizontal twin: retreats and wedge back-outs are open loop and
        # happily push the airframe through the planning box wall into the
        # very shelving the box exists to exclude (run 027 reached x=-5.8
        # against a box floor of -4.6 and ground on the west shelf). Outside
        # the box plus margin, bias the WORLD velocity back toward it.
        if self._odom is not None and self._box_xy is not None:
            p = self._odom.pose.pose.position
            yaw = _yaw_from_quaternion(self._odom.pose.pose.orientation)
            (x_lo, x_hi, y_lo, y_hi) = self._box_xy
            push_wx = (0.2 if p.x < x_lo else (-0.2 if p.x > x_hi else 0.0))
            push_wy = (0.2 if p.y < y_lo else (-0.2 if p.y > y_hi else 0.0))
            if push_wx != 0.0 or push_wy != 0.0:
                cos, sin = math.cos(yaw), math.sin(yaw)
                twist.linear.x += cos * push_wx + sin * push_wy
                twist.linear.y += -sin * push_wx + cos * push_wy
        self._cmd.publish(twist)

    def _publish(self, command):
        # type: (object) -> None
        """Put the twist on the wire and report how well it is tracking."""
        twist = Twist()
        twist.linear.x = command.vx
        twist.linear.y = command.vy
        twist.linear.z = command.vz
        twist.angular.z = command.yaw_rate
        self._send(twist)

        # Flat array rather than a custom message so nothing has to be built
        # into the container to record it. Order is fixed and documented in the
        # package README; `postmortem.py` reads it positionally.
        self._diagnostics.publish(Float32MultiArray(data=[
            command.position_error_m, command.along_track_lag_m,
            command.cross_track_error_m, command.yaw_error_rad,
            command.world_vx, command.world_vy, command.world_vz,
            command.yaw_rate, float(command.trajectory_id),
            command.reference_time_s,
            1.0 if command.saturated else 0.0,
            1.0 if command.holding else 0.0,
            1.0 if command.past_end else 0.0]))


def main():
    # type: () -> None
    """Entry point."""
    rospy.init_node("bspline_follower")
    BsplineFollowerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
