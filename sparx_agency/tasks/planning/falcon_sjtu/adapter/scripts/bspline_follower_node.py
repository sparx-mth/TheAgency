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
from sparx_agency.core.planning.safety.clearance_envelope import (
    ClearanceEnvelope, ClearanceEnvelopeConfig,
)
from sparx_agency.core.planning.local_planners.corridor_centering import (
    CorridorCentering, CorridorCenteringConfig,
)
from sparx_agency.core.control.reference.params import ReferenceParams
from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory
from sparx_agency.core.planning.trajectories.bspline.projection import ProjectionParams

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
        # The scan rebuilds the MAP. It does nothing whatever for a planner that
        # has the map and cannot ROUTE through it, and telling those two apart
        # from here is impossible -- both look like silence. So the recovery is
        # capped: measured, a run where A* was failing every query spent 93% of
        # its ticks turning on the spot, seven consecutive recoveries deep,
        # until the watchdog ended it for not moving. Past the cap the follower
        # holds instead, which is honest (it has nothing left to try), lets
        # FALCON's dead-end guard see a stationary aircraft, and lets the
        # watchdog reach a verdict on a mission that is genuinely stuck rather
        # than on one that is busy spinning.
        self._max_resurveys = int(rospy.get_param("~max_resurveys", 4))
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
        # Set later, from the map's own box: see the altitude guard below.
        self._escape_climb_max_z = float(
            rospy.get_param("~escape_climb_max_z", 0.0))
        # How far above itself the climb looks before committing. One z layer
        # of the gate (0.2 m) plus a little, so the test asks about occupancy
        # the climb would actually reach rather than about where it already is.
        self._escape_climb_lookahead_m = float(
            rospy.get_param("~escape_climb_lookahead_m", 0.25))
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
        # ── trusting the plan ────────────────────────────────────────────
        # The reflexes below exist for what the PLANNER CANNOT SEE: obstacles
        # inside the depth near clip, and surprises the voxel map has not
        # caught up with. When the aircraft is accurately tracking a FRESH
        # plan it is, by definition, where a planner holding safe_distance
        # clearance and the whole map intended it to be -- so a reflex firing
        # there is overruling the planner with strictly less information.
        #
        # Measured in the hospital: "brake: depth 0.45 m in corridor; holding
        # (cross-track 0.08 m)". The aircraft was 8 cm off a curve planned
        # with 0.40 m of clearance and was stopped dead. At a 0.90 m doorway
        # the frame is ALWAYS inside the bumper's 0.70 m corridor, so that
        # veto fires on every doorway pass -- the aircraft cannot fly through
        # a door it is perfectly lined up with.
        #
        # On plan, a brake may still SLOW the aircraft; it may not stop it and
        # it may not send it backwards. Off plan, everything re-arms.
        self._last_cmd_speed = None     # type: object
        self._trust_plan_xtrack_m = float(
            rospy.get_param("~trust_plan_xtrack_m", 0.25))
        # Must EXCEED FALCON's replan cadence or the rule is dead most of
        # every cycle: measured in the hospital at mean 2.52 s between
        # plans, max 3.20 s. At the first attempt this was 1.5 s -- below
        # the mean -- and the vetoes kept firing at cross-track 0.00 m.
        # 4.0 s clears the observed maximum and still sits under
        # hold_silent_s (6.0), which is the separate "the planner died"
        # judgement, so a genuinely dead planner still re-arms everything.
        self._trust_plan_fresh_s = float(
            rospy.get_param("~trust_plan_fresh_s", 4.0))
        # ── clearance-relative reflexes ──────────────────────────────────
        # These replace prox_stop_m / prox_slope / prox_floor, which were an
        # absolute speed-versus-distance curve and are gone rather than left
        # inert: a parameter that is still declared but no longer read is how
        # obstacles_inflation wasted an afternoon in this same stack.
        # Every threshold above is an absolute distance, and an absolute
        # distance is the wrong shape: the room a correctly flown aircraft has
        # is a property of the CORRIDOR. The same numbers leave a 1.4 m
        # warehouse aisle (0.70 m of half width) untouched and make a 0.90 m
        # hospital doorway (0.45 m) unflyable, which is why this stack has
        # needed a different tuning per world and why fixing one broke the
        # other. The envelope compares the aircraft's clearance against the
        # PLAN's own clearance instead, so one configuration serves both.
        # See core/planning/safety/clearance_envelope.py.
        self._envelope = ClearanceEnvelope(ClearanceEnvelopeConfig(
            hard_floor_m=float(rospy.get_param("~clearance_hard_floor_m", 0.30)),
            tolerance_m=float(rospy.get_param("~clearance_tolerance_m", 0.10)),
            deficit_span_m=float(rospy.get_param("~clearance_span_m", 0.25)),
            breach_deficit_m=float(rospy.get_param("~clearance_breach_m", 0.20)),
            floor_speed=float(rospy.get_param("~creep_speed", 0.12)),
            open_clearance_m=float(rospy.get_param("~clearance_open_m", 0.90))))
        # Radius the clearance queries search. Must exceed open_clearance_m or
        # a genuinely open point reports the search limit as its clearance and
        # every deficit is measured against a ceiling rather than the geometry.
        self._clearance_r = float(rospy.get_param("~clearance_search_m", 1.0))
        # Free width across travel below which the aircraft counts as inside a
        # passage, where the wedge reflex must not fire: see the in_passage
        # rule in _tick. 1.20 m sits above every opening in either world
        # (hospital doorways 0.930 and 1.500, warehouse aisles 0.909-1.216) and
        # below any room. Measured with rays to BOTH sides, not with clearance:
        # clearance is direction-agnostic, so beside a single wall both probes
        # return that same wall and the pair reads as a corridor that is not
        # there.
        self._passage_width_m = float(
            rospy.get_param("~passage_width_m", 1.20))
        self._plan_clearance = None     # type: object  # cached, see _clearances
        self._plan_clearance_at = None  # type: object
        # ── centre of the opening ────────────────────────────────────────
        # The planner's distance cost is soft, so its curve is NEAR the middle
        # of a doorway rather than ON it, and the follower's own tracking error
        # lands on top of that. Neither is a fault; together they are a jamb
        # strike. This probes the free space laterally and biases toward its
        # peak, bounded and only where it is tight.
        self._centering_enabled = bool(rospy.get_param("~centering", True))
        self._centering = CorridorCentering(CorridorCenteringConfig(
            engage_clearance_m=float(
                rospy.get_param("~centering_engage_m", 0.85)),
            probe_m=float(rospy.get_param("~centering_probe_m", 0.25)),
            gain=float(rospy.get_param("~centering_gain", 0.6)),
            max_speed=float(rospy.get_param("~centering_max_speed", 0.15)),
            min_asymmetry_m=float(rospy.get_param("~centering_min_asym_m", 0.05))))
        # Trajectories by id, so the reference POINT can be recovered for the
        # clearance comparison: the servo reports where on the curve it is
        # tracking (reference_time_s) but not the point itself, and the point
        # is what carries the planner's clearance. Two are kept because a
        # command can still name the previous curve on the tick a new one is
        # promoted.
        self._trajectories = {}         # type: dict
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
            # The airframe's HALF-HEIGHT, and it was 0.35 m by default against a
            # collision mesh that measures z -0.040..0.070 -- 0.11 m tall in
            # total. The gate tests only the z layers the airframe can strike at
            # its current altitude, so a half-height six times too large means
            # the aircraft must clear every obstacle by 0.35 m instead of by its
            # own body. In the hospital that made a 1.79 m clutter pile
            # unflyable at ANY altitude the 1.9 m box allows (it demanded
            # 2.14 m), and run 002 ground against that one pile for 27 of its
            # contacts before the watchdog ended it. In the warehouse it forced
            # 1.8 m of altitude to overfly a 1.10 m pile that 1.5 m clears.
            body_halfheight_m=float(
                rospy.get_param("~gate_body_halfheight_m", 0.15)),
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
        # 30 s at creep_speed is 3.6 m: enough to cross an approach AND the
        # opening at the end of it, which 20 s (2.4 m) was not. A longer window
        # costs nothing when things go well, because creep_clear_m releases it
        # spatially as soon as the aircraft is 2.0 m from the contested spot --
        # the timer is only the backstop for a concession that gets nowhere.
        self._creep_time_s = float(rospy.get_param("~creep_time_s", 30.0))
        # Distance from the contested spot at which the concession is over.
        # 2.0 m is past the 1.5 m radius _note_block_episode groups episodes
        # by, so leaving cancels the creep without immediately re-arming it.
        self._creep_clear_m = float(rospy.get_param("~creep_clear_m", 2.0))
        self._creep_from = None         # type: object
        self._creep_until = None            # type: object
        self._block_episodes = []           # type: list  # (t_s, x, y)
        # ── giving up on an approach that keeps ending in contact ─────────
        # A bumper contact retreats the aircraft, and nothing stops FALCON
        # re-issuing the same route, so an obstacle standing IN a doorway
        # produces an unbounded strike-retreat-strike loop: measured, 55 bumper
        # reports on two objects at one hospital doorway, the mission ending on
        # the confinement watchdog at 216 m3.
        #
        # Retreating is right the first time and the second. By the third it is
        # evidence that this approach does not work, and the only actor that can
        # do anything about it is FALCON, whose dead-end guard retires a
        # viewpoint the aircraft stays within 2 m of for 25 s without reaching.
        # HOLDING STATION IS THEREFORE THE ACTION, not a failure to act: it is
        # exactly the condition that guard watches for, so it converts a grind
        # into a retired viewpoint and a coverage tour that moves on.
        self._contact_spots = []            # type: list  # (t_s, x, y)
        self._give_up_repeats = int(rospy.get_param("~give_up_repeats", 3))
        self._give_up_radius_m = float(rospy.get_param("~give_up_radius_m", 1.5))
        self._give_up_window_s = float(rospy.get_param("~give_up_window_s", 120.0))
        self._give_up_hold_s = float(rospy.get_param("~give_up_hold_s", 30.0))
        # Capped below the watchdog's no-movement patience (180 s), because a
        # hold is not a plan: if the planner has not moved on by then, the
        # mission is stuck and the watchdog is entitled to say so.
        self._give_up_hold_max_s = float(
            rospy.get_param("~give_up_hold_max_s", 90.0))
        self._give_ups = 0
        self._give_up_until = None          # type: object

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
            # The cross-section the airframe actually sweeps. core defaults
            # both to 0.35, which describes a body 0.70 m wide and 0.70 m tall
            # against a collision mesh that measures 0.52 x 0.52 x 0.11. The
            # height is the expensive half: at 0.35 the brake vetoes on
            # anything within 0.35 m below the camera axis, so cruising at
            # 1.30 m it brakes for a 1.14 m reception desk it clears by
            # 0.11 m -- and the hospital is full of 0.7-1.2 m furniture.
            # Matched to the voxel gate's body_halfheight so the two brakes
            # agree about the shape of the aircraft they are protecting.
            corridor_halfwidth_m=float(
                rospy.get_param("~depth_corridor_halfwidth_m", 0.30)),
            corridor_halfheight_m=float(
                rospy.get_param("~depth_corridor_halfheight_m", 0.15)),
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
        # The vertical escape ceiling is the flight box's, not a number. It was
        # 1.65 -- chosen when the warehouse box topped out at 1.8 and correct
        # only for that box. A constant here is wrong in both directions: it
        # climbs OUT of a box whose ceiling is lower (and a planner cannot plan
        # from outside its own box, so the escape strands the mission it was
        # meant to rescue), and it wastes most of the headroom of a box whose
        # ceiling is higher. The box already knows; ask it.
        if self._escape_climb_max_z <= 0.0:
            self._escape_climb_max_z = self._alt_max
        else:
            self._escape_climb_max_z = min(self._escape_climb_max_z,
                                           self._alt_max)
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
        # The mission watchdog's two verdicts. A NUDGE is recoverable and the
        # follower is the only node that can act on it: a fresh survey turn
        # rebuilds the local map and re-arms FALCON's frontier finder on a
        # region it has stopped seeing frontiers in. An ABORT is terminal --
        # hold station so the aircraft is parked and level when the harness
        # tears the stack down, rather than carrying a plan into a wall while
        # the containers die around it.
        rospy.Subscriber("/mission/nudge", String, self._on_nudge, queue_size=1)
        rospy.Subscriber("/mission/abort", String, self._on_abort, queue_size=1)
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
        # How far the projector may advance the reference in ONE call.
        # It defaults to 1.5 s of curve, but resolve() runs at 50 Hz
        # while the aircraft covers ~0.02 s of curve per tick, so the
        # window is three orders of magnitude wider than the motion it
        # is tracking. At a corner tighter than search_ahead_s * speed
        # (0.375 m at 0.25 m/s) the Euclidean-nearest point on the curve
        # is on the OUTGOING leg, the projection snaps past the apex,
        # and the correction then steers straight at it -- the aircraft
        # cuts the turn and clips the inside corner. Hospital doorways
        # and corridor junctions are exactly that geometry.
        #
        # 0.30 s is 7.5 cm of curve at cruise: too short to skip a
        # corner, and still 15 s of curve per wall-clock second at
        # 50 Hz, so recovering after a hold or a retreat is unaffected.
        reference = ReferenceParams(
            projection=ProjectionParams(
                search_back_s=float(rospy.get_param("~proj_back_s", 0.5)),
                search_ahead_s=float(rospy.get_param("~proj_ahead_s", 0.30))))
        return VelocityServoParams(
            plant=plant, limits=limits, vertical_pid=vertical_pid,
            reference=reference,
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
            # Keep the curve so the reference POINT can be recovered later:
            # the clearance comparison needs where the plan is, and the servo
            # reports only how far along it the reference sits.
            self._trajectories[int(trajectory.traj_id)] = trajectory
            if len(self._trajectories) > 2:
                for stale in sorted(self._trajectories)[:-2]:
                    del self._trajectories[stale]

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

    def _on_nudge(self, msg):
        # type: (String) -> None
        """The watchdog says the mission is going nowhere; break the fixation.

        A survey turn is the one intervention this node owns that changes what
        the PLANNER sees rather than only what the aircraft does: it rebuilds
        the local map from every bearing, which is what lets a frontier finder
        that has stopped firing in this region fire again. Ignored while a
        retreat owns the aircraft -- that maneuver is already an intervention,
        and interrupting it mid-back-out leaves the drone against the wall it
        was leaving.
        """
        if self._finished or self._retreat_until is not None:
            return
        if self._resurveys >= self._max_resurveys:
            # Same cap as the silence recovery, and for the same reason: a scan
            # that has not helped four times will not help a fifth, and turning
            # on the spot is indistinguishable from progress to nothing except
            # the coverage number the watchdog is already watching.
            rospy.logwarn_throttle(
                30.0, "[follower] watchdog nudge ignored: %d surveys have not "
                "restored progress", self._resurveys)
            return
        self._begin_resurvey(rospy.Time.now(), "watchdog nudge (%s)" % msg.data)

    def _on_abort(self, msg):
        # type: (String) -> None
        """The watchdog has ended the mission. Park the aircraft.

        Deliberately the same latch the finish uses: hold POSITION, not zero
        velocity. A velocity-commanded airframe parked on zero drifts, and the
        harness may take a few seconds to tear the stack down.
        """
        if self._finished:
            return
        self._finished = True
        self._servo.reset()
        self._hold_point = None
        rospy.logwarn("[follower] mission aborted by the watchdog (%s); "
                      "holding station", msg.data)

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
                and silent_s > self._resurvey_after_s
                and self._resurveys < self._max_resurveys):
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
        #
        # The INFERRED wedge is gated on clearance; the bumper is not. This was
        # the dominant backwards path and it was the only one the trust rule did
        # not cover -- it is evaluated here, several branches before `on_plan`
        # is even computed, so an aircraft deliberately crawling through a
        # doorway at a brake-limited speed reads as pinned and gets driven 1.3 m
        # back out of it. "Going nowhere" and "going slowly on purpose" are the
        # same measurement; only the clearance says which one it is. A real
        # contact stays ungated, because a bumper report is not an inference.
        if follow and not command.holding:
            contact = self._contact_recent(now)
            wedged = self._is_wedged(now, position, command)
            if wedged and not contact:
                plan_age = (now - self._last_bspline_at).to_sec() \
                    if self._last_bspline_at is not None else 1e9
                plan_clear, actual_clear = self._clearances(now, position, command)
                on_clearance = (
                    self._envelope.budget(plan_clear, actual_clear).deficit_m <= 0.0
                    and plan_age < self._trust_plan_fresh_s)
                # Three ways this is not a wedge, in the order they were found
                # to be needed. The reflex exists for an obstacle the MAP DOES
                # NOT HAVE -- something inside the depth near clip that FALCON
                # planned straight through -- and backing out is right there,
                # because the aircraft is somewhere nobody knew about. None of
                # the three below is that.
                #
                # 1. ON CLEARANCE: the aircraft is no nearer anything than its
                #    own reference. Necessary, and on its own not sufficient:
                #    with the plan on the centre line of a 0.90 m opening a
                #    0.15 m tracking error is already a deficit, so this rule
                #    released exactly where it was needed most.
                # 2. IN A PASSAGE: there is structure on BOTH sides across the
                #    direction of travel. In a 0.93 m doorway the map has both
                #    jambs, the aircraft is where it should be and merely slow,
                #    and a 1.3 m back-out only replays the approach -- measured
                #    in run 003, whose retreat counter reached 35 while it
                #    cycled at one doorway with coverage frozen at 1565.9 m3,
                #    each cycle costing about 12 s. See _in_passage for why
                #    this is cast as rays rather than read off the plan.
                # 3. CREEPING: the follower has already conceded this spot and
                #    chosen to push through slowly. Reading the resulting lack
                #    of progress as a wedge undoes the concession on the tick
                #    after it is made -- concede, no progress for 3 s, retreat,
                #    replan the identical curve, concede again.
                #
                # All three are bounded, and by mechanisms that can actually
                # resolve a doorway rather than reverse out of one: a real
                # bumper contact is still ungated, the map gate's own 4 s
                # hard-block escalation still retreats, creep's 20 s timer
                # re-arms the wedge, and the mission watchdog still ends a run
                # that has stopped making map.
                in_passage = (plan_age < self._trust_plan_fresh_s
                              and self._in_passage(position, command))
                if on_clearance or in_passage or (
                        self._creep_until is not None
                        and now < self._creep_until):
                    wedged = False
                    self._note_limiter("tight_pass")
            if contact and self._note_contact_spot(now, position):
                # Third strike in the same place: stop attempting it. See the
                # note by _contact_spots -- holding is what lets FALCON's
                # dead-end guard retire the viewpoint and re-route the tour.
                #
                # The hold is armed here but does NOT act here, and the ordering
                # is the whole correctness of it: the aircraft is IN CONTACT at
                # this instant, so holding station now parks it against whatever
                # it just hit, the bumper keeps reporting, and the hold re-arms
                # itself forever. Measured, when this returned early: 71 bumper
                # reports and 46 give-ups in one run, the aircraft pinned. Fall
                # through to the retreat, which backs it 1.3 m clear first; the
                # hold below then applies from the next tick on which nothing
                # fresh has been struck -- and 1.3 m clear of an unreached
                # viewpoint is exactly where FALCON's guard wants it.
                # The hold LENGTHENS each time the same place defeats it. A
                # fixed 30 s hold ends, FALCON routes the aircraft straight
                # back, and it touches the same thing again: measured, 76
                # bumper reports on one hanging-scrubs prop across seven
                # give-ups, coverage frozen. Each repeat is evidence the
                # planner needs longer to conclude the viewpoint is
                # unreachable, so give it longer -- bounded, because a hold is
                # not a plan and the mission watchdog is entitled to a verdict.
                self._give_ups += 1
                hold_s = min(self._give_up_hold_s * self._give_ups,
                             self._give_up_hold_max_s)
                self._give_up_until = now + rospy.Duration(hold_s)
                rospy.logwarn("[follower] %d contacts at (%.1f, %.1f) within "
                              "%.1f m; this approach does not work -- backing "
                              "out and holding %.0fs (give-up #%d) so the "
                              "planner retires the viewpoint",
                              self._give_up_repeats, float(position[0]),
                              float(position[1]), self._give_up_radius_m,
                              hold_s, self._give_ups)
            if contact or wedged:
                # Look only if the MAP does not already have whatever stopped
                # it. The turn and dwell are 7.5 s of a 12 s manoeuvre and they
                # buy nothing against an obstacle the mapper has already fused.
                self._begin_retreat(position, command,
                                    look=not self._map_knows_ahead(position, command))
                return
            if self._give_up_until is not None and now < self._give_up_until:
                self._note_limiter("gave_up")
                self._hold_station(position, yaw)
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
            # Tracking a fresh plan closely: the planner's clearance stands.
            plan_age = (now - self._last_bspline_at).to_sec() \
                if self._last_bspline_at is not None else 1e9
            # How much of the plan's own margin the aircraft has spent. This
            # replaces a cross-track test, and the difference is what lets one
            # configuration fly both worlds: cross-track is a PROXY for having
            # room, and it is a proxy with a different meaning in every
            # corridor. 0.25 m off a curve down the middle of a 1.4 m aisle is
            # 0.45 m of remaining clearance and perfectly safe; the same 0.25 m
            # in a 0.90 m doorway is a strike. The deficit is the thing itself.
            plan_clear, actual_clear = self._clearances(now, position, command)
            budget = self._envelope.budget(plan_clear, actual_clear)
            on_plan = (budget.deficit_m <= 0.0
                       and plan_age < self._trust_plan_fresh_s
                       and not command.holding)
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
            # A breach is a near-contact: the aircraft is materially closer to
            # something than the curve it is flying ever was, so the planner's
            # clearance no longer covers it. A fixed personal-space bubble used
            # to make this call, and a fixed bubble in a 0.90 m doorway fires on
            # every pass -- the walls are 0.45 m away and there is nowhere in
            # the opening that satisfies it. Judging the DEFICIT instead leaves
            # an intended tight pass alone and still catches a drift into a
            # jamb.
            if not creeping and not on_plan and budget.breached:
                self._note_limiter("bubble_breach")
                rospy.logwarn("[follower] clearance breach: %.2f m of room "
                              "against a plan that held %.2f m (deficit %.2f m); "
                              "retreating (cross-track %.2f m)",
                              -1.0 if budget.actual_clearance_m is None
                              else budget.actual_clearance_m,
                              budget.plan_clearance_m, budget.deficit_m,
                              command.cross_track_error_m)
                # No look: the clearance that objected was measured FROM the
                # map, so the obstacle is already in it.
                self._begin_retreat(position, command, look=False)
                return

            # Thread the middle of the opening. Applied BEFORE the brakes so
            # they still guard the biased command, and only where it is tight
            # (open space returns an idle bias), so warehouse cruise is
            # untouched.
            if self._centering_enabled and self._gate_enabled:
                command = self._centre_in_opening(command, position, yaw)

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
            #
            # The cap is now relative to the plan rather than absolute, and the
            # measured cost of the absolute version is why. Its knee sat at the
            # airframe radius with a fixed slope, so in the hospital -- where
            # every corridor puts a wall inside 0.6 m -- it was the binding
            # limiter for 71-77% of ticks at 0.36x the planned speed, throttling
            # an aircraft that was flying exactly where it had been told to.
            # Below, an aircraft no closer than its own reference is not
            # throttled at all, and one that has drifted closer is slowed in
            # proportion to the margin it has spent. The absolute floor lives
            # in the envelope's hard_floor_m and still stops it dead.
            if self._gate_enabled and not hard_blocked:
                if budget.hard_stop:
                    hard_blocked = True
                    block_why = "clearance %.2f m at the airframe floor" % (
                        budget.actual_clearance_m
                        if budget.actual_clearance_m is not None else -1.0)
                elif budget.speed_scale < 1.0:
                    speed = math.hypot(command.vx, command.vy)
                    cap = self._envelope.speed_cap(budget, speed)
                    if speed > cap > 0.0:
                        f = cap / speed
                        if f < worst:
                            worst, binding = f, "proximity_cap"
                        command = dataclasses.replace(
                            command, vx=command.vx * f, vy=command.vy * f,
                            world_vx=command.world_vx * f,
                            world_vy=command.world_vy * f)

            if hard_blocked and on_plan:
                # Slow to the creep speed and keep going THROUGH, rather than
                # stopping in the doorway and handing the aircraft to a
                # retreat that drives it back out the way it came.
                speed = math.hypot(command.vx, command.vy)
                if speed > self._creep_speed:
                    f = self._creep_speed / speed
                    if f < worst:
                        worst, binding = f, "on_plan_crawl"
                    command = dataclasses.replace(
                        command, vx=command.vx * f, vy=command.vy * f,
                        world_vx=command.world_vx * f,
                        world_vy=command.world_vy * f)
                hard_blocked = False
                self._gate_blocked_since = None

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
                    rospy.logwarn("[follower] brake: %s; holding (cross-track "
                                  "%.2f m, gap %.2f m)", block_why,
                                  command.cross_track_error_m,
                                  command.position_error_m)
                elif (now - self._gate_blocked_since).to_sec() > self._gate_block_retreat_s:
                    self._gate_blocked_since = None
                    # No look: whichever brake held this stop for 4 s was
                    # reading an obstacle it can already see -- the voxel map
                    # for the gate, the live depth frame for the bumper.
                    self._begin_retreat(position, command, look=False)
                    return
                self._note_limiter(
                    "hard_block:%s" % ("depth" if block_why.startswith("depth")
                                       else "map_gate"), 0.0)
                stop = Twist()
                stop.linear.z = command.vz          # altitude hold stays live
                stop.angular.z = command.yaw_rate   # keep looking along the path
                # Centring survives the stop, and it is the only thing here
                # that can END the stop. A dead halt in a doorway resolves in
                # exactly one way -- the aircraft moves back toward the middle
                # of the opening -- and zeroing every horizontal axis removes
                # the one motion that would clear the block, leaving the 4 s
                # escalation above to reverse the aircraft out of a door it was
                # 0.1 m off centre in. Forward drive stays zero; only the
                # across-track component is allowed through.
                if self._centering_enabled and self._gate_enabled:
                    bias = self._centering.bias(
                        self._clearance_at,
                        (float(position[0]), float(position[1]),
                         float(position[2])),
                        (command.world_vx, command.world_vy))
                    if bias.engaged:
                        cos, sin = math.cos(yaw), math.sin(yaw)
                        stop.linear.x = cos * bias.world_vx + sin * bias.world_vy
                        stop.linear.y = -sin * bias.world_vx + cos * bias.world_vy
                # A DELIBERATE stop is not a wedge, and the wedge detector
                # cannot tell the difference on its own: it compares the last
                # commanded speed against the distance travelled, and the
                # centring bias above puts 0.1-0.15 m/s on the wire while the
                # aircraft is meant to be holding position and inching
                # sideways. Three seconds of that reads as "commanding motion,
                # going nowhere" and retreats the aircraft 1.3 m back out of
                # the opening it is lining up on (measured at the hospital's
                # NW doorway, retreats 7 and 8 of run 002). Clearing the track
                # keeps the window from ever filling while the follower is the
                # one holding the aircraft still; the block's own 4 s
                # escalation above is what handles a stop that will not clear.
                self._track = []
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

    def _note_contact_spot(self, now, position):
        # type: (object, np.ndarray) -> bool
        """Record a contact; report whether this place has now failed enough.

        Grouped by position rather than by object, because the aircraft cannot
        identify what it touched -- and the geometry is what repeats. Returns
        True on the ``give_up_repeats``-th contact within ``give_up_radius_m``
        inside ``give_up_window_s``.
        """
        now_s = now.to_sec()
        x, y = float(position[0]), float(position[1])
        self._contact_spots = [s for s in self._contact_spots
                               if now_s - s[0] < self._give_up_window_s]
        self._contact_spots.append((now_s, x, y))
        near = sum(1 for s in self._contact_spots
                   if math.hypot(x - s[1], y - s[2]) < self._give_up_radius_m)
        return near >= self._give_up_repeats

    def _in_passage(self, position, command):
        # type: (np.ndarray, object) -> bool
        """Whether the aircraft is inside a passage rather than beside a wall.

        Both look identical to a clearance reading and they want opposite
        responses: in a doorway a 1.3 m back-out only replays the approach,
        while beside a wall in a room it is exactly right. The separator is
        whether there is structure on BOTH sides across the direction of
        travel, which is what ``across_width`` casts rays to find out.
        """
        if not self._gate_enabled:
            return False
        speed = math.hypot(command.world_vx, command.world_vy)
        if speed < 1e-3:
            return False
        width = self._centering.across_width(
            self._gate.blocked_distance,
            (float(position[0]), float(position[1]), float(position[2])),
            (command.world_vx / speed, command.world_vy / speed),
            self._passage_width_m)
        return width is not None and width <= self._passage_width_m

    def _map_knows_ahead(self, position, command):
        # type: (np.ndarray, object) -> bool
        """Whether the voxel map already carries whatever is stopping the aircraft.

        Decides the turn-and-look phase of a retreat. That manoeuvre exists to
        put an obstacle the mapper has NEVER SEEN in front of the camera -- the
        near-clip case, where FALCON routes through a wall because no depth ray
        ever reached it. Against an obstacle already in the map it buys nothing
        and costs 7.5 s of a 12 s retreat plus a yaw slew off the route.
        """
        if not self._gate_enabled:
            return False
        speed = math.hypot(command.world_vx, command.world_vy)
        if speed < 1e-3:
            return False
        blocked = self._gate.blocked_distance(
            (float(position[0]), float(position[1]), float(position[2])),
            (command.world_vx / speed, command.world_vy / speed),
            self._retreat_clear_m)
        return blocked is not None

    def _clearance_at(self, x, y, z):
        # type: (float, float, float) -> object
        """Room at a world point, metres, or None when nothing is within reach.

        The one measurement both the envelope and the centring probe are built
        on. ``None`` genuinely means "more than the search radius", which both
        callers read as open space -- so ``clearance_search_m`` must stay above
        the envelope's ``open_clearance_m`` or every open point reports the
        search limit and no deficit can ever be measured honestly.
        """
        if not self._gate_enabled:
            return None
        return self._gate.nearest_occupied((x, y, z), self._clearance_r)

    def _clearances(self, now, position, command):
        # type: (object, np.ndarray, object) -> tuple
        """``(plan_clearance, actual_clearance)``, metres, either possibly None.

        The plan's clearance is measured at the point on the curve the servo is
        currently tracking, which is the planner's own statement of how much
        room it believed this part of the route has. Recovering that point is
        why the node keeps the trajectories: the command carries how far along
        the curve the reference sits, not where that is.

        It is cached for ``clearance_plan_period_s`` because the reference
        travels at plan speed (0.25 m/s: 2.5 cm between recomputations) while
        this runs at 50 Hz, so recomputing it every tick buys nothing and costs
        a radial search inside the flight loop.
        """
        actual = self._clearance_at(float(position[0]), float(position[1]),
                                    float(position[2]))
        if (self._plan_clearance_at is not None
                and (now - self._plan_clearance_at).to_sec() < 0.1):
            return self._plan_clearance, actual
        trajectory = self._trajectories.get(int(command.trajectory_id))
        if trajectory is None:
            self._plan_clearance = None
            self._plan_clearance_at = now
            return None, actual
        reference = trajectory.position_at(command.reference_time_s)
        self._plan_clearance = self._clearance_at(
            float(reference[0]), float(reference[1]), float(reference[2]))
        self._plan_clearance_at = now
        return self._plan_clearance, actual

    def _centre_in_opening(self, command, position, yaw):
        # type: (object, np.ndarray, float) -> object
        """Bias the command toward the middle of a tight opening.

        The bias REDIRECTS rather than adds: the resulting speed is scaled back
        to what it was, so threading a doorway costs forward progress instead of
        buying sideways motion with extra energy. That keeps the whole thing
        inside the stopping-distance budget the brakes downstream are sized
        against, and means the centring can never make the aircraft faster than
        the planner asked for.
        """
        speed = math.hypot(command.world_vx, command.world_vy)
        if speed < 1e-3:
            return command
        bias = self._centering.bias(
            self._clearance_at,
            (float(position[0]), float(position[1]), float(position[2])),
            (command.world_vx, command.world_vy))
        if not bias.engaged:
            return command
        wx = command.world_vx + bias.world_vx
        wy = command.world_vy + bias.world_vy
        mixed = math.hypot(wx, wy)
        if mixed < 1e-6:
            return command
        scale = speed / mixed
        wx, wy = wx * scale, wy * scale
        cos, sin = math.cos(yaw), math.sin(yaw)
        return dataclasses.replace(
            command, world_vx=wx, world_vy=wy,
            vx=cos * wx + sin * wy, vy=-sin * wx + cos * wy)

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
        # Judge against what was ACTUALLY COMMANDED last tick, not against
        # the raw plan. Every brake in this node scales the command down,
        # and the deliberate slow modes (on_plan_crawl and creep, both
        # ~0.12 m/s) then look identical to being pinned: at the plan's
        # 0.25 m/s the test expects 0.75 m of travel in the 3 s window,
        # while a legitimate crawl covers 0.36 m and is called wedged.
        # That is what pulled the aircraft back OUT of doorways it was
        # crawling through -- the retreat drives 1.3 m the way it came.
        wanted = self._last_cmd_speed if self._last_cmd_speed is not None \
            else math.hypot(command.world_vx, command.world_vy)
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

    def _begin_retreat(self, position, command, look=True):
        # type: (np.ndarray, object, bool) -> None
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
        # Arm the repeat-block concession on EVERY retreat, not only on one
        # that failed to move. A retreat that backs out cleanly and then
        # re-enters the same doorway on the same legal plan is the pure
        # loop: breach at ~0.10 m of cross-track, back out 1.3 m, face,
        # dwell, fly the identical curve back in, breach again -- forever,
        # because nothing recorded that this SPOT is contested. Recording it
        # lets the second attempt concede and creep through instead.
        self._note_block_episode(rospy.Time.now(), position)
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
        #
        # ``look`` is False when the MAP is what stopped the aircraft. The turn
        # and dwell exist to put an obstacle the mapper has never seen in front
        # of the camera -- the near-clip case, where FALCON plans through a wall
        # because no depth ray ever reached it. When the voxel gate or the
        # clearance envelope is the thing objecting, the obstacle is already IN
        # the map by definition, so the manoeuvre buys nothing and costs 7.5 s
        # of the 13 s retreat plus a yaw slew away from the route. Run 002 spent
        # 143 s of its first 330 in retreats at a single doorway.
        if drive is not None and look:
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
        # A concession armed by THIS retreat has been running on the clock the
        # whole time the retreat was flying, and a retreat is 5 to 12 s of a
        # 20 s window. What survived was too short to be the concession it was
        # meant to be: the aircraft came out of the manoeuvre with ~8 s of
        # creep, which at creep_speed is under a metre, ran out short of the
        # opening, re-armed the wedge and retreated again. Measured: 28 retreats
        # cycling on one doorway approach with coverage frozen at 630 m3.
        # Restarting it here spends the window on FLYING rather than on backing
        # up, which is what conceding was for.
        if self._creep_until is not None:
            self._creep_until = rospy.Time.now() + rospy.Duration(self._creep_time_s)
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
            futile = False
            if self._gate_enabled:
                # The one absolute in the clearance model: the distance at
                # which the airframe is about to touch something. A climb has
                # bought its way out the moment nothing is inside it.
                floor_m = self._envelope.config.hard_floor_m
                here = (float(position[0]), float(position[1]),
                        float(position[2]))
                clear = not self._gate.bubble_blocked(here, floor_m)
                # A climb that would not clear it either is not an escape, it
                # is a slow grind up the face of the obstacle. Measured: the
                # aircraft rose to 1.80 m -- the flight ceiling -- against a
                # clutter pile whose top is 1.79 m, and scraped along it for
                # 100 s, eight bumper contacts, coverage frozen. Looking one
                # step ahead costs one query and turns that into a concession.
                if not clear:
                    ahead = min(self._escape_climb_max_z,
                                float(position[2]) + self._escape_climb_lookahead_m)
                    futile = (ahead <= float(position[2]) + 1e-3
                              or self._gate.bubble_blocked(
                                  (here[0], here[1], ahead), floor_m))
            if futile:
                self._note_block_episode(now, position)
                self._send(Twist())
                self._end_retreat("climbing would not clear it either at %.2f m"
                                  % float(position[2]))
                return
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

    def _note_cmd_speed(self, twist):
        # type: (object) -> None
        """Remember the horizontal speed actually put on the wire."""
        self._last_cmd_speed = math.hypot(twist.linear.x, twist.linear.y)

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
        self._note_cmd_speed(twist)
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
