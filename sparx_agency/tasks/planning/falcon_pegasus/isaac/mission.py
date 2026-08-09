"""Fly one FALCON exploration run, from arming to touchdown.

FALCON decides *where* to go; this decides nothing about the route at all. What
it owns is everything FALCON's own simulator never had to do:

1. **Get airborne first, and hand over cleanly.** FALCON's odometry stream is
   deliberately held back until the aircraft is at cruise altitude and settled.
   Its FSM waits for odometry before it will plan, so withholding it *is* the
   handover: the first trajectory is then planned from a stable hover rather than
   from an aircraft halfway through a climb.
2. **Map during the climb.** Depth frames start flowing as soon as the aircraft
   arms, so by the time FALCON is allowed to plan it already has a TSDF and a
   frontier set, and the first plan is immediate instead of two seconds of
   staring at an empty map.
3. **Close the loop FALCON assumes is already closed.** Upstream, the position
   command is fed straight back as the aircraft's state, so tracking error is
   identically zero. Here a real airframe lags, so
   :class:`~sparx_agency.core.planning.trackers.reference_tracker_3d.ReferenceTracker3D`
   turns the reference into a velocity PX4 can actually fly.
4. **Notice when it is over.** Exploration ends when FALCON's frontier set
   empties, and the way it announces that is by killing its own trajectory
   server -- so the command stream simply stops. An aircraft that is not watching
   for it hovers until a timeout.

Every way the flight can end has a named outcome, because a run that quietly
records fifteen minutes of a drone against a wall is worse than one that stops.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.control.airframe import AirframeController
from sparx_agency.core.control.flatness import matrix_from_quaternion
from sparx_agency.core.control.thrust_model import ThrustModelParams
from sparx_agency.core.control.trajectory_tracking import TrajectoryTrackerParams
from sparx_agency.core.planning.trackers.reference_tracker_3d import (
    ReferenceTracker3D, ReferenceTrackerParams,
)
from sparx_agency.tasks.planning.falcon_pegasus.isaac import sensing
from sparx_agency.tasks.planning.falcon_pegasus.link import protocol
from sparx_agency.tasks.planning.falcon_pegasus.link.sim_clock import SimClock
from sparx_agency.tasks.planning.sim_flight_recording.episode import (
    CRASH_HOLD_S, CRASH_TILT_DEG, arm_for_offboard, attitude_deg, hold_velocity,
    slew_towards,
)
from sparx_agency.tasks.planning.sim_flight_recording.path_follower import FollowSpec

CONTROL_ATTITUDE = "attitude"
"""Rebuild FALCON's trajectory here and command PX4 an attitude and a throttle."""
CONTROL_VELOCITY = "velocity"
"""Follow FALCON's 100 Hz sampled command and send PX4 world velocities."""
CONTROL_MODES = (CONTROL_ATTITUDE, CONTROL_VELOCITY)


def _default_thrust_model():
    # type: () -> ThrustModelParams
    """Thrust-model seed for the Pegasus Iris on PX4 SITL.

    0.62 rather than PX4's own 0.5 default, because that is roughly where this
    airframe actually hovers. The estimator finds the truth within a second or
    two whatever the seed, but those are the seconds immediately after handover,
    when the aircraft is at cruise height over furniture.
    """
    return ThrustModelParams(hover_throttle=0.62)


ODOMETRY_EVERY_N_STEPS = 5      # 50 Hz at the 250 Hz physics rate
CLIMB_TOLERANCE_M = 0.15
CLIMB_TIMEOUT_S = 45.0
TURN_TOLERANCE_RAD = math.radians(8.0)
SETTLE_BEFORE_HANDOVER_S = 3.0
# One full turn on the spot before handing over. Not a flourish: the camera sees
# a 90-degree wedge, so an aircraft that has only ever pointed one way hands
# FALCON a map that is one wedge of free space in a 30 x 74 m building. Its
# coverage tour then picks a cell fifteen metres away through unknown space, A*
# cannot route to it, and the FSM sits in PLAN_TRAJ reporting "No path to next
# viewpoint" forever -- measured, before this existed. One turn gives it a closed
# bubble of free space to plan out of, which is also what a real exploration
# drone does on take-off.
SURVEY_TURN_RATE = math.radians(35.0)
SURVEY_TURNS = 1.0
# FALCON needs one odometry message plus two seconds before it leaves INIT, then
# a frontier set before it leaves WAIT_TRIGGER. Thirty seconds is generous for
# both and short enough that a stack that is not going to plan says so.
FIRST_COMMAND_TIMEOUT_S = 30.0
# How long to hold station after the command stream dies before landing. Long
# enough to ride out a slow replan, short enough not to be a hover.
PLANNER_GONE_GRACE_S = 5.0
LAND_TIMEOUT_S = 90.0
POST_LAND_S = 2.0
STATUS_EVERY_S = 10.0
# Sustained divergence means the aircraft is no longer flying the plan; FALCON
# keeps replanning from where it really is, so this is a last resort rather than
# a first response.
#
# RAISED 30 -> 75 s, because at 30 it became the thing that ended healthy
# flights. The best run of the session was still climbing -- 324 s, 224 m,
# 929 m3, only two contacts, holding 1.7% of ticks -- when this fired and threw
# away three quarters of the flight budget. And its premise is weak now that
# FALCON is on /clock and replans from the aircraft's TRUE position several
# times a second: a large gap is self-correcting by construction, not evidence
# the flight is lost.
#
# Safe to relax because it is not the only guard. STALL_WINDOW_S catches an
# aircraft that has stopped moving, the crash detector catches an upset, and
# PLANNER_STALL_S catches a dead planner. This one only has to catch the case
# where all of those look fine and the aircraft is still not flying the plan.
DIVERGENCE_ABORT_S = 75.0
# PX4 leaves offboard on its own whenever a failsafe fires, and then ignores
# every setpoint sent to it. From outside that is indistinguishable from an
# aircraft that has fallen a long way behind its reference -- which is exactly
# how it was first seen here, as a 7 m "divergence" that was really an autopilot
# no longer listening. Ask for it back, but not forever.
OFFBOARD_LOST_S = 10.0
OFFBOARD_RETRY_S = 1.0
# An aircraft that has not moved this far in this long is not flying its plan,
# whatever the autopilot says: it is wedged against something. Worth its own
# outcome, because "stuck on a desk" and "cannot keep up with the reference"
# look identical from the tracking error alone and want completely different
# fixes.
STALL_DISTANCE_M = 0.5
STALL_WINDOW_S = 25.0
# A wedge is not necessarily the end of the flight. Before giving up, back the
# aircraft out along the path it just flew and let FALCON plan from somewhere
# else -- see _unwedge. Two of three consecutive soak rounds ended wedged, one
# after loitering 527 s in a single 2 x 2 m cell, so the difference between
# "recoverable" and "fatal" here is most of the flight budget.
UNWEDGE_ATTEMPTS = 3
# 2.5 m. A longer retreat (4.0 m, 18 s) was tried on the theory that backing off
# further would stop FALCON re-planning the aircraft straight back into the same
# wall -- observed three times in one flight at (-1.3, 14-16). It measured
# WORSE, 385 m3 against 567, on one flight each: the extra seconds spent flying
# backwards are seconds not exploring, and it did not stop the return trip.
# Reverted rather than left in on a hunch. The thing actually worth trying is
# not a bigger retreat but refusing the route back for a few seconds
# afterwards -- see RESUME.md.
UNWEDGE_RETREAT_M = 2.5
UNWEDGE_TIMEOUT_S = 12.0
# How much flown path to keep, and how often to sample it. The retreat target is
# chosen from this, so it must cover more than UNWEDGE_RETREAT_M of travel even
# when the aircraft is crawling.
BREADCRUMB_EVERY_S = 0.5
BREADCRUMB_KEEP_S = 60.0
# A CONTACT REFLEX, on the same retreat the wedge detector uses but triggered in
# a fifth of a second instead of twenty-five. Every crashed soak round has the
# same shape: the aircraft touches something, keeps pushing because the plan
# still says forward, touches it again, and flips. By the time the stall
# watchdog notices it has been grinding for 25 s and the attitude is already
# gone. These thresholds are the ones postmortem.py uses to identify a contact
# from a recording, where they were checked against real crashes.
CONTACT_REVERSAL_DEG = 120.0
CONTACT_WINDOW_S = 0.25
CONTACT_SPEED_MPS = 0.6
# Deceleration no command can produce: the tilt ceiling is 35 deg, so the most
# the controller can ask for is g*tan(35) = 6.9 m/s^2, and it cannot ask
# instantly. Past this the building applied it.
CONTACT_ARREST_MPS2 = 8.0
# Bounded, because a reflex that fires forever is a hover -- but bounded HIGH.
# At 6 the cap became the thing that ended flights: the best clock-fixed run
# spent all six recovering successfully (876 m3, its coverage still climbing)
# and then died on the seventh contact with the reflex disabled. Surviving a
# contact is cheap, a few seconds of retreat; not surviving one ends the
# flight. The stall and divergence watchdogs still bound a flight that is
# genuinely going nowhere, so this only has to stop an infinite loop, not
# ration a scarce resource.
CONTACT_REFLEXES = 25
# A PIN, which the reflex above structurally cannot see. Both its signatures
# need SPEED -- a 120 deg course reversal or an 8 m/s^2 arrest -- and an
# aircraft grinding against a wall produces neither. Measured on the flight
# that finally explained all this: 107 s pinned, never once reaching the
# 0.6 m/s the reversal branch needs, largest arrest 1.88 m/s^2 against the 8.0
# threshold. The controller was asking for 10-15 degrees of tilt the whole
# time and the aircraft was moving 0.16 m/s. "Commanded hard, going nowhere" is
# the signature, and it is the one the status line has been printing all along.
PINNED_TILT_DEG = 5.0
PINNED_SPEED_MPS = 0.20
PINNED_HOLD_S = 2.0
# BLOCKAGE MEMORY. Backing off a wall achieves nothing if the next trajectory
# flies straight back into it, and that is exactly what happened: six contacts
# in one flight, each followed by a successful retreat and a return within ten
# seconds. The reflex is reactive and always will be; this is the preventive
# half. A place that has just been struck is refused for a while, which makes
# the aircraft hold, which makes FALCON -- replanning several times a second --
# choose something else.
#
# It EXPIRES, and that matters more than the radius. FALCON's map improves as
# it flies, so a route that was a mistake thirty seconds ago may be the right
# one now; a permanent blacklist would carve the building up and strand the
# frontier set. The same reasoning as `blockage_memory` in
# core/planning/environment.
BLOCKAGE_RADIUS_M = 1.2
BLOCKAGE_MEMORY_S = 25.0
# How far along the trajectory to look. Far enough to refuse before arriving,
# short enough that a curve merely PASSING a struck wall at a safe distance is
# not refused -- the check is on proximity, not on direction.
BLOCKAGE_LOOKAHEAD_S = 3.0
# ESCALATION, for an aircraft that is not merely touching a wall but INSIDE it.
# A horizontal retreat assumes the aircraft can still translate; embedded in
# geometry it cannot, and the retreat measures as ~0 m of movement. Going UP is
# the one direction a wall does not block, so a retreat that achieves nothing
# is retried from a higher altitude.
ESCALATE_CLIMB_M = 0.6
ESCALATE_MOVED_M = 0.4
# WHEN FALCON STOPS PLANNING. The planner going quiet is a legitimate ending
# once the space is covered -- the best flight of the session ended exactly
# that way with 1517 m3 -- but it is a failure when it happens early, and
# ending the flight makes it unrecoverable either way. Nudging the aircraft
# somewhere new gives the frontier search something it has not already
# rejected, which is the only lever available: FALCON cannot be told anything.
PLANNER_NUDGES = 3
PLANNER_NUDGE_M = 3.0
# Fraction of the flight budget within which a silent planner is treated as a
# problem rather than as completion. See the gate in _explore.
PLANNER_NUDGE_UNTIL = 0.6
# How long the trajectory id may stand still before the planner is declared
# finished with this flight.
#
# This is the one that catches FALCON dying. Its exploration node segfaults
# inside the vendored LKH TSP solver (`LinKernighan` <- `solveTSPLKH` <-
# `solveTSP` <- `planExploreMotionHGrid`, per its own backward-cpp stack trace)
# on some coverage-tour instances -- twice in this campaign, both times mid-
# flight, both times after minutes of healthy exploration. It is third-party C
# with global state and is not fixable from here.
#
# What makes that dangerous is that it is INVISIBLE from the aircraft: traj_server
# outlives the planner and keeps publishing the final point of the last
# trajectory forever, so commands never stop, the tracking error stays at a
# centimetre, and every other health check reads perfect. The trajectory id is
# the only thing that stops moving.
PLANNER_STALL_S = 30.0

TRACE_COLUMNS = (
    "t", "x", "y", "z", "yaw", "vx", "vy", "vz",
    "ref_x", "ref_y", "ref_z", "ref_yaw", "ref_vx", "ref_vy", "ref_vz",
    "cmd_yaw", "err_m", "lag_m", "xte_m", "traj_id", "rtf", "holding",
)
"""Column layout of the per-tick flight trace. See ``_record_trace``.

Kept beside the writer so the two cannot drift apart, and written out next to
``result.json`` as ``trace_columns.json`` so a reader never has to guess.
"""

OUTCOME_EXPLORED = "explored"
OUTCOME_TIMEOUT = "flight_timeout"
OUTCOME_CRASHED = "crashed"
OUTCOME_DIVERGED = "diverged"
OUTCOME_ARM_FAILED = "arm_failed"
OUTCOME_NO_COMMANDS = "no_commands"
OUTCOME_LINK_LOST = "link_lost"
OUTCOME_CLIMB_FAILED = "climb_failed"
OUTCOME_OFFBOARD_LOST = "offboard_lost"
OUTCOME_STALLED = "stalled"
OUTCOME_PLANNER_STOPPED = "planner_stopped"
GOOD_OUTCOMES = (OUTCOME_EXPLORED, OUTCOME_TIMEOUT, OUTCOME_PLANNER_STOPPED)


@dataclass
class MissionResult:
    """What one exploration run did.

    Attributes:
        outcome: One of the ``OUTCOME_*`` constants. :data:`OUTCOME_EXPLORED` is
            FALCON declaring the space covered; :data:`OUTCOME_TIMEOUT` is a
            healthy flight that ran out of budget, which for a demonstration is
            also a success.
        detail: Human-readable explanation, empty on a clean finish.
        flight_s: Simulated seconds from arming to touchdown.
        explore_s: Simulated seconds spent flying FALCON's trajectory.
        distance_m: Path length actually flown.
        commands: Reference states received.
        depth_frames: Depth frames sent.
        dropped_frames: Depth frames superseded before they could be sent.
        mean_tracking_error_m: Average distance between the reference and the
            aircraft while exploring. The number that says whether the outer loop
            is doing its job.
        max_tracking_error_m: Worst of the same.
        landed: Whether PX4's land detector saw a touchdown.
    """

    outcome: str
    detail: str = ""
    flight_s: float = 0.0
    explore_s: float = 0.0
    distance_m: float = 0.0
    commands: int = 0
    depth_frames: int = 0
    dropped_frames: int = 0
    mean_tracking_error_m: float = 0.0
    max_tracking_error_m: float = 0.0
    landed: bool = False
    trajectories: int = 0

    @property
    def ok(self) -> bool:
        """True if the aircraft flew a real exploration and came back down."""
        return self.outcome in GOOD_OUTCOMES


@dataclass
class MissionSpec:
    """The run's parameters, as read from its YAML.

    Attributes:
        name: Run name, for logs.
        scene: Isaac scene key.
        spawn_xy: Where the aircraft starts.
        spawn_yaw: The heading it takes off on, radians.
        cruise_altitude_m: Altitude to climb to before handing over to FALCON.
        frame_rate_hz: Depth frames per second sent to FALCON.
        max_flight_s: Simulated seconds of exploration before landing regardless.
        control_mode: Which cut into PX4 to fly, ``"attitude"`` or
            ``"velocity"``.

            ``attitude`` is the default and the deeper cut: the trajectory is
            rebuilt on this side, the outer loop emits an acceleration, and PX4
            is left with the attitude loop, the rate loop and the mixer. It
            removes PX4's own velocity controller from the chain -- a loop that
            runs at tens of Hz off the same position estimate the outer loop
            already has, and so contributes lag without contributing anything.

            ``velocity`` is the older path, following the 100 Hz sampled
            command and sending world velocities. Kept because it is the
            baseline the campaign's numbers were measured against, and a
            comparison against a baseline you can no longer run is not one.
        tracker: Gains and limits for the velocity-cut outer loop.
        tracking: Gains and limits for the attitude-cut outer loop.
        thrust: Bounds and learning rate for the thrust model. Its
            ``hover_throttle`` seed should be near the airframe's real one; the
            estimator converges away from a bad seed in a second or two, but
            those are the seconds just after handover.
    """

    name: str
    scene: str
    spawn_xy: tuple
    spawn_yaw: float
    cruise_altitude_m: float
    frame_rate_hz: float
    max_flight_s: float
    control_mode: str = CONTROL_ATTITUDE
    tracker: ReferenceTrackerParams = field(default_factory=ReferenceTrackerParams)
    tracking: TrajectoryTrackerParams = field(default_factory=TrajectoryTrackerParams)
    thrust: ThrustModelParams = field(default_factory=_default_thrust_model)

    def __post_init__(self):
        # type: () -> None
        """Reject an unknown control mode here rather than mid-flight."""
        if self.control_mode not in CONTROL_MODES:
            raise ValueError("control_mode must be one of %r, got %r"
                             % (CONTROL_MODES, self.control_mode))


class ExplorationMission:
    """Runs one aircraft through one FALCON exploration.

    Args:
        loop: The :class:`~sim_loop.SimLoop` driving the simulation.
        px4: The autopilot link, already booted and configured.
        adapter: The ``PegasusIrisVehicle`` being flown.
        link: The connected :class:`~.falcon_client.FalconLink`.
        spec: The run's parameters.
        recorder: Optional ``FlightRecorder`` for the onboard/chase video.
        verbose: Print a status line every few simulated seconds.
    """

    def __init__(self, loop, px4, adapter, link, spec: MissionSpec, recorder=None,
                 verbose: bool = True):
        self.loop = loop
        self.px4 = px4
        self.adapter = adapter
        self.link = link
        self.spec = spec
        self.recorder = recorder
        self.verbose = verbose
        # Both controllers exist whichever mode is flown. They are cheap, and
        # having both constructed means the climb and the hold phases -- which
        # are velocity-commanded regardless -- do not have to care which cut the
        # exploration phase will use.
        self.tracker = ReferenceTracker3D(spec.tracker)
        self.controller = AirframeController(tracker=spec.tracking, thrust=spec.thrust)
        # hold_velocity/slew_towards want a FollowSpec for their ceilings; the
        # climb is the only phase that uses them, and it wants the same limits
        # the tracker flies with so the aircraft does not change character at
        # handover.
        self._follow = FollowSpec(
            cruise_speed=min(1.0, spec.tracker.limits.max_speed_xy),
            max_climb_rate=spec.tracker.limits.max_speed_z,
            turn_yaw_rate=spec.tracker.limits.max_yaw_rate,
        )
        self._errors = []
        self._distance = 0.0
        self._last_position = None
        self._last_velocity = None
        self._streaming_depth = False
        self._streaming_odometry = False
        self._next_frame_at = 0.0
        self._finished = False
        self._planner_gone_at = None
        self._unsafe_trajectory = None   # trajectory id FALCON condemned
        # FALCON plans on the wall clock; this aircraft lives in Isaac Sim's,
        # which runs slower. The map between them is what makes the schedule
        # flyable -- see link/sim_clock.py.
        self.clock = SimClock()
        self._trace = []                 # see _record_trace
        self._breadcrumbs = []           # (sim_time, position) -- see _unwedge
        self._breadcrumb_at = -1.0
        self._recent_velocity = []       # (sim_time, vx, vy) -- see _touched
        self._pinned_since = None        # see _pinned
        self._struck = []                # (sim_time, point) -- see _leads_into_blockage
        self._nudge_from = -1            # trajectory id when a nudge began
        self._collider_camera = None     # set by enable_collider_fusion()
        self._traced_trajectory = None   # the curve the tracker is actually on

    # ── phases ───────────────────────────────────────────────────────────

    def fly(self) -> MissionResult:
        """Arm, climb, hand over to FALCON, explore, land.

        Returns:
            The :class:`MissionResult`. A video and a map recording are written
            either way -- a partial run is still worth watching, and the result
            says which it is.
        """
        position = self.adapter.vehicle.state.position
        failure = arm_for_offboard(self.loop, self.px4, position, self.spec.spawn_yaw,
                                   self.spec.cruise_altitude_m)
        if failure is not None:
            return self._result(OUTCOME_ARM_FAILED, failure)

        self._streaming_depth = True
        self._say("armed into offboard; streaming depth to FALCON while climbing")
        armed_at = self.loop.sim_time

        outcome, detail = self._climb()
        if outcome is not None:
            return self._result(outcome, detail, armed_at)

        outcome, detail = self._survey_turn()
        if outcome is not None:
            return self._result(outcome, detail, armed_at)

        outcome, detail = self._handover()
        if outcome is not None:
            return self._result(outcome, detail, armed_at)

        explore_started = self.loop.sim_time
        outcome, detail = self._explore()
        explore_s = self.loop.sim_time - explore_started

        self._land()
        return self._result(outcome, detail, armed_at, explore_s)

    def _climb(self):
        """Climb to cruise altitude over the take-off point, turning as we go.

        Both at once: they hold the same horizontal position, so doing them in
        series only adds dead time. The horizontal *position* is held rather than
        commanding zero velocity -- velocity control has no position feedback, and
        an aircraft told to hold zero was measured drifting three metres sideways
        during a five-second climb.
        """
        takeoff = self.adapter.vehicle.state.position
        target = (float(takeoff[0]), float(takeoff[1]), self.spec.cruise_altitude_m)
        yaw = self._true_yaw()
        started = self.loop.sim_time
        while True:
            position = self.adapter.vehicle.state.position
            yaw = slew_towards(yaw, self.spec.spawn_yaw, self._follow.turn_yaw_rate,
                               self.loop.dt)
            velocity = hold_velocity(position, target, self._follow)
            self.px4.send_velocity_world(velocity[0], velocity[1], velocity[2], yaw)
            self._tick()

            at_altitude = position[2] >= self.spec.cruise_altitude_m - CLIMB_TOLERANCE_M
            lined_up = abs(normalize_angle(self.spec.spawn_yaw - yaw)) < TURN_TOLERANCE_RAD
            if at_altitude and lined_up:
                self._say("at %.2f m on %.0f deg -- settling before handing over to FALCON"
                          % (position[2], math.degrees(yaw)))
                settle_until = self.loop.sim_time + SETTLE_BEFORE_HANDOVER_S
                while self.loop.sim_time < settle_until:
                    self.px4.send_velocity_world(
                        *hold_velocity(self.adapter.vehicle.state.position, target,
                                       self._follow), yaw)
                    self._tick()
                return None, ""
            if self.loop.sim_time - started > CLIMB_TIMEOUT_S:
                return OUTCOME_CLIMB_FAILED, (
                    "never reached %.1f m within %.0f s (stuck at %.2f m)"
                    % (self.spec.cruise_altitude_m, CLIMB_TIMEOUT_S, position[2]))
            if not self.link.alive:
                return OUTCOME_LINK_LOST, "the FALCON bridge went away during the climb"

    def _survey_turn(self):
        """Turn once on the spot, so FALCON is handed a bubble and not a wedge.

        See :data:`SURVEY_TURNS`. Slow enough (35 deg/s) that consecutive depth
        frames overlap: at 12 Hz that is 3 degrees between frames against a
        90-degree field of view, so nothing is missed between them. It ends
        facing the way it started, which is the heading the run config chose.
        """
        target = tuple(self.adapter.vehicle.state.position)
        yaw = self._true_yaw()
        turned = 0.0
        self._say("surveying: one turn on the spot to map what is around us")
        while turned < SURVEY_TURNS * 2.0 * math.pi:
            step = SURVEY_TURN_RATE * self.loop.dt
            yaw = normalize_angle(yaw + step)
            turned += step
            self.px4.send_velocity_world(
                *hold_velocity(self.adapter.vehicle.state.position, target, self._follow),
                yaw)
            self._tick()
            if not self.link.alive:
                return OUTCOME_LINK_LOST, "the FALCON bridge went away during the survey"
        # Come back to the run's chosen heading before the first plan, so the
        # first trajectory starts from the pose the config describes.
        while abs(normalize_angle(self.spec.spawn_yaw - yaw)) > TURN_TOLERANCE_RAD:
            yaw = slew_towards(yaw, self.spec.spawn_yaw, self._follow.turn_yaw_rate,
                               self.loop.dt)
            self.px4.send_velocity_world(
                *hold_velocity(self.adapter.vehicle.state.position, target, self._follow),
                yaw)
            self._tick()
        return None, ""

    def _handover(self):
        """Start the odometry stream and wait for FALCON's first reference.

        This is the moment FALCON is allowed to plan. Its FSM needs one odometry
        message plus two seconds to leave INIT, and a non-empty frontier set to
        leave WAIT_TRIGGER -- which the depth streamed during the climb has
        already given it.
        """
        self._streaming_odometry = True
        here = sensing.nav_position(self.adapter)
        self.tracker.reset(yaw=self._true_yaw(), hold_position=here)
        # The integrators are cleared, because they hold a bias learned during a
        # climb; the thrust scale is not, because the airframe's mass and its
        # battery did not change when the mission phase did.
        self.controller.reset(yaw=self._true_yaw(), hold_position=here)
        self._say("odometry is flowing -- FALCON has control (%s cut, hover throttle %.2f)"
                  % (self.spec.control_mode, self.controller.hover_throttle))
        target = tuple(self.adapter.vehicle.state.position)
        started = self.loop.sim_time
        while not self.link.has_trajectory:
            self.px4.send_velocity_world(
                *hold_velocity(self.adapter.vehicle.state.position, target, self._follow),
                self._true_yaw())
            self._tick()
            if not self.link.alive:
                return OUTCOME_LINK_LOST, "the FALCON bridge went away before it planned"
            if self.loop.sim_time - started > FIRST_COMMAND_TIMEOUT_S:
                return OUTCOME_NO_COMMANDS, (
                    "FALCON planned no trajectory within %.0f s of receiving odometry. "
                    "Its FSM is probably still in WAIT_TRIGGER, which means it found no "
                    "frontiers -- check that depth frames are being fused (rostopic hz "
                    "/uav_simulator/depth_image) and that the exploration box contains "
                    "unknown space." % FIRST_COMMAND_TIMEOUT_S)
        self._say("first trajectory received -- exploring")
        return None, ""

    def _control(self, position):
        """Fly one tick, by whichever cut into PX4 the run selected.

        Both paths take the same measured state and both honour the same
        condemned-trajectory hold; what differs is how deep into PX4 they reach.

        The tracker closes on the **sensor's** position, because that is the
        frame FALCON's plan is expressed in. Comparing a reference meant for the
        camera against the body origin would bias every command by the camera's
        20 cm mount offset, rotated by whatever heading the aircraft happened to
        have.

        A condemned trajectory is withheld rather than followed: the controller
        brakes toward a latched point instead of carrying its momentum into the
        obstacle FALCON has just found while FALCON re-plans.

        Args:
            position: The sensor's world position this tick.

        Note there is no time argument. Everything here is timed on
        ``loop.sim_time``, because FALCON now stamps its trajectories on the
        same clock -- see patches/allow_sim_time.sh. A wall-clock instant
        compared against any of it would reintroduce exactly the mismatch that
        change removed.

        Returns:
            An object exposing ``holding`` and ``position_error_m`` -- the two
            things the watchdogs above need, and all they need.
        """
        velocity = tuple(float(v) for v in self.adapter.vehicle.state.linear_velocity)
        # Two reasons not to follow: FALCON condemned its own curve, or this one
        # leads back to somewhere the aircraft has just been stopped. Both make
        # the controller brake to a latched point and wait for a replacement,
        # which is what gives FALCON the chance to choose differently.
        follow = (self._unsafe_trajectory is None
                  and not self._leads_into_blockage(self.link.trajectory))
        if self.spec.control_mode == CONTROL_VELOCITY:
            reference = self.link.reference if follow else None
            command = self.tracker.update(
                reference, position, self._true_yaw(), self.loop.dt,
                # PhysX hands over the true world-frame velocity, so the
                # damping term acts on a measured signal rather than a
                # difference of positions.
                velocity=velocity,
                reference_age=self.link.reference_age_s(self.loop.sim_time))
            self.px4.send_velocity_world(command.vx, command.vy, command.vz, command.yaw)
            return command

        if self.link.trajectory is not None:
            # Idempotent: the controller rejects anything not newer than what it
            # already holds, so re-offering the same curve every tick is free.
            #
            # NO re-basing any more. FALCON now runs on /clock, which the
            # bridge publishes from the very stamps this mission sends, so a
            # trajectory's start time is already on the aircraft's own clock.
            # link/sim_clock.py compensated for the mismatch when FALCON was on
            # the wall clock; with the mismatch gone, converting again would
            # reintroduce it.
            rebased = self.link.trajectory
            if self.controller.set_trajectory(rebased) or self._traced_trajectory is None:
                self._traced_trajectory = rebased
        command = self.controller.update(position, velocity, self._true_yaw(),
                                         self.loop.dt, self.loop.sim_time,
                                         follow=follow)
        self.px4.send_attitude_target(command.attitude.quaternion_wxyz(),
                                      command.throttle,
                                      yaw_rate=command.attitude.yaw_rate)
        self._learn_thrust(command.throttle, velocity)
        return command

    def _learn_thrust(self, throttle, velocity):
        """Feed the thrust model what the throttle just bought.

        The acceleration is differenced from PhysX's own velocity rather than
        read from an accelerometer, because there is no accelerometer on this
        path and the simulator's velocity is exact. On a real aircraft this is
        the IMU, and the estimator's outlier rejection is written for that noise
        rather than for this.

        The thrust axis is the aircraft's **measured** one, not the commanded
        one: using the command would make the estimate agree with itself and
        learn nothing.
        """
        # Only while PX4 is actually flying the commanded throttle. Out of
        # offboard -- a failsafe, a mode change, the retry window in _explore --
        # PX4 ignores the setpoint and holds position or lands on its own, so
        # the acceleration measured next tick was bought by *its* throttle, not
        # ours. Folding that in teaches the estimator a scale for a throttle
        # that was never applied, and it is the vertical axis that pays.
        #
        # The previous velocity is dropped too, not just the observation: the
        # first tick back in offboard would otherwise difference across the
        # whole gap and hand the estimator one enormous acceleration.
        if not self.px4.in_offboard:
            self._last_velocity = None
            return
        if self._last_velocity is not None:
            measured = tuple((velocity[i] - self._last_velocity[i]) / self.loop.dt
                             for i in range(3))
            attitude = self.adapter.vehicle.state.attitude
            body_z = matrix_from_quaternion(
                (attitude[3], attitude[0], attitude[1], attitude[2]))[:, 2]
            self.controller.observe_thrust(throttle, measured, body_z, self.loop.dt)
        self._last_velocity = velocity

    def _explore(self):
        """Track FALCON's reference until the space is explored or time runs out."""
        started = self.loop.sim_time
        tilted_since = None
        diverged_since = None
        offboard_lost_since = None
        last_offboard_request = self.loop.sim_time
        last_status = self.loop.sim_time
        progress_mark = sensing.nav_position(self.adapter)
        progress_at = self.loop.sim_time
        last_trajectory = self.link.trajectory_id
        trajectory_at = self.loop.sim_time
        unwedge_attempts = 0
        reflexes = 0
        nudges = 0

        while True:
            # The tracker closes on the SENSOR's position, because that is the
            # frame FALCON's reference is expressed in (see sensing.nav_position).
            # Comparing a reference for the sensor against the body origin would
            # bias every command by the camera's 20 cm mount offset, rotated by
            # whatever the aircraft's heading happened to be.
            position = sensing.nav_position(self.adapter)
            self._drop_breadcrumb(position)
            # Checked BEFORE the next command is computed: the whole point is
            # not to issue another one pushing into whatever was just hit.
            if reflexes < CONTACT_REFLEXES and self._touched():
                reflexes += 1
                self._say("contact at (%.1f, %.1f, %.1f) -- backing off (%d of %d)"
                          % (position[0], position[1], position[2],
                             reflexes, CONTACT_REFLEXES))
                self._remember_blockage(position)
                self._unwedge()
                self._recent_velocity = []
                progress_mark = sensing.nav_position(self.adapter)
                progress_at = self.loop.sim_time
                continue
            command = self._control(position)
            if not command.holding:
                self._errors.append(command.position_error_m)
            self._record_trace(position, command)
            self._tick()

            if reflexes < CONTACT_REFLEXES and self._pinned(command):
                reflexes += 1
                here = sensing.nav_position(self.adapter)
                self._say("pinned at (%.1f, %.1f, %.1f) -- commanded hard and not "
                          "moving; backing off (%d of %d)"
                          % (here[0], here[1], here[2], reflexes, CONTACT_REFLEXES))
                self._remember_blockage(here)
                self._unwedge()
                self._pinned_since = None
                self._recent_velocity = []
                progress_mark = sensing.nav_position(self.adapter)
                progress_at = self.loop.sim_time
                continue

            if self._finished:
                return OUTCOME_EXPLORED, ""
            if not self.link.alive:
                return OUTCOME_LINK_LOST, "the FALCON bridge closed the link"

            roll, pitch, _ = attitude_deg(self.adapter.vehicle)
            if max(abs(roll), abs(pitch)) > CRASH_TILT_DEG:
                tilted_since = tilted_since or self.loop.sim_time
                if self.loop.sim_time - tilted_since >= CRASH_HOLD_S:
                    return OUTCOME_CRASHED, (
                        "tilted past %.0f deg for %.0f s at (%.1f, %.1f, %.1f)"
                        % (CRASH_TILT_DEG, CRASH_HOLD_S, position[0], position[1],
                           position[2]))
            else:
                tilted_since = None

            # Checked before divergence, because an autopilot that has stopped
            # listening produces a growing position error and would otherwise be
            # reported as the aircraft failing to fly a plan it never received.
            if self.px4.in_offboard:
                offboard_lost_since = None
            else:
                offboard_lost_since = offboard_lost_since or self.loop.sim_time
                if self.loop.sim_time - last_offboard_request >= OFFBOARD_RETRY_S:
                    self.px4.set_offboard_mode()
                    last_offboard_request = self.loop.sim_time
                if self.loop.sim_time - offboard_lost_since >= OFFBOARD_LOST_S:
                    return OUTCOME_OFFBOARD_LOST, (
                        "PX4 left offboard mode (now %s) and would not come back for "
                        "%.0f s; it said: %s"
                        % (self.px4.main_mode, OFFBOARD_LOST_S,
                           "; ".join(self.px4.drain_status_texts()[-4:]) or "(nothing)"))

            if self.link.trajectory_id != last_trajectory:
                last_trajectory = self.link.trajectory_id
                trajectory_at = self.loop.sim_time
                if self._unsafe_trajectory is not None:
                    self._unsafe_trajectory = None
                    self._say("new trajectory #%d -- following again" % last_trajectory)
            elif self.loop.sim_time - trajectory_at >= PLANNER_STALL_S:
                # ONLY EARLY IN THE FLIGHT. A silent planner late on almost
                # always means the space is covered -- the best flight of the
                # session ended exactly that way, planner_stopped at 1517 m3,
                # a full map. Nudging that case would spend three recoveries
                # and a minute of budget arguing with a planner that is right,
                # and risks turning a clean flight into a failed one.
                young = (self.loop.sim_time - started) < PLANNER_NUDGE_UNTIL * self.spec.max_flight_s
                if young and nudges < PLANNER_NUDGES:
                    nudges += 1
                    self._say("FALCON has not planned for %.0f s -- moving somewhere "
                              "new to give it a different view (%d of %d)"
                              % (PLANNER_STALL_S, nudges, PLANNER_NUDGES))
                    self._nudge()
                    trajectory_at = self.loop.sim_time
                    progress_mark = sensing.nav_position(self.adapter)
                    progress_at = self.loop.sim_time
                    continue
                return OUTCOME_PLANNER_STOPPED, (
                    "FALCON published no new trajectory for %.0f s (still on #%d). "
                    "Its exploration node has stopped planning -- everything flown "
                    "up to here is real." % (PLANNER_STALL_S, last_trajectory))

            if math.dist(position, progress_mark) >= STALL_DISTANCE_M:
                progress_mark = position
                progress_at = self.loop.sim_time
            elif self._unsafe_trajectory is not None:
                # Holding on purpose is not a stall. When FALCON condemns its own
                # trajectory the controller brakes to a latched point and waits
                # for a replacement, so standing still is the CORRECT behaviour;
                # reporting it as "wedged, not lagging" sends the next hour into
                # the flight controller for what was a planner event. Four of the
                # six soak rounds so far were misdiagnosed from an outcome string,
                # which is expensive enough already.
                #
                # Nothing is lost by deferring: a hold only persists while no new
                # trajectory arrives, and the planner-stall check immediately
                # above returns OUTCOME_PLANNER_STOPPED after PLANNER_STALL_S of
                # exactly that -- with the right diagnosis attached.
                progress_at = self.loop.sim_time
            elif self.loop.sim_time - progress_at >= STALL_WINDOW_S:
                if unwedge_attempts < UNWEDGE_ATTEMPTS:
                    unwedge_attempts += 1
                    self._say("wedged at (%.1f, %.1f, %.1f) -- backing out along "
                              "the way in (attempt %d of %d)"
                              % (position[0], position[1], position[2],
                                 unwedge_attempts, UNWEDGE_ATTEMPTS))
                    self._unwedge()
                    progress_mark = sensing.nav_position(self.adapter)
                    progress_at = self.loop.sim_time
                    continue
                return OUTCOME_STALLED, (
                    "moved less than %.1f m in %.0f s at (%.1f, %.1f, %.1f) while "
                    "FALCON kept replanning -- it is wedged, not lagging"
                    % (STALL_DISTANCE_M, STALL_WINDOW_S, position[0], position[1],
                       position[2]))

            if command.diverged:
                diverged_since = diverged_since or self.loop.sim_time
                if self.loop.sim_time - diverged_since >= DIVERGENCE_ABORT_S:
                    return OUTCOME_DIVERGED, (
                        "%.1f m behind FALCON's reference for %.0f s -- the aircraft is "
                        "no longer flying the plan"
                        % (command.position_error_m, DIVERGENCE_ABORT_S))
            else:
                diverged_since = None

            if self._planner_gone_at is not None:
                if self.loop.sim_time - self._planner_gone_at >= PLANNER_GONE_GRACE_S:
                    return OUTCOME_NO_COMMANDS, (
                        "FALCON's trajectory server stopped publishing and did not "
                        "come back within %.0f s" % PLANNER_GONE_GRACE_S)

            if self.loop.sim_time - started > self.spec.max_flight_s:
                return OUTCOME_TIMEOUT, (
                    "reached the %.0f s flight budget with exploration still running"
                    % self.spec.max_flight_s)

            if self.verbose and self.loop.sim_time - last_status >= STATUS_EVERY_S:
                last_status = self.loop.sim_time
                # The gap is printed split, because the halves mean opposite
                # things: `lag` is how far behind schedule the aircraft is, which
                # is benign, and `xte` is how far off the path it is, which is
                # what hits walls. A run that is 1.5 m behind but on the line is
                # healthy; one that is 1.5 m sideways is about to end.
                # The command is printed alongside the error, because the two
                # together are what separate "the controller is not asking" from
                # "the controller is asking and the aircraft is not going". A
                # metre of error with a degree of tilt is the first; a metre of
                # error with ten degrees of tilt is the second, and they want
                # completely different fixes.
                tilt_deg = math.degrees(command.attitude.tilt_rad) \
                    if self.spec.control_mode == CONTROL_ATTITUDE else float("nan")
                throttle = command.throttle \
                    if self.spec.control_mode == CONTROL_ATTITUDE else float("nan")
                self._say("t=%5.1fs pos=(%6.2f,%6.2f,%5.2f) err=%4.2fm lag=%5.2fm "
                          "xte=%4.2fm tilt=%4.1fdeg thr=%.2f rtf=%.2f%s traj#%d "
                          "frames=%d/%d"
                          % (self.loop.sim_time - started, position[0], position[1],
                             position[2], command.position_error_m,
                             command.along_track_lag_m, command.cross_track_error_m,
                             tilt_deg, throttle, self.clock.real_time_factor,
                             " HOLD" if command.holding else "",
                             self.link.trajectory_id,
                             self.link.frames_sent, self.link.frames_dropped))

    def _land(self) -> None:
        """Put the aircraft down where it finished."""
        self._say("landing")
        self.link.send_event(protocol.EVENT_MISSION_OVER, "landing")
        self.px4.land()
        started = self.loop.sim_time
        while self.loop.sim_time - started < LAND_TIMEOUT_S:
            self._tick()
            if self.px4.on_ground and self.loop.sim_time - started >= POST_LAND_S:
                self._say("on the ground")
                return
            if self.loop.sim_time - started >= LAND_TIMEOUT_S / 2:
                self.px4.land()   # a failsafe may have taken it out of LAND
        self._say("still airborne %.0f s after the land command" % LAND_TIMEOUT_S)

    # ── the per-step work ────────────────────────────────────────────────

    def _tick(self) -> None:
        """Advance one physics step and service the link.

        The link is polled *before* the step so the reference the next control
        computation reads is the newest one that had arrived, and depth is sent
        *after* the step because only a rendered step produces a fresh frame.
        """
        for name, detail in self.link.poll():
            self._on_event(name, detail)
        rendered = self.loop.step()
        # After the step, so both clocks are read at the same instant and the
        # factor measures a step that has actually happened.
        self.clock.update(time.time(), self.loop.sim_time)
        self._accumulate_distance()

        if self._streaming_odometry and self.loop.step_index % ODOMETRY_EVERY_N_STEPS == 0:
            # THE SIMULATOR'S CLOCK, not the wall clock. The bridge republishes
            # these stamps as ROS /clock, so this is what FALCON plans on -- see
            # patches/allow_sim_time.sh. Sending wall time here is what made
            # FALCON's schedule 1.6x faster than the aircraft could fly it.
            self.link.send_odometry(self.loop.sim_time,
                                    *sensing.vehicle_state(self.adapter))

        if rendered:
            if self.recorder is not None:
                self.recorder.capture(stamp_s=self.loop.sim_time)
            if self._streaming_depth and self.loop.sim_time >= self._next_frame_at:
                self._next_frame_at = self.loop.sim_time + 1.0 / self.spec.frame_rate_hz
                self._send_frame()

    def enable_collider_fusion(self, scene: str, camera_name: str) -> None:
        """Make the depth the aircraft sends agree with the physics.

        Off unless called, because it costs a raycast per frame and because a
        scene whose renderer and colliders already agree does not need it.
        """
        import numpy as _np
        from sparx_agency.robots.PEGASUS.adapters import scene_map
        from sparx_agency.tasks.planning.falcon_pegasus.stub.voxel_camera import (
            VoxelDepthCamera,
        )

        path = Path(scene_map.MAP_DIR) / ("%s_voxels.npz" % scene)
        if not path.exists():
            raise FileNotFoundError(
                "collider fusion needs the surveyed map for %r at %s -- it is the "
                "only description of what the aircraft can actually hit" % (scene, path))
        data = _np.load(str(path))
        from sparx_agency.robots.PEGASUS.adapters.vehicle import camera_intrinsics
        intrinsics = camera_intrinsics(name=camera_name)
        # A quarter grid: this only has to catch surfaces the renderer missed
        # entirely, not resolve them finely, and it runs every frame.
        self._collider_camera = VoxelDepthCamera(
            data["voxels"], data["origin"], float(data["resolution"]), intrinsics,
            ray_shape=(intrinsics.width // 4, intrinsics.height // 4))
        self._say("depth fused with the surveyed colliders -- glass will read solid")

    def _send_frame(self) -> None:
        """Send one depth frame and the camera pose that took it.

        Both are read here, in one place, immediately after the render that
        produced the image, and they leave carrying a single timestamp. FALCON
        refuses to fuse a depth image it cannot pair with a camera pose to within
        a millisecond, so this is not a convenience -- it is the pairing.
        """
        translation, quaternion = sensing.camera_pose(self.adapter)
        if self._collider_camera is None:
            payload, width, height = sensing.depth_bytes(self.adapter)
        else:
            # GLASS. The rendered depth sees through it; PhysX does not. Fusing
            # with a raycast of the surveyed COLLIDERS stops FALCON planning
            # routes through glass doors and partitions it cannot pass. See
            # sensing.fuse_with_colliders.
            from sparx_agency.core.common.spatial_math import quat_to_rot
            from sparx_agency.robots.PEGASUS.adapters.camera_pose import BODY_TO_OPTICAL
            import numpy as _np
            raw = self.adapter._camera._camera.get_depth()
            if raw is None:
                payload, width, height = sensing.depth_bytes(self.adapter)
            else:
                # THE BODY attitude, then BODY_TO_OPTICAL -- exactly what the
                # stub does. `quaternion` above is already the OPTICAL frame
                # (camera_pose returns T_w_c), so rotating it again was a
                # double rotation: the raycast pointed somewhere the camera was
                # not looking, and min() then fused those bogus ranges into
                # FALCON's map. It corrupted the mapping visibly.
                body = _np.asarray(quat_to_rot(*self.adapter.vehicle.state.attitude))
                optical = body.dot(BODY_TO_OPTICAL)
                fused = sensing.fuse_with_colliders(
                    _np.asarray(raw, dtype=_np.float32), self._collider_camera,
                    translation, optical)
                from sparx_agency.tasks.planning.falcon_pegasus.link.depth_codec \
                    import encode_depth
                encoded = encode_depth(fused)
                payload = encoded.tobytes()
                height, width = encoded.shape
        self.link.send_frame(self.loop.sim_time, width, height, translation,
                             quaternion, payload)

    def _accumulate_distance(self) -> None:
        position = self.adapter.vehicle.state.position
        if self._last_position is not None:
            self._distance += math.sqrt(sum(
                (float(position[i]) - self._last_position[i]) ** 2 for i in range(3)))
        self._last_position = tuple(float(v) for v in position)

    def _on_event(self, name: str, detail: str) -> None:
        if name == protocol.EVENT_EXPLORATION_FINISHED:
            self._finished = True
            self._say("FALCON has explored the whole box (%s)" % detail)
        elif name == protocol.EVENT_PLANNER_GONE:
            self._planner_gone_at = self._planner_gone_at or self.loop.sim_time
            self._say("FALCON stopped commanding (%s) -- holding" % detail)
        elif name == protocol.EVENT_TRAJECTORY_UNSAFE:
            # Latched on the trajectory that was live when FALCON condemned it.
            # Cleared in _explore() the moment a new one arrives, which is the
            # only thing that makes the reference safe to follow again.
            self._unsafe_trajectory = self.link.trajectory_id
            self._say("FALCON found an obstacle on the live trajectory (%s) -- "
                      "holding until it replans" % detail)

    def _true_yaw(self) -> float:
        qx, qy, qz, qw = self.adapter.vehicle.state.attitude
        return math.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))

    def _say(self, message: str) -> None:
        if self.verbose:
            print("    [%s] %s" % (self.spec.name, message), flush=True)

    def _touched(self) -> bool:
        """True if the aircraft was just stopped or turned by the building.

        Two signatures, because a collision has two shapes -- the same pair
        ``postmortem.py`` uses to find contacts in a recording, where they were
        checked against real crashes. A glancing blow spins the velocity round;
        a square-on strike simply arrests it, leaving no course to reverse.

        Read from the aircraft's own velocity rather than from any command, so
        it cannot be fooled by what the controller *wanted* to happen.
        """
        velocity = self.adapter.vehicle.state.linear_velocity
        now = self.loop.sim_time
        self._recent_velocity.append((now, float(velocity[0]), float(velocity[1])))
        while self._recent_velocity and now - self._recent_velocity[0][0] > CONTACT_WINDOW_S:
            self._recent_velocity.pop(0)
        if len(self._recent_velocity) < 2:
            return False

        then, was_x, was_y = self._recent_velocity[0]
        dt = now - then
        if dt <= 0.0:
            return False
        before = math.hypot(was_x, was_y)
        after = math.hypot(velocity[0], velocity[1])

        if before >= CONTACT_SPEED_MPS and after >= CONTACT_SPEED_MPS:
            swing = abs(math.degrees(normalize_angle(
                math.atan2(velocity[1], velocity[0]) - math.atan2(was_y, was_x))))
            if swing >= CONTACT_REVERSAL_DEG:
                return True
        return (before - after) / dt >= CONTACT_ARREST_MPS2

    def _nudge(self) -> None:
        """Move somewhere new when FALCON has stopped planning.

        A silent planner is a legitimate ending once the space is covered, and
        a failure when it happens early -- and from the aircraft the two look
        identical. Ending the flight forecloses on both. Moving a few metres
        gives the frontier search a viewpoint it has not already rejected,
        which is the only lever there is: FALCON takes no instruction, it only
        reacts to where the aircraft is and what the camera has seen.

        Backwards along the flown path, for the same reason the unwedge
        retreat goes that way -- it is the one direction known to have been
        clear seconds ago. Falls back to backing out along -body-x.
        """
        self._nudge_from = self.link.trajectory_id
        position = sensing.nav_position(self.adapter)
        target = self._retreat_target(position)
        if target is None:
            yaw_now = self._true_yaw()
            target = (position[0] - PLANNER_NUDGE_M * math.cos(yaw_now),
                      position[1] - PLANNER_NUDGE_M * math.sin(yaw_now),
                      self.spec.cruise_altitude_m)
        target = (target[0], target[1], self.spec.cruise_altitude_m)
        yaw = self._true_yaw()
        started = self.loop.sim_time
        while self.loop.sim_time - started < UNWEDGE_TIMEOUT_S:
            where = self.adapter.vehicle.state.position
            self.px4.send_velocity_world(
                *hold_velocity(where, target, self._follow), yaw)
            self._tick()
            if not self.link.alive or self._finished:
                break
            # Stop the moment FALCON starts planning again -- that is the whole
            # point, and carrying on would just waste budget.
            if self.link.trajectory_id != self._nudge_from:
                break
            if math.dist(tuple(float(v) for v in where), target) < 0.4:
                break
        here = sensing.nav_position(self.adapter)
        self.controller.reset(yaw=self._true_yaw(), hold_position=here)
        self.tracker.reset(yaw=self._true_yaw(), hold_position=here)

    def _remember_blockage(self, point) -> None:
        """Record where the aircraft was just stopped, and forget the stale ones."""
        self._struck.append((self.loop.sim_time, tuple(float(v) for v in point)))
        cutoff = self.loop.sim_time - BLOCKAGE_MEMORY_S
        self._struck = [(t, p) for t, p in self._struck if t >= cutoff]

    def _leads_into_blockage(self, trajectory) -> bool:
        """Whether this curve goes back to somewhere just struck.

        Sampled forward over a few seconds rather than tested at the endpoint:
        what matters is whether the aircraft would ARRIVE at the obstacle, and
        a curve can pass through one on its way somewhere else.

        Refusing makes the mission hold, and holding is the whole mechanism --
        FALCON replans continuously from the aircraft's true position, so a
        few seconds of not following is usually enough for it to pick a
        different viewpoint. Nothing here tells FALCON anything; it cannot be
        told.
        """
        if not self._struck or trajectory is None:
            return False
        cutoff = self.loop.sim_time - BLOCKAGE_MEMORY_S
        self._struck = [(t, p) for t, p in self._struck if t >= cutoff]
        if not self._struck:
            return False
        elapsed = trajectory.elapsed(self.loop.sim_time)
        horizon = min(elapsed + BLOCKAGE_LOOKAHEAD_S, trajectory.duration)
        step = 0.25
        when = max(0.0, elapsed)
        while when <= horizon:
            point = trajectory.position_at(when)
            for _stamp, struck in self._struck:
                if math.dist((point[0], point[1], point[2]), struck) < BLOCKAGE_RADIUS_M:
                    return True
            when += step
        return False

    def _pinned(self, command) -> bool:
        """True when the controller is pushing hard and nothing is happening.

        The complement of :meth:`_touched`: that one catches the moment of a
        strike, this one catches the aftermath nobody was watching -- an
        aircraft held against something, commanded into it continuously, for as
        long as the stall watchdog takes to notice (25 s, by which time the
        attitude has usually gone).

        Deliberately reads the COMMAND, not the plan. A large tracking error
        alone proves nothing -- the aircraft may simply be flying to catch up.
        What is diagnostic is a sustained tilt demand producing no motion,
        because that combination has only one physical explanation.
        """
        tilt = getattr(getattr(command, "attitude", None), "tilt_rad", 0.0)
        velocity = self.adapter.vehicle.state.linear_velocity
        moving = math.hypot(float(velocity[0]), float(velocity[1]))
        pushing = math.degrees(tilt) >= PINNED_TILT_DEG and moving < PINNED_SPEED_MPS
        if not pushing or getattr(command, "holding", False):
            self._pinned_since = None
            return False
        if self._pinned_since is None:
            self._pinned_since = self.loop.sim_time
            return False
        return self.loop.sim_time - self._pinned_since >= PINNED_HOLD_S

    def _drop_breadcrumb(self, position) -> None:
        """Remember where the aircraft has been, so it can retreat along it."""
        if self.loop.sim_time - self._breadcrumb_at < BREADCRUMB_EVERY_S:
            return
        self._breadcrumb_at = self.loop.sim_time
        self._breadcrumbs.append((self.loop.sim_time, tuple(float(v) for v in position)))
        cutoff = self.loop.sim_time - BREADCRUMB_KEEP_S
        while self._breadcrumbs and self._breadcrumbs[0][0] < cutoff:
            self._breadcrumbs.pop(0)

    def _retreat_target(self, position):
        """The most recent place the aircraft was, at least a retreat away.

        Walked BACKWARDS from now, so the target is the nearest such point
        rather than the oldest -- a short step off whatever it is caught on, not
        a flight back across the building.

        Returns None when there is no history far enough back, which happens if
        the aircraft wedged almost immediately after handover.
        """
        for _stamp, point in reversed(self._breadcrumbs):
            if math.dist(position, point) >= UNWEDGE_RETREAT_M:
                return point
        return None

    def _unwedge(self) -> None:
        """Back out along the path just flown, then hand control back to FALCON.

        The aircraft is wedged: FALCON keeps planning and nothing moves. Ending
        the flight there throws away the remaining budget, and the fix costs a
        few seconds -- FALCON replans continuously, so it only has to be
        somewhere *else* for the deadlock to break.

        **Retreating along the flown path is what makes this safe.** The
        aircraft was at every one of those points seconds ago, so the corridor
        back is known clear in a way no other escape direction is. Nothing here
        invents a heading or pushes into unmapped space.

        Velocity control, not the trajectory tracker: the tracker's whole job is
        to fly FALCON's curve, and this is the one moment that curve is the
        problem. The controller is reset afterwards so the position integrators
        do not carry the wedge's accumulated error into the next trajectory.
        """
        position = sensing.nav_position(self.adapter)
        target = self._retreat_target(position)
        if target is None:
            # NO BREADCRUMBS LEFT, and this is the case that matters most.
            # After BREADCRUMB_KEEP_S of not moving, every surviving crumb is
            # within UNWEDGE_RETREAT_M of here, so _retreat_target returns None
            # and the whole recovery used to degrade to a hold -- measured,
            # three times in one flight, printing "no flown path to retreat
            # along" while the aircraft stayed on the wall for 107 s.
            #
            # Backing straight out along -body-x is the right fallback. The
            # aircraft is pinned nose-first (the camera is 0.2 m forward and
            # reads the obstacle at ~0.1 m), and it FLEW IN forwards, so the
            # space behind it is the one direction known to have been clear
            # seconds ago -- the same argument the breadcrumb trail makes,
            # without needing the trail.
            yaw_now = self._true_yaw()
            target = (position[0] - UNWEDGE_RETREAT_M * math.cos(yaw_now),
                      position[1] - UNWEDGE_RETREAT_M * math.sin(yaw_now),
                      self.spec.cruise_altitude_m)
            self._say("no flown path left -- backing straight out to (%.1f, %.1f)"
                      % (target[0], target[1]))
        # Retreat at cruise height whatever height the wedge happened at: a
        # wedge under a desk or between shelves is escaped by leaving, not by
        # reproducing the altitude that caused it.
        target = (target[0], target[1], self.spec.cruise_altitude_m)

        started = self.loop.sim_time
        yaw = self._true_yaw()
        while self.loop.sim_time - started < UNWEDGE_TIMEOUT_S:
            here = self.adapter.vehicle.state.position
            self.px4.send_velocity_world(
                *hold_velocity(here, target, self._follow), yaw)
            self._tick()
            if not self.link.alive or self._finished:
                break
            if math.dist(tuple(float(v) for v in here), target) < 0.4:
                break

        # DID IT ACTUALLY MOVE? A horizontal retreat assumes the aircraft can
        # still translate. Embedded in a wall it cannot, and this measures as
        # near-zero displacement -- the case where the recovery looks like it
        # ran and changed nothing. Climbing is the one direction a wall does
        # not block, so try again from higher up.
        moved = math.dist(tuple(float(v) for v in sensing.nav_position(self.adapter)),
                          tuple(float(v) for v in position))
        if moved < ESCALATE_MOVED_M:
            self._say("retreat moved only %.2f m -- climbing %.1f m and retrying"
                      % (moved, ESCALATE_CLIMB_M))
            climb = (target[0], target[1],
                     self.spec.cruise_altitude_m + ESCALATE_CLIMB_M)
            started = self.loop.sim_time
            while self.loop.sim_time - started < UNWEDGE_TIMEOUT_S:
                where = self.adapter.vehicle.state.position
                self.px4.send_velocity_world(
                    *hold_velocity(where, climb, self._follow), yaw)
                self._tick()
                if not self.link.alive or self._finished:
                    break
                if math.dist(tuple(float(v) for v in where), climb) < 0.4:
                    break

        # Whatever happened, resume from where the aircraft actually is.
        here = sensing.nav_position(self.adapter)
        self.controller.reset(yaw=self._true_yaw(), hold_position=here)
        self.tracker.reset(yaw=self._true_yaw(), hold_position=here)
        # PRUNE the trail around the place it was stuck, and keep the rest.
        # Discarding the whole trail -- which this did at first -- leaves the
        # next retreat with no history to use: measured on a real flight, the
        # first contact backed off 1.8 m, the second 0.5 m, and the third
        # reported "no flown path to retreat along" and simply held. Pruning
        # keeps the route back out of the room while making sure no later
        # retreat aims at the obstacle just escaped.
        self._breadcrumbs = [(stamp, point) for stamp, point in self._breadcrumbs
                             if math.dist(point, position) > UNWEDGE_RETREAT_M]
        self._say("back at (%.1f, %.1f, %.1f) -- following FALCON again "
                  "(%d breadcrumbs kept)"
                  % (here[0], here[1], here[2], len(self._breadcrumbs)))

    def _record_trace(self, position, command) -> None:
        """Keep one row per tick of what the plan asked for and what happened.

        Twelve status lines per flight is not enough to tell a tracking failure
        from a collision, and that distinction has decided every soak round so
        far. The reference is re-sampled from the trajectory rather than read
        off the command, because the command carries only scalars and half the
        questions worth asking are about a *direction*.

        Deliberately unconditional. A flight costs half an hour and cannot be
        replayed; the one that finally fails in a new way should not be the one
        that was not being recorded.
        """
        trajectory = self._traced_trajectory
        if trajectory is None:
            return
        track = command.tracking if hasattr(command, "tracking") else command
        # A hold is NOT a tracking sample. _hold_station reports lag and
        # cross-track as zero because it has no curve to measure against, and
        # folding those fabricated zeros into the flight statistics flatters
        # exactly the numbers a reader is trying to judge. Recorded, but flagged,
        # so the analysis can drop them.
        holding = 1.0 if getattr(track, "holding", False) else 0.0
        reference = trajectory.sample(track.reference_time_s)
        velocity = self.adapter.vehicle.state.linear_velocity
        self._trace.append((
            self.loop.sim_time, position[0], position[1], position[2],
            self._true_yaw(),
            float(velocity[0]), float(velocity[1]), float(velocity[2]),
            reference.x, reference.y, reference.z,
            float("nan") if reference.yaw is None else reference.yaw,
            reference.vx, reference.vy, reference.vz,
            track.yaw, track.position_error_m, track.along_track_lag_m,
            track.cross_track_error_m, float(track.trajectory_id),
            self.clock.real_time_factor, holding,
        ))

    def _result(self, outcome: str, detail: str = "", armed_at: Optional[float] = None,
                explore_s: float = 0.0) -> MissionResult:
        errors = self._errors or [0.0]
        return MissionResult(
            outcome=outcome, detail=detail,
            flight_s=0.0 if armed_at is None else self.loop.sim_time - armed_at,
            explore_s=explore_s, distance_m=self._distance,
            commands=self.link.commands_received,
            depth_frames=self.link.frames_sent,
            dropped_frames=self.link.frames_dropped,
            mean_tracking_error_m=sum(errors) / len(errors),
            max_tracking_error_m=max(errors),
            landed=bool(self.px4.on_ground),
            trajectories=self.link.trajectory_id,
        )
