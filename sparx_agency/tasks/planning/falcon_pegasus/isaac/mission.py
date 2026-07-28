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
from dataclasses import dataclass, field
from typing import Optional

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.trackers.reference_tracker_3d import (
    ReferenceTracker3D, ReferenceTrackerParams,
)
from sparx_agency.tasks.planning.falcon_pegasus.isaac import sensing
from sparx_agency.tasks.planning.falcon_pegasus.link import protocol
from sparx_agency.tasks.planning.sim_flight_recording.episode import (
    CRASH_HOLD_S, CRASH_TILT_DEG, arm_for_offboard, attitude_deg, hold_velocity,
    slew_towards,
)
from sparx_agency.tasks.planning.sim_flight_recording.path_follower import FollowSpec

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
# a first response. Generous, because a metre or two of lag through a fast
# corner is normal and self-correcting.
DIVERGENCE_ABORT_S = 30.0
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
        tracker: Gains and limits for the outer loop.
    """

    name: str
    scene: str
    spawn_xy: tuple
    spawn_yaw: float
    cruise_altitude_m: float
    frame_rate_hz: float
    max_flight_s: float
    tracker: ReferenceTrackerParams = field(default_factory=ReferenceTrackerParams)


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
        self.tracker = ReferenceTracker3D(spec.tracker)
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
        self._streaming_depth = False
        self._streaming_odometry = False
        self._next_frame_at = 0.0
        self._finished = False
        self._planner_gone_at = None

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
        self.tracker.reset(yaw=self._true_yaw(),
                           hold_position=sensing.nav_position(self.adapter))
        self._say("odometry is flowing -- FALCON has control")
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

        while True:
            # The tracker closes on the SENSOR's position, because that is the
            # frame FALCON's reference is expressed in (see sensing.nav_position).
            # Comparing a reference for the sensor against the body origin would
            # bias every command by the camera's 20 cm mount offset, rotated by
            # whatever the aircraft's heading happened to be.
            position = sensing.nav_position(self.adapter)
            now = time.time()
            command = self.tracker.update(
                self.link.reference, position, self._true_yaw(), self.loop.dt,
                reference_age=self.link.reference_age_s(now))
            self.px4.send_velocity_world(command.vx, command.vy, command.vz, command.yaw)
            if not command.holding:
                self._errors.append(command.position_error_m)
            self._tick()

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
            elif self.loop.sim_time - trajectory_at >= PLANNER_STALL_S:
                return OUTCOME_PLANNER_STOPPED, (
                    "FALCON published no new trajectory for %.0f s (still on #%d). "
                    "Its exploration node has stopped planning -- everything flown "
                    "up to here is real." % (PLANNER_STALL_S, last_trajectory))

            if math.dist(position, progress_mark) >= STALL_DISTANCE_M:
                progress_mark = position
                progress_at = self.loop.sim_time
            elif self.loop.sim_time - progress_at >= STALL_WINDOW_S:
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
                self._say("t=%5.1fs pos=(%6.2f,%6.2f,%5.2f) err=%4.2fm xte=%4.2fm "
                          "traj#%d frames=%d/%d"
                          % (self.loop.sim_time - started, position[0], position[1],
                             position[2], command.position_error_m,
                             command.cross_track_error_m, self.link.trajectory_id,
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
        self._accumulate_distance()

        if self._streaming_odometry and self.loop.step_index % ODOMETRY_EVERY_N_STEPS == 0:
            self.link.send_odometry(time.time(), *sensing.vehicle_state(self.adapter))

        if rendered:
            if self.recorder is not None:
                self.recorder.capture(stamp_s=self.loop.sim_time)
            if self._streaming_depth and self.loop.sim_time >= self._next_frame_at:
                self._next_frame_at = self.loop.sim_time + 1.0 / self.spec.frame_rate_hz
                self._send_frame()

    def _send_frame(self) -> None:
        """Send one depth frame and the camera pose that took it.

        Both are read here, in one place, immediately after the render that
        produced the image, and they leave carrying a single timestamp. FALCON
        refuses to fuse a depth image it cannot pair with a camera pose to within
        a millisecond, so this is not a convenience -- it is the pairing.
        """
        payload, width, height = sensing.depth_bytes(self.adapter)
        translation, quaternion = sensing.camera_pose(self.adapter)
        self.link.send_frame(time.time(), width, height, translation, quaternion, payload)

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

    def _true_yaw(self) -> float:
        qx, qy, qz, qw = self.adapter.vehicle.state.attitude
        return math.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))

    def _say(self, message: str) -> None:
        if self.verbose:
            print("    [%s] %s" % (self.spec.name, message), flush=True)

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
