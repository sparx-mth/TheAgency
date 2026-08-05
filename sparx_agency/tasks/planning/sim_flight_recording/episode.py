"""Fly one planned mission end to end, autonomously, and record it.

Arm, take off, track the planned route, land at the goal. That is one episode
and one recording. A campaign is this, repeatedly, with a fresh goal each time
(:mod:`collect`).

Routes come from :mod:`episode_plan`, off a surveyed map, but *when* they are
planned matters as much as how. This module asks for one **after the climb**,
from the pose the simulator actually has, and asks again whenever the aircraft
diverges -- see :func:`fly_episode`. It is written on the assumption that a
data-collection run is unattended: every way a flight can fail has an explicit
detection and a named outcome, because a campaign that silently records ninety
minutes of a drone lying against a wall is worse than one that stops.

Three behaviours are worth calling out because they shape the data:

* **The route is flown as one continuous curve**, not a list of stops. The
  planner's corners are smoothed into a spline and chased with a moving carrot
  (:mod:`path_follower`), so the aircraft holds cruise speed the whole way
  instead of decelerating onto every waypoint. Flying the corners literally
  produced stop-and-go motion and a camera that swung at each one.
* **The aircraft is flown on velocity, not position.** The follower's output is
  a world-frame velocity and PX4 is left to be an inner-loop velocity
  controller. PX4's offboard *position* path was tried first and did not work
  here: given a setpoint one metre away, in a healthy offboard mode, with no
  failsafe and its own estimate tracking truth to 30 cm, it closed the gap at
  one centimetre per second until the flight timed out. Closing the loop on
  ground truth also makes the aircraft's *true* path the thing that converges,
  so PX4's estimator drift stops mattering -- and it is how every other
  controller in this repo flies a drone.
* **Recording starts at arming and ends at touchdown**, so a recording opens
  with the climb and closes on the ground. The frames before arming are a
  stationary drone on the floor, which is not navigation data.

The flight runs as three phases -- climb straight up, turn to face the route,
then follow it -- because doing any two of them at once is how an aircraft clips
what it took off next to, or pans the camera hard across the first few metres of
every recording. The route is planned at the boundary between the first two,
which is the only moment at which the aircraft is both level and stationary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from sparx_agency.core.common.types import Pose3D, normalize_angle
from sparx_agency.tasks.planning.sim_flight_recording.path_follower import (
    FollowSpec, PathFollower, build_trajectory,
)

SETPOINT_PRIME_S = 2.0     # offboard needs a setpoint stream flowing before it engages
ARM_RETRY_S = 1.0          # PX4 rejects arming until its own pre-flight checks pass
ARM_TIMEOUT_S = 60.0       # simulated seconds of rejected arming before giving up
LAND_TIMEOUT_S = 60.0      # PX4's descent is slow; this is a backstop, not the normal path
POST_LAND_S = 1.0          # keep recording briefly after touchdown
STATUS_EVERY_S = 5.0
# A multirotor held past this tilt is not flying, it is lying against something.
CRASH_TILT_DEG = 60.0
CRASH_HOLD_S = 3.0
# An aircraft that has not moved this far in this long is not flying its route,
# whatever PX4's mode says. Catching it early is what stops a campaign filling
# up with long recordings of a stationary drone.
STALL_DISTANCE_M = 0.5
STALL_WINDOW_S = 20.0
# Simulated seconds of being armed but not in offboard before giving up. PX4
# declines a mode request silently, so the only sign is that nothing happens.
OFFBOARD_LOST_S = 10.0
OFFBOARD_RETRY_S = 1.0
# Budget per metre of planned route, plus a fixed allowance for takeoff, arming
# and landing. Generous: the point is to catch a wedged aircraft, not to police
# a slow one.
SECONDS_PER_METRE = 4.0
FIXED_OVERHEAD_S = 90.0
# Position hold, used only while climbing and turning. The route itself is flown
# by path_follower, which is a different and much smoother thing.
HOLD_GAIN = 0.8             # 1/s
HOLD_DEADBAND_M = 0.15      # inside this, stop commanding and let it settle
CLIMB_GAIN = 1.0            # 1/s
TAKEOFF_TOLERANCE_M = 0.15  # how close to cruise altitude counts as airborne

MAX_REPLANS = 6
"""Replans allowed in one flight before it is given up on.

Enough to recover from the two things that actually cause divergence -- a gust
of PX4 overshoot on a tight corner, and a route the follower cuts wider than the
planner intended -- without letting an aircraft that simply cannot track its
route grind through its whole time budget re-deriving the same answer.

Raised from 3: a flight that drifts off the route should be **replanned from
where it is**, not abandoned, and three was low enough that a long route through
several doorways could exhaust them and be recorded as ``off_route`` while still
perfectly flyable. The interval floor below is what keeps this from becoming a
loop.
"""

REPLAN_MIN_INTERVAL_S = 5.0
"""Simulated seconds between replans.

The follower reports divergence on every step once it has diverged, so without a
floor here one bad moment would consume every replan in three consecutive
iterations and none of them would have been given time to work.
"""
TAKEOFF_TIMEOUT_S = 30.0
TURN_TOLERANCE_RAD = math.radians(8.0)  # lined up enough to set off

PHASE_CLIMB = "climb"
PHASE_TURN = "turn"
PHASE_FOLLOW = "follow"
# How close to the goal counts as having got there. Wider than the mission's own
# arrival radius because PX4's descent drifts.
GOAL_TOLERANCE_M = 2.0

OUTCOME_LANDED = "landed"
OUTCOME_CRASHED = "crashed"
OUTCOME_ARM_TIMEOUT = "arm_timeout"
OUTCOME_OFFBOARD_LOST = "offboard_lost"
OUTCOME_STALLED = "stalled"
OUTCOME_FLIGHT_TIMEOUT = "flight_timeout"
OUTCOME_LAND_TIMEOUT = "land_timeout"
OUTCOME_MISSED_GOAL = "missed_goal"
OUTCOME_OFF_ROUTE = "off_route"
GOOD_OUTCOMES = (OUTCOME_LANDED,)


@dataclass
class EpisodeResult:
    """What one episode did.

    Attributes:
        outcome: One of the ``OUTCOME_*`` constants. Only
            :data:`OUTCOME_LANDED` is a clean flight.
        frames: Frames recorded.
        duration_s: Simulated seconds from arming to the end of the recording.
        route_remaining_m: Along-path distance still unflown when the episode
            ended. Near zero on a clean flight.
        cross_track_error_m: How far off the planned route the aircraft was at
            the end, metres.
        final_xy: Where the aircraft ended up.
        goal_error_m: Horizontal distance from the goal at the end.
        estimator_drift_m: How far PX4's position estimate moved relative to
            ground truth over the flight. Tens of centimetres is healthy;
            metres means the aircraft was being commanded to the wrong place.
        replans: How many times the route was planned again in flight. Zero on
            a flight that tracked its first route all the way.
        flown_plan: The route actually flown -- planned after the climb, from
            where the aircraft really was, and replaced again on every replan.
            The caller records *this* rather than the pre-takeoff plan, so a
            recording's metadata describes the route its images were taken on.
        detail: Human-readable explanation, empty on a clean flight.
        px4_messages: ``STATUSTEXT`` lines PX4 emitted, which is where a refused
            arming says why.
    """

    outcome: str
    frames: int = 0
    duration_s: float = 0.0
    route_remaining_m: float = 0.0
    cross_track_error_m: float = 0.0
    final_xy: Tuple[float, float] = (0.0, 0.0)
    goal_error_m: float = 0.0
    estimator_drift_m: float = 0.0
    replans: int = 0
    flown_plan: object = None
    detail: str = ""
    px4_messages: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if the flight completed and landed."""
        return self.outcome in GOOD_OUTCOMES


def flight_budget_s(path_length_m: float) -> float:
    """Simulated seconds to allow a route before declaring the aircraft stuck."""
    return FIXED_OVERHEAD_S + SECONDS_PER_METRE * path_length_m


def attitude_deg(vehicle) -> Tuple[float, float, float]:
    """The vehicle's roll, pitch and yaw in degrees."""
    from scipy.spatial.transform import Rotation

    qx, qy, qz, qw = vehicle.state.attitude
    roll, pitch, yaw = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=True)
    return float(roll), float(pitch), float(yaw)


def hold_velocity(position, target, spec: FollowSpec) -> Tuple[float, float, float]:
    """World-frame velocity that holds ``position`` at ``target``.

    Used only while climbing and turning, where the aircraft must stay put
    rather than travel. Velocity control has no position feedback of its own, so
    "hold still" has to be commanded as "fly to where you already are".

    Args:
        position: True world-frame ``(x, y, z)``.
        target: World-frame ``(x, y, z)`` to hold.
        spec: Flight parameters, for the speed and climb-rate ceilings.

    Returns:
        ``(vx, vy, vz)`` in the world frame, m/s, ``vz`` positive up.
    """
    dx, dy = target[0] - position[0], target[1] - position[1]
    distance = math.hypot(dx, dy)
    if distance > HOLD_DEADBAND_M:
        speed = min(max(HOLD_GAIN * distance, spec.min_speed), spec.cruise_speed)
        vx, vy = dx / distance * speed, dy / distance * speed
    else:
        vx = vy = 0.0
    climb = CLIMB_GAIN * (target[2] - position[2])
    return vx, vy, max(-spec.max_climb_rate, min(spec.max_climb_rate, climb))


def slew_towards(current: float, target: float, max_rate: float, dt: float) -> float:
    """Step ``current`` toward ``target`` by at most ``max_rate * dt`` radians.

    The commanded heading is always slewed, never stepped: a stepped setpoint
    hands the slew rate back to the autopilot's own limiter, and how fast the
    world rotates in the recorded imagery stops being something this code
    controls.
    """
    error = normalize_angle(target - current)
    step = max_rate * dt
    return normalize_angle(current + max(-step, min(step, error)))


def _true_yaw(vehicle) -> float:
    """The vehicle's exact heading, radians CCW from +X (world ENU / body FLU)."""
    import numpy as np

    qx, qy, qz, qw = vehicle.state.attitude
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))




def arm_for_offboard(loop, px4, hold_xyz, yaw: float, cruise_altitude: float) -> Optional[str]:
    """Prime the offboard stream, request OFFBOARD, and arm.

    PX4 refuses to *enter* offboard mode until a setpoint stream is already
    flowing, and drops out of it if the stream stops -- so setpoints go out
    every step here, before the mode request as well as after.

    The primed setpoint is the aircraft's own position at cruise altitude: the
    moment offboard engages PX4 flies to whatever it was last told, and a
    setpoint anywhere else would make the aircraft lunge off the pad.

    **Both conditions are waited for, not just arming.** An earlier version
    returned as soon as ``px4.armed`` was true, which meant an aircraft that
    arrived already armed (a previous episode's failsafe, say) never had
    OFFBOARD requested at all -- and then sat on the pad ignoring every setpoint
    while the mission timed out waypoint by waypoint.

    Args:
        loop: The :class:`~sim_loop.SimLoop` to step.
        px4: The autopilot link.
        hold_xyz: The aircraft's current world position. Unused now that arming
            holds zero velocity rather than a position, kept for the signature.
        yaw: Heading to hold while arming.
        cruise_altitude: Altitude to climb to once armed, metres.

    Returns:
        None once armed *and* in offboard, or a failure detail string.
    """
    started_at = loop.sim_time
    last_request = -ARM_RETRY_S

    while not (px4.armed and px4.in_offboard):
        position = loop.vehicle.state.position
        # Keep re-latching the heading bias while the aircraft sits still: PX4's
        # estimator is settling, and standing still is the only time a live
        # measurement is safe (see PX4Offboard.latch_frame).
        px4.latch_frame(position, _true_yaw(loop.vehicle))
        px4.send_velocity_world(0.0, 0.0, 0.0, yaw)
        loop.step()

        elapsed = loop.sim_time - started_at
        if elapsed < SETPOINT_PRIME_S:
            continue
        if loop.sim_time - last_request >= ARM_RETRY_S:
            px4.set_offboard_mode()
            px4.arm()
            last_request = loop.sim_time
        if elapsed > ARM_TIMEOUT_S + SETPOINT_PRIME_S:
            state = ("armed but stuck in mode %s" % px4.main_mode if px4.armed
                     else "not armed")
            return (f"PX4 would not arm into offboard for {ARM_TIMEOUT_S:.0f} simulated "
                    f"seconds ({state}); PX4 said: "
                    f"{'; '.join(px4.drain_status_texts()[-4:]) or '(nothing)'}")
    return None


def fly_episode(loop, px4, adapter, plan, recorder, follow_spec: FollowSpec = None,
                verbose: bool = True, replan=None) -> EpisodeResult:
    """Arm, fly ``plan``, land at its goal, recording throughout.

    **The route that is flown is planned after the climb, not before it.** A
    route planned from the ground starts where the aircraft *was*, and the climb
    moves it: measured over 16 flights, the median displacement between the
    planned start and the first sample at cruise altitude was 0.54 m, the 90th
    percentile 1.42 m and the worst 4.52 m -- against a planner standoff of only
    0.6 m. So the aircraft routinely reached cruise altitude already outside the
    corridor its route had been planned for, and spent the first leg cutting
    back onto a spline that began somewhere it no longer was. Flights that
    diverged had a median takeoff drift of 4.52 m against 0.52 m for those that
    landed. Planning once the aircraft is level, from its true position, removes
    that error at its source rather than asking the follower to absorb it.

    **A route that diverges is replanned, not abandoned.** Cross-track error
    past the follower's tolerance used to end the episode outright; now it
    re-plans from wherever the aircraft actually is, up to
    :data:`MAX_REPLANS`. Only when those are used up is the flight given up on.

    Args:
        loop: The :class:`~sim_loop.SimLoop` driving the simulation.
        px4: The autopilot link, already booted and configured.
        adapter: The :class:`PegasusIrisVehicle` being flown.
        plan: The :class:`~episode_plan.EpisodePlan` to execute. Its *goal* is
            what the flight is for; its waypoints are only a fallback for when
            replanning is unavailable or fails.
        recorder: A :class:`~flight_session.FlightRecorder` to capture into.
        verbose: Print a status line every few simulated seconds.
        replan: ``f(x, y, yaw) -> EpisodePlan | None``, planning a fresh route
            from a live pose to the same goal. ``None`` keeps the old
            plan-once-before-takeoff behaviour, which is what the unit tests
            that have no map exercise.

    Returns:
        The :class:`EpisodeResult`. A recording is written either way -- a
        partial recording of a flight that hit something is still worth having,
        and ``meta.json`` says which it is.
    """
    vehicle = adapter.vehicle
    cruise_altitude = plan.waypoints[0][2]
    follow_spec = follow_spec or FollowSpec()

    start_position = vehicle.state.position
    # Where the aircraft *is*, not where the route says it should be. The two
    # agree when the plan was made from a live pose, and the yaw setpoint is the
    # one place a disagreement is expensive: it is a position command, so opening
    # a flight with the plan's heading asks PX4 to rotate to it as fast as it
    # can, unbounded by the follower's yaw rate. The position half of this has
    # always come from the vehicle; the heading half now does too.
    start_yaw = adapter.yaw()
    failure = arm_for_offboard(loop, px4, start_position, start_yaw, cruise_altitude)
    if failure is not None:
        return EpisodeResult(
            outcome=OUTCOME_ARM_TIMEOUT, detail=failure,
            final_xy=(float(start_position[0]), float(start_position[1])),
            goal_error_m=math.hypot(start_position[0] - plan.goal.x,
                                    start_position[1] - plan.goal.y),
            flown_plan=plan,
            px4_messages=px4.drain_status_texts(),
        )
    if verbose:
        offset = px4.frame_offset
        print(f"    armed into offboard; PX4 frame offset "
              f"({offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}) m, "
              f"heading bias {math.degrees(px4.heading_bias):+.1f} deg", flush=True)

    def build_follower(from_x: float, from_y: float, from_yaw: float, waypoints):
        """A follower whose spline starts under the aircraft, not where it was."""
        return PathFollower(
            build_trajectory(Pose3D(float(from_x), float(from_y), cruise_altitude,
                                    float(from_yaw)),
                             waypoints, follow_spec),
            follow_spec, initial_yaw=float(from_yaw))

    flown_plan = plan
    replans = 0
    last_replan_at = -REPLAN_MIN_INTERVAL_S
    follower = build_follower(start_position[0], start_position[1], start_yaw,
                              plan.waypoints)
    # Only a placeholder until the post-climb plan exists: during the climb the
    # aircraft turns toward the goal's straight-line bearing, which is close
    # enough that the turn after replanning is short, and costs no planning.
    route_heading = math.atan2(plan.goal.y - start_position[1],
                               plan.goal.x - start_position[0])

    armed_at = loop.sim_time
    budget = flight_budget_s(plan.path_length_m)
    takeoff_xy = (float(start_position[0]), float(start_position[1]))
    # Where the aircraft should hold while it climbs and turns. It starts as the
    # take-off point -- translating near the floor is how an aircraft clips what
    # it took off next to -- and moves to the post-climb position once the route
    # has been planned from there.
    hold_xy = takeoff_xy
    phase = PHASE_CLIMB
    yaw = start_yaw
    tilted_since = None
    offboard_lost_since = None
    last_offboard_request = loop.sim_time
    landing_started = None
    last_status = loop.sim_time
    progress_mark = (takeoff_xy[0], takeoff_xy[1], float(start_position[2]))
    progress_at = loop.sim_time
    follow = None
    outcome = None
    detail = ""

    while True:
        position = vehicle.state.position

        if landing_started is None:
            if phase == PHASE_CLIMB:
                # Climb and turn at the same time. Both hold position, so doing
                # them in series only adds dead time to the recording.
                yaw = slew_towards(yaw, route_heading, follow_spec.turn_yaw_rate,
                                   loop.dt)
                # Climb straight up before setting off. Translating while still
                # near the floor is how an aircraft clips whatever it took off
                # next to -- so the climb target is the take-off point itself,
                # which *holds* the horizontal position. Merely commanding zero
                # horizontal velocity is not the same thing: velocity control
                # has no position feedback, and the aircraft was measured
                # drifting three metres sideways during a five-second climb.
                velocity = hold_velocity(
                    position, (takeoff_xy[0], takeoff_xy[1], cruise_altitude),
                    follow_spec)
                at_altitude = position[2] >= cruise_altitude - TAKEOFF_TOLERANCE_M
                if at_altitude:
                    # Level at cruise height: this is where the route is planned
                    # from, using the pose the simulator actually has rather
                    # than the one the aircraft had before it left the ground.
                    fresh = replan(float(position[0]), float(position[1]),
                                   float(yaw)) if replan is not None else None
                    if fresh is not None:
                        flown_plan = fresh
                        follower = build_follower(position[0], position[1], yaw,
                                                  fresh.waypoints)
                        budget = flight_budget_s(fresh.path_length_m)
                        last_replan_at = loop.sim_time
                        if verbose:
                            drift = math.hypot(position[0] - plan.start.x,
                                               position[1] - plan.start.y)
                            print(f"    at {position[2]:.1f} m, {drift:.2f} m from where "
                                  f"it took off -- planned {fresh.path_length_m:.1f} m "
                                  f"from here over {len(fresh.waypoints)} waypoints",
                                  flush=True)
                    elif replan is not None:
                        # The pre-takeoff route was planned from a snapped ground
                        # position and a heading the aircraft has since changed;
                        # flying it because the real plan failed means flying a
                        # route to somewhere the aircraft is not. Refuse instead.
                        outcome = OUTCOME_OFF_ROUTE
                        detail = (f"no route from ({position[0]:.1f}, "
                                  f"{position[1]:.1f}) at cruise altitude -- the "
                                  f"pre-takeoff route is not flyable from here")
                        break
                    route_heading = follower.initial_heading()
                    hold_xy = (float(position[0]), float(position[1]))
                    phase = PHASE_TURN

                lined_up = abs(normalize_angle(route_heading - yaw)) < TURN_TOLERANCE_RAD
                if at_altitude and lined_up:
                    phase = PHASE_FOLLOW
                    follower.yaw = yaw
                    if verbose:
                        print(f"    on {math.degrees(route_heading):.0f} deg -- "
                              f"following the route", flush=True)
                elif not at_altitude and loop.sim_time - armed_at > TAKEOFF_TIMEOUT_S:
                    # Only a failure to *climb* times out here. Reaching altitude
                    # and still turning is progress, and the turn has its own
                    # phase; without the guard a slow climb that just levelled
                    # off would be reported as never having got off the ground.
                    outcome = OUTCOME_FLIGHT_TIMEOUT
                    detail = (f"never reached its {cruise_altitude:.1f} m cruise altitude "
                              f"within {TAKEOFF_TIMEOUT_S:.0f} s (stuck at "
                              f"{position[2]:.2f} m)")
                    break
            elif phase == PHASE_TURN:
                # Line up with the route before moving off. Turning while under
                # way would pan the camera hard across the first few metres of
                # every flight, which is the opposite of what the recording is
                # for.
                velocity = hold_velocity(
                    position, (hold_xy[0], hold_xy[1], cruise_altitude),
                    follow_spec)
                yaw = slew_towards(yaw, route_heading, follow_spec.turn_yaw_rate,
                                   loop.dt)
                if abs(normalize_angle(route_heading - yaw)) < TURN_TOLERANCE_RAD:
                    phase = PHASE_FOLLOW
                    follower.yaw = yaw
                    if verbose:
                        print(f"    lined up on {math.degrees(route_heading):.0f} deg "
                              f"-- following the route", flush=True)
            else:
                follow = follower.update(position, _true_yaw(vehicle),
                                         vehicle.state.linear_velocity, loop.dt)
                velocity, yaw = follow.velocity, follow.yaw
                if follow.failed:
                    # Diverging is a reason to plan again from where the
                    # aircraft is, not to throw the flight away. The interval
                    # stops a route that cannot be tracked from replanning every
                    # step; the count stops it going on forever.
                    fresh = None
                    if (replan is not None and replans < MAX_REPLANS
                            and loop.sim_time - last_replan_at >= REPLAN_MIN_INTERVAL_S):
                        fresh = replan(float(position[0]), float(position[1]),
                                       float(_true_yaw(vehicle)))
                    if fresh is None:
                        outcome = OUTCOME_OFF_ROUTE
                        detail = (f"{follow.cross_track_error:.1f} m off the planned route "
                                  f"at ({position[0]:.1f}, {position[1]:.1f}) after "
                                  f"{replans} replan(s)")
                        break
                    replans += 1
                    flown_plan = fresh
                    last_replan_at = loop.sim_time
                    follower = build_follower(position[0], position[1],
                                              follower.yaw, fresh.waypoints)
                    # The clock keeps running, so extend the budget by what the
                    # new route costs rather than restarting it -- otherwise a
                    # flight that replans repeatedly never times out.
                    budget += SECONDS_PER_METRE * fresh.path_length_m
                    route_heading = follower.initial_heading()
                    if verbose:
                        print(f"    {follow.cross_track_error:.1f} m off route at "
                              f"({position[0]:.1f}, {position[1]:.1f}) -- replanned "
                              f"{fresh.path_length_m:.1f} m from here "
                              f"({replans}/{MAX_REPLANS})", flush=True)
                    # Drive this step off the new follower rather than skipping
                    # it: the aircraft is under way and must be given a
                    # velocity every iteration. Not stopping to turn is
                    # deliberate -- the follower's rate-limited yaw brings it
                    # round while it keeps moving.
                    follow = follower.update(position, _true_yaw(vehicle),
                                             vehicle.state.linear_velocity, loop.dt)
                    velocity, yaw = follow.velocity, follow.yaw
                    if follow.failed:
                        outcome = OUTCOME_OFF_ROUTE
                        detail = (f"still {follow.cross_track_error:.1f} m off route "
                                  f"immediately after replanning at "
                                  f"({position[0]:.1f}, {position[1]:.1f})")
                        break
                if follow.done:
                    px4.land()
                    landing_started = loop.sim_time
                    if verbose:
                        print("    route complete -- landing", flush=True)
            px4.send_velocity_world(velocity[0], velocity[1], velocity[2], yaw)

        rendered = loop.step()
        if rendered:
            recorder.capture(stamp_s=loop.sim_time - armed_at)

        roll, pitch, _ = attitude_deg(vehicle)
        if max(abs(roll), abs(pitch)) > CRASH_TILT_DEG:
            tilted_since = tilted_since if tilted_since is not None else loop.sim_time
            if loop.sim_time - tilted_since >= CRASH_HOLD_S:
                outcome = OUTCOME_CRASHED
                detail = (f"tilted past {CRASH_TILT_DEG:.0f} deg for {CRASH_HOLD_S:.0f} s "
                          f"at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")
                break
        else:
            tilted_since = None

        if landing_started is None:
            # PX4 leaves offboard on its own whenever a failsafe fires, and then
            # ignores every setpoint. Ask for it back, but do not do so forever.
            if px4.in_offboard:
                offboard_lost_since = None
            else:
                offboard_lost_since = (offboard_lost_since if offboard_lost_since is not None
                                       else loop.sim_time)
                if loop.sim_time - last_offboard_request >= OFFBOARD_RETRY_S:
                    px4.set_offboard_mode()
                    last_offboard_request = loop.sim_time
                if loop.sim_time - offboard_lost_since >= OFFBOARD_LOST_S:
                    outcome = OUTCOME_OFFBOARD_LOST
                    detail = (f"PX4 left offboard mode (now {px4.main_mode}) and would not "
                              f"return for {OFFBOARD_LOST_S:.0f} s; PX4 said: "
                              f"{'; '.join(px4.drain_status_texts()[-4:]) or '(nothing)'}")
                    break

            if phase != PHASE_FOLLOW or (follow is not None and follow.turning):
                # Climbing, lining up, or deliberately stopped to rotate onto the
                # next leg. All three are stationary on purpose, and a stall
                # detector that cannot tell them from a wedged aircraft would end
                # every flight that has a sharp corner in it.
                progress_at = loop.sim_time
            elif math.dist(tuple(position[:3]), progress_mark) >= STALL_DISTANCE_M:
                progress_mark = (float(position[0]), float(position[1]), float(position[2]))
                progress_at = loop.sim_time
            elif loop.sim_time - progress_at >= STALL_WINDOW_S:
                outcome = OUTCOME_STALLED
                detail = (f"moved less than {STALL_DISTANCE_M:.1f} m in "
                          f"{STALL_WINDOW_S:.0f} s at ({position[0]:.1f}, "
                          f"{position[1]:.1f}, {position[2]:.1f}) -- it is not flying")
                break

            if loop.sim_time - armed_at > budget:
                outcome = OUTCOME_FLIGHT_TIMEOUT
                remaining = f"{follow.distance_to_goal:.1f} m" if follow else "the whole route"
                detail = (f"route not finished within its {budget:.0f} s budget "
                          f"({remaining} still to fly)")
                break
        else:
            if px4.on_ground and loop.sim_time - landing_started >= POST_LAND_S:
                outcome = OUTCOME_LANDED
                break
            if loop.sim_time - landing_started > LAND_TIMEOUT_S:
                outcome = OUTCOME_LAND_TIMEOUT
                detail = f"still airborne {LAND_TIMEOUT_S:.0f} s after the land command"
                break
            if loop.sim_time - landing_started >= LAND_TIMEOUT_S / 2:
                px4.land()  # a failsafe may have taken the aircraft out of LAND

        if verbose and loop.sim_time - last_status >= STATUS_EVERY_S:
            last_status = loop.sim_time
            speed = math.hypot(vehicle.state.linear_velocity[0],
                               vehicle.state.linear_velocity[1])
            print(f"    t={loop.sim_time - armed_at:6.1f}s {phase:<6s} "
                  f"pos=({position[0]:6.2f},{position[1]:6.2f},{position[2]:5.2f}) "
                  f"speed={speed:4.2f} yaw={math.degrees(_true_yaw(vehicle)):6.1f} "
                  f"left={follow.distance_to_goal if follow else 0.0:5.1f}m "
                  f"xte={follow.cross_track_error if follow else 0.0:4.2f} "
                  f"mode={px4.main_mode} frames={recorder.frames}", flush=True)

    position = vehicle.state.position
    goal_error = math.hypot(position[0] - flown_plan.goal.x, position[1] - flown_plan.goal.y)
    # A landing on the spot the aircraft never left is still a landing, and it
    # is not a flight to the goal. Judge the recording by where it actually
    # ended up, not by the state machine having run to the end.
    if outcome == OUTCOME_LANDED and goal_error > GOAL_TOLERANCE_M:
        outcome = OUTCOME_MISSED_GOAL
        detail = f"landed {goal_error:.1f} m from the goal"

    return EpisodeResult(
        outcome=outcome,
        frames=recorder.frames,
        duration_s=loop.sim_time - armed_at,
        route_remaining_m=follow.distance_to_goal if follow else flown_plan.path_length_m,
        cross_track_error_m=follow.cross_track_error if follow else 0.0,
        final_xy=(float(position[0]), float(position[1])),
        goal_error_m=goal_error,
        estimator_drift_m=px4.frame_drift(position),
        replans=replans,
        flown_plan=flown_plan,
        detail=detail,
        px4_messages=px4.drain_status_texts(),
    )


def settle_after_landing(loop, px4, descend_timeout_s: float = 40.0,
                         settle_s: float = 3.0, verbose: bool = True) -> None:
    """Get the aircraft on the ground and disarmed before the next episode.

    Never force-disarms an aircraft that is still flying. That looks like a
    tidy way to end a failed episode and is in fact a way to drop the airframe
    from cruise altitude: it lands on its side, the next episode's arming is
    refused on attitude, and the campaign is over. So an airborne aircraft is
    commanded to land and given time to do it, and the override is a last
    resort that says so out loud.

    The setpoint stream stops here, which also drops PX4 out of offboard mode --
    exactly what should happen between missions.

    Args:
        loop: The simulation loop to step.
        px4: The autopilot link.
        descend_timeout_s: Simulated seconds to allow for a descent.
        settle_s: Simulated seconds to let the airframe come to rest afterwards.
        verbose: Announce a forced disarm.
    """
    started = loop.sim_time
    while px4.armed and not px4.on_ground:
        if loop.sim_time - started > descend_timeout_s:
            if verbose:
                print(f"    WARNING: still airborne {descend_timeout_s:.0f} s after the "
                      f"land command -- force-disarming, the airframe will drop",
                      flush=True)
            px4.disarm(force=True)
            break
        px4.land()
        loop.run_for(1.0)

    px4.disarm()
    loop.run_for(settle_s)
    if px4.armed:
        px4.disarm(force=True)
        loop.run_for(1.0)
