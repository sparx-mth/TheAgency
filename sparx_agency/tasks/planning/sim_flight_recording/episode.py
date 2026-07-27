"""Fly one planned mission end to end, autonomously, and record it.

Arm, take off, track the planned route, land at the goal. That is one episode
and one recording. A campaign is this, repeatedly, with a fresh goal each time
(:mod:`collect`).

Nothing here decides *where* to fly -- :mod:`episode_plan` did that, off a
surveyed map, before the simulator was involved. This module only executes, and
it is written on the assumption that a data-collection run is unattended: every
way a flight can fail has an explicit detection and a named outcome, because a
campaign that silently records ninety minutes of a drone lying against a wall is
worse than one that stops.

Three behaviours are worth calling out because they shape the data:

* **The aircraft is flown on velocity, not position.** :func:`guidance_velocity`
  turns the world-frame error between the *simulator's exact position* and the
  current waypoint into a clamped velocity command, and PX4 is left to be an
  inner-loop velocity controller. PX4's offboard *position* path was tried first
  and did not work here: given a setpoint one metre away, in a healthy offboard
  mode, with no failsafe and its own estimate tracking truth to 30 cm, it closed
  the gap at one centimetre per second until the flight timed out. Closing the
  loop on ground truth also makes the aircraft's *true* path the thing that
  converges, so PX4's estimator drift stops mattering -- and it is how every
  other controller in this repo flies a drone.
* **The aircraft looks along the leg it is flying**, using the heading the
  planner attached to each waypoint. A live "point at the waypoint" heading was
  tried first and is a trap: the commanded yaw depends on the position error,
  the position response depends on the yaw, and the two close a loop. The
  aircraft flew a stable 1.5 m circle around its waypoint for a hundred seconds
  -- yaw rotating at a steady 19 deg/s, body-frame velocity pinned at 0.37 m/s
  to the right, the commanded heading permanently 15 deg ahead of the actual
  one, chasing its own tail. A per-leg constant has no such path, and points
  the camera at the same thing anyway.
* **Recording starts at arming and ends at touchdown**, so a recording opens
  with the climb and closes on the ground. The frames before arming are a
  stationary drone on the floor, which is not navigation data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from sparx_agency.tasks.planning.sim_flight_recording.px4_params import INDOOR_LIMITS
from sparx_agency.tasks.planning.sim_flight_recording.waypoint_mission import (
    ARRIVAL_RADIUS_M, FINAL_RADIUS_M, WaypointMission,
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
# Guidance: a proportional law on the world-frame position error, clamped to the
# same envelope PX4 is configured for. Kept slower than PX4's own inner loops so
# the aircraft is never asked for a velocity step it cannot make smoothly.
GUIDANCE_GAIN = 0.8         # 1/s
# Never command a speed so small that the autopilot ignores it. A pure
# proportional law tapers to nothing near the waypoint, and below roughly this
# the aircraft simply does not respond -- it hovers a couple of centimetres per
# second short of where it was sent. This is the same "minimum decisive
# command" the FALCON followers use for the same reason.
MIN_GUIDANCE_SPEED = 0.3    # m/s
CLIMB_GAIN = 1.0            # 1/s
MAX_CLIMB_RATE = 0.8        # m/s
TAKEOFF_TOLERANCE_M = 0.15  # how close to cruise altitude counts as airborne
TAKEOFF_TIMEOUT_S = 30.0
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
GOOD_OUTCOMES = (OUTCOME_LANDED,)


@dataclass
class EpisodeResult:
    """What one episode did.

    Attributes:
        outcome: One of the ``OUTCOME_*`` constants. Only
            :data:`OUTCOME_LANDED` is a clean flight.
        frames: Frames recorded.
        duration_s: Simulated seconds from arming to the end of the recording.
        waypoints_reached: How many of the plan's waypoints were actually
            reached, excluding any that timed out.
        waypoints_skipped: How many timed out. Non-zero means the aircraft cut
            a corner somewhere, which is worth filtering on even when the
            flight otherwise succeeded.
        final_xy: Where the aircraft ended up.
        goal_error_m: Horizontal distance from the goal at the end.
        estimator_drift_m: How far PX4's position estimate moved relative to
            ground truth over the flight. Tens of centimetres is healthy;
            metres means the aircraft was being commanded to the wrong place.
        detail: Human-readable explanation, empty on a clean flight.
        px4_messages: ``STATUSTEXT`` lines PX4 emitted, which is where a refused
            arming says why.
    """

    outcome: str
    frames: int = 0
    duration_s: float = 0.0
    waypoints_reached: int = 0
    waypoints_skipped: int = 0
    final_xy: Tuple[float, float] = (0.0, 0.0)
    goal_error_m: float = 0.0
    estimator_drift_m: float = 0.0
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


def guidance_velocity(position, target, cruise_speed: float,
                      arrival_radius_m: float = 0.0) -> Tuple[float, float, float]:
    """World-frame velocity that flies ``position`` toward ``target``.

    Proportional, clamped, and computed from the simulator's exact position --
    so the aircraft's *true* path is what converges, whatever PX4's estimate is
    doing. The horizontal and vertical axes are limited separately because the
    airframe's climb authority and its cruise speed are different numbers.

    Args:
        position: True world-frame ``(x, y, z)``.
        target: World-frame ``(x, y, z)`` to fly to.
        cruise_speed: Horizontal speed ceiling, m/s.
        arrival_radius_m: Inside this the horizontal command is zero, so the
            aircraft settles instead of hunting around the waypoint.

    Returns:
        ``(vx, vy, vz)`` in the world frame, m/s, ``vz`` positive up.
    """
    dx, dy = target[0] - position[0], target[1] - position[1]
    distance = math.hypot(dx, dy)
    if distance > arrival_radius_m:
        speed = min(max(GUIDANCE_GAIN * distance, MIN_GUIDANCE_SPEED), cruise_speed)
        vx, vy = dx / distance * speed, dy / distance * speed
    else:
        # Inside the acceptance radius the job is done; commanding a residual
        # speed here only pushes the aircraft past the waypoint and back.
        vx = vy = 0.0
    climb = CLIMB_GAIN * (target[2] - position[2])
    return vx, vy, max(-MAX_CLIMB_RATE, min(MAX_CLIMB_RATE, climb))


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


def fly_episode(loop, px4, adapter, plan, recorder, verbose: bool = True) -> EpisodeResult:
    """Arm, fly ``plan``, land at its goal, recording throughout.

    Args:
        loop: The :class:`~sim_loop.SimLoop` driving the simulation.
        px4: The autopilot link, already booted and configured.
        adapter: The :class:`PegasusIrisVehicle` being flown.
        plan: The :class:`~episode_plan.EpisodePlan` to execute.
        recorder: A :class:`~flight_session.FlightRecorder` to capture into.
        verbose: Print a status line every few simulated seconds.

    Returns:
        The :class:`EpisodeResult`. A recording is written either way -- a
        partial recording of a flight that hit something is still worth having,
        and ``meta.json`` says which it is.
    """
    vehicle = adapter.vehicle
    cruise_altitude = plan.waypoints[0][2]

    start_position = vehicle.state.position
    failure = arm_for_offboard(loop, px4, start_position, plan.start.yaw, cruise_altitude)
    if failure is not None:
        return EpisodeResult(
            outcome=OUTCOME_ARM_TIMEOUT, detail=failure,
            final_xy=(float(start_position[0]), float(start_position[1])),
            goal_error_m=math.hypot(start_position[0] - plan.goal.x,
                                    start_position[1] - plan.goal.y),
            px4_messages=px4.drain_status_texts(),
        )
    if verbose:
        offset = px4.frame_offset
        print(f"    armed into offboard; PX4 frame offset "
              f"({offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}) m, "
              f"heading bias {math.degrees(px4.heading_bias):+.1f} deg", flush=True)

    mission = WaypointMission(plan.waypoints)
    armed_at = loop.sim_time
    budget = flight_budget_s(plan.path_length_m)
    cruise_speed = cruise_speed_hint()
    takeoff_xy = (float(start_position[0]), float(start_position[1]))
    climbed = False
    yaw = plan.start.yaw
    tilted_since = None
    offboard_lost_since = None
    last_offboard_request = loop.sim_time
    landing_started = None
    last_status = loop.sim_time
    progress_mark = (float(start_position[0]), float(start_position[1]),
                     float(start_position[2]))
    progress_at = loop.sim_time
    outcome = None
    detail = ""

    while True:
        position = vehicle.state.position
        target = mission.current()

        if landing_started is None:
            reference = target if target is not None else (
                plan.goal.x, plan.goal.y, cruise_altitude, yaw)
            yaw = reference[3]
            # Climb straight up before setting off. Translating while still near
            # the floor is how an aircraft clips whatever it took off next to --
            # so the climb target is the take-off point itself, which holds the
            # horizontal position rather than merely not commanding one. Simply
            # commanding zero horizontal velocity is not the same thing: velocity
            # control has no position feedback, and the aircraft was measured
            # drifting three metres sideways during a five-second climb.
            if position[2] < cruise_altitude - TAKEOFF_TOLERANCE_M and not climbed:
                velocity = guidance_velocity(
                    position, (takeoff_xy[0], takeoff_xy[1], cruise_altitude),
                    cruise_speed)
                if loop.sim_time - armed_at > TAKEOFF_TIMEOUT_S:
                    outcome = OUTCOME_FLIGHT_TIMEOUT
                    detail = (f"never reached its {cruise_altitude:.1f} m cruise altitude "
                              f"within {TAKEOFF_TIMEOUT_S:.0f} s (stuck at "
                              f"{position[2]:.2f} m)")
                    break
            else:
                climbed = True
                velocity = guidance_velocity(
                    position, reference, cruise_speed,
                    arrival_radius_m=(FINAL_RADIUS_M if mission.on_final
                                      else ARRIVAL_RADIUS_M))
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

            if not climbed:
                progress_at = loop.sim_time
            elif target is not None and math.dist(
                    tuple(position[:3]), tuple(target[:3])) <= 2.0 * ARRIVAL_RADIUS_M:
                progress_at = loop.sim_time   # settling onto a waypoint, not stalled
            elif math.dist(tuple(position[:3]), progress_mark) >= STALL_DISTANCE_M:
                progress_mark = (float(position[0]), float(position[1]), float(position[2]))
                progress_at = loop.sim_time
            elif loop.sim_time - progress_at >= STALL_WINDOW_S:
                outcome = OUTCOME_STALLED
                detail = (f"moved less than {STALL_DISTANCE_M:.1f} m in "
                          f"{STALL_WINDOW_S:.0f} s at ({position[0]:.1f}, "
                          f"{position[1]:.1f}, {position[2]:.1f}) -- it is not flying")
                break

            if climbed and mission.update(position, loop.sim_time) and verbose:
                print(f"    waypoint {mission.index}/{len(plan.waypoints)} at "
                      f"({position[0]:.1f}, {position[1]:.1f})", flush=True)
            if mission.finished:
                px4.land()
                landing_started = loop.sim_time
                if verbose:
                    print("    route complete -- landing", flush=True)
            elif loop.sim_time - armed_at > budget:
                outcome = OUTCOME_FLIGHT_TIMEOUT
                detail = (f"route not finished within its {budget:.0f} s budget "
                          f"(reached {mission.index}/{len(plan.waypoints)} waypoints)")
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
            aim = target if target is not None else (plan.goal.x, plan.goal.y, 0.0, 0.0)
            print(f"    t={loop.sim_time - armed_at:6.1f}s "
                  f"pos=({position[0]:6.2f},{position[1]:6.2f},{position[2]:5.2f}) "
                  f"->({aim[0]:6.2f},{aim[1]:6.2f}) "
                  f"v=({vehicle.state.linear_velocity[0]:5.2f},"
                  f"{vehicle.state.linear_velocity[1]:5.2f}) "
                  f"rpy=({roll:5.1f},{pitch:5.1f},{math.degrees(_true_yaw(vehicle)):6.1f}) "
                  f"drift={px4.frame_drift(position):4.2f} "
                  f"wp={mission.index}/{len(plan.waypoints)} mode={px4.main_mode} "
                  f"frames={recorder.frames}", flush=True)

    position = vehicle.state.position
    goal_error = math.hypot(position[0] - plan.goal.x, position[1] - plan.goal.y)
    # A mission "finishes" when its last waypoint is reached OR times out, and a
    # landing on the spot the aircraft never left is still a landing. Neither is
    # a flight to the goal. Judge the recording by where it actually ended up,
    # not by the state machine having run to the end. Skipped waypoints are
    # reported separately rather than failing the episode: a flight that took
    # too long over one corner but still arrived is good data, while a flight
    # that cut a corner is something the caller may want to filter on.
    if outcome == OUTCOME_LANDED and goal_error > GOAL_TOLERANCE_M:
        outcome = OUTCOME_MISSED_GOAL
        detail = f"landed {goal_error:.1f} m from the goal"

    return EpisodeResult(
        outcome=outcome,
        frames=recorder.frames,
        duration_s=loop.sim_time - armed_at,
        waypoints_reached=mission.index - mission.skipped,
        waypoints_skipped=mission.skipped,
        final_xy=(float(position[0]), float(position[1])),
        goal_error_m=goal_error,
        estimator_drift_m=px4.frame_drift(position),
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


def cruise_speed_hint() -> float:
    """The cruise speed PX4 is configured for, m/s. For time estimates."""
    return float(INDOOR_LIMITS["MPC_XY_CRUISE"])
