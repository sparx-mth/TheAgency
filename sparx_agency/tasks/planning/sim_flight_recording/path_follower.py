"""Fly a planned route as one continuous curve instead of a list of stops.

A route from the planner is a handful of corner points. Flying it literally --
aim at the next corner, decelerate onto it, accept it, aim at the one after --
produces stop-and-go motion: the aircraft pauses at every waypoint, the camera
swings, and the recording is a sequence of lurches rather than a flight. That is
poor demonstration data whatever the geometry says.

This composes two things the repo already has into a continuous follower:

* :class:`~sparx_agency.core.planning.smoothers.hermite.HermiteSmoother3D` turns
  the corner points into a **G1-continuous cubic Hermite spline**, resampled
  densely and parameterised by arc length at a constant nominal speed. Corners
  become curves.
* :class:`~sparx_agency.core.planning.trackers.pure_pursuit.PurePursuitTracker3D`
  chases a **carrot** running ahead of the aircraft along that spline. Because
  the carrot never stops moving until the very end, the commanded speed never
  tapers to zero mid-route. Its lookahead shortens on tight curves and lengthens
  with speed, which is what keeps a fast pass through a doorway from cutting the
  corner.

Both are the same components the real drones fly, which is the point: the
simulated expert is demonstrating the stack's own idea of a good flight.

The heading is **integrated from the tracker's rate-limited yaw command**, not
snapped to the path tangent. A rate limit is the whole reason the camera pans
instead of whipping, and it is the one knob that decides how fast the world
rotates in the recorded imagery.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from sparx_agency.core.common.types import (
    KinematicLimits, Path3D, Pose3D, State3D, Trajectory, Twist3D, normalize_angle,
)
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest3D
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest
from sparx_agency.core.planning.smoothers.hermite import HermiteSmoother3D
from sparx_agency.core.planning.smoothers.hermite.params import HermiteParams3D
from sparx_agency.core.planning.trackers.pure_pursuit import PurePursuitTracker3D
from sparx_agency.core.planning.trackers.pure_pursuit.params import PurePursuitParams3D


@dataclass(frozen=True)
class FollowSpec:
    """How the aircraft should fly a route.

    Attributes:
        cruise_speed: Speed held along the route, m/s. The whole flight runs at
            this except the final approach.
        max_speed: Ceiling the tracker may command, m/s. Must leave headroom
            under PX4's own ``MPC_XY_VEL_MAX`` or the autopilot clips the
            command and the aircraft quietly falls behind the carrot.
        min_speed: Floor, m/s. A pure proportional taper dies in the autopilot's
            deadband; this stops the aircraft creeping the last metre.
        max_climb_rate: Vertical speed ceiling, m/s.
        max_yaw_rate: How fast the aircraft may rotate *while flying the route*,
            rad/s. **This is the knob that sets how fast the world spins in the
            recorded imagery**, and the only one of the two that shapes the
            data. Raised from 0.25 to 0.6: at 0.25 a 90-degree corner needed 6.4
            seconds of yawing, and the aircraft flew a wide arc through all of
            it. Smooth footage is not worth missing a doorway.
        turn_yaw_rate: How fast it may rotate while stationary, lining up on the
            route before setting off, rad/s. Deliberately faster: the aircraft
            is holding position and going nowhere, so every second spent here is
            dead time in the recording rather than smooth footage. At the cruise
            rate a 180-degree line-up took thirteen seconds, which on a short
            flight was most of it.
        base_lookahead: Carrot distance at rest, m. Larger cuts corners and
            flies smoother; smaller tracks the spline more tightly. Shortened
            from 1.2 to 0.8 (and the ceiling from 2.5 to 1.6) because cutting
            corners is exactly what stops the aircraft getting through a door.
        max_lookahead: Ceiling on the speed-scaled carrot distance, m.
        slow_down_distance: Distance from the end over which the aircraft
            decelerates, m. The only place the speed profile is not flat.
        goal_tolerance: How close to the end of the spline counts as arrived, m.
        path_tolerance: Cross-track error at which the follower gives up, m.
            A genuine divergence detector: the aircraft is no longer on the
            route it was cleared to fly.
        corner_speed_factor: How much to slow for curvature. 0 holds cruise
            speed through every corner.
        tangent_scale: Spline tangent magnitude, and the one parameter here that
            can fly the aircraft into a wall. The smoothed curve cuts *outside*
            the planner's corner points, and the planner's obstacle standoff is
            the entire budget for that. Measured on right-angle corners the
            worst-case bulge is close to ``1.8 * tangent_scale`` metres, so the
            default of 0.2 spends about 0.36 m of a 0.6 m standoff. Raising it
            makes for prettier curves and less clearance; do not raise it
            without raising ``EpisodeSpec.inflate_radius_m`` too. Lowered from
            0.2 to 0.12 to tighten corners: the smoothed bulge was spending most
            of the standoff, which is what put the aircraft on the wrong side of
            a doorframe.
        yaw_lookahead: How far along the route the heading aims, m. Must exceed
            ``max_lookahead`` or the heading is just chasing the carrot again --
            which is what made the flown path weave from side to side, since the
            aircraft only ever pointed at the next point and ignored the rest of
            the route.
        stop_turn_rad: Heading error at which the aircraft stops translating and
            rotates on the spot. This is the "sometimes you can stop, turn and
            continue" behaviour: a continuous flight is better, but not at the
            price of a corner taken so wide it misses the gap. 40 degrees leaves
            ordinary curves entirely alone -- they never reach it -- and catches
            real corners and reversals.
        resume_turn_rad: Heading error at which it starts translating again.
            Below ``stop_turn_rad`` on purpose: a single threshold chatters.
    """

    cruise_speed: float = 1.2
    max_speed: float = 1.5
    min_speed: float = 0.3
    max_climb_rate: float = 0.8
    max_yaw_rate: float = 0.6
    turn_yaw_rate: float = 0.7
    base_lookahead: float = 0.8
    max_lookahead: float = 1.6
    slow_down_distance: float = 1.5
    goal_tolerance: float = 0.6
    path_tolerance: float = 2.5
    corner_speed_factor: float = 0.8
    tangent_scale: float = 0.12
    yaw_lookahead: float = 3.5
    stop_turn_rad: float = math.radians(40.0)
    resume_turn_rad: float = math.radians(12.0)


@dataclass
class FollowState:
    """What the follower did on one step.

    Attributes:
        velocity: World-frame ``(vx, vy, vz)`` to command, m/s.
        yaw: World-frame heading to command, radians CCW from +X. Integrated
            from the rate-limited yaw command, so it never steps.
        distance_to_goal: Along-path distance remaining, m.
        cross_track_error: How far off the spline the aircraft is, m.
        speed: Commanded speed, m/s.
        turning: The follower is holding position and rotating, because the
            heading is too far from where the route goes to fly it accurately.
            Deliberate, and the reason a stall detector must not treat a
            stationary aircraft as a wedged one.
        done: The end of the route has been reached.
        failed: The aircraft diverged from the route.
        reason: Why it stopped, empty while flying.
    """

    velocity: Tuple[float, float, float]
    yaw: float
    distance_to_goal: float
    cross_track_error: float
    speed: float
    turning: bool = False
    done: bool = False
    failed: bool = False
    reason: str = ""


def build_trajectory(start: Pose3D, waypoints, spec: FollowSpec) -> Trajectory:
    """Smooth a planned route into a dense, arc-length-parameterised trajectory.

    Args:
        start: Where the aircraft is starting from, at cruise altitude. Included
            as the first spline point so the curve begins under the aircraft
            rather than at the first corner.
        waypoints: The planner's ``(x, y, z, yaw)`` tuples. Only the position is
            used -- the heading comes from the path tangent, which after
            smoothing is a better answer than the per-leg bearing.
        spec: Flight parameters.

    Returns:
        A :class:`Trajectory` sampled densely enough for the tracker.

    Raises:
        ValueError: If fewer than two distinct points survive.
    """
    points = [start] + [Pose3D(w[0], w[1], w[2], 0.0) for w in waypoints]
    path = Path3D(points=tuple(points), frame_id="world")
    smoother = HermiteSmoother3D(HermiteParams3D(
        nominal_speed_xy=spec.cruise_speed,
        nominal_speed_z=spec.max_climb_rate,
        tangent_scale=spec.tangent_scale,
        zero_endpoint_velocity=False,
    ))
    return smoother.smooth(SmootherRequest3D(path=path))


class PathFollower:
    """Chases a carrot along a smoothed route, producing world-frame velocity.

    Args:
        trajectory: The smoothed route, from :func:`build_trajectory`.
        spec: Flight parameters.
        initial_yaw: The aircraft's heading when following starts, which the
            integrated yaw command continues from.
    """

    def __init__(self, trajectory: Trajectory, spec: FollowSpec, initial_yaw: float):
        self.trajectory = trajectory
        self.spec = spec
        self.yaw = float(initial_yaw)
        self._tracker = PurePursuitTracker3D(
            PurePursuitParams3D(
                base_lookahead=spec.base_lookahead,
                min_lookahead=max(spec.base_lookahead * 0.5, 0.3),
                max_lookahead=spec.max_lookahead,
                cruise_speed=spec.cruise_speed,
                min_speed=spec.min_speed,
                max_speed=spec.max_speed,
                max_speed_z=spec.max_climb_rate,
                curvature_speed_factor=spec.corner_speed_factor,
                slow_down_distance=spec.slow_down_distance,
                goal_tolerance=spec.goal_tolerance,
                path_tolerance=spec.path_tolerance,
                max_yaw_rate=spec.max_yaw_rate,
                yaw_lookahead=spec.yaw_lookahead,
                stop_turn_rad=spec.stop_turn_rad,
                resume_turn_rad=spec.resume_turn_rad,
            ),
            default_limits=KinematicLimits(
                max_speed_xy=spec.max_speed,
                max_speed_z=spec.max_climb_rate,
                max_yaw_rate=spec.max_yaw_rate,
            ),
        )

    def initial_heading(self) -> float:
        """The direction the route sets off in, for turning to face it first.

        Turning on the spot before setting off, rather than while under way, is
        what keeps the recorded imagery from panning hard across the first few
        metres of every flight -- and what stops the first corner being taken
        wide before the aircraft has even lined up.

        Measured over :attr:`FollowSpec.yaw_lookahead` of route, not over the
        first 30 cm of spline. The first 30 cm of a smoothed curve points wherever
        the tangent happens to leave the aircraft, which on a route that turns
        immediately is not the direction of travel at all: one flight lined itself
        up to within 8 degrees of *that* and then still swung wide, because the
        route it was actually about to fly went somewhere else.
        """
        points = self.trajectory.sample_by_time(0.1)
        if len(points) < 2:
            return self.yaw
        origin = points[0]
        travelled, previous = 0.0, origin
        for point in points[1:]:
            travelled += math.hypot(point.x - previous.x, point.y - previous.y)
            previous = point
            if travelled >= self.spec.yaw_lookahead:
                break
        dx, dy = previous.x - origin.x, previous.y - origin.y
        if math.hypot(dx, dy) < 0.3:
            return self.yaw
        return math.atan2(dy, dx)

    def update(self, position, yaw: float, velocity, dt: float) -> FollowState:
        """Advance the follower one step.

        Args:
            position: The aircraft's true world ``(x, y, z)``.
            yaw: Its true heading, radians.
            velocity: Its true world-frame ``(vx, vy, vz)``.
            dt: Seconds since the last call, for integrating the yaw command.

        Returns:
            The :class:`FollowState` to act on.
        """
        state = State3D(
            pose=Pose3D(float(position[0]), float(position[1]), float(position[2]),
                        float(yaw)),
            twist=Twist3D(float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0),
        )
        result = self._tracker.step(
            TrackerRequest(state=state, trajectory=self.trajectory, t=0.0))

        command = result.command
        meta = result.metadata
        # Integrate the tracker's rate-limited yaw rate rather than jumping to
        # its target heading: the rate limit is the whole point, and a stepped
        # setpoint would hand the slew back to the autopilot's own limiter.
        self.yaw = normalize_angle(self.yaw + command.yaw_rate * dt)

        return FollowState(
            velocity=(command.x, command.y, command.z),
            yaw=self.yaw,
            distance_to_goal=float(meta.get("dist_to_goal", 0.0)),
            cross_track_error=float(meta.get("cross_track_error", 0.0)),
            speed=float(meta.get("speed_cmd", 0.0)),
            turning=bool(meta.get("turning", False)),
            done=bool(meta.get("done", False)),
            failed=bool(meta.get("failed", False)),
            reason=str(meta.get("reason", "")),
        )
