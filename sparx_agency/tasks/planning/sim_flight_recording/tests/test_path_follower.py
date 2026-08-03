"""Continuous route following: the point is that the aircraft never stops.

These fly the follower against a perfect vehicle (whatever velocity is commanded
is achieved exactly) at 50 Hz. That is not a claim about the simulator -- it
isolates the guidance law, which is what decides whether the flight is smooth.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.common.types import Pose3D
from sparx_agency.tasks.planning.sim_flight_recording.path_follower import (
    FollowSpec, PathFollower, build_trajectory,
)

ALTITUDE = 1.5
# A route with a straight run, two right-angle corners and a doubling back.
ROUTE = [(3.0, 0.0, ALTITUDE, 0.0), (6.0, 0.0, ALTITUDE, 0.0),
         (9.0, 2.0, ALTITUDE, 0.0), (9.0, 6.0, ALTITUDE, 0.0),
         (6.0, 8.0, ALTITUDE, 0.0)]
START = Pose3D(0.0, 0.0, ALTITUDE, 0.0)


def _fly(spec: FollowSpec, route=ROUTE, dt: float = 0.02, max_steps: int = 20000):
    """Fly ``route`` with a perfect vehicle; return the flight log."""
    trajectory = build_trajectory(START, route, spec)
    follower = PathFollower(trajectory, spec, initial_yaw=START.yaw)
    follower.yaw = follower.initial_heading()

    position = np.array([START.x, START.y, START.z])
    yaw = follower.yaw
    speeds, yaw_rates, errors, path = [], [], [], [position.copy()]
    state = None
    for _ in range(max_steps):
        state = follower.update(position, yaw, (0.0, 0.0, 0.0), dt)
        if state.done or state.failed:
            break
        velocity = np.array(state.velocity)
        position = position + velocity * dt
        yaw_rates.append(abs(state.yaw - yaw) / dt)
        yaw = state.yaw
        speeds.append(float(np.linalg.norm(velocity[:2])))
        errors.append(state.cross_track_error)
        path.append(position.copy())
    return {"state": state, "speed": np.array(speeds), "yaw_rate": np.array(yaw_rates),
            "cte": np.array(errors), "path": np.array(path),
            "time": len(speeds) * dt}


def test_the_route_is_flown_to_the_end():
    flight = _fly(FollowSpec())
    assert flight["state"].done
    assert not flight["state"].failed
    end = flight["path"][-1]
    assert math.dist(end[:2], ROUTE[-1][:2]) < 1.0


def _cruising(speed, spec):
    """The middle of a flight: past the ramp from rest, before the final approach."""
    ramp = int(1.0 / 0.02)                      # the first second, accelerating
    approach = int(len(speed) * 0.15)
    return speed[ramp:-approach]


def test_the_aircraft_never_stops_mid_route():
    """The whole reason this exists: no deceleration onto every waypoint.

    The floor is 0.55x rather than 1.0x because the tracker deliberately eases
    off through tight corners (``corner_speed_factor``), and ``compute_velocity_3d``
    additionally throttles forward speed by how well the aircraft's actual
    heading matches the bearing to the lookahead point (yaw-then-fly, not
    holonomic strafing) -- both are smoothing, not stopping. Measured worst
    case on this route is 0.63x.
    """
    spec = FollowSpec()
    cruising = _cruising(_fly(spec)["speed"], spec)
    assert cruising.min() > 0.55 * spec.cruise_speed, "it stopped somewhere mid-route"
    assert cruising.mean() > 0.85 * spec.cruise_speed


def test_speed_holds_near_cruise_rather_than_sawtoothing():
    spec = FollowSpec()
    cruising = _cruising(_fly(spec)["speed"], spec)
    assert cruising.std() < 0.1 * spec.cruise_speed


def test_a_faster_cruise_speed_actually_flies_faster():
    slow = _fly(FollowSpec(cruise_speed=0.6, max_speed=0.9))
    fast = _fly(FollowSpec(cruise_speed=1.5, max_speed=1.9))
    assert fast["time"] < slow["time"] * 0.75
    assert fast["speed"].mean() > slow["speed"].mean() * 1.7


def test_the_heading_never_turns_faster_than_its_limit():
    """This is the knob that decides how fast the world spins in the imagery."""
    spec = FollowSpec(max_yaw_rate=0.3)
    yaw_rate = _fly(spec)["yaw_rate"]
    assert yaw_rate.max() <= spec.max_yaw_rate + 1e-6


def test_a_lower_yaw_rate_limit_really_does_rotate_more_slowly():
    quick = _fly(FollowSpec(max_yaw_rate=0.8))["yaw_rate"]
    calm = _fly(FollowSpec(max_yaw_rate=0.2))["yaw_rate"]
    assert calm.max() < quick.max()
    assert calm.mean() < quick.mean()


def test_the_aircraft_stays_on_the_smoothed_route():
    spec = FollowSpec()
    cte = _fly(spec)["cte"]
    assert cte.max() < spec.path_tolerance
    assert cte.mean() < 0.5


def test_the_smoothed_route_does_not_bulge_far_outside_its_corners():
    """The spline eats into the planner's standoff; it must not eat much."""
    spec = FollowSpec()
    corners = np.array([[START.x, START.y]] + [[w[0], w[1]] for w in ROUTE])
    samples = build_trajectory(START, ROUTE, spec).sample_by_time(0.05)
    curve = np.array([[p.x, p.y] for p in samples])

    # Distance from each spline sample to the nearest point on the corner polyline.
    worst = 0.0
    for point in curve:
        segs = []
        for a, b in zip(corners, corners[1:]):
            ab = b - a
            t = np.clip(np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
            segs.append(np.linalg.norm(point - (a + t * ab)))
        worst = max(worst, min(segs))
    # The planner's default standoff is 0.6 m; the spline may not spend it all.
    assert worst < 0.4, f"spline bulges {worst:.2f} m outside the planned corners"


def test_a_bigger_tangent_scale_bulges_further_outside_the_corners():
    """The knob that trades curve prettiness against obstacle clearance."""
    def bulge(tangent_scale):
        spec = FollowSpec(tangent_scale=tangent_scale)
        corners = np.array([[START.x, START.y]] + [[w[0], w[1]] for w in ROUTE])
        samples = build_trajectory(START, ROUTE, spec).sample_by_time(0.05)
        worst = 0.0
        for point in np.array([[p.x, p.y] for p in samples]):
            segs = []
            for a, b in zip(corners, corners[1:]):
                ab = b - a
                t = np.clip(np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
                segs.append(np.linalg.norm(point - (a + t * ab)))
            worst = max(worst, min(segs))
        return worst

    assert bulge(0.15) < bulge(0.3) < bulge(0.5)


def test_the_initial_heading_faces_the_way_the_route_sets_off():
    spec = FollowSpec()
    follower = PathFollower(build_trajectory(START, ROUTE, spec), spec, initial_yaw=3.0)
    # ROUTE leaves along +x from the origin.
    assert abs(follower.initial_heading()) < math.radians(20)


def test_a_two_point_route_still_flies():
    """The planner can legitimately emit a single straight leg."""
    flight = _fly(FollowSpec(), route=[(4.0, 0.0, ALTITUDE, 0.0)])
    assert flight["state"].done


def test_an_aircraft_dragged_off_the_route_is_reported_not_hidden():
    spec = FollowSpec(path_tolerance=1.0)
    trajectory = build_trajectory(START, ROUTE, spec)
    follower = PathFollower(trajectory, spec, initial_yaw=0.0)

    state = follower.update((0.0, 40.0, ALTITUDE), 0.0, (0.0, 0.0, 0.0), 0.02)

    assert state.failed
    assert "diverged" in state.reason


def test_the_on_the_spot_turn_is_brisker_than_the_cruise_rotation():
    """Different jobs: one shapes the footage, the other is dead time before it."""
    spec = FollowSpec()
    assert spec.turn_yaw_rate > spec.max_yaw_rate


def test_the_cruise_yaw_rate_is_gentle_enough_for_usable_footage():
    """Measured: 45 deg/s whipped the camera; this is the calm end of the range."""
    assert math.degrees(FollowSpec().max_yaw_rate) <= 20.0
