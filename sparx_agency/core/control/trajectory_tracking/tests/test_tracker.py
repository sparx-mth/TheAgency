"""The outer loop, flown closed-loop against an airframe with lag.

Unit-testing a controller by inspecting one tick's output says almost nothing --
the interesting properties are what happens over a hundred ticks against
something that does not obey instantly. So most of these tests fly a simple
second-order airframe: a body whose acceleration chases the command through a
first-order lag, which is what the attitude and rate loops underneath look like
from the outside.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.control.trajectory_tracking import (
    TrajectoryTracker, TrajectoryTrackerParams,
)
from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory

DT = 0.004


class LaggingAirframe:
    """A point mass whose acceleration reaches the command through a lag.

    ``tau`` stands in for everything below the outer loop -- the attitude loop
    building a tilt, the rotors spinning up. Zero lag would make any controller
    look perfect, which is exactly the mistake the real stack was making.
    """

    def __init__(self, position, velocity=None, tau=0.15):
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.tau = float(tau)

    def step(self, command, dt):
        alpha = dt / (self.tau + dt)
        self.acceleration += alpha * (np.asarray(command, dtype=float) - self.acceleration)
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt


def _spline(points, knot_span=0.4, start_time_s=0.0, traj_id=1, yaws=None):
    """Build a trajectory from control points, the way FALCON transmits one."""
    knots = [(-3 + i) * knot_span for i in range(len(points) + 4)]
    return BsplineTrajectory.from_falcon(
        3, knots, points, yaws if yaws is not None else [0.0] * 6,
        knot_span, start_time_s, traj_id)


def _line(length=16, spacing=0.4, **kwargs):
    return _spline([[i * spacing, 0.0, 1.4] for i in range(length)], **kwargs)


def _corner(knot_span=0.35, **kwargs):
    """East along y=0, then a hard left turn north."""
    points = ([[i * 0.5, 0.0, 1.4] for i in range(7)]
              + [[3.0, j * 0.5, 1.4] for j in range(1, 8)])
    return _spline(points, knot_span=knot_span, **kwargs)


def _at_start(trajectory):
    """The curve's own first point.

    A cubic B-spline does not pass through its first control point, so an
    aircraft placed at ``control_points[0]`` starts a few tens of centimetres
    off the plan. Every closed-loop test here wants to measure tracking, not a
    fixture's initial transient.
    """
    start = trajectory.sample(0.0)
    return [start.x, start.y, start.z]


def _fly(tracker, trajectory, airframe, seconds, start_s=0.0, follow=True):
    """Run the loop, returning the commands issued and the path actually flown."""
    tracker.set_trajectory(trajectory)
    commands = []
    flown = []
    now = start_s
    for _ in range(int(seconds / DT)):
        command = tracker.update(airframe.position, airframe.velocity, 0.0, DT, now,
                                 follow=follow)
        airframe.step(command.acceleration(), DT)
        commands.append(command)
        flown.append(airframe.position.copy())
        now += DT
    return commands, flown


def _worst_distance_from_curve(trajectory, flown, samples=600):
    """Greatest distance from the flown path to the planned curve, by dense scan.

    Measured independently of the controller's own diagnostics, because the two
    control modes under comparison report that number by different routes -- one
    projects, the other decomposes a time-indexed error vector -- and comparing
    a controller's opinion of itself against another's is not a comparison.
    """
    curve = np.array([trajectory.position_at(float(t))
                      for t in np.linspace(0.0, trajectory.duration, samples)])
    return max(float(np.min(np.linalg.norm(curve - point, axis=1))) for point in flown)


def test_it_tracks_a_straight_line_to_within_centimetres():
    """The baseline. An airframe with lag, on a plan it can fly, stays on it.

    Handed over at the plan's own initial state, which is what happens for real:
    FALCON plans its first curve from the aircraft's measured position and
    velocity, so there is no step to absorb at the start. Starting from rest
    against a curve that begins at a metre per second measures the fixture's
    step response, not the controller.
    """
    trajectory = _line()
    start = trajectory.sample(0.0)
    airframe = LaggingAirframe([start.x, start.y, start.z],
                               velocity=[start.vx, start.vy, start.vz])
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    commands, _ = _fly(tracker, trajectory, airframe, trajectory.duration)
    errors = [c.position_error_m for c in commands]
    assert max(errors) < 0.05
    assert sum(errors) / len(errors) < 0.02


def test_the_schedule_catch_up_settles_rather_than_biasing_the_speed():
    """A lagging aircraft catches up and then stops catching up.

    The catch-up term exists because projection throws the plan's timing away.
    It has to converge: a term that keeps pushing once the schedule is recovered
    would fly the whole route fast and overrun the end of every trajectory,
    which is exactly what a lookahead does and why there is not one.

    What is asserted is *convergence*, not a particular residual. The residual
    is a deliberate trade and it moved once: ``max_catchup_speed`` went
    0.5 -> 0.15 because on a slow simulator the lag is largely an
    artefact of the clock rather than a real deficit, and chasing it flew the
    aircraft half again as fast as the route was cleared for. A tenth of a metre
    behind on a 1.0 m/s plan is a tenth of a second late, which is the benign
    kind of wrong -- and it costs nothing laterally, because the reference is
    the NEAREST point on the curve, not the one the schedule names.
    """
    trajectory = _line()
    start = _at_start(trajectory)
    airframe = LaggingAirframe([start[0] - 0.5, start[1], start[2]])
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    commands, _ = _fly(tracker, trajectory, airframe, trajectory.duration)
    tail = commands[-len(commands) // 4:]
    # Bounded, and no longer growing -- the second half of the tail is not worse
    # than the first, which is what "settles" means and what a term that kept
    # pushing would fail.
    assert max(abs(c.along_track_lag_m) for c in tail) < 0.15
    assert max(c.position_error_m for c in tail) < 0.20
    half = len(tail) // 2
    early = max(abs(c.along_track_lag_m) for c in tail[:half])
    late = max(abs(c.along_track_lag_m) for c in tail[half:])
    assert late <= early + 1e-3


def test_the_feedforward_carries_the_command_not_the_feedback():
    """On a well-tracked plan the correction terms are a small share of the total.

    If they are not, the plan is not being replayed properly and raising the
    gains would be treating the symptom.
    """
    trajectory = _line()
    airframe = LaggingAirframe(_at_start(trajectory))
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    commands, _ = _fly(tracker, trajectory, airframe, trajectory.duration)
    mid = commands[len(commands) // 2]
    reference = trajectory.sample(mid.reference_time_s)
    planned = np.array([reference.ax, reference.ay, reference.az])
    issued = np.array(mid.acceleration())
    assert float(np.linalg.norm(issued - planned)) < 0.6


def _worst_departure(use_projection, hold_s=0.0, hold_from=1.2, tau=0.35):
    """Fly the corner, optionally holding mid-route, and report the worst departure."""
    tracker = TrajectoryTracker(TrajectoryTrackerParams(use_projection=use_projection))
    trajectory = _corner()
    tracker.reset(yaw=0.0)
    start = trajectory.sample(0.0)
    airframe = LaggingAirframe([start.x, start.y, start.z],
                               velocity=[start.vx, start.vy, start.vz], tau=tau)
    flown = []
    now = 0.0
    tracker.set_trajectory(trajectory)
    for _ in range(int((trajectory.duration + hold_s) / DT)):
        follow = not (hold_from <= now < hold_from + hold_s)
        command = tracker.update(airframe.position, airframe.velocity, 0.0, DT, now,
                                 follow=follow)
        airframe.step(command.acceleration(), DT)
        flown.append(airframe.position.copy())
        now += DT
    return _worst_distance_from_curve(trajectory, flown)


def test_projection_is_not_worse_on_a_smooth_trajectory():
    """Nominal tracking of a dynamically feasible spline, either way of indexing.

    Worth stating plainly, because the opposite was assumed: on a curve FALCON's
    optimiser has already made feasible, and with the modest lag of a healthy
    inner loop, tracking the nearest point is **not** better than tracking the
    point at time *t*. Measured here, it is marginally worse. The corner-cutting
    that projection is supposed to remove needs a reference that runs away
    around a bend, and a smooth curve with 0.3 m of lag does not provide one.

    So this asserts only that both stay on the plan. The case where the choice
    does matter is the next test.
    """
    assert _worst_departure(True) < 0.5
    assert _worst_departure(False) < 0.5


def test_projection_rejoins_the_route_after_a_hold_instead_of_skipping_ahead():
    """Where projection actually earns its place, and it is not corners.

    FALCON condemns its own live trajectory whenever it finds an obstacle on it,
    and the aircraft holds until a replacement arrives. The plan's clock keeps
    running through that hold, so on resuming, a time-indexed reference sits
    seconds further down the route -- around the corner, through whatever is
    between -- and the aircraft is pulled at it across the intervening space.
    Projection resumes from where the aircraft actually is and flies the part of
    the route it had not flown yet.

    That is a displacement in *time*, which is the thing projection is really
    for; a corner alone is not.
    """
    assert _worst_departure(True, hold_s=2.0) < _worst_departure(False, hold_s=2.0)


def test_a_queued_trajectory_waits_for_its_own_start_time():
    """FALCON starts each curve in the future so it joins the one being flown.

    Adopting it the moment it arrives would jump the reference to a point the
    aircraft has not reached.
    """
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    tracker.set_trajectory(_line(traj_id=1, start_time_s=0.0))
    tracker.update([0.0, 0.0, 1.4], [0.0, 0.0, 0.0], 0.0, DT, 0.5)
    assert tracker.trajectory_id == 1

    assert tracker.set_trajectory(_line(traj_id=2, start_time_s=2.0))
    tracker.update([0.0, 0.0, 1.4], [0.0, 0.0, 0.0], 0.0, DT, 1.0)
    assert tracker.trajectory_id == 1
    tracker.update([0.0, 0.0, 1.4], [0.0, 0.0, 0.0], 0.0, DT, 2.1)
    assert tracker.trajectory_id == 2


def test_a_misordered_trajectory_is_refused():
    """A re-sent or out-of-order curve would restart the plan mid-flight."""
    tracker = TrajectoryTracker()
    assert tracker.set_trajectory(_line(traj_id=5))
    assert not tracker.set_trajectory(_line(traj_id=5))
    assert not tracker.set_trajectory(_line(traj_id=4))
    assert tracker.set_trajectory(_line(traj_id=6))


def test_withholding_the_trajectory_brakes_to_a_latched_point():
    """How a caller acts on FALCON condemning its own live trajectory.

    The aircraft must stop, not coast: it is being told an obstacle sits on the
    curve it is flying.
    """
    trajectory = _line()
    airframe = LaggingAirframe(_at_start(trajectory))
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    _fly(tracker, trajectory, airframe, 2.0)
    assert float(np.linalg.norm(airframe.velocity)) > 0.5

    now = 2.0
    for _ in range(int(4.0 / DT)):
        command = tracker.update(airframe.position, airframe.velocity, 0.0, DT, now,
                                 follow=False)
        assert command.holding
        airframe.step(command.acceleration(), DT)
        now += DT
    assert float(np.linalg.norm(airframe.velocity)) < 0.05


def test_holding_returns_to_the_latched_point_rather_than_drifting():
    """The hold point is latched once, so the aircraft does not ratchet away."""
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    airframe = LaggingAirframe([1.0, 2.0, 1.4], velocity=[0.8, 0.0, 0.0])
    now = 0.0
    for _ in range(int(8.0 / DT)):
        command = tracker.update(airframe.position, airframe.velocity, 0.0, DT, now)
        airframe.step(command.acceleration(), DT)
        now += DT
    # It overshot past the latch on the way to stopping, then came back to it.
    assert airframe.position == pytest.approx([1.0, 2.0, 1.4], abs=0.05)


def test_running_past_the_end_brakes_to_a_hover_on_the_final_point():
    """A gap between replans is normal; the aircraft finishes the curve it has.

    The failure this guards is not academic. The projection search returns a
    time marginally *inside* the curve, and sampling the trajectory there hands
    back the plan's full cruise velocity as a feedforward -- so before the
    reference was pinned to the endpoint, the aircraft sailed a metre and a half
    past the end of its trajectory at cruise, into space FALCON had never
    checked against the map, with only the position term mildly objecting.
    """
    params = TrajectoryTrackerParams(max_trajectory_age_s=5.0)
    trajectory = _line(length=8)
    start = trajectory.sample(0.0)
    airframe = LaggingAirframe([start.x, start.y, start.z],
                               velocity=[start.vx, start.vy, start.vz])
    tracker = TrajectoryTracker(params)
    tracker.reset(yaw=0.0)
    commands, _ = _fly(tracker, trajectory, airframe, trajectory.duration + 4.0)
    assert commands[-1].past_end
    assert not commands[-1].holding
    final = trajectory.sample(trajectory.duration)
    assert airframe.position[0] == pytest.approx(final.x, abs=0.1)
    assert float(np.linalg.norm(airframe.velocity)) < 0.1


def test_a_long_silence_gives_up_and_holds_station():
    """Past the grace period, flying to a stale endpoint is a guess."""
    params = TrajectoryTrackerParams(max_trajectory_age_s=0.5)
    tracker = TrajectoryTracker(params)
    tracker.reset(yaw=0.0)
    trajectory = _line(length=8)
    airframe = LaggingAirframe(_at_start(trajectory))
    commands, _ = _fly(tracker, trajectory, airframe, trajectory.duration + 1.5)
    assert commands[-1].holding


def test_divergence_is_reported_but_the_loop_keeps_trying():
    """Advisory, not a mode change -- the mission decides what to do about it."""
    params = TrajectoryTrackerParams(max_position_error_m=0.5)
    tracker = TrajectoryTracker(params)
    tracker.reset(yaw=0.0)
    trajectory = _line()
    tracker.set_trajectory(trajectory)
    # The route runs along y = 0, so an aircraft five metres to the +y side is
    # pulled back toward negative y.
    command = tracker.update([0.0, 5.0, 1.4], [0.0, 0.0, 0.0], 0.0, DT, 1.0)
    assert command.diverged
    assert not command.holding
    assert command.ay < 0.0


def test_the_position_error_clamp_bounds_the_pull_without_turning_it():
    """The clamp limits how hard the loop pulls, not which way it pulls.

    Tested on the clamp itself rather than through a flight, because the
    commanded acceleration also carries the feedforward and the damping term and
    those would drown the property being checked.

    Per-axis clamping breaks this contract: an error of (5.0, 1.0) clamps to
    (1.0, 1.0), which points 33.7 degrees away from the reference, with 45 as
    the worst case -- most wrong exactly when the aircraft is furthest off the
    plan. The horizontal pair therefore has to be scaled together, the way
    ``limit_acceleration`` already scales the horizontal command.
    """
    tracker = TrajectoryTracker(TrajectoryTrackerParams(position_error_clamp_m=1.0))

    for error in ([5.0, 1.0, 0.0], [3.0, 0.5, 0.0], [-4.0, 2.0, 0.0], [0.2, 6.0, 0.0]):
        clamped = tracker._clamp_error(np.array(error, dtype=float))
        wanted = np.array(error[:2]) / np.linalg.norm(error[:2])
        got = clamped[:2] / np.linalg.norm(clamped[:2])
        angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(wanted, got))))))
        assert angle < 1e-6, "clamp rotated %s by %.1f deg" % (error, angle)
        assert float(np.linalg.norm(clamped[:2])) == pytest.approx(1.0, abs=1e-9)

    # Inside the clamp nothing is touched at all.
    small = tracker._clamp_error(np.array([0.3, -0.4, 0.2]))
    assert small == pytest.approx([0.3, -0.4, 0.2])

    # Vertical is clipped on its own: altitude is not interchangeable with
    # sideways drift, and the two axes have their own gains and limits.
    tall = tracker._clamp_error(np.array([0.0, 0.0, 9.0]))
    assert tall[2] == pytest.approx(1.0)


def test_a_cross_track_offset_is_pushed_back_along_its_own_axis():
    """The flight-level consequence: sideways error produces sideways correction."""
    tracker = TrajectoryTracker(TrajectoryTrackerParams(use_projection=False))
    tracker.reset(yaw=0.0)
    trajectory = _line()
    tracker.set_trajectory(trajectory)
    on_plan = trajectory.position_at(1.0)
    reference = trajectory.sample(1.0)
    command = tracker.update([on_plan[0], on_plan[1] + 3.0, on_plan[2]],
                             [reference.vx, reference.vy, reference.vz], 0.0, DT, 1.0)
    # Flying at the plan's speed with a pure +y offset: the correction is -y,
    # and nothing significant is asked of x.
    assert command.ay < -0.5
    assert abs(command.ax) < 0.3


def test_reset_stops_it_flying_the_plan():
    """``reset(hold_position=...)`` must actually hold that position.

    It did not: the corrections were cleared but the loaded trajectory was not,
    so the very next tick found a usable curve and carried on following it. The
    one call a caller has to say "stop flying the plan" quietly did nothing.
    """
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    trajectory = _line()
    tracker.set_trajectory(trajectory)
    flying = tracker.update(_at_start(trajectory), [0.0, 0.0, 0.0], 0.0, DT, 1.0)
    assert not flying.holding
    assert tracker.trajectory_id == 1

    somewhere = [0.0, 0.0, 1.4]
    tracker.reset(yaw=0.0, hold_position=somewhere)
    held = tracker.update([0.3, 0.2, 1.4], [0.0, 0.0, 0.0], 0.0, DT, 1.2)
    assert held.holding
    assert tracker.trajectory_id == -1
    # ... and it holds the point it was given, not wherever it happens to be.
    assert held.position_error_m == pytest.approx(math.hypot(0.3, 0.2), abs=1e-6)


def test_lag_reads_as_along_track_and_offset_reads_as_cross_track():
    """Being late and being sideways are different problems with one magnitude.

    Projection is what makes the split exact: the off-path distance is measured
    to the nearest point on the curve, and the lag is the schedule error turned
    into metres at the planned speed.
    """
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    trajectory = _line()
    tracker.set_trajectory(trajectory)

    behind = trajectory.position_at(0.6)
    late = tracker.update(behind, [0.0, 0.0, 0.0], 0.0, DT, 1.0)
    assert late.along_track_lag_m > 0.2
    assert late.cross_track_error_m < 0.02

    tracker.reset(yaw=0.0)
    tracker.set_trajectory(_line(traj_id=2))
    on_time = trajectory.position_at(1.0)
    sideways = tracker.update([on_time[0], on_time[1] + 0.35, on_time[2]],
                              [0.0, 0.0, 0.0], 0.0, DT, 1.0)
    assert sideways.cross_track_error_m == pytest.approx(0.35, abs=0.02)
    assert abs(sideways.along_track_lag_m) < 0.05


def test_yaw_follows_the_plan_and_is_rate_limited():
    """Heading is the plan's, but a jump in it is slewed rather than snapped."""
    params = TrajectoryTrackerParams(max_yaw_rate=math.radians(30.0), yaw_rate_margin=1.0)
    tracker = TrajectoryTracker(params)
    tracker.reset(yaw=0.0)
    tracker.set_trajectory(_line(yaws=[2.0] * 6))
    command = tracker.update([0.0, 0.0, 1.4], [0.0, 0.0, 0.0], 0.0, DT, 1.0)
    assert 0.0 < command.yaw <= math.radians(30.0) * DT + 1e-9


def test_jerk_is_passed_through_unchanged_for_the_stage_below():
    """The outer loop has no use for jerk; the flatness stage does.

    Checked as an equality against the curve rather than as "non-zero": on the
    straight leg of a route the jerk genuinely is zero, and a test that demanded
    otherwise would be testing the fixture.
    """
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    trajectory = _corner()
    tracker.set_trajectory(trajectory)
    here = trajectory.position_at(1.8)
    command = tracker.update(here, [1.0, 0.0, 0.0], 0.0, DT, 1.8)
    reference = trajectory.sample(command.reference_time_s)
    assert command.jerk() == pytest.approx((reference.jx, reference.jy, reference.jz))
    # 1.8 s in, the corner trajectory is turning, so this is a live value.
    assert max(abs(v) for v in command.jerk()) > 0.0


def test_a_non_positive_timestep_is_refused():
    """The integrator and the yaw slew are both functions of dt."""
    tracker = TrajectoryTracker()
    with pytest.raises(ValueError, match="dt must be > 0"):
        tracker.update([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 0.0, 0.0)


def test_lag_is_seen_when_the_aircraft_is_behind_a_freshly_planned_curve():
    """The regression that made the catch-up term inert in flight.

    FALCON plans each curve from its **own previous curve**, not from the
    aircraft, so a lagging aircraft is behind the new curve's *start*. A curve
    has no negative time, so measuring the deficit as a difference of times on
    that curve -- ``elapsed - projected_time`` -- clamps to zero and the lag
    vanishes. Measured in flight: a true 1.30 m read as 0.03 m, the catch-up
    contributed 0.03 m/s instead of its 0.5 m/s ceiling, and because FALCON
    replans about four times a second no lag ever survived on one curve long
    enough to be noticed.

    The aircraft here sits squarely behind the start of a straight curve, so the
    answer is unambiguous: all lag, no cross-track.
    """
    trajectory = _line()
    start = trajectory.sample(0.0)
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    tracker.set_trajectory(trajectory)

    behind = [start.x - 1.3, start.y, start.z]
    command = tracker.update(behind, [start.vx, start.vy, start.vz], 0.0, DT, 0.05)

    # 1.35 rather than 1.30: the plan moved on for the 0.05 s of elapsed time,
    # which is the point -- the deficit is measured against where the plan says
    # the aircraft should be NOW, not against where this curve began.
    assert command.along_track_lag_m == pytest.approx(1.35, abs=0.05)
    assert command.cross_track_error_m < 0.05
    # ... and it does something about it: the command pushes forward, harder
    # than the plan's own acceleration on a constant-speed straight line.
    assert command.ax > 1.0


def test_the_three_error_numbers_are_one_consistent_decomposition():
    """``along**2 + cross**2 == gap**2``, so a gap is always fully attributable.

    Both halves come from the same displacement, resolved along and across the
    direction of travel. An earlier version measured cross-track as the distance
    to the nearest point on the curve, which is honest about the curve and
    useless as a measure of being off the path: with the aircraft directly
    behind the start of a straight line it reported the entire 1.3 m of lateness
    as cross-track error.
    """
    trajectory = _line()
    tracker = TrajectoryTracker()
    for offset in ([-1.3, 0.0, 0.0], [0.0, 0.7, 0.0], [-0.9, 0.6, 0.2], [0.4, -0.3, -0.5]):
        tracker.reset(yaw=0.0)
        tracker.set_trajectory(_line(traj_id=int(abs(offset[1]) * 1000) + 7))
        on_plan = trajectory.position_at(1.0)
        here = [on_plan[i] + offset[i] for i in range(3)]
        command = tracker.update(here, [0.0, 0.0, 0.0], 0.0, DT, 1.0)
        recombined = math.hypot(command.along_track_lag_m, command.cross_track_error_m)
        assert recombined == pytest.approx(command.position_error_m, abs=1e-6)


def test_the_catch_up_closes_a_deficit_that_falcon_keeps_re_creating():
    """The flight failure, reproduced: replan constantly, always from the plan.

    Each new curve starts from the *previous plan's* position rather than the
    aircraft's, which is what FALCON does and what locks a lag in place. The
    aircraft must still converge.
    """
    tracker = TrajectoryTracker()
    tracker.reset(yaw=0.0)
    trajectory = _line(length=40)
    start = trajectory.sample(0.0)
    airframe = LaggingAirframe([start.x - 1.3, start.y, start.z],
                               velocity=[start.vx, start.vy, start.vz])

    now = 0.0
    traj_id = 1
    # First replan one cadence in, so the opening tick flies the trajectory it
    # was given rather than being handed a replacement that has not started yet.
    next_replan = 0.25
    lags = []
    tracker.set_trajectory(trajectory)
    for _ in range(int(12.0 / DT)):
        # FALCON's cadence: a new curve every 0.25 s, cut from the live plan at
        # now + 0.1 s -- never from where the aircraft actually is.
        if now >= next_replan:
            next_replan = now + 0.25
            traj_id += 1
            resumed = trajectory.sample(min(now + 0.1, trajectory.duration))
            tracker.set_trajectory(_spline(
                [[resumed.x + i * 0.4, resumed.y, resumed.z] for i in range(12)],
                start_time_s=now + 0.1, traj_id=traj_id))
        command = tracker.update(airframe.position, airframe.velocity, 0.0, DT, now)
        airframe.step(command.acceleration(), DT)
        lags.append(command.along_track_lag_m)
        now += DT

    assert lags[0] > 1.0
    assert lags[-1] < 0.35, "the aircraft never caught up: %.2f m still behind" % lags[-1]


def test_the_speed_ceiling_stops_the_position_loop_running_away():
    """FALCON's clearance is computed for FALCON's speed; flying faster spends it.

    The position loop has no natural ceiling -- a metre of error asks for
    ``kp * clamp`` of acceleration and the damping term only balances it once
    the aircraft is ``kp * clamp / kd`` faster than the plan. Measured in flight
    before this existed: 42% of the time above 1.1 m/s on a 0.6 m/s plan,
    peaking at 2.85, and the flight ended embedded in a desk at cruise height.

    Measured across the catch-up itself, which is when the position term is
    saturated and the overspeed actually happens; once the lag is closed both
    settings simply cruise at the plan's speed and tell you nothing.
    """
    trajectory = _line(length=40)
    start = trajectory.sample(0.0)
    planned = math.hypot(start.vx, start.vy)

    def peak_speed(max_overspeed):
        tracker = TrajectoryTracker(TrajectoryTrackerParams(
            max_overspeed=max_overspeed,
            max_catchup_speed=min(0.5, max_overspeed)))
        tracker.reset(yaw=0.0)
        airframe = LaggingAirframe([start.x - 1.3, start.y, start.z],
                                   velocity=[start.vx, start.vy, start.vz])
        speeds = []
        now = 0.0
        tracker.set_trajectory(trajectory)
        for _ in range(int(10.0 / DT)):
            command = tracker.update(airframe.position, airframe.velocity, 0.0, DT, now)
            airframe.step(command.acceleration(), DT)
            speeds.append(float(np.linalg.norm(airframe.velocity)))
            now += DT
        return max(speeds)

    tight = peak_speed(0.5)
    loose = peak_speed(3.0)
    # A soft ceiling: the governor brakes the excess off, but no controller can
    # stop an airframe instantly, so a couple of tenths of transient survive.
    assert tight <= planned + 0.5 + 0.2, "ceiling not respected: %.2f m/s" % tight
    # And it is the ceiling doing the limiting, not the airframe: the same
    # aircraft closing the same lag goes materially faster without it.
    assert loose > tight + 0.15, "ceiling had no effect (%.2f vs %.2f)" % (loose, tight)


def _ungoverned(catchup=0.1):
    """A tracker whose speed ceiling can never engage, for differencing against.

    The governor's contribution is only visible as a DIFFERENCE. Asserting that
    a fast aircraft is braking proves nothing: at three times the plan speed the
    damping term alone commands about -4.4 m/s^2, so the assertion passes with
    the governor deleted outright -- which an earlier version of this test did.
    """
    return TrajectoryTracker(TrajectoryTrackerParams(max_overspeed=1000.0,
                                                     max_catchup_speed=catchup))


def _one_tick(tracker, trajectory, offset, velocity):
    tracker.reset(yaw=0.0)
    tracker.set_trajectory(trajectory)
    on_plan = trajectory.position_at(1.0)
    return tracker.update([on_plan[0] + offset, on_plan[1], on_plan[2]],
                          velocity, 0.0, DT, 1.0)


def test_the_ceiling_never_blocks_braking():
    """It removes only the accelerating component, so it cannot deadlock a stop.

    An aircraft already over the ceiling still needs to be able to slow down --
    a limiter that zeroed the whole command would leave it coasting.
    """
    trajectory = _line()
    governed = TrajectoryTracker(TrajectoryTrackerParams(max_overspeed=0.1,
                                                         max_catchup_speed=0.1))
    a = _one_tick(governed, trajectory, -0.5, [3.0, 0.0, 0.0])
    b = _one_tick(_ungoverned(), trajectory, -0.5, [3.0, 0.0, 0.0])
    assert a.ax < 0.0, "should be braking, got ax=%.2f" % a.ax
    # and the governor must have made it MORE braking, not less
    assert a.ax < b.ax


def test_the_active_braking_branch_does_something():
    """Above the ceiling the excess is braked off, not merely left alone.

    Deleting that branch used to break no test at all. Differenced against a
    tracker that cannot engage its ceiling, it has to show up.
    """
    trajectory = _line()
    governed = TrajectoryTracker(TrajectoryTrackerParams(max_overspeed=0.1,
                                                         max_catchup_speed=0.1))
    over = _one_tick(governed, trajectory, 0.0, [2.0, 0.0, 0.0])
    free = _one_tick(_ungoverned(), trajectory, 0.0, [2.0, 0.0, 0.0])
    assert over.ax < free.ax - 0.5


def test_the_governor_shapes_the_command_as_documented():
    """The three branches of the speed governor, exercised directly.

    Tested on ``_limit_speed`` rather than through a flight, deliberately. The
    branches are only reachable when the command is ACCELERATING while the
    aircraft is already near the ceiling, and on a straight plan that
    combination does not occur -- projection puts the reference at the
    aircraft's own along-track position, so a lagging aircraft has no position
    error to accelerate towards and the damping term is already braking. A
    closed-loop fixture therefore exercises none of this, which is exactly how
    the taper and the active-braking branch came to have no coverage at all
    while a test claimed to guard them.
    """
    tracker = TrajectoryTracker(TrajectoryTrackerParams(max_overspeed=0.5))
    plan = 1.0
    ceiling = plan + 0.5
    taper = 0.5 * 0.6                       # _GOVERNOR_TAPER
    accelerating = np.array([2.0, 0.0, 0.0])
    braking = np.array([-2.0, 0.0, 0.0])

    def limit(wanted, speed):
        return tracker._limit_speed(wanted, np.array([speed, 0.0, 0.0]), plan)

    # well below the ceiling: untouched
    assert limit(accelerating, ceiling - taper - 0.2)[0] == pytest.approx(2.0)
    # inside the taper band: the accelerating component is partly removed
    faded = limit(accelerating, ceiling - taper / 2.0)[0]
    assert 0.0 < faded < 2.0
    # at the ceiling: fully removed
    assert limit(accelerating, ceiling)[0] == pytest.approx(0.0, abs=1e-9)
    # past it: actively braked, beyond merely removing the request
    assert limit(accelerating, ceiling + 0.4)[0] < -0.5
    # braking is never blocked, at any speed
    for speed in (0.5, ceiling - taper / 2.0, ceiling, ceiling + 0.4):
        assert limit(braking, speed)[0] <= -2.0 + 1e-9


def test_the_ceiling_moves_with_the_plan_rather_than_capping_absolutely():
    """An absolute clamp under the planner's limit would leave it permanently behind.

    The ceiling is planned speed *plus* a margin, so a faster plan is simply
    flown faster.
    """
    params = TrajectoryTrackerParams(max_overspeed=0.5)
    tracker = TrajectoryTracker(params)
    tracker.reset(yaw=0.0)
    # 0.8 m spacing over a 0.4 s knot span is a 2 m/s plan -- far above any
    # sane absolute clamp.
    fast = _line(length=40, spacing=0.8)
    start = fast.sample(0.0)
    airframe = LaggingAirframe([start.x, start.y, start.z],
                               velocity=[start.vx, start.vy, start.vz])
    commands, _ = _fly(tracker, fast, airframe, 6.0)
    assert float(np.linalg.norm(airframe.velocity)) > 1.8
    assert max(c.position_error_m for c in commands[len(commands) // 2:]) < 0.2


def test_bad_overspeed_settings_are_refused():
    """A ceiling below the catch-up cancels the catch-up it exists to allow."""
    with pytest.raises(ValueError, match="max_overspeed"):
        TrajectoryTrackerParams(max_overspeed=0.2, max_catchup_speed=0.5)


# ── the standing-force feedforward and the attitude lead ──────────────────

def _on_plan_tick(params, trajectory, t=1.0):
    """One tick with the aircraft exactly on the plan, at the plan's velocity.

    Feedback terms all read ~zero there, so what comes back is the feedforward
    path alone -- the clean way to observe it without disentangling the PID.

    The projector is WALKED to ``t`` rather than teleported: its search window
    is 1.5 s ahead of wherever it last was, so a fresh tracker asked to resolve
    t=2.0 clamps to 1.5 and every feedback term reads the 0.7 m gap between
    the two -- which is the projector doing its job, not a fixture to fight.
    """
    tracker = TrajectoryTracker(params)
    tracker.reset(yaw=0.0)
    tracker.set_trajectory(trajectory)
    step = 0.5
    when = step
    while when < t - 1e-9:
        ref = trajectory.sample(when)
        tracker.update([ref.x, ref.y, ref.z], [ref.vx, ref.vy, ref.vz],
                       0.0, DT, when)
        when += step
    ref = trajectory.sample(t)
    return tracker.update([ref.x, ref.y, ref.z], [ref.vx, ref.vy, ref.vz],
                          0.0, DT, t)


def test_drag_feedforward_adds_the_measured_curve_along_travel():
    trajectory = _line()          # straight +x at ~1 m/s
    plain = _on_plan_tick(TrajectoryTrackerParams(), trajectory)
    dragged = _on_plan_tick(TrajectoryTrackerParams(drag_per_mps=0.176,
                                                    drag_offset_mps2=0.121),
                            trajectory)
    speed = np.hypot(trajectory.sample(1.0).vx, trajectory.sample(1.0).vy)
    expected = 0.176 * speed + 0.121
    assert dragged.ax - plain.ax == pytest.approx(expected, abs=1e-6)
    assert dragged.ay - plain.ay == pytest.approx(0.0, abs=1e-6)


def test_a_hover_feeds_no_drag_forward():
    """The offset term acts along travel and must vanish with it.

    A constant 0.121 m/s^2 with no direction to attach to would push a
    hovering aircraft sideways forever.
    """
    trajectory = _line()
    params = TrajectoryTrackerParams(drag_per_mps=0.176, drag_offset_mps2=0.121)
    tracker = TrajectoryTracker(params)
    tracker.reset(yaw=0.0)
    tracker.set_trajectory(trajectory)
    # Past the end the reference is the stopped endpoint: planned velocity zero.
    end = trajectory.sample(trajectory.duration)
    command = tracker.update([end.x, end.y, end.z], [0.0, 0.0, 0.0],
                             0.0, DT, trajectory.duration + 0.5)
    assert abs(command.ax) < 0.05 and abs(command.ay) < 0.05


def test_the_attitude_lead_samples_the_feedforward_ahead():
    """With lead, the commanded acceleration is the plan's at t + lead.

    Verified on a corner, where acceleration actually changes -- on a straight
    line the lead is invisible, which is exactly why it is safe there.
    """
    trajectory = _corner()
    lead = 0.25
    t = 2.0                       # mid-corner: acceleration varying
    plain = _on_plan_tick(TrajectoryTrackerParams(), trajectory, t=t)
    led = _on_plan_tick(TrajectoryTrackerParams(attitude_lead_s=lead),
                        trajectory, t=t)
    ahead = trajectory.sample(plain.reference_time_s + lead)
    # The led feedforward should match the plan AHEAD, not the plan now.
    assert led.ax == pytest.approx(ahead.ax, abs=0.05)
    assert led.ay == pytest.approx(ahead.ay, abs=0.05)
    assert (abs(led.ax - plain.ax) + abs(led.ay - plain.ay)) > 0.01
    # And the jerk it carries forward is the led one too.
    assert led.jx == pytest.approx(ahead.jx, abs=1e-9)


def test_negative_drag_or_lead_is_rejected():
    for bad in ({"drag_per_mps": -0.1}, {"drag_offset_mps2": -0.1},
                {"attitude_lead_s": -0.05}):
        with pytest.raises(ValueError):
            TrajectoryTrackerParams(**bad)
