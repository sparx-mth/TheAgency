"""The whole chain, flown against a body that only accepts attitude and throttle.

This is the first test in the stack where nothing is assumed about the layer
below: the airframe is given a tilt and a throttle and works out its own
acceleration, exactly as a real one does. Anything wrong with the flatness
algebra, the thrust scale or the way the three stages are wired shows up here as
an aircraft that will not hold altitude.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.control.airframe import AirframeController
from sparx_agency.core.control.constants import GRAVITY_MPS2
from sparx_agency.core.control.flatness import matrix_from_quaternion
from sparx_agency.core.control.thrust_model import ThrustModelParams
from sparx_agency.core.control.trajectory_tracking import TrajectoryTrackerParams
from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory

DT = 0.004


class AttitudeAirframe:
    """A body that accepts a tilt and a throttle, and nothing else.

    The attitude loop underneath is modelled as a first-order lag on the *thrust
    axis*: told to point somewhere, the aircraft gets there over ``tau``. That is
    what PX4's attitude and rate loops look like from above, and it is the only
    dynamics that matters here -- where the thrust points, and how much there is.

    ``true_hover_throttle`` is the airframe's real thrust curve, which the
    controller does not know and has to learn.
    """

    def __init__(self, position, velocity=None, tau=0.15, true_hover_throttle=0.62):
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float)
        self.acceleration = np.zeros(3)
        self.body_z = np.array([0.0, 0.0, 1.0])
        self.tau = float(tau)
        self.full_scale = GRAVITY_MPS2 / float(true_hover_throttle)

    def step(self, quaternion_wxyz, throttle, dt):
        wanted = matrix_from_quaternion(quaternion_wxyz)[:, 2]
        alpha = dt / (self.tau + dt)
        self.body_z += alpha * (wanted - self.body_z)
        self.body_z /= np.linalg.norm(self.body_z)
        thrust = float(throttle) * self.full_scale
        self.acceleration = thrust * self.body_z - np.array([0.0, 0.0, GRAVITY_MPS2])
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt


def _line(length=16, spacing=0.4, knot_span=0.4, traj_id=1):
    points = [[i * spacing, 0.0, 1.4] for i in range(length)]
    knots = [(-3 + i) * knot_span for i in range(length + 4)]
    return BsplineTrajectory.from_falcon(3, knots, points, [0.0] * 6, knot_span, 0.0, traj_id)


def _fly(controller, airframe, seconds, trajectory=None, start_s=0.0):
    """Run the full chain, feeding the thrust model as a real caller would."""
    if trajectory is not None:
        controller.set_trajectory(trajectory)
    commands = []
    now = start_s
    for _ in range(int(seconds / DT)):
        command = controller.update(airframe.position, airframe.velocity, 0.0, DT, now)
        airframe.step(command.attitude.quaternion_wxyz(), command.throttle, DT)
        controller.observe_thrust(command.throttle, airframe.acceleration,
                                  airframe.body_z, DT)
        commands.append(command)
        now += DT
    return commands


def test_it_holds_a_hover_with_nothing_to_follow():
    """No trajectory means hold station -- and holding altitude needs the thrust
    scale to be right, which is the whole point of the third stage."""
    airframe = AttitudeAirframe([1.0, 2.0, 1.4])
    controller = AirframeController()
    controller.reset(yaw=0.0)
    _fly(controller, airframe, 10.0)
    assert airframe.position == pytest.approx([1.0, 2.0, 1.4], abs=0.05)


def test_it_tracks_a_trajectory_end_to_end():
    """Attitude and throttle only, and the aircraft still flies the plan."""
    trajectory = _line()
    start = trajectory.sample(0.0)
    airframe = AttitudeAirframe([start.x, start.y, start.z],
                                velocity=[start.vx, start.vy, start.vz])
    controller = AirframeController()
    controller.reset(yaw=0.0)
    commands = _fly(controller, airframe, trajectory.duration, trajectory)
    settled = [c.position_error_m for c in commands[len(commands) // 4:]]
    assert max(settled) < 0.12
    assert sum(settled) / len(settled) < 0.06


def test_a_badly_seeded_thrust_model_still_holds_altitude():
    """The reason the thrust scale is measured rather than assumed.

    Seeded at 0.50 against an airframe that hovers at 0.70, the aircraft starts
    by sinking -- it is commanding 30% less thrust than it needs. The estimator
    finds the real curve within seconds and the altitude comes back. Without it,
    the sink is permanent, the position integrator quietly absorbs it, and every
    gain above is then tuned against a bias.
    """
    airframe = AttitudeAirframe([0.0, 0.0, 5.0], true_hover_throttle=0.70)
    controller = AirframeController(thrust=ThrustModelParams(hover_throttle=0.50,
                                                             learn_tau_s=2.0))
    controller.reset(yaw=0.0)
    commands = _fly(controller, airframe, 30.0)
    assert controller.hover_throttle == pytest.approx(0.70, abs=0.02)
    assert airframe.position[2] == pytest.approx(5.0, abs=0.1)
    assert commands[-1].hover_throttle == pytest.approx(controller.hover_throttle)


def test_the_learned_thrust_scale_survives_a_phase_handover():
    """Integrators must be cleared between phases; the thrust curve must not.

    The airframe's mass and its battery did not change because the mission moved
    from a climb to an exploration, and re-learning the scale at handover puts a
    transient into the first seconds of the part that matters.
    """
    airframe = AttitudeAirframe([0.0, 0.0, 2.0], true_hover_throttle=0.68)
    controller = AirframeController(thrust=ThrustModelParams(hover_throttle=0.5,
                                                             learn_tau_s=1.0))
    controller.reset(yaw=0.0)
    _fly(controller, airframe, 15.0)
    learned = controller.hover_throttle
    assert learned > 0.6

    controller.reset(yaw=0.0)
    assert controller.hover_throttle == pytest.approx(learned)
    controller.reset(yaw=0.0, forget_thrust=True)
    assert controller.hover_throttle == pytest.approx(0.5)


def test_every_command_is_sendable():
    """A unit quaternion and an in-range throttle, at every tick of a real flight."""
    trajectory = _line()
    start = trajectory.sample(0.0)
    airframe = AttitudeAirframe([start.x, start.y, start.z])
    controller = AirframeController()
    controller.reset(yaw=0.0)
    for command in _fly(controller, airframe, trajectory.duration, trajectory):
        norm = math.sqrt(sum(v * v for v in command.attitude.quaternion_wxyz()))
        assert norm == pytest.approx(1.0, abs=1e-9)
        assert 0.0 < command.throttle < 1.0
        assert command.attitude.specific_thrust_mps2 > 0.0


def test_the_limits_are_shared_between_the_stages():
    """One set of acceleration ceilings, or the two stages disagree silently."""
    params = TrajectoryTrackerParams()
    controller = AirframeController(tracker=params)
    assert controller.limits is params.limits


def test_holding_brakes_a_moving_aircraft():
    """The response to FALCON condemning its live trajectory, through the whole chain."""
    trajectory = _line()
    start = trajectory.sample(0.0)
    airframe = AttitudeAirframe([start.x, start.y, start.z],
                                velocity=[start.vx, start.vy, start.vz])
    controller = AirframeController()
    controller.reset(yaw=0.0)
    _fly(controller, airframe, 2.0, trajectory)
    assert float(np.linalg.norm(airframe.velocity)) > 0.5

    now = 2.0
    for _ in range(int(6.0 / DT)):
        command = controller.update(airframe.position, airframe.velocity, 0.0, DT, now,
                                    follow=False)
        assert command.holding
        airframe.step(command.attitude.quaternion_wxyz(), command.throttle, DT)
        controller.observe_thrust(command.throttle, airframe.acceleration,
                                  airframe.body_z, DT)
        now += DT
    assert float(np.linalg.norm(airframe.velocity)) < 0.05


def test_it_reports_the_same_diagnostics_as_the_velocity_cut_controller():
    """The two controllers must be readable through the same names.

    A mission that can be flown either way reads its status line off whichever
    command it got. When this chain was first wired in, it was missing
    ``cross_track_error_m`` -- and because the two control paths have separate
    status lines, nothing caught it until a status print raised an
    AttributeError two minutes into a real Isaac Sim flight. This pins the
    surface so the next omission fails in a unit test instead.
    """
    from sparx_agency.core.planning.trackers.reference_tracker_3d.types import (
        TrackedSetpoint,
    )

    shared = {name for name in vars(TrackedSetpoint).get("__annotations__", {})
              if not name.startswith("_")}
    # The two disagree about the *command* fields on purpose -- one emits a
    # velocity, the other an attitude -- so only the diagnostics are compared.
    shared -= {"vx", "vy", "vz"}

    controller = AirframeController()
    controller.reset(yaw=0.0)
    command = controller.update([0.0, 0.0, 1.4], [0.0, 0.0, 0.0], 0.0, DT, 0.0)
    missing = sorted(name for name in shared if not hasattr(command, name))
    assert not missing, "AirframeCommand is missing diagnostics: %s" % (missing,)
