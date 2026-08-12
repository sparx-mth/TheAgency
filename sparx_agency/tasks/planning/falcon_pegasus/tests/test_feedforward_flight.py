"""The drag and attitude-lead feedforwards, measured on the stand-in airframe.

The stub's :class:`AttitudeAircraft` models the two effects these terms exist
for -- drag at the measured curve, and a thrust axis that reaches its command
through a ~0.18 s lag -- so flying it A/B on a FIXED trajectory is the honest
test. A live stub run cannot be: FALCON's exploration is nondeterministic and
its run-to-run variance (mean errors 0.18 to 2.61 on identical control code)
swamps the effect being measured.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.control.airframe import AirframeController
from sparx_agency.core.control.thrust_model import ThrustModelParams
from sparx_agency.core.control.trajectory_tracking import TrajectoryTrackerParams
from sparx_agency.core.planning.trajectories.bspline import BsplineTrajectory
from sparx_agency.tasks.planning.falcon_pegasus.stub.airframe import AttitudeAircraft

DT = 0.02          # the stub's control rate


def _route():
    """A cruise with a corner -- long enough for drag to bite, bent enough
    for the lead to matter."""
    points = ([[i * 0.5, 0.0, 1.4] for i in range(10)]
              + [[4.5, j * 0.5, 1.4] for j in range(1, 10)])
    span = 0.45
    knots = [(-3 + i) * span for i in range(len(points) + 4)]
    return BsplineTrajectory.from_falcon(3, knots, points, [0.0] * 6, span, 0.0, 1)


def _fly(tracker_params):
    """Fly the whole curve closed-loop; return the mean and max tracking error."""
    trajectory = _route()
    start = trajectory.sample(0.0)
    aircraft = AttitudeAircraft([start.x, start.y, start.z], 0.0,
                                true_hover_throttle=0.62)
    aircraft.velocity = np.array([start.vx, start.vy, start.vz])
    controller = AirframeController(tracker=tracker_params,
                                    thrust=ThrustModelParams(hover_throttle=0.62))
    controller.set_trajectory(trajectory)
    errors = []
    now = 0.0
    for _ in range(int(trajectory.duration / DT)):
        command = controller.update(tuple(aircraft.position),
                                    tuple(aircraft.velocity),
                                    aircraft.yaw, DT, now)
        aircraft.step_attitude(command.attitude.quaternion_wxyz(),
                               command.throttle, command.tracking.yaw, DT)
        controller.observe_thrust(command.throttle, aircraft.acceleration,
                                  aircraft.body_z, DT)
        errors.append(command.tracking.position_error_m)
        now += DT
    return float(np.mean(errors)), float(np.max(errors))


def test_the_feedforwards_measurably_improve_tracking():
    """Drag + lead on beats off, against the airframe that has both effects."""
    plain_mean, plain_max = _fly(TrajectoryTrackerParams())
    fed_mean, fed_max = _fly(TrajectoryTrackerParams(drag_per_mps=0.176,
                                                     drag_offset_mps2=0.121,
                                                     attitude_lead_s=0.18))
    assert fed_mean < plain_mean * 0.85, (
        "expected at least 15%% mean improvement, got %.3f -> %.3f"
        % (plain_mean, fed_mean))
    assert fed_max <= plain_max * 1.10          # and no new worst-case


def test_the_terms_divide_the_work_as_measured():
    """What each term is for, pinned as measured -- not as first assumed.

    Drag is the workhorse: alone it cuts the mean error ~4x (0.189 -> 0.041),
    because the standing force is the dominant residual. The lead ALONE
    slightly worsens the mean (0.189 -> 0.198) while trimming the worst case
    (0.343 -> 0.272): on an aircraft riding behind the plan, the led sample
    does not correspond to where it actually is. TOGETHER they multiply --
    0.020 mean / 0.055 max -- because once drag is cancelled the aircraft
    rides ON the plan and the led feedforward matches reality. An earlier
    version of this test asserted each term improves the mean on its own;
    the lead does not, and the pair is what ships.
    """
    plain_mean, plain_max = _fly(TrajectoryTrackerParams())
    drag_mean, _ = _fly(TrajectoryTrackerParams(drag_per_mps=0.176,
                                                drag_offset_mps2=0.121))
    lead_mean, lead_max = _fly(TrajectoryTrackerParams(attitude_lead_s=0.18))
    assert drag_mean < plain_mean * 0.5          # the workhorse
    assert lead_max < plain_max                  # the worst-case trimmer
    assert lead_mean < plain_mean * 1.10         # and no real mean regression
