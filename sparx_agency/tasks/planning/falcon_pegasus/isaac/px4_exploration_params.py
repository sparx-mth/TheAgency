"""PX4 settings an exploration flight needs on top of the collection set.

The base set (``sim_flight_recording/px4_params.py``) was tuned for a campaign
that flies smooth A-to-B routes for training data, where the aircraft's heading
is chosen to make the footage readable. Exploration is a different job with a
different constraint: FALCON plans yaw to point the camera at the frontier it
intends to observe next, so the heading is part of the plan, and an autopilot
that cannot turn as fast as the plan asks does not merely look different -- it
arrives at a viewpoint facing the wrong way, sees nothing, and the frontier
survives to be chosen again.

Only what changes is here. Everything else comes from the base set, so a fix
there reaches this too.
"""
from __future__ import annotations

from sparx_agency.tasks.planning.sim_flight_recording import px4_params

EXPLORATION_OVERRIDES = {
    # The collection set holds this at 25 deg/s so the recorded world turns
    # slowly. FALCON's own model allows 90 deg/s and its trajectories use it:
    # at 25 the aircraft is still rotating when the reference has finished
    # turning, so it observes the previous viewpoint's view at the new
    # viewpoint's position. 60 gives headroom over the 51.5 deg/s the launch
    # file caps FALCON at, without letting the camera whip.
    "MPC_YAWRAUTO_MAX": 60.0,   # deg/s
    # The B-spline is continuous in acceleration and FALCON plans up to
    # 1.5 m/s^2; leaving PX4's jerk limit at the collection value made the
    # tracker's velocity command the binding constraint through every corner.
    "MPC_JERK_AUTO": 8.0,       # m/s^3
    # Exploration flies close to walls by design -- a frontier is, definitionally,
    # at the edge of what is known. A shallower tilt keeps the camera pointing
    # where the yaw plan says rather than at the floor or the ceiling during
    # acceleration, which is what the frontier model assumes.
    "MPC_TILTMAX_AIR": 20.0,    # deg
}
"""What an exploration flight changes about the collection parameter set."""


def all_params() -> dict:
    """Every PX4 parameter an exploration run should fly with.

    ``vision=False``: this stack still flies on PX4's simulated GNSS and
    magnetometer. The collection stack moved off them because a stale compass had
    PX4 raising a failsafe and force-landing mid-route (see
    ``sim_flight_recording/px4_params.GPS_ESTIMATOR``), and the same fix would
    help here -- but it needs a ``VisionPoseSender`` in this package's own loop
    (``isaac/setup.py``), not just a parameter set, and sending the parameters
    without the pose would leave EKF2 waiting for data that never arrives.
    Changing that is its own piece of work.

    Returns:
        Name to value, the collection set with :data:`EXPLORATION_OVERRIDES`
        applied. The value's **Python type selects the MAVLink parameter type**
        (see ``PX4Offboard.set_params``), so the int/float split matters.
    """
    merged = px4_params.all_params(vision=False)
    merged.update(EXPLORATION_OVERRIDES)
    return merged
