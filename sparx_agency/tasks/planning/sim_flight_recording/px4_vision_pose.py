"""Feed PX4 the simulator's true pose as a mocap fix, instead of simulated GPS.

*Why it exists.* The accurate position was never the hard part -- the simulator
knows it exactly and every follower in this package already reads it straight off
``vehicle.state.position``. The problem was that **PX4** had no access to it.
Pegasus simulates a GPS receiver and a magnetometer and PX4 runs its own EKF2 on
top of them, and PX4 holds the veto: when its estimate disagreed with reality it
raised a failsafe and force-landed the aircraft out from under the follower,
mid-route, while the follower carried on steering. One 700-episode campaign was
lost to it -- ``Landing at current position`` 3694 times across 120 of 139
workers, ``Compass needs calibration - Land now!`` in 79, and a median
``estimator_drift_m`` of 8.4 m on the flights that ended in a crash. The
recordings looked like a badly-flying expert because the expert was not flying.

This module closes that gap the short way round: PX4's estimator is switched onto
external vision (:data:`~px4_params.VISION_ESTIMATOR`) and fed the pose the
simulator already has, so PX4's estimate *is* the number the rest of the stack
uses and cannot drift away from it. The magnetometer and the barometer are then
not merely ignored but removed, which is what stops them refusing an arming.

*What this is not.* It is not a model of a real VIO system -- there is no noise,
no latency and no scale error, on purpose. A simulated flight exists to produce a
clean expert demonstration; studying estimator error is a different experiment,
and ``px4_params.GPS_ESTIMATOR`` is still there for it.

## The four ways this failed before it worked

All four are fixed, and all four failed *silently* -- PX4 accepted everything and
flew on GPS, or refused to arm with a message about something else entirely.

1. **Parameter type mismatch.** Most of these EKF2 parameters are INT32. Sending
   one as REAL32 makes PX4 reject it (``ERROR [mavlink] param types mismatch``)
   and keep the old value, and the rejection appears only in PX4's own console.
   See :meth:`px4_offboard.PX4Offboard.set_params` and
   :func:`px4_launch.format_param_value` -- the int/float split in
   :mod:`px4_params` is load-bearing on both channels.
2. **The wrong MAVLink link.** Pegasus's own ``send_vision_msgs`` writes to the
   HIL connection, where PX4's ``simulator_mavlink`` consumes only ``HIL_*``
   messages and drops everything else without a word. It has to go over the
   companion link, which is what :class:`px4_offboard.PX4Offboard` holds.
3. **The socket direction.** PX4 *binds* ``14580 + instance`` and *sends to*
   ``14540 + instance``. A ``udpin`` socket on 14540 replies to PX4's source port
   and is fine -- which is what ``PX4Offboard`` opens -- but a raw ``udpout`` to
   14540 is silently dropped. Confirm arrival with ``listener
   vehicle_visual_odometry`` in PX4's console; nothing printed means nothing
   arrived.
4. **``ekf2 missing data`` does not mean what it looks like.** It comes from
   ``estimatorCheck.cpp`` and fires when the ``estimator_status`` uORB topic has
   never been advertised -- i.e. ekf2 has not completed a single ``update()``
   since boot. It is not an "EV data missing" message, and if it appears the
   vision path is a red herring.

## Three constraints PX4 v1.14.3 does not warn about

* **The innovation gates have to be opened, and that is not a shortcut.** EKF2
  puts every vision sample through a consistency test, which is right for a sensor
  that might be wrong and actively harmful for one that cannot be. Two situations
  produce a gap the stock gates never close. At start-up the filter has no yaw
  reference at all -- the magnetometer is gone -- so it aligns to zero and marks
  itself aligned; a correct vision yaw then arrives as an innovation of up to 180
  degrees. And once samples are rejected the state is unaided, drifts on IMU
  integration, and the innovation grows, which rejects it harder. Both were
  measured: at the stock gates the filter sat 6.8 m and 130 degrees from truth with
  the right answer arriving 50 times a second, rejecting every one, and PX4 refused
  to arm on "Yaw estimate error". ``EKF2_EVP_GATE`` and ``EKF2_HDG_GATE`` in
  :data:`px4_params.VISION_ESTIMATOR` are widened for exactly this. The *noise*
  parameters stay tight, so steady-state accuracy is unaffected -- what changes is
  only whether a good measurement can be thrown away.
  (PX4's own mechanism for this is ``VISION_POSITION_ESTIMATE``'s
  ``reset_counter``, which makes EKF2 snap its state to the sample instead of
  fusing it. The pymavlink in the isaac-sim container predates that extension
  field, so it is unavailable -- see
  :meth:`px4_offboard.PX4Offboard.send_vision_pose`.)
* Consecutive samples must be **less than 200 ms apart** (``EV_MAX_INTERVAL``) or
  fusion never starts, and a 400 ms gap stops it. :data:`SEND_RATE_HZ` satisfies
  that with room to spare *in simulated time*, which is the clock that matters
  under lockstep -- but only if :meth:`VisionPoseSender.send` really is called
  every step. :class:`~sim_loop.SimLoop` does that, which is why the sender lives
  there rather than in each flight script.
* ``EKF2_EV_QMIN`` must stay 0. ``handle_message_vision_position_estimate`` never
  sets ``odom.quality``, so it is always 0 for a ``VISION_POSITION_ESTIMATE`` and
  any positive minimum blocks fusion forever.

Frames: the simulator is ENU with an FLU body; MAVLink's vision estimate is NED
with an FRD body. :func:`enu_flu_to_ned_frd` is the only place that is converted.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

SEND_RATE_HZ = 50.0
"""How often to send, in *simulated* seconds.

Comfortably inside EKF2's 200 ms ``EV_MAX_INTERVAL`` even if a few steps are
missed, and well under the physics rate, so the companion link is not carrying a
pose per 4 ms step that the estimator would only throw away.
"""

_TOLERANCE_S = 1e-9
"""Slack on the due-time comparison.

The simulation clock is a running sum of ``1/250``, so a time that should equal a
multiple of the send interval is routinely a few parts in 10^16 below it. Without
this the step that is exactly due is deferred to the next one.
"""


def enu_flu_to_ned_frd(position, roll: float, pitch: float, yaw: float
                       ) -> Tuple[float, float, float, float, float, float]:
    """Convert a simulator pose (ENU world, FLU body) to MAVLink's NED/FRD.

    Args:
        position: World-frame ``(x_east, y_north, z_up)``, metres.
        roll: Roll about the FLU body x (forward) axis, radians.
        pitch: Pitch about the FLU body y (left) axis, radians.
        yaw: Heading, radians CCW from East.

    Returns:
        ``(north, east, down, roll_frd, pitch_frd, yaw_ned)``.
    """
    east, north, up = position
    return north, east, -up, roll, -pitch, math.pi / 2.0 - yaw


def find_px4_backend(vehicle):
    """The vehicle's ``PX4MavlinkBackend``, or None if it has none.

    Matched on the attribute the sender actually needs rather than on the class,
    so this does not require ``pegasus.simulator`` to be importable -- which it is
    not outside a running Kit app, and this module has to stay readable and
    testable outside one.

    Args:
        vehicle: A Pegasus ``Vehicle``.

    Returns:
        The backend, or None.
    """
    for backend in getattr(vehicle, "_backends", ()):
        if hasattr(backend, "_current_utime"):
            return backend
    return None


class VisionPoseSender:
    """Pushes the vehicle's true pose into PX4's estimator as a vision fix.

    Construct one per aircraft and call :meth:`send` every physics step.
    :class:`~sim_loop.SimLoop` does that when it is given one, which is the only
    wiring any flight script should need.

    Args:
        px4: The :class:`~px4_offboard.PX4Offboard` companion link to send over.
            Pegasus's own ``send_vision_msgs`` cannot be used -- see this
            module's docstring, failure 2.
        vehicle: The Pegasus vehicle to read ground truth from. Its PX4 backend
            is found automatically and read only for its simulation clock.
        rate_hz: How often to send, in simulated time.

    Raises:
        ValueError: If the vehicle has no PX4 backend, which means the pose would
            be stamped on a clock PX4 is not running on. Sending it anyway is the
            failure this class exists to prevent, so it is refused rather than
            degraded.
    """

    def __init__(self, px4, vehicle, rate_hz: float = SEND_RATE_HZ):
        backend = find_px4_backend(vehicle)
        if backend is None:
            raise ValueError(
                "this vehicle has no PX4 MAVLink backend, so there is no lockstep "
                "clock to stamp a vision pose with -- spawn it with use_px4=True"
            )
        self._px4 = px4
        self._vehicle = vehicle
        self._backend = backend
        self._interval = 1.0 / rate_hz
        # A *due* time, advanced by whole intervals rather than snapped to the
        # last send. The naive "sim_time - last_sent < interval" form loses phase:
        # the caller can only send on a step boundary, so each send lands a
        # fraction late, that fraction is carried into the next comparison, and
        # the effective rate falls below the requested one -- measured at 44 Hz
        # for a requested 50. Harmless against EKF2's 200 ms timeout, but a rate
        # limiter that quietly does not hold its rate is not worth having.
        self._due: Optional[float] = None
        self.sent = 0

    def send(self, sim_time: float) -> bool:
        """Send one vision pose if enough simulated time has passed.

        Must be called *after* the backend's own ``update()`` for this step, so
        the timestamp is on the clock PX4 has actually reached: the estimator
        drops a sample stamped in its future, and it is the sensor stream that
        moves PX4's clock forward.

        Args:
            sim_time: Elapsed simulation time, seconds.

        Returns:
            True if a pose was sent this call.
        """
        if self._due is not None and sim_time < self._due - _TOLERANCE_S:
            return False
        # From the due time, not from now, so the phase does not creep. max()
        # re-bases it if the caller was away for longer than one interval, which
        # otherwise leaves a backlog of due times to burn through.
        self._due = max(sim_time, self._due or sim_time) + self._interval

        from scipy.spatial.transform import Rotation

        state = self._vehicle.state
        qx, qy, qz, qw = state.attitude
        roll, pitch, yaw = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz")
        north, east, down, roll_frd, pitch_frd, yaw_ned = enu_flu_to_ned_frd(
            state.position, roll, pitch, yaw,
        )
        # _current_utime is the clock the backend stamps HIL_SENSOR with, and
        # therefore the clock PX4 itself is running on under lockstep.
        self._px4.send_vision_pose(
            north, east, down, roll_frd, pitch_frd, yaw_ned,
            self._backend._current_utime,
        )
        self.sent += 1
        return True
