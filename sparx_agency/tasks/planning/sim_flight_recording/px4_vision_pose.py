"""Feed PX4 a precise vision/mocap pose instead of simulated GPS.

**INCOMPLETE — opt in with ``fly_px4.py --vision``, and expect it not to arm.**
See "Current state" below before using it.

*Why it exists.* Pegasus's PX4 backend simulates a GPS receiver, noise and all.
Outdoors that is the right model; indoors it is fatal. Flying ``office`` on
GPS, PX4's position hold wandered 2-5 m around each setpoint -- more than the
clearance between the desks -- and the drone hit furniture on a leg the route
planner had verified as clear. Real indoor drones do not use GPS for this; they
fuse a visual-inertial or motion-capture pose. This module supplies the
simulated equivalent, from the simulator's ground truth.

*Current state.* With :data:`VISION_EKF_PARAMS` applied, PX4 accepts the
parameters but then refuses to arm with ``Preflight Fail: ekf2 missing data``
-- the estimator has been told to depend on vision and is not receiving any.
Two causes have been found and fixed and the symptom persists, so at least one
more remains:

1. **Parameter type mismatch.** Four of these EKF2 parameters are INT32.
   Sending them as REAL32 made PX4 reject them (``ERROR [mavlink] param types
   mismatch``) and silently keep using GPS. Fixed -- see
   :meth:`px4_offboard.PX4Offboard.set_params`.
2. **Wrong MAVLink link.** Pegasus's own ``send_vision_msgs`` writes to the HIL
   connection, where PX4's ``simulator_mavlink`` ignores everything that is not
   a ``HIL_*`` message. Fixed by sending over the companion link instead.

Remaining suspects, untested: the ``time_usec`` stamp not lining up with PX4's
lockstep clock closely enough for the EKF's acceptance window, and EKF2
requiring a reboot rather than a runtime parameter change to switch aiding
source.

Frames: the simulator is ENU with an FLU body; MAVLink's vision estimate is NED
with an FRD body.
"""
from __future__ import annotations

import math

# EKF2 settings that switch PX4 from GPS to external vision, applied over
# MAVLink before arming.
#
# The int/float split here is load-bearing, not cosmetic: PX4 rejects a
# parameter whose MAVLink type does not match its own ("ERROR [mavlink] param
# types mismatch param: EKF2_GPS_CTRL"), and the rejection is only visible in
# PX4's console -- the flight simply carries on using GPS. Sending all seven of
# these as REAL32 silently left vision fusion switched off entirely.
VISION_EKF_PARAMS = {
    "EKF2_EV_CTRL": 11,     # INT32 bitmask: horizontal position (1) + vertical (2) + yaw (8)
    "EKF2_GPS_CTRL": 0,     # INT32; stop fusing GPS entirely
    "EKF2_HGT_REF": 3,      # INT32; take the height reference from vision, not baro/GPS
    "EKF2_EV_NOISE_MD": 1,  # INT32; use the parameter noise below, not the message covariance
    "EKF2_EV_DELAY": 0.0,   # ms; the simulated pose is not delayed
    "EKF2_EVP_NOISE": 0.02,  # m, position noise -- this pose is ground truth
    "EKF2_EVA_NOISE": 0.02,  # rad, angle noise
}

# The pose only needs to arrive fast enough for the EKF; matching the physics
# rate would just flood the link.
SEND_RATE_HZ = 30.0


def enu_flu_to_ned_frd(position, roll: float, pitch: float, yaw: float):
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


class VisionPoseSender:
    """Pushes the vehicle's true pose into PX4's estimator as a vision fix.

    Args:
        px4: The :class:`~px4_offboard.PX4Offboard` companion link to send over.
            Pegasus's own ``send_vision_msgs`` cannot be used: it writes to the
            HIL connection, where PX4 ignores everything that is not a ``HIL_*``
            message.
        backend: The vehicle's ``PX4MavlinkBackend``, read only for its
            simulation clock.
        rate_hz: How often to send, in simulated time.
    """

    def __init__(self, px4, backend, rate_hz: float = SEND_RATE_HZ):
        self._px4 = px4
        self._backend = backend
        self._interval = 1.0 / rate_hz
        self._last_sent = -1.0

    def send(self, vehicle, sim_time: float) -> bool:
        """Send one vision pose if enough simulated time has passed.

        Must be called *after* the backend's own ``update()`` for the step, so
        the timestamp matches the sensor stream PX4's lockstep clock is
        following.

        Args:
            vehicle: The Pegasus vehicle to read ground truth from.
            sim_time: Elapsed simulation time, seconds.

        Returns:
            True if a pose was sent this call.
        """
        if sim_time - self._last_sent < self._interval:
            return False
        self._last_sent = sim_time

        from scipy.spatial.transform import Rotation

        qx, qy, qz, qw = vehicle.state.attitude
        roll, pitch, yaw = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz")
        north, east, down, roll_frd, pitch_frd, yaw_ned = enu_flu_to_ned_frd(
            vehicle.state.position, roll, pitch, yaw,
        )

        # _current_utime is the clock the backend stamps HIL_SENSOR with, and
        # therefore the clock PX4 itself is running on under lockstep; the EKF
        # rejects vision samples that do not line up with it.
        self._px4.send_vision_pose(
            north, east, down, roll_frd, pitch_frd, yaw_ned,
            self._backend._current_utime,
        )
        return True
