"""The PX4 parameter sets a simulated indoor drone needs to fly reliably.

PX4 ships configured for a real aircraft under open sky with a pilot and a radio
attached. Every one of those assumptions is false here, and each false one costs
either a failed arming or a flight into furniture. These are the three groups
that make the difference, kept apart because they answer three different
questions and you will want to change them independently.

Two things about *how* they are applied matter as much as the values:

* **Nothing here needs a restart, and that is deliberate.** EKF2 sizes its
  observation buffers exactly once, on the first IMU sample after ``ekf2
  start``, so every ``*_DELAY`` parameter is read once and then ignored -- a
  runtime change to one is silently inert. Rather than build a
  push-parameters-then-reboot dance into the collector (which would mean tearing
  down and re-establishing the simulator's HIL link mid-campaign), the set below
  is restricted to parameters that take effect immediately.
  :data:`REBOOT_REQUIRED` names the ones deliberately left out.
* **PX4 persists every parameter it is sent.** A run that experiments with, say,
  the estimator's aiding source silently leaves that setting in place for every
  later run in the same working directory. That is a real trap this repo has
  already fallen into once. The collector deletes the parameter file before each
  campaign so a run's configuration is exactly what is written here.
"""
from __future__ import annotations

INDOOR_LIMITS = {
    # These are ceilings, not targets: the route is flown by the guidance law in
    # path_follower, and PX4 only has to be able to deliver what it asks for.
    # Each must sit comfortably ABOVE FollowSpec's corresponding limit -- if PX4
    # is the binding constraint the aircraft silently falls behind its carrot
    # instead of tracking the route.
    "MPC_XY_VEL_MAX": 2.0,      # m/s, horizontal speed ceiling
    "MPC_XY_CRUISE": 1.5,       # m/s
    "MPC_ACC_HOR_MAX": 2.5,     # m/s^2, enough to hold speed through a curve
    "MPC_ACC_HOR": 2.0,         # m/s^2
    "MPC_JERK_AUTO": 4.0,       # m/s^3, smooths the corners
    "MPC_TILTMAX_AIR": 25.0,    # deg, shallow tilt keeps the camera useful
    "MPC_Z_VEL_MAX_UP": 1.0,    # m/s
    "MPC_Z_VEL_MAX_DN": 0.7,    # m/s
    # The follower slews the heading itself, so this only has to not be the
    # binding limit. How fast the world actually rotates in the recorded
    # imagery is FollowSpec.max_yaw_rate.
    "MPC_YAWRAUTO_MAX": 25.0,   # deg/s
    # Takeoff is the least stable moment: MPC_TILTMAX_AIR does not govern the
    # takeoff ramp, and snapping to full climb thrust while still in ground
    # contact has tipped the airframe onto its back (roll -150 deg two seconds
    # after arming). Ramp the thrust in and climb slowly instead.
    "MPC_TKO_RAMP_T": 3.0,      # s, thrust ramp-up time
    "MPC_TKO_SPEED": 0.5,       # m/s, initial climb rate
    "MPC_LAND_SPEED": 0.4,      # m/s, touchdown rate
}
"""Flight envelope for a furnished room.

PX4's defaults are tuned for open sky: ~12 m/s cruise, 45 deg of tilt,
aggressive acceleration. Indoors that overshoots every waypoint into a wall --
a ``simple_room`` run reached the far wall at 4.35 m and finished the flight
nose-down on the floor. These are the limits an operator would set on a real
indoor drone: slow, gentle, shallow.
"""

SIM_ESTIMATOR = {
    # The simulated GPS is exact (see robots/PEGASUS/adapters/sensors.py), so
    # tell the estimator to believe it. The stock 0.5 m position noise models a
    # real receiver and makes PX4 average away a fix that has nothing to average.
    # Not pushed to the 0.01 floor: an estimator that trusts one sensor
    # absolutely has nowhere to put a transient disagreement except into its
    # bias states, which is how an accel bias runs away.
    "EKF2_GPS_P_NOISE": 0.1,    # m,   default 0.5
    "EKF2_GPS_V_NOISE": 0.1,    # m/s, default 0.3
    "EKF2_GPS_P_GATE": 10.0,    # SD, wide enough that a good fix is never rejected
    "EKF2_GPS_V_GATE": 10.0,
    # PX4's GNSS quality gates read fields a simulated receiver invents
    # (satellite count, DOP, drift-while-stationary). Gating a perfect fix on
    # synthetic quality metrics only ever costs a delayed or refused arming.
    "EKF2_GPS_CHECK": 0,
    "EKF2_REQ_GPS_H": 0.5,      # s, don't sit through 10 s of "waiting for healthy GPS"
    # Drop the barometer out of height fusion entirely. Height already comes
    # from the exact GPS (EKF2_HGT_REF defaults to 1 = GNSS), and the barometer
    # is the one Pegasus sensor whose noise is not configurable -- worse, PX4
    # sees two sensor_baro instances in this setup and was observed switching to
    # the stale one mid-flight ("BARO switch from #0 -> #1", then "BARO #1
    # failed: STALE"). A height source that goes stale under the estimator is
    # exactly how a vertical accel bias grows.
    "EKF2_BARO_CTRL": 0,
    # The magnetometer is kept -- it is noiseless here, and it is what makes yaw
    # observable while the aircraft is stationary on the ground, which GNSS-only
    # yaw is not. Only its field-strength sanity check goes: Pegasus's simulated
    # field does not have to agree with PX4's world magnetic model.
    "EKF2_MAG_CHECK": 0,
    # Estimate a gyro bias but not an accelerometer bias (bit 0 only, default 3).
    # The simulated IMU has no accelerometer bias to find, so the state has
    # nothing to converge on and everything to absorb: PhysX's contact jitter
    # while the aircraft rests on the floor drove it past PX4's arming limit and
    # blocked every flight after the first with "Preflight Fail: High
    # Accelerometer Bias". A bias state is only worth having when there is a
    # bias.
    "EKF2_IMU_CTRL": 1,
    "EKF2_ABL_LIM": 1.0,        # m/s^2, default 0.4 -- belt and braces for the same check
}
"""Make PX4's estimator believe the (now exact) simulated sensors.

Left deliberately alone: ``EKF2_GPS_CTRL``, ``EKF2_HGT_REF``, ``EKF2_MAG_TYPE``
and every IMU process-noise parameter. Switching the aiding source is what the
unfinished external-vision work (``px4_vision_pose.py``) does, and it is a much
larger change than making the existing source accurate.

Tightening the IMU process noise to match the (noiseless) simulated IMU was
tried and **reverted**: it makes the filter trust the accelerometer absolutely,
so every transient disagreement with GPS lands in the accel-bias state instead.
The result was ``Preflight Fail: High Accelerometer Bias`` from the moment of
takeoff and a refusal to re-arm for the next episode. Exact sensors do not want
an over-confident filter on top of them.
"""

SIM_SENSOR_CALIBRATION = {
    # PX4 continuously learns sensor offsets while disarmed. That is right for
    # hardware and wrong here twice over: the simulated sensors have no offsets
    # to find, and the aircraft creeps slightly on the floor between flights, so
    # what gets "learned" is the creep. One run taught the gyro a 0.026 rad/s
    # offset (1.5 deg/s) that the estimator then had to fight for the rest of
    # the campaign.
    "IMU_GYRO_CAL_EN": 0,
    "SENS_IMU_AUTOCAL": 0,
    "SENS_MAG_AUTOCAL": 0,
}
"""Stop PX4 calibrating away offsets that do not exist."""

SIM_ARMING = {
    # No radio, no pilot, no ground station. Every check that assumes one is a
    # refused arming with no way to clear it.
    "COM_RC_IN_MODE": 4,        # stick input disabled entirely
    "NAV_RCL_ACT": 0,           # no RC-loss failsafe
    "NAV_DLL_ACT": 0,           # no data-link-loss failsafe
    "COM_ARM_MAG_STR": 0,       # don't refuse to arm on "magnetic interference"
    "COM_ARM_MAG_ANG": -1,      # no inter-magnetometer consistency check
    "COM_ARM_SDCARD": 0,        # SITL has no SD card
    "COM_ARM_ARSP_EN": 0,       # multicopter: no airspeed sensor
    "COM_ARM_WO_GPS": 1,
    "COM_LOW_BAT_ACT": 0,       # simulated battery, warning only
    # An indoor flight legitimately holds position tightly for a long time; the
    # stock position failsafe is tuned for losing satellites outdoors.
    "COM_POS_FS_DELAY": 100,
    "GF_ACTION": 0,             # no geofence -- the map is the fence
    # The estimator innovation gates, opened to their maximum. These are the
    # checks that decide whether the aircraft may arm *again* after a flight,
    # and a campaign lives or dies on them: a landing leaves the estimator
    # briefly unsettled, and at the stock 0.5 the next episode is refused with
    # "position estimate error" / "horizontal velocity unstable" and the worker
    # stops with one recording. 1.0 is the documented maximum, not a bypass --
    # the aircraft still cannot arm without a valid local position at all, and
    # that check is not parameterisable.
    "COM_ARM_EKF_POS": 1.0,
    "COM_ARM_EKF_VEL": 1.0,
    "COM_ARM_EKF_HGT": 1.0,
    "COM_ARM_EKF_YAW": 1.0,
    "COM_ARM_IMU_ACC": 1.0,
    "COM_ARM_IMU_GYR": 0.3,
}
"""Preflight checks and failsafes that only make sense with a pilot present.

None of these relax anything that protects the *flight*: the aircraft still
refuses to arm without a valid local position estimate (``modeCheck.cpp``), and
that check cannot be parameterised away -- which is exactly as it should be,
since a data-collection run with a broken estimator produces poisoned data
rather than a visible failure.
"""

REBOOT_REQUIRED = ("EKF2_GPS_DELAY", "EKF2_MAG_TYPE", "EKF2_DECL_TYPE")
"""Parameters that would help, but that EKF2 only reads at startup.

``EstimatorInterface::initialise_interface`` runs on the first IMU sample after
``ekf2 start`` and its ``_initialised`` flag is never cleared, so setting any
``*_DELAY`` at runtime changes the stored value and nothing else.

The one that costs something to leave out is ``EKF2_GPS_DELAY``: its 110 ms
default models a real receiver, while HIL_GPS has no latency at all, so the
estimator fuses each fix as if it were 110 ms stale. That is a position error
proportional to speed -- about 11 cm at the 1 m/s this flies at, well inside the
planner's standoff, and it buys back a larger observation buffer. Not worth a
restart. Set it by hand in PX4's console (``param set EKF2_GPS_DELAY 20`` then
``ekf2 stop && ekf2 start``) if you ever need the last centimetres.
"""


def all_params() -> dict:
    """Every parameter a simulated collection flight should run with.

    Returns:
        Name to value. The value's **Python type selects the MAVLink parameter
        type** -- see :meth:`px4_offboard.PX4Offboard.set_params` -- so the
        int/float split in the dicts above is load-bearing, not cosmetic.
    """
    merged = {}
    for group in (INDOOR_LIMITS, SIM_ESTIMATOR, SIM_SENSOR_CALIBRATION, SIM_ARMING):
        merged.update(group)
    return merged


def needs_restart(params: dict) -> bool:
    """Whether applying ``params`` requires PX4 to be restarted to take effect."""
    return any(name in params for name in REBOOT_REQUIRED)
