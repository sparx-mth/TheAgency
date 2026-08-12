"""The PX4 parameter sets a simulated indoor drone needs to fly reliably.

PX4 ships configured for a real aircraft under open sky with a pilot and a radio
attached. Every one of those assumptions is false here, and each false one costs
either a failed arming or a flight into furniture. The groups below are kept
apart because they answer different questions and you will want to change them
independently.

**There are two channels, and which one a parameter belongs on is a property of
the parameter, not a preference.**

* :func:`boot_params` is written into PX4's ``px4-rc.params`` startup hook and
  applied *before ``ekf2`` starts* -- see :func:`px4_launch.write_boot_parameters`.
  Anything PX4 marks ``@reboot_required`` has to go here: EKF2 sizes its
  observation buffers and picks its height reference exactly once, on the first
  IMU sample after ``ekf2 start``, so a runtime change to one of those is
  silently inert. This channel did not exist while the estimator ran on GPS,
  which is why the notes in :data:`REBOOT_REQUIRED` used to end in "not worth a
  restart".
* :func:`all_params` is pushed over MAVLink after the first heartbeat, for
  everything that takes effect immediately.

**PX4 persists every parameter it is sent.** A run that experiments with, say,
the estimator's aiding source silently leaves that setting in place for every
later run in the same working directory. That is a real trap this repo has
already fallen into once, in both directions: ``EKF2_GPS_CTRL=0`` outlived the
flag that set it and broke every later flight. The collector deletes both the
parameter file *and* any stale boot script before each campaign, so a run's
configuration is exactly what is written here.
"""
from __future__ import annotations

from typing import Dict

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
    "EKF2_ABL_LIM": 1.0,        # m/s^2, default 0.4 -- belt and braces for the same check
}
"""Estimator settings that hold whatever the aiding source is.

``EKF2_IMU_CTRL`` deliberately lives in the two aiding groups instead of here:
which IMU bias states are worth estimating depends on whether the estimator has
an absolute attitude reference to correct a wrong one against, and the two
configurations differ on exactly that.

Tightening the IMU process noise to match the (noiseless) simulated IMU was
tried and **reverted**: it makes the filter trust the accelerometer absolutely,
so every transient disagreement with the aiding source lands in the accel-bias
state instead. The result was ``Preflight Fail: High Accelerometer Bias`` from
the moment of takeoff and a refusal to re-arm for the next episode. Exact
sensors do not want an over-confident filter on top of them.
"""

GPS_ESTIMATOR = {
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
    # Keep the magnetometer's field-strength sanity check off: Pegasus's
    # simulated field does not have to agree with PX4's world magnetic model.
    "EKF2_MAG_CHECK": 0,
    # Estimate a gyro bias but not an accelerometer bias (bit 0 only, default 3).
    # The simulated IMU has no accelerometer bias to find, so the state has
    # nothing to converge on and everything to absorb: PhysX's contact jitter
    # while the aircraft rests on the floor drove it past PX4's arming limit and
    # blocked every flight after the first with "Preflight Fail: High
    # Accelerometer Bias". A bias state is only worth having when there is a
    # bias. The gyro-bias state survives here only because the magnetometer keeps
    # heading observable, so a wrong one gets corrected -- see
    # VISION_ESTIMATOR, where it does not.
    "EKF2_IMU_CTRL": 1,
}
"""Aiding from the simulated GNSS receiver and the magnetometer.

**This is the configuration that produced the corrupted campaign**, and it is
kept only so the two can be compared. It leaves yaw observable *only* through
the magnetometer, and Pegasus publishes one magnetometer where PX4's SITL
airframe declares two (``CAL_MAG0_ID`` and ``CAL_MAG1_ID`` in
``init.d-posix/rcS``). Measured over 139 worker runs of one campaign:
``MAG #0 failed: STALE`` in 128 of them, ``Compass needs calibration - Land
now!`` in 79, ``Failsafe activated`` in 108, and ``Landing at current position``
3694 times across 120 -- PX4 seizing the aircraft mid-route and force-landing it
while our follower was still steering. ``estimator_drift_m`` came out at 0.56 m
median on the flights that survived and 8.4 m on the ones that crashed.

Prefer :data:`VISION_ESTIMATOR`. See ``README.md``.
"""

VISION_ESTIMATOR = {
    # Fuse the external pose for horizontal position (1), vertical position (2)
    # and yaw (8). Not 3D velocity (4): VISION_POSITION_ESTIMATE carries no
    # velocity, and asking EKF2 to fuse a field the message cannot supply stops
    # fusion starting at all. The build's own default is 15, i.e. EV aiding was
    # always enabled -- it simply never had data.
    "EKF2_EV_CTRL": 11,
    # And nothing else. Both of the sources that used to provide the horizontal
    # reference are now off, which is the entire point: the pose being fused is
    # the simulator's ground truth, so a second, worse opinion can only pull the
    # estimate away from it.
    "EKF2_GPS_CTRL": 0,
    "EKF2_BARO_CTRL": 0,
    "EKF2_HGT_REF": 3,          # height reference = vision. @reboot_required.
    # Magnetometer gone, at both levels. EKF2_MAG_TYPE 5 ("None") stops the
    # estimator fusing it under any circumstance, and SYS_HAS_MAG 0 stops PX4
    # expecting the sensor at all -- without the second one the aircraft still
    # refuses to arm on a stale compass it is no longer using. Yaw comes from
    # the vision pose, which is exact and observable standing still, so there is
    # nothing left for a magnetometer to contribute. Both @reboot_required.
    "EKF2_MAG_TYPE": 5,
    "SYS_HAS_MAG": 0,
    # Same argument for the barometer: out of height fusion above, and out of
    # the sensor set here, which is what silences the "BARO #0 failed: STALE" /
    # "BARO switch #0 -> #1" churn rather than merely ignoring its output.
    "SYS_HAS_BARO": 0,
    "EKF2_EV_DELAY": 0.0,       # ms; the pose is generated in step, not delayed
    # Take the observation noise from the two parameters below rather than from
    # the message covariance, which handle_message_vision_position_estimate
    # never fills in.
    "EKF2_EV_NOISE_MD": 1,
    # Do NOT tighten these towards zero because the pose is exact. Measured, at
    # EKF2_EVP_NOISE 0.02 m with the stock 5 SD gate: EV position fused exactly
    # once (test ratio 0.42) and every sample afterwards was rejected with ratios
    # of 20 to 780, while the aircraft sat still and the observation never moved.
    # A measurement declared that precise makes the innovation gate narrower than
    # the filter's own prediction error, so the first transient disagreement locks
    # the measurement out; the position is then unaided, drifts on pure IMU
    # integration, and the innovation grows -- which rejects it harder. It ended
    # 6.8 m and 33 deg from ground truth with 2256 perfect poses delivered.
    #
    # 0.1 m / 10 SD is deliberately the same treatment GPS_ESTIMATOR gives the
    # (equally exact) simulated GNSS, for the same reason its docstring gives:
    # an estimator that trusts one sensor absolutely has nowhere to put a
    # disagreement. Still two orders of magnitude better than real VIO.
    "EKF2_EVP_NOISE": 0.1,      # m,   default 0.1. Param floor 0.01.
    "EKF2_EVA_NOISE": 0.05,     # rad, default 0.1. 0.05 *is* the param's floor --
                                # anything smaller is silently clamped up to it.
    # The gates, opened wide. This is the pair that decides whether a *correct*
    # measurement can be thrown away, and with ground truth on the wire the answer
    # must be no. The noise above stays tight, so steady-state accuracy is
    # unchanged -- only the rejection behaviour is.
    #
    # EKF2_HDG_GATE is the one that mattered most and is easy to miss, because its
    # name says "magnetic heading": fuseYaw() lives in mag_fusion.cpp and EV yaw
    # goes through it too. With no magnetometer the filter aligns yaw to zero and
    # declares itself aligned, so the first vision yaw arrives as an innovation of
    # up to 180 deg. At the stock 2.6 SD that tested at 15x the gate and was
    # rejected forever: PX4 held a heading 130 deg from truth, with the right
    # answer arriving 50 times a second, and refused to arm on "Yaw estimate
    # error". 30 SD accepts even a 180 deg initial misalignment, after which
    # ordinary fusion holds it.
    "EKF2_EVP_GATE": 30.0,      # SD, default 5
    "EKF2_HDG_GATE": 30.0,      # SD, default 2.6
    # Must stay 0. handle_message_vision_position_estimate never sets
    # odom.quality, so it is always 0 for a VISION_POSITION_ESTIMATE and any
    # positive minimum blocks fusion forever.
    "EKF2_EV_QMIN": 0,
    # The pose describes the airframe origin, not a camera mounted off it.
    "EKF2_EV_POS_X": 0.0,
    "EKF2_EV_POS_Y": 0.0,
    "EKF2_EV_POS_Z": 0.0,
    # Estimate no IMU bias at all (default 3: gyro + accel). GPS_ESTIMATOR keeps
    # the gyro-bias state because the magnetometer keeps heading observable, so a
    # wrong bias is corrected. With the magnetometer gone there is nothing to
    # correct it against for the first seconds of a run -- external vision has not
    # started fusing yet -- and the aircraft spends exactly those seconds dropping
    # the last centimetres onto the floor. Measured: 0.54 rad/s of real PhysX
    # contact rotation at t=0.2 s was absorbed as a gyro bias that saturated
    # EKF2_GYR_B_LIM at 0.150 rad/s on all three axes and stayed there. PX4's yaw
    # estimate then rotated at 8.6 deg/s for the rest of the run, sweeping past
    # the correct heading roughly every 35 s, while a perfect vision yaw arrived
    # 50 times a second with too little authority to stop it. The simulated gyro
    # is exactly bias-free, so the state has nothing to find and everything to
    # absorb -- the same argument GPS_ESTIMATOR already makes for the
    # accelerometer.
    "EKF2_IMU_CTRL": 0,
}
"""Aiding from the simulator's own ground-truth pose, fed in as mocap.

The accurate position was always available -- ``vehicle.state.position`` is
exact and every follower in this package already reads it. What was missing is
that *PX4* had no access to it, and PX4 holds the veto: its estimator ran on a
simulated magnetometer and a simulated GNSS receiver, and when those disagreed
with reality it declared a failsafe and landed the aircraft out from under the
follower. This group closes that gap by making PX4's estimate the same number
the rest of the stack already uses.

Every one of these must be applied **before ``ekf2`` starts** -- three are
``@reboot_required`` outright and the rest are only useful if fusion begins
cleanly on the first update -- so they belong on :func:`boot_params`, never on
the MAVLink push. See :mod:`px4_vision_pose` for the sender and for the four
ways this went wrong before it worked.
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

REBOOT_REQUIRED = (
    "EKF2_HGT_REF", "EKF2_MAG_TYPE", "EKF2_EV_DELAY", "EKF2_GPS_DELAY",
    "EKF2_DECL_TYPE", "SYS_HAS_MAG", "SYS_HAS_BARO",
)
"""Parameters EKF2 (or the sensors module) only reads at startup.

``EstimatorInterface::initialise_interface`` runs on the first IMU sample after
``ekf2 start`` and its ``_initialised`` flag is never cleared, so setting any
``*_DELAY`` at runtime changes the stored value and nothing else. PX4 marks
``EKF2_HGT_REF``, ``EKF2_MAG_TYPE`` and ``EKF2_EV_DELAY`` ``@reboot_required``
itself; ``SYS_HAS_*`` gate which sensors the ``sensors`` module instantiates,
which likewise happens once.

These are no longer "deliberately left out" -- :func:`boot_params` applies them
through PX4's own ``px4-rc.params`` hook, which ``rcS`` sources after the
airframe file and *before* ``rc.vehicle_setup`` starts ``ekf2``. That is the only
window in which they mean anything, and it needs no restart because nothing has
started yet.

``EKF2_GPS_DELAY`` is listed for completeness and is irrelevant under
:data:`VISION_ESTIMATOR`, which does not fuse GNSS at all.
"""


def boot_params(vision: bool) -> Dict[str, object]:
    """Parameters that must be in place before ``ekf2`` starts.

    Written into an instance-private ``px4-rc.params`` by
    :func:`px4_launch.write_boot_parameters`.

    Args:
        vision: Fuse the simulator's ground-truth pose as external vision
            (:data:`VISION_ESTIMATOR`). False leaves PX4 on GNSS + magnetometer,
            which needs nothing set this early -- see :data:`GPS_ESTIMATOR` for
            what that costs.

    Returns:
        Name to value, possibly empty. The value's Python type selects how it is
        written, so the int/float split in the dicts above is load-bearing.
    """
    return dict(VISION_ESTIMATOR) if vision else {}


def all_params(vision: bool) -> Dict[str, object]:
    """Every parameter a collection flight pushes over MAVLink once PX4 is up.

    Neither this nor :func:`boot_params` has a default, deliberately. The aiding
    source decides whether a flight is trustworthy, the two channels have to
    agree about it, and a wrong answer is invisible in flight -- so every caller
    states it. That is one line of noise against the class of bug that cost a
    700-episode campaign.

    Args:
        vision: Match :func:`boot_params`. When True the GNSS aiding tuning is
            left out, because nothing fuses GNSS; passing it anyway would be
            harmless but would leave the parameter store describing a
            configuration that is not the one flying.

    Returns:
        Name to value. The value's **Python type selects the MAVLink parameter
        type** -- see :meth:`px4_offboard.PX4Offboard.set_params` -- so the
        int/float split in the dicts above is load-bearing, not cosmetic.

    Raises:
        ValueError: If a ``@reboot_required`` parameter reached this set, where
            it would be accepted, acknowledged, saved and ignored. That silent
            no-op cost a day the first time; it is an error now.
    """
    groups = [INDOOR_LIMITS, SIM_ESTIMATOR, SIM_SENSOR_CALIBRATION, SIM_ARMING]
    if not vision:
        groups.insert(2, GPS_ESTIMATOR)
    merged: Dict[str, object] = {}
    for group in groups:
        merged.update(group)

    inert = sorted(set(merged) & set(REBOOT_REQUIRED))
    if inert:
        raise ValueError(
            f"{', '.join(inert)} only take effect at ekf2 start, so pushing them "
            f"over MAVLink does nothing visible. Move them to boot_params()."
        )
    return merged


def needs_restart(params: dict) -> bool:
    """Whether applying ``params`` requires PX4 to be restarted to take effect."""
    return any(name in params for name in REBOOT_REQUIRED)
