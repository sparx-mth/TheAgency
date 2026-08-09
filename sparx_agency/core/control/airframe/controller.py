"""The three control stages, assembled.

Trajectory and measured state in; attitude and throttle out. This exists so that
the composition -- which stage feeds which, what resets when, where the thrust
measurement goes -- lives in ``core`` with tests, rather than being reassembled
by hand in each robot's mission loop where it cannot be exercised without a
simulator.

.. code-block:: text

    trajectory ─┐
    state ──────┴─► trajectory_tracking ─► acceleration + heading
                                                  │
                                              flatness ─► attitude + specific thrust
                                                  │
                                            thrust_model ─► throttle

The caller is left with exactly two jobs, and both are platform-specific: put
the attitude and throttle on the wire in whatever frame and units the autopilot
wants, and feed :meth:`observe_thrust` a measured acceleration so the thrust
scale stays honest.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from sparx_agency.core.control.airframe.types import AirframeCommand
from sparx_agency.core.control.flatness import AccelerationLimits, acceleration_to_attitude
from sparx_agency.core.control.thrust_model import ThrustModel, ThrustModelParams
from sparx_agency.core.control.trajectory_tracking import (
    TrajectoryTracker, TrajectoryTrackerParams,
)


class AirframeController:
    """Follows a planned trajectory by commanding attitude and thrust.

    One instance per aircraft, stepped at a steady rate. Stateful throughout --
    the tracker's integrators, the projector's position along the curve, the
    learned thrust scale -- so :meth:`reset` whenever the aircraft stops flying
    the plan.

    Args:
        tracker: Outer-loop gains and limits. Its acceleration limits are the
            ones the flatness stage applies, so the two cannot disagree.
        thrust: Thrust-model bounds and learning rate.
    """

    def __init__(self, tracker=None, thrust=None):
        # type: (Optional[TrajectoryTrackerParams], Optional[ThrustModelParams]) -> None
        self.tracker = TrajectoryTracker(tracker)
        self.thrust_model = ThrustModel(thrust)
        self._deliverable = None      # type: Optional[AccelerationLimits]

    @property
    def limits(self):
        # type: () -> AccelerationLimits
        """The acceleration ceilings, owned by the tracker and shared downward."""
        return self.tracker.params.limits

    def deliverable_limits(self):
        # type: () -> AccelerationLimits
        """The envelope with its thrust ceiling cut to what the airframe can buy.

        ``AccelerationLimits.max_specific_thrust`` is a statement about the
        airframe, but the throttle that actually reaches it is clamped at
        ``max_throttle``, so the real ceiling is
        ``max_throttle * full_scale`` -- and the two are not reconciled
        anywhere. With this aircraft's learned scale they disagree by about
        0.15 g: the limiter permits 15.70 m/s^2 while 0.9 throttle against a
        15.82 m/s^2 scale can only produce 14.24, and the gap widens as the
        learned hover throttle rises.

        Left unreconciled the shortfall does not fall where the module's own
        priority says it should. ``limit_acceleration`` gives horizontal away
        first *precisely* so that altitude survives; but a thrust the throttle
        cannot deliver still produces the commanded ATTITUDE, so the tilt is
        honoured and the missing thrust comes off the vertical instead -- the
        exact inversion, and only at the top of the envelope where it is least
        affordable.

        Recomputed as the scale is learned, and cached so the frozen dataclass
        is not rebuilt at every tick of a 250 Hz loop.
        """
        base = self.limits
        ceiling = (self.thrust_model.params.max_throttle
                   * self.thrust_model.full_scale_mps2)
        if ceiling >= base.max_specific_thrust:
            return base
        # An airframe that cannot deliver even the floor is a configuration
        # error, not a flight condition; keep the dataclass constructible and
        # let the saturation flag report it rather than raising mid-flight.
        ceiling = max(ceiling, base.min_specific_thrust * 1.000001)
        if (self._deliverable is None
                or abs(self._deliverable.max_specific_thrust - ceiling) > 1e-3):
            self._deliverable = replace(base, max_specific_thrust=ceiling)
        return self._deliverable

    @property
    def trajectory_id(self):
        # type: () -> int
        """Id of the trajectory being flown, or -1 when none is."""
        return self.tracker.trajectory_id

    @property
    def hover_throttle(self):
        # type: () -> float
        """Throttle currently believed to hold a hover."""
        return self.thrust_model.hover_throttle

    def reset(self, yaw=None, hold_position=None, forget_thrust=False):
        # type: (Optional[float], Optional[object], bool) -> None
        """Clear the accumulated state before a new phase of flight.

        Args:
            yaw: Heading to start slewing the command from.
            hold_position: Where to hold while nothing is being followed.
            forget_thrust: Also discard the learned thrust scale. Off by
                default, and that default is deliberate: the scale is a property
                of the airframe and its battery, not of the manoeuvre, so it
                survives a handover that the position integrators must not. Turn
                it on between *flights*, not between phases.
        """
        self.tracker.reset(yaw=yaw, hold_position=hold_position)
        if forget_thrust:
            self.thrust_model.reset()

    def set_trajectory(self, trajectory):
        # type: (object) -> bool
        """Queue a newly planned trajectory. See ``TrajectoryTracker``."""
        return self.tracker.set_trajectory(trajectory)

    def update(self, position, velocity, yaw, dt, now_s, follow=True):
        # type: (object, object, float, float, float, bool) -> AirframeCommand
        """Advance one control tick through all three stages.

        Args:
            position: Measured world ``(x, y, z)``, metres.
            velocity: Measured world ``(vx, vy, vz)``, m/s.
            yaw: Measured heading, radians CCW from world +x.
            dt: Seconds since the previous call. Must be > 0.
            now_s: Current time, on the clock the trajectories are stamped on.
            follow: False holds station instead of tracking.

        Returns:
            The attitude and throttle to command, with every stage's diagnostics.
        """
        # ONE envelope, both stages. The thrust ceiling the throttle can
        # actually buy is below the configured one from the 0.62 seed onward, so
        # if only the flatness stage knew about it the tracker would report
        # `saturated` False while its command was being trimmed -- and the
        # integrator, which freezes on that flag, would keep charging against a
        # correction that never reaches the airframe.
        limits = self.deliverable_limits()
        tracking = self.tracker.update(position, velocity, yaw, dt, now_s,
                                       follow=follow, limits=limits)
        attitude = acceleration_to_attitude(
            tracking.acceleration(), tracking.yaw,
            jerk=tracking.jerk(), yaw_rate=tracking.yaw_rate, limits=limits)
        return AirframeCommand(
            attitude=attitude,
            throttle=self.thrust_model.normalized(attitude.specific_thrust_mps2),
            tracking=tracking,
            hover_throttle=self.thrust_model.hover_throttle)

    def observe_thrust(self, commanded_throttle, acceleration_world, body_z_world, dt):
        # type: (float, object, object, float) -> bool
        """Feed the thrust model one throttle-versus-acceleration measurement.

        Call once per tick with the throttle actually sent and the acceleration
        that resulted. Skipping it is not fatal -- the model falls back to its
        seed hover throttle -- but the seed is a guess, and a wrong one is a
        standing bias on the vertical axis that the position integrator absorbs
        and then hides.

        Args:
            commanded_throttle: The throttle that was sent, in [0, 1].
            acceleration_world: Measured world acceleration, m/s^2, +z up.
            body_z_world: The aircraft's measured thrust axis in world
                coordinates, unit length. Its *measured* one, not the commanded
                one -- using the command would make the estimate agree with
                itself.
            dt: Seconds since the previous observation.

        Returns:
            True if the observation was accepted.
        """
        return self.thrust_model.observe(commanded_throttle, acceleration_world,
                                         body_z_world, dt)
