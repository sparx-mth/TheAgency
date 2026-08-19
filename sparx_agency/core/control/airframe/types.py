"""What the assembled control chain emits.

Everything the three stages produced, kept together rather than collapsed to the
two numbers the autopilot needs. The extra fields cost nothing to carry and are
the difference between "the flight diverged" and knowing which stage let go --
whether the outer loop was already off the plan, or the command saturated, or
the thrust model was still learning.
"""
from __future__ import annotations

from dataclasses import dataclass

from sparx_agency.core.control.flatness.types import AttitudeThrustCommand
from sparx_agency.core.control.trajectory_tracking.types import AccelerationCommand


@dataclass(frozen=True)
class AirframeCommand:
    """One tick of the whole chain: what to send, and how it was arrived at.

    Attributes:
        attitude: The attitude and specific thrust from the flatness stage.
        throttle: Normalized collective thrust, ready for the autopilot.
        tracking: The outer loop's output and diagnostics, unmodified.
        hover_throttle: What the thrust model currently believes a hover costs.
            Logged every tick because it should move *slowly*: a jump means an
            observation got through that should not have, and every gain above
            it is implicitly tuned against this number.
    """

    attitude: AttitudeThrustCommand
    throttle: float
    tracking: AccelerationCommand
    hover_throttle: float

    # The tracker's diagnostics are re-exposed here rather than left one
    # attribute deeper. A caller that swaps this chain in for the velocity-cut
    # controller reads the same names off the same place, so the swap is a
    # one-line change at the call site instead of a rename at every use --
    # and forgetting one of them is an AttributeError minutes into a flight,
    # from a status line, which is where this was actually found.

    @property
    def holding(self):
        # type: () -> bool
        """True when the chain is holding station rather than following a plan."""
        return self.tracking.holding

    @property
    def position_error_m(self):
        # type: () -> float
        """Distance to where the plan says the aircraft should be right now."""
        return self.tracking.position_error_m

    @property
    def reference_z_m(self):
        # type: () -> float
        """Altitude the plan asked for, world frame. NaN while holding station.

        Paired with the aircraft's measured altitude this separates a planned
        descent from a sagging one, which is the difference between a planning
        fault and a control fault when the aircraft ends up on top of an
        obstacle.
        """
        return self.tracking.reference_z_m

    @property
    def along_track_lag_m(self):
        # type: () -> float
        """How far behind schedule the aircraft is. Benign; being late is fine."""
        return self.tracking.along_track_lag_m

    @property
    def cross_track_error_m(self):
        # type: () -> float
        """How far off the path it is. This is the one that flies into walls."""
        return self.tracking.cross_track_error_m

    @property
    def yaw_error_rad(self):
        # type: () -> float
        """Reference heading minus measured, wrapped."""
        return self.tracking.yaw_error_rad

    @property
    def yaw(self):
        # type: () -> float
        """Commanded heading, radians CCW from world +x."""
        return self.tracking.yaw

    @property
    def trajectory_id(self):
        # type: () -> int
        """Which FALCON trajectory is being flown, or -1 for none."""
        return self.tracking.trajectory_id

    @property
    def diverged(self):
        # type: () -> bool
        """True while the aircraft is too far off the plan to be tracking it."""
        return self.tracking.diverged

    @property
    def past_end(self):
        # type: () -> bool
        """True when the trajectory has run out and the endpoint is being flown to."""
        return self.tracking.past_end

    @property
    def saturated(self):
        # type: () -> bool
        """True when the command hit the airframe's acceleration limits."""
        return self.tracking.saturated
