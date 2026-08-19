"""What the velocity backend emits, and what it reports about itself.

The command is a **body-frame velocity plus a yaw rate** -- the twist an
autopilot that owns its own velocity loop expects. It is deliberately the last
thing produced: everything upstream of here is world-frame, and the single
rotation into the body happens once, at the end, after both clamps. See
``limits.py`` for why that ordering is load-bearing.

The diagnostics are the same decomposition the acceleration backend reports, on
purpose. A flight flown on either backend must produce numbers that can be laid
side by side, or the two can never be compared.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BodyTwistCommand:
    """One tick of velocity-backend output, plus everything needed to judge it.

    Attributes:
        vx: Body-forward velocity command, m/s.
        vy: Body-left velocity command, m/s. Body FLU (REP-103), so a positive
            value moves the aircraft to its own left.
        vz: Body-up velocity command, m/s. Identical to the world-up component
            while the aircraft is level, which is the only attitude this backend
            commands.
        yaw_rate: Heading rate command, rad/s, positive counter-clockwise.
        world_vx: The same command before rotation, along world +x. Carried for
            logging: a body twist cannot be compared against a plan without
            undoing the rotation, and doing that in an analysis script invites
            using a different yaw than the controller did.
        world_vy: The same along world +y.
        world_vz: The same along world +z.
        commanded_yaw: The heading the plan asked for at this instant, radians,
            or the measured heading when the plan expresses none.
        position_error_m: Distance from the aircraft to where the plan says it
            should be now. The single number that says whether it is flying the
            plan.
        along_track_lag_m: Component of that error along the direction of
            travel. Positive means late, which is benign.
        cross_track_error_m: Component perpendicular to it. This is the one that
            flies into walls -- same magnitude, entirely different meaning.
        yaw_error_rad: Reference heading minus measured, wrapped.
        reference_time_s: Where on the trajectory the reference was taken,
            seconds from its start.
        trajectory_id: FALCON's id for the trajectory being flown, or -1.
        diverged: True while the position error exceeds its ceiling. Advisory --
            the loop keeps trying.
        holding: True when no trajectory is being followed and the loop is
            holding station instead.
        past_end: True when the trajectory has run out and the aircraft is
            flying to its final point. Normal for a second between replans; a
            standing condition means the planner stopped.
        saturated: True when the command hit a speed ceiling.
        rate_limited: True when the command hit the slew ceiling. Distinct from
            ``saturated`` because they mean different things: one is asking to
            go too fast, the other is asking to change too abruptly, and a
            replan blend that trips this every tick is a blend that is not
            working.
        reference_z_m: Altitude the plan asked for at this instant, world frame.
            Carried purely so a contact can be attributed: ``position_error_m``
            says the aircraft is off its plan but not in which direction, and
            for a strike on top of an obstacle the direction is the whole
            question. A descent onto a crate with the reference already down
            there is a PLANNING fault -- the curve was routed through the
            obstacle. The same descent with the reference holding station above
            is a CONTROL fault -- altitude sagging out from under a good plan.
            The two need opposite fixes. NaN while holding, where there is no
            plan to quote.
    """

    vx: float
    vy: float
    vz: float
    yaw_rate: float
    world_vx: float = 0.0
    world_vy: float = 0.0
    world_vz: float = 0.0
    commanded_yaw: float = 0.0
    position_error_m: float = 0.0
    along_track_lag_m: float = 0.0
    cross_track_error_m: float = 0.0
    yaw_error_rad: float = 0.0
    reference_time_s: float = 0.0
    trajectory_id: int = -1
    diverged: bool = False
    holding: bool = False
    past_end: bool = False
    saturated: bool = False
    rate_limited: bool = False
    reference_z_m: float = float("nan")

    def body_velocity(self):
        # type: () -> tuple
        """The command as a plain body-frame ``(vx, vy, vz)`` triple."""
        return self.vx, self.vy, self.vz

    def world_velocity(self):
        # type: () -> tuple
        """The command as a plain world-frame ``(vx, vy, vz)`` triple."""
        return self.world_vx, self.world_vy, self.world_vz
