"""Rules for choosing which point of the plan is the reference.

These settle three questions that are about the *plan*, not about the airframe,
which is why they live here rather than in either backend's tuning: when a newly
arrived curve takes over, how long a finished curve is still worth flying to,
and whether the reference is the nearest point on the curve or the point at the
current time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sparx_agency.core.planning.trajectories.bspline.projection import ProjectionParams


@dataclass(frozen=True)
class ReferenceParams:
    """How the reference point is chosen from the trajectory being flown.

    Attributes:
        projection: Nearest-point search settings.
        use_projection: Track the nearest point on the curve rather than the
            point at the current time.

            On by default, but not for the reason usually given: measured, it
            does *not* beat time indexing through a bend on a trajectory
            FALCON's optimiser has already made feasible. It wins when the
            aircraft has been displaced in time from the plan -- a hold while
            FALCON replans, a stall -- where a time-indexed reference has moved
            seconds down the route and pulls the aircraft across everything
            between. Off reproduces the time-indexed behaviour, and is worth
            keeping to measure against rather than as a fallback.
        max_trajectory_age_s: How long past a trajectory's end the aircraft will
            keep flying to its final point before the feed declares it unusable.
            Covers the normal gap between replans; a longer silence means the
            planner died, and flying to a stale endpoint then is a guess.
    """

    projection: ProjectionParams = field(default_factory=ProjectionParams)
    use_projection: bool = True
    max_trajectory_age_s: float = 2.0

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the feed relies on."""
        if self.max_trajectory_age_s <= 0.0:
            raise ValueError("max_trajectory_age_s must be > 0, got %r"
                             % (self.max_trajectory_age_s,))
