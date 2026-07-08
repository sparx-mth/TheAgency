"""Re-search / recovery policy: where to look when the target track is lost.

The reference orchestrator simply hovered on track loss and waited for the
detector to re-fire — blind to *where* the target went. This policy instead reads
the last valid track's position and image-plane velocity to infer the exit side
and actively yaw the camera back toward it, so a target lost to drift or a fast
move is re-framed quickly rather than by luck.

Pure and clock-free: fed the last :class:`Track2D` and how long the track has been
lost, it returns a body-frame yaw command (REP-103: ``+yaw_rate`` CCW) plus the
inferred exit side and a give-up flag. A short initial hold lets an in-flight
re-detection recover the lock before the drone starts sweeping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sparx_agency.core.common.types import ControlCommand
from sparx_agency.core.common.types.perception import Track2D
from sparx_agency.core.common.math.bbox import center_offset_norm


@dataclass(frozen=True)
class ReSearchConfig:
    """Tuning for :class:`ReSearchPolicy`.

    Attributes:
        search_yaw_rate: Yaw rate magnitude while sweeping to re-acquire (rad/s).
        hold_before_search_s: Hold (hover, no yaw) this long after loss before
            sweeping, giving an in-flight re-detection a chance first.
        max_search_s: Give up (report ``give_up``) after this long lost.
        velocity_weight: Weight on the last image-plane velocity vs the last
            position when inferring the exit side.
        default_direction: Exit side to assume when there is no prior track
            (+1 => target assumed to the left, sweep CCW; -1 => right, sweep CW).
    """

    search_yaw_rate: float = 0.5
    hold_before_search_s: float = 0.3
    max_search_s: float = 8.0
    velocity_weight: float = 0.5
    default_direction: float = 1.0

    def __post_init__(self) -> None:
        if self.search_yaw_rate <= 0.0:
            raise ValueError("search_yaw_rate must be > 0.")
        if self.default_direction not in (-1.0, 1.0):
            raise ValueError("default_direction must be +1.0 or -1.0.")


@dataclass(frozen=True)
class ReSearchDecision:
    """One recovery tick.

    Attributes:
        command: Body-frame command (pure yaw sweep, or zero during the hold).
        exit_side: +1 if the target is inferred to have left to the **left** of
            frame, -1 to the **right** (0 during the initial hold / unknown).
        phase: "hold" | "search".
        give_up: True once ``max_search_s`` has elapsed with no re-acquisition.
    """

    command: ControlCommand
    exit_side: float
    phase: str
    give_up: bool


def infer_exit_side(last_track: Optional[Track2D], frame_w: int, frame_h: int,
                    velocity_weight: float, default_direction: float) -> float:
    """Infer which side the target left: +1 left, -1 right.

    Combines the last box's horizontal offset with its image-plane x-velocity
    (px/s): a box near the right edge (``ox > 0``) or moving right (``vx > 0``)
    yields a POSITIVE score, which maps to an exit to the right (return ``-1``).
    Returns ``default_direction`` when there is no prior track.
    """
    if last_track is None:
        return default_direction
    ox, _oy = center_offset_norm(last_track.bbox_xyxy, frame_w, frame_h)
    vx_px = last_track.velocity_px[0]
    half_w = max(1.0, 0.5 * float(frame_w))
    # Normalise velocity by ~half-image-per-second so it is comparable to ox.
    vx_norm = vx_px / half_w
    score = ox + velocity_weight * vx_norm    # >0 => right, <0 => left
    if score > 0.0:
        return -1.0   # exited right
    if score < 0.0:
        return 1.0    # exited left
    return default_direction


class ReSearchPolicy:
    """Turn a lost track into an active re-search yaw command."""

    name = "re_search"

    def __init__(self, config: Optional[ReSearchConfig] = None) -> None:
        self.cfg = config or ReSearchConfig()

    def command(self, last_track: Optional[Track2D], lost_for_s: float,
                frame_w: int, frame_h: int) -> ReSearchDecision:
        """Recovery command for a track lost ``lost_for_s`` seconds ago."""
        c = self.cfg
        give_up = lost_for_s >= c.max_search_s

        if lost_for_s < c.hold_before_search_s:
            cmd = ControlCommand.velocity(0.0, 0.0, 0.0, 0.0, source=self.name,
                                          phase="hold")
            return ReSearchDecision(command=cmd, exit_side=0.0, phase="hold",
                                    give_up=False)

        side = infer_exit_side(last_track, frame_w, frame_h, c.velocity_weight,
                               c.default_direction)
        # side=+1 (target to the left) => yaw CCW (+); side=-1 (right) => yaw CW (-).
        wz = c.search_yaw_rate * side
        cmd = ControlCommand.velocity(0.0, 0.0, 0.0, wz, source=self.name,
                                      phase="search")
        return ReSearchDecision(command=cmd, exit_side=side, phase="search",
                                give_up=give_up)
