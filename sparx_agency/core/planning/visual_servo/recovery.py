"""Re-search / recovery policy: how to move to re-see a target the track just lost.

Drives the RECOVER state. The mission reaches RECOVER only after the tracker has
already coasted (dead-reckoned) through a brief dropout and still cannot hold the
box — so by here the target has genuinely left view (occluded, behind an object,
motion-blurred, or off-frame). This policy turns the *last thing we saw* into a
small, bounded manoeuvre to bring it back, reading the last valid track's position
and image-plane velocity to decide which way to look.

Two manoeuvres, chosen from where/how the target vanished:

  * **directional** — it clearly left to one side (its last box was off-centre
    and/or moving that way): yaw toward that side and lean (crab) gently after it,
    so a target that ran left is chased by turning/leaning left, and one that ran
    right by turning/leaning right.
  * **peek** — it vanished from near the *centre* of the frame, which means it is
    most likely hidden behind an object straight ahead rather than gone sideways.
    A pure yaw would never see behind that object, so instead the drone sidesteps
    to change its viewing angle and looks *around* the occluder, alternating sides
    so both edges get checked. A small one-off forward nudge helps clear the edge.

Both are deliberately restrained for wall safety: yaw dominates (rotating in place
does not translate into a wall), the crab speeds are small, the peek oscillates so
it stays near the loss position instead of drifting away, and the forward nudge is
one bounded pulse, not a sustained advance. And the whole episode is time-bounded
by ``max_search_s`` (mirrored to the FSM's ``recover_timeout_s``): if the target is
not re-acquired in time, ``give_up`` is raised and the mission returns to
SEARCH/SCAN. A short initial hold lets an in-flight re-detection recover the lock
before any motion starts.

Pure and clock-free: fed the last :class:`Track2D` and how long the track has been
lost, it returns a body-frame command (REP-103: ``+vx`` forward, ``+vy`` left,
``+yaw_rate`` CCW). Command magnitudes are floored/capped downstream by the node's
force shaper and kinematic limits.
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
        hold_before_search_s: Hold (hover, no motion) this long after loss before
            moving, giving an in-flight re-detection a chance first.
        max_search_s: Give up (report ``give_up``) after this long lost; set to the
            FSM's ``recover_timeout_s`` so recovery does not outlast the RECOVER state.
        velocity_weight: Weight on the last image-plane velocity vs the last
            position when scoring which side the target left.
        default_direction: Side to assume when there is no prior track
            (+1 => sweep as if the target went left/CCW; -1 => right/CW).
        center_exit_frac: If the (position + velocity) exit score is weaker than
            this the target is treated as having vanished from the **centre**
            (likely occluded) -> peek manoeuvre; stronger than this it clearly left
            a side -> directional manoeuvre.
        directional_roll_speed: Gentle crab speed toward the exit side while yawing
            after a target that left sideways (m/s). Kept small for wall safety.
        peek_forward_speed: Speed of the single forward nudge that starts a peek,
            to help clear the edge of a central occluder (m/s).
        peek_forward_s: Duration of that forward nudge (s); after it, the peek is
            pure sidestep+yaw with no further advance (bounds forward travel).
        peek_roll_speed: Sidestep (crab) speed while peeking around an occluder (m/s).
        peek_period_s: Full left->right->left oscillation period of the peek (s);
            oscillating keeps net drift near zero so the drone stays put.
        peek_orbit: If True (default) the peek yaws *opposite* its sidestep, so as it
            slides to one side it keeps looking back toward the occluded spot and
            sees around the object's edge. If False it yaws *with* the sidestep.
    """

    search_yaw_rate: float = 0.5
    hold_before_search_s: float = 0.3
    max_search_s: float = 8.0
    velocity_weight: float = 0.5
    default_direction: float = 1.0
    center_exit_frac: float = 0.25
    directional_roll_speed: float = 0.05
    peek_forward_speed: float = 0.06
    peek_forward_s: float = 0.6
    peek_roll_speed: float = 0.10
    peek_period_s: float = 2.0
    peek_orbit: bool = True

    def __post_init__(self) -> None:
        if self.search_yaw_rate <= 0.0:
            raise ValueError("search_yaw_rate must be > 0.")
        if self.default_direction not in (-1.0, 1.0):
            raise ValueError("default_direction must be +1.0 or -1.0.")
        if self.center_exit_frac < 0.0:
            raise ValueError("center_exit_frac must be >= 0.")
        for name in ("directional_roll_speed", "peek_forward_speed",
                     "peek_forward_s", "peek_roll_speed"):
            if getattr(self, name) < 0.0:
                raise ValueError("%s must be >= 0." % name)
        if self.peek_period_s <= 0.0:
            raise ValueError("peek_period_s must be > 0.")


@dataclass(frozen=True)
class ReSearchDecision:
    """One recovery tick.

    Attributes:
        command: Body-frame command (zero during the hold, else the manoeuvre).
        exit_side: The direction the manoeuvre favours: +1 left, -1 right
            (0 during the hold). For a peek it is the current sidestep direction.
        phase: "hold" | "directional" | "peek".
        give_up: True once ``max_search_s`` has elapsed with no re-acquisition.
    """

    command: ControlCommand
    exit_side: float
    phase: str
    give_up: bool


def _exit_score(last_track: Track2D, frame_w: int, frame_h: int,
                velocity_weight: float) -> float:
    """Signed strength of which side the target left: ``> 0`` right, ``< 0`` left.

    Combines the last box's horizontal offset with its image-plane x-velocity
    (normalised by half the image width so the two are comparable).
    """
    ox, _oy = center_offset_norm(last_track.bbox_xyxy, frame_w, frame_h)
    vx_px = last_track.velocity_px[0]
    half_w = max(1.0, 0.5 * float(frame_w))
    return ox + velocity_weight * (vx_px / half_w)


def infer_exit_side(last_track: Optional[Track2D], frame_w: int, frame_h: int,
                    velocity_weight: float, default_direction: float) -> float:
    """Infer which side the target left: +1 left, -1 right.

    A box near the right edge (``ox > 0``) or moving right (``vx > 0``) scores
    POSITIVE, which maps to an exit to the right (return ``-1``). Returns
    ``default_direction`` when there is no prior track or the score is exactly zero.
    """
    if last_track is None:
        return default_direction
    score = _exit_score(last_track, frame_w, frame_h, velocity_weight)
    if score > 0.0:
        return -1.0   # exited right
    if score < 0.0:
        return 1.0    # exited left
    return default_direction


class ReSearchPolicy:
    """Turn a lost track into a bounded re-search manoeuvre (directional or peek)."""

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

        # No prior track at all -> sweep toward the configured default side.
        if last_track is None:
            return self._directional(c.default_direction, give_up)

        # Clear side exit -> chase it; vanished near centre -> peek around occluder.
        strength = abs(_exit_score(last_track, frame_w, frame_h, c.velocity_weight))
        if strength >= c.center_exit_frac:
            side = infer_exit_side(last_track, frame_w, frame_h,
                                   c.velocity_weight, c.default_direction)
            return self._directional(side, give_up)
        return self._peek(lost_for_s, give_up)

    # ── manoeuvres ────────────────────────────────────────────────────
    def _directional(self, side: float, give_up: bool) -> ReSearchDecision:
        """Yaw toward the exit side and lean gently after it (side +1 left / -1 right)."""
        c = self.cfg
        wz = c.search_yaw_rate * side           # +1 left -> yaw CCW (turn left)
        vy = c.directional_roll_speed * side    # +vy is left -> crab toward the side
        cmd = ControlCommand.velocity(0.0, vy, 0.0, wz, source=self.name,
                                      phase="directional")
        return ReSearchDecision(command=cmd, exit_side=side, phase="directional",
                                give_up=give_up)

    def _peek(self, lost_for_s: float, give_up: bool) -> ReSearchDecision:
        """Sidestep + yaw to look around a central occluder, alternating sides."""
        c = self.cfg
        t = lost_for_s - c.hold_before_search_s
        step = 1.0 if (t % c.peek_period_s) < 0.5 * c.peek_period_s else -1.0
        vx = c.peek_forward_speed if t < c.peek_forward_s else 0.0  # one bounded nudge
        vy = c.peek_roll_speed * step                               # sidestep (roll)
        yaw_sign = -step if c.peek_orbit else step   # orbit: look back at the occluder
        wz = c.search_yaw_rate * yaw_sign
        cmd = ControlCommand.velocity(vx, vy, 0.0, wz, source=self.name, phase="peek")
        return ReSearchDecision(command=cmd, exit_side=step, phase="peek",
                                give_up=give_up)
