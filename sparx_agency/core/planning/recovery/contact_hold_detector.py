"""Detecting that the airframe is being HELD by geometry, not flying.

There is a failure this stack could not see. The aircraft touches something --
a wall, a ledge, a door frame -- and ends up leaning against it at 40-80 deg,
airborne, and completely rigid: measured over 969 flights it wanders **2 cm**
while held, for a median of 10 s and as long as 133 s. It happens in **37 % of
flights** and consumes **3.6 % of all flight time**; the worst single flight lost
87 % of its window to it.

Why nothing already here catches it
-----------------------------------
Every existing detector judges *commanded versus achieved* motion --
:class:`~sparx_agency.core.planning.recovery.stuck_detector.StuckDetector` needs
``min_cmd_vx`` before a tick even counts as trying, and
:class:`~sparx_agency.core.planning.trackers.drift_pid.blockage.BlockageMonitor`
is built the same way. That is the right test for "told to go, did not go", and
it is structurally blind here: the follower's tilt reflex reacts to the excursion
by commanding **zero**, so from that moment nothing is being asked, no axis is
under test, and no commanded-vs-achieved detector can fire by construction. The
aircraft is stuck precisely while the stack has stopped asking it to move.

So this detector deliberately does NOT look at commands. It is the one judgement
available when nothing is being commanded, and it is a *statics* argument rather
than a kinematic one:

  A multirotor cannot hold 50 deg of tilt and stay at a constant height. If the
  attitude is sustained and the aircraft is neither translating nor descending,
  something outside the airframe is carrying its weight.

The optional ``support_ratio`` corroborates that geometrically: a down-facing
ranger on an aircraft tilted by theta reads ``height / cos(theta)``, so the ratio
of ranger to localized height should be ``1 / cos(theta)``. Measured across the
episodes it is 1.70 against a reported ~54 deg, which agrees. Pass it when a
ranger is available and the check tightens; leave it ``None`` and the verdict
rests on attitude and motion alone.

This detector reports. It does not steer, and it deliberately recommends nothing:
the escape ladder that already exists was measured burning half a flight on 38
attempts that never restored motion, so what to *do* about a confirmed hold is an
open question this class exists to gather evidence for, not to prejudge.

Python 3.8 compatible (the FALCON Noetic adapter imports ``core`` under 3.8): no
PEP 604 unions, no ``match``/``case``, no builtin generics at runtime, stdlib
only -- no numpy, no scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Optional


@dataclass(frozen=True)
class ContactHoldParams:
    """Tuning for :class:`ContactHoldDetector`.

    Attributes:
        enabled: False disables the detector outright; it never confirms.
        tilt_deg: Tilt above which a level hover is no longer a plausible
            explanation (deg). 40 sits well above the 21.7 deg p90 of ordinary
            aggressive flight and below the 45-80 deg the held episodes show.
        max_speed_mps: Ground speed below which the aircraft counts as not
            translating (m/s). Held episodes wander 2 cm total, so this is
            generous.
        max_climb_mps: Vertical speed magnitude below which the aircraft counts
            as still supported (m/s). This is the discriminator that keeps a
            genuine free capsize out: a capsizing aircraft is *falling*, and
            must keep its existing reflex rather than be reported as contact.
        confirm_s: Seconds the signature must hold continuously before the
            verdict flips to held. Wall-clock rather than ticks because the
            evidence is a sustained physical state, not an accumulation of
            attempts.
        clear_s: Seconds the signature must be absent before a standing hold is
            released, so a single clean sample does not end an episode.
        support_ratio_tol: Tolerance on ``ranger / height`` versus
            ``1 / cos(tilt)`` when a ratio is supplied (dimensionless). Wide,
            because the ranger sees whatever surface is under it and the tilt is
            the airframe's, not the beam's.
    """

    enabled: bool = True
    tilt_deg: float = 40.0
    max_speed_mps: float = 0.05
    max_climb_mps: float = 0.15
    confirm_s: float = 2.0
    clear_s: float = 0.5
    support_ratio_tol: float = 0.6

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the detector relies on."""
        for name in ("tilt_deg", "max_speed_mps", "max_climb_mps",
                     "confirm_s", "support_ratio_tol"):
            if getattr(self, name) <= 0.0:
                raise ValueError("ContactHoldParams." + name + " must be > 0")
        if self.clear_s < 0.0:
            raise ValueError("ContactHoldParams.clear_s must be >= 0")
        if self.tilt_deg >= 90.0:
            raise ValueError("ContactHoldParams.tilt_deg must be < 90")


@dataclass(frozen=True)
class ContactVerdict:
    """What the detector believes about the airframe being held.

    Attributes:
        held: True while a contact hold is confirmed and not yet cleared.
        since_s: Seconds the current hold has been confirmed for, 0.0 when not
            held. This is the number worth logging: the cost of the event.
        tilt_deg: Tilt at the moment of the verdict (deg).
        corroborated: True when a ``support_ratio`` was supplied and agreed with
            the tilt. False means either none was supplied or it disagreed; the
            verdict still stands on attitude and motion, this only records how
            much independent evidence backs it.
    """

    held: bool = False
    since_s: float = 0.0
    tilt_deg: float = 0.0
    corroborated: bool = False


class ContactHoldDetector:
    """Watches attitude and motion, and reports being held against geometry.

    Stateful across ticks: feed it every control tick. It keeps only the start
    of the current candidate signature and of the current gap, so it costs
    nothing to run.
    """

    def __init__(self, params=None):
        # type: (Optional[ContactHoldParams]) -> None
        """Initialize the detector.

        Args:
            params: Tuning; defaults to :class:`ContactHoldParams`.
        """
        self.params = params if params is not None else ContactHoldParams()
        self._signature_since = None   # type: Optional[float]
        self._gap_since = None         # type: Optional[float]
        self._held_since = None        # type: Optional[float]
        self._corroborated = False

    def reset(self):
        # type: () -> None
        """Forget the current episode. Call when the aircraft is known to fly."""
        self._signature_since = None
        self._gap_since = None
        self._held_since = None
        self._corroborated = False

    def update(self, now, tilt_deg, speed_mps, climb_mps, support_ratio=None):
        # type: (float, float, float, float, Optional[float]) -> ContactVerdict
        """Fold one tick in and return the current verdict.

        Args:
            now: Monotonic-ish timestamp, seconds.
            tilt_deg: ``max(|roll|, |pitch|)`` of the airframe, degrees.
            speed_mps: Measured horizontal ground speed, m/s.
            climb_mps: Measured vertical speed, m/s (sign ignored).
            support_ratio: Optional ranger-over-height ratio for corroboration.

        Returns:
            The :class:`ContactVerdict` for this tick.
        """
        p = self.params
        if not p.enabled:
            self.reset()
            return ContactVerdict()

        signature = (tilt_deg >= p.tilt_deg
                     and abs(speed_mps) <= p.max_speed_mps
                     and abs(climb_mps) <= p.max_climb_mps)

        if signature:
            self._gap_since = None
            if self._signature_since is None:
                self._signature_since = now
                self._corroborated = False
            if support_ratio is not None and self._supports(tilt_deg, support_ratio):
                self._corroborated = True
            if (self._held_since is None
                    and now - self._signature_since >= p.confirm_s):
                self._held_since = self._signature_since
        else:
            self._signature_since = None
            if self._held_since is not None:
                if self._gap_since is None:
                    self._gap_since = now
                if now - self._gap_since >= p.clear_s:
                    self.reset()

        if self._held_since is None:
            return ContactVerdict(tilt_deg=tilt_deg)
        return ContactVerdict(held=True,
                              since_s=max(0.0, now - self._held_since),
                              tilt_deg=tilt_deg,
                              corroborated=self._corroborated)

    def _supports(self, tilt_deg, support_ratio):
        # type: (float, float) -> bool
        """Whether a ranger/height ratio agrees with the reported tilt.

        Args:
            tilt_deg: Airframe tilt, degrees.
            support_ratio: Measured ranger divided by localized height.

        Returns:
            True when the ratio is within tolerance of ``1 / cos(tilt)``.
        """
        c = cos(radians(min(tilt_deg, 89.0)))
        if c <= 1e-6 or support_ratio <= 0.0:
            return False
        return abs(support_ratio - 1.0 / c) <= self.params.support_ratio_tol
