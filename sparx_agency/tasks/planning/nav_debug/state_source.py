"""The aircraft's measured state, from whichever source the run actually has.

The debug screen has always shown what the planner *asked for* in numbers and
what the aircraft *did* only as a shape on a map. Putting the two side by side
needs position, velocity and heading as measurements, and this package records
them in three different places depending on which halves of the recording ran:

===========  ===================================  ==================================
``source``   where it comes from                  present when
===========  ===================================  ==================================
``odom``     the follower's own ``/odom_world``,  the ROS1 trace carries a ``state``
             sampled at the instant it built      section (runs recorded after the
             the command                          follower started tracing it)
``truth``    ``/R1/velocity_truth`` +             the ROS2 half of the recording was
             ``/R1/sphera/state``                 collected and joined
``pose_diff``differentiated from the recorded     always -- it needs only the spine
             pose spine
===========  ===================================  ==================================

The order above is the order of preference, and it is deliberately *not* the
order of accuracy. ``truth`` is the more authoritative measurement, but it is
written by a second process on a second clock and reaches a frame through a
cross-recorder join; ``odom`` is the state the controller itself was reacting
to, which is what a control-loop debug view is actually asking about. So
``odom`` wins when present, and :attr:`DroneState.source` says which one drew
the numbers -- a screen that silently swaps measurement sources mid-replay is
worse than one that shows a dash.

``pose_diff`` exists because the interesting flights are the ones already on
disk. A run recorded before any of this instrumentation still has a pose spine,
and a pose spine still knows how fast the aircraft was going.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

from sparx_agency.tasks.planning.nav_debug.frame import DroneState
from sparx_agency.tasks.planning.nav_debug.sources import to_float

#: Half-width of the central-difference window, in seconds. The spine runs at
#: 15-20 Hz with 4-decimal rounding, so differencing adjacent samples turns
#: ~0.1 mm of quantisation into ~1.5 mm/s of noise per sample; widening the
#: window to ~a fifth of a second trades a little lag for a readable number.
DIFF_HALF_WINDOW_S = 0.12

#: A gap wider than this is a recording dropout, not a sample interval: two
#: poses either side of it say nothing about the velocity between them.
DIFF_MAX_GAP_S = 1.0

ODOM = "odom"
TRUTH = "truth"
POSE_DIFF = "pose_diff"
#: Position and heading are recorded, velocity is not derivable here (the ends
#: of the run, or either side of a dropout). Measured pose, unknown velocity.
POSE_ONLY = "pose"


class StateSource:
    """Resolves one frame's measured state from the best source the run has.

    Args:
        rows: The spine rows (see :mod:`.timeline`), time-ordered. Used for the
            ``pose_diff`` fallback, which is computed once for the whole run.
    """

    def __init__(self, rows: Sequence[dict]) -> None:
        self._rows = rows
        self._derived = derive_velocities(rows)

    def at(self, i: int, control_row: Optional[dict],
           truth_row: Optional[dict]) -> Optional[DroneState]:
        """The measured state for frame ``i``.

        Args:
            i: Frame index into the spine.
            control_row: This instant's ``control.jsonl`` row, if any -- it may
                carry the follower's own ``state`` section.
            truth_row: This instant's ``ros2/truth.jsonl`` row, if any.

        Returns:
            A :class:`~.frame.DroneState`, or ``None`` when the frame has no
            pose at all (which the spine guarantees it does, so in practice
            only for an empty run).
        """
        pose = self._pose(i)
        return (self._from_odom(control_row, pose)
                or self._from_truth(truth_row, pose)
                or self._from_pose_diff(i, pose))

    # ── the three sources ────────────────────────────────────────────────────
    @staticmethod
    def _from_odom(control_row: Optional[dict], pose: dict) -> Optional[DroneState]:
        """The follower's own view of the aircraft, recorded beside the command."""
        section = control_row.get("state") if isinstance(control_row, dict) else None
        if not isinstance(section, dict):
            return None
        state = _state(section, ODOM, pose)
        # A state section that carried no velocity is not a velocity source; fall
        # through to one that did rather than report an all-dash "measurement".
        return state if state.vx is not None or state.wz is not None else None

    @staticmethod
    def _from_truth(truth_row: Optional[dict], pose: dict) -> Optional[DroneState]:
        """Sphera's ground truth: the velocity lane plus its own pose report."""
        if not isinstance(truth_row, dict):
            return None
        sphera = truth_row.get("sphera")
        merged = dict(sphera) if isinstance(sphera, dict) else {}
        merged.update({k: v for k, v in truth_row.items() if k != "sphera"})
        merged["wz"] = truth_row.get("yaw_rate")
        state = _state(merged, TRUTH, pose)
        return state if state.vx is not None else None

    def _from_pose_diff(self, i: int, pose: dict) -> Optional[DroneState]:
        """Velocity differentiated from the pose spine -- the universal fallback."""
        if not pose:
            return None
        derived = self._derived[i] if 0 <= i < len(self._derived) else None
        fields = dict(pose)
        if derived is not None:
            fields.update(derived)
        return _state(fields, POSE_DIFF if derived is not None else POSE_ONLY,
                      pose)

    def _pose(self, i: int) -> dict:
        """The spine's own pose for frame ``i`` -- always recorded, never derived."""
        if not 0 <= i < len(self._rows):
            return {}
        row = self._rows[i]
        return {"x": to_float(row.get("x")), "y": to_float(row.get("y")),
                "z": to_float(row.get("z")), "yaw": to_float(row.get("yaw"))}


def _state(fields: dict, source: str, pose: dict) -> DroneState:
    """Build a :class:`DroneState`, falling back to the spine pose per field.

    A velocity source may report where it thinks the aircraft is (Sphera does)
    or not report position at all (an odometry twist need not); either way the
    pose the rest of the screen is drawn from stays the pose shown here, so the
    table cannot disagree with the map beside it.
    """
    def pick(key):
        value = to_float(fields.get(key))
        return pose.get(key) if value is None else value

    return DroneState(
        x=pick("x"), y=pick("y"), z=pick("z"), yaw=pick("yaw"),
        vx=to_float(fields.get("vx")), vy=to_float(fields.get("vy")),
        vz=to_float(fields.get("vz")), wz=to_float(fields.get("wz")),
        source=source)


def derive_velocities(rows: Sequence[dict]) -> List[Optional[dict]]:
    """Per-row ``{vx, vy, vz, wz}`` differentiated from the pose spine.

    A central difference over :data:`DIFF_HALF_WINDOW_S` either side of each
    sample, so the estimate is centred on the frame rather than lagging it by
    half a window. Rows at the ends of the run, or either side of a recording
    dropout wider than :data:`DIFF_MAX_GAP_S`, get ``None`` -- an unknown
    velocity, not a zero one.

    Args:
        rows: Time-ordered spine rows carrying ``t``, ``x``, ``y`` and
            optionally ``z`` and ``yaw``.

    Returns:
        One entry per row, aligned with ``rows``.
    """
    stamps = [to_float(r.get("t")) for r in rows]
    out = [None] * len(rows)       # type: List[Optional[dict]]
    for i, t in enumerate(stamps):
        if t is None:
            continue
        lo = _reach(stamps, i, -1, t)
        hi = _reach(stamps, i, +1, t)
        if lo is None or hi is None:
            continue
        dt = stamps[hi] - stamps[lo]
        if dt <= 0.0 or dt > 2.0 * DIFF_MAX_GAP_S:
            continue
        out[i] = _rate(rows[lo], rows[hi], dt)
    return out


def _reach(stamps: Sequence[Optional[float]], i: int, step: int,
           t: float) -> Optional[int]:
    """Walk from ``i`` to the first sample a full half-window away.

    Returns ``None`` at the end of the run or across a dropout, so the caller
    reports "unknown" rather than dividing by an interval that spans a gap.
    """
    j = i
    k = i + step
    while 0 <= k < len(stamps):
        if stamps[k] is None:
            return None
        if abs(stamps[k] - stamps[j]) > DIFF_MAX_GAP_S:
            return None
        j = k
        if abs(stamps[j] - t) >= DIFF_HALF_WINDOW_S:
            return j
        k += step
    return None


def _rate(lo: dict, hi: dict, dt: float) -> dict:
    """The signed rate of change of pose between two spine rows."""
    return {"vx": _delta(lo, hi, "x") / dt, "vy": _delta(lo, hi, "y") / dt,
            "vz": _delta(lo, hi, "z") / dt,
            "wz": _wrap(_delta(lo, hi, "yaw")) / dt}


def _delta(lo: dict, hi: dict, key: str) -> float:
    a, b = to_float(lo.get(key)), to_float(hi.get(key))
    return 0.0 if a is None or b is None else b - a


def _wrap(radians: float) -> float:
    """A heading difference folded to ``(-pi, pi]``, so a wrap is not a spike."""
    return math.atan2(math.sin(radians), math.cos(radians))
