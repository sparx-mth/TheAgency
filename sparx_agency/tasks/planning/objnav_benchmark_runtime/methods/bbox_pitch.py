"""Image-space vertical servo for the camera pitch: where the target's BOX sits in the frame.

The dynamic pitch controller of ``target_closing`` used to read the target's
*geometry* alone -- the 3-D memory's height against the camera's -- and tilt
toward it. That prediction is only as good as the memory, and the memory is
biased exactly when it matters: a frame cut by the lower edge lands its depth
patch on the upper part of the object (the toilets of the 1,000-episode run,
2026-10-11). The box itself is the direct observation. Here the pitch is
driven by the box's place in the frame, with no class, no memory and no
height involved: a box in the LOWER region of the frame, or about to be cut
by the lower edge on the next step toward it, asks for LOOK_DOWN; one in the
UPPER region, or about to be cut by the top, asks for LOOK_UP; one in the
centre band asks for nothing. The geometric prediction remains the fallback
for a frame WITHOUT a fresh box (an occluded approach step), where there is
nothing to servo on.

Everything is in rows and angles of the pinhole: a row ``v`` lies at
``atan2(v - cy, fy)`` below the optical axis; a LOOK_DOWN of ``d`` moves the
image content UP by ``fy * tan(.)`` of that angle; stepping ``a`` metres toward
an object ``r`` metres away at a height offset ``h`` moves a row at depression
``e = atan2(h, r)`` to ``atan2(h, r - a)``. Pure functions, standard library
only; the caller owns the settings and the counters.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]

#: What the servo saw in the (predicted) frame.
REGION_CENTRE = "centre"
REGION_LOWER = "lower"
REGION_UPPER = "upper"
REGION_CUT_BOTTOM = "cut_bottom"
REGION_CUT_TOP = "cut_top"
REGION_CUT_BOTH = "cut_both"


@dataclass(frozen=True)
class VerticalFix:
    """The servo's verdict on one fresh box.

    Attributes:
        steps: Signed tilt steps to take now: positive LOOK_DOWN, negative
            LOOK_UP, 0 none. Never more than the pitch limits allow.
        region: Where the box (after the planned advance, if any) sits:
            ``centre``, ``lower``, ``upper``, ``cut_bottom``, ``cut_top`` or
            ``cut_both``.
        centre_row: The box centre row the verdict was read at (the predicted
            one when an advance was planned).
        reason: One line for the diagnostics.
    """

    steps: int
    region: str
    centre_row: float
    reason: str

    @property
    def wanted(self) -> bool:
        """Whether a tilt was asked for."""
        return self.steps != 0


def row_after_tilt(v: float, cy: float, fy: float, tilt_rad: float) -> float:
    """Where image row ``v`` lands after the camera tilts DOWN by ``tilt_rad`` (negative tilts up).

    Exact for a pinhole: the row's angle below the optical axis loses the
    tilt. Rows past 90 degrees of the axis saturate at the far edge rather
    than wrapping.
    """
    angle = math.atan2(v - cy, fy) - tilt_rad
    if angle >= math.pi / 2 - 1e-6:
        return float("inf")
    if angle <= -math.pi / 2 + 1e-6:
        return float("-inf")
    return cy + fy * math.tan(angle)


def box_after_tilt(box: Sequence[float], cy: float, fy: float, tilt_rad: float) -> Box:
    """The box after the camera tilts DOWN by ``tilt_rad``; the columns do not move."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1, row_after_tilt(y1, cy, fy, tilt_rad), x2, row_after_tilt(y2, cy, fy, tilt_rad))


def box_after_advance(box: Sequence[float], cy: float, fy: float, pitch_rad: float,
                      range_m: float, advance_m: float, min_range_m: float = 0.3) -> Box:
    """Where the box will be after stepping ``advance_m`` straight toward it, the camera as it is.

    Each edge row is a ray at depression ``e = atan2(v - cy, fy) + pitch``
    below the horizon; at planar range ``range_m`` that ray meets the object
    at height offset ``h = range_m * tan(e)`` (positive below the camera),
    and from ``range_m - advance_m`` the same point lies at
    ``atan2(h, range_m - advance_m)``. The whole object is taken at one
    planar range -- the near edge of the surface in view, when the caller
    has it -- which is what the lower edge of a floor-standing object needs
    and slightly overstates how fast its top rises. Columns are not
    predicted: the caller's yaw servo owns them.

    Args:
        box: ``(x1, y1, x2, y2)`` pixels.
        cy: Principal row.
        fy: Vertical focal length, pixels.
        pitch_rad: Camera pitch now, REP-103 (positive looks down).
        range_m: Planar range to the object now, metres; must be positive.
        advance_m: How far the next action carries the agent toward it; 0
            returns the box as it is.
        min_range_m: The range after the advance is never predicted below
            this (the object cannot be walked into).
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    if advance_m <= 0.0 or not (range_m > 0.0) or not math.isfinite(range_m):
        return (x1, y1, x2, y2)
    after = max(min_range_m, range_m - advance_m)

    def row(v: float) -> float:
        depression = math.atan2(v - cy, fy) + pitch_rad
        height = range_m * math.tan(depression)
        return cy + fy * math.tan(math.atan2(height, after) - pitch_rad)

    return (x1, row(y1), x2, row(y2))


def box_fits(box: Sequence[float], height: int, margin_px: int) -> bool:
    """Whether the box lies whole inside the frame's vertical margins (columns are not judged here)."""
    _, y1, _, y2 = (float(v) for v in box)
    return y1 >= margin_px and y2 <= height - margin_px


def pitch_within(pitch_rad: float, min_pitch_rad: Optional[float], max_pitch_rad: Optional[float]) -> bool:
    """Whether a pitch lies within the action spec's limits (``None`` is unclamped)."""
    if min_pitch_rad is not None and pitch_rad < min_pitch_rad - 1e-9:
        return False
    return max_pitch_rad is None or pitch_rad <= max_pitch_rad + 1e-9


def vertical_fix(box: Sequence[float], width: int, height: int, cy: float, fy: float,
                 pitch_rad: float, tilt_rad: float, region: float, margin_px: int,
                 min_pitch_rad: Optional[float] = None, max_pitch_rad: Optional[float] = None,
                 range_m: Optional[float] = None, advance_m: float = 0.0) -> VerticalFix:
    """The tilt that keeps the box whole and near the frame's vertical centre, or none.

    Read on the box as it will be after the planned advance (``advance_m``
    toward an object ``range_m`` away; 0 for a frame taken standing still):

    * cut by BOTH the top and the bottom margins -- the object fills the
      frame vertically; no tilt helps -- nothing;
    * cut by the bottom margin alone -- LOOK_DOWN by the steps that bring its
      centre nearest the axis (one at least), provided the top edge stays
      inside the frame afterwards: trading a bottom cut for a top cut is
      no gain; symmetrically LOOK_UP for a top cut;
    * whole but with its centre in the lower ``region`` of the frame's rows
      -- LOOK_DOWN by the rounded steps (one at least), provided the whole
      box stays inside the margins afterwards and its centre does not land
      in the upper region (fewer steps are tried first); symmetrically for
      the upper region;
    * otherwise -- the centre band -- nothing.

    A tilt past the pitch limits is cut back to what the limits allow, down
    to none. ``region`` is a fraction of the frame height: 0.25 makes the
    lower region the bottom quarter of the rows and the upper region the top
    quarter. The centre band is never narrower than the rounding's own dead
    band -- ``fy * tan(tilt / 2)`` rows either side of the principal row,
    where one tilt lands a centre -- so the servo cannot hunt whatever
    ``region`` says (:func:`region_compatible` tells whether it had to widen).

    Args:
        box: ``(x1, y1, x2, y2)`` pixels, the fresh detection.
        width: Frame width, pixels (recorded only; columns are the yaw servo's).
        height: Frame height, pixels.
        cy: Principal row.
        fy: Vertical focal length, pixels.
        pitch_rad: Camera pitch now, REP-103.
        tilt_rad: What one LOOK tilts, radians; positive.
        region: The lower/upper region, as a fraction of the frame height in
            ``(0, 0.5)``.
        margin_px: The border margin a box must clear to count as whole.
        min_pitch_rad: The most upward pitch allowed, or None.
        max_pitch_rad: The most downward pitch allowed, or None.
        range_m: Planar range to the object, for the advance prediction.
        advance_m: The next action's advance toward the object, metres.

    Raises:
        ValueError: On a non-positive tilt, a region outside ``(0, 0.5)`` or
            a frame without rows.
    """
    if not (tilt_rad > 0.0) or not math.isfinite(tilt_rad):
        raise ValueError("tilt_rad must be positive and finite")
    if not (0.0 < region < 0.5):
        raise ValueError("region must lie in (0, 0.5)")
    if height <= 0 or width <= 0:
        raise ValueError("the frame needs rows and columns")
    predicted = box_after_advance(box, cy, fy, pitch_rad, range_m if range_m is not None else 0.0, advance_m)
    _, y1, _, y2 = predicted
    centre = 0.5 * (y1 + y2)
    bottom_cut, top_cut = y2 > height - margin_px, y1 < margin_px
    half_band = fy * math.tan(0.5 * tilt_rad)                     # where one rounded tilt lands a centre
    lower_edge = max(height * (1.0 - region), cy + half_band)
    upper_edge = min(height * region, cy - half_band)

    def allowed(steps: int) -> bool:
        return pitch_within(pitch_rad + steps * tilt_rad, min_pitch_rad, max_pitch_rad)

    def rounded(sign: int) -> int:
        """The steps that bring the centre nearest the axis, at least one in ``sign``'s direction."""
        magnitude = int(round(abs(math.atan2(centre - cy, fy)) / tilt_rad))
        return sign * max(1, magnitude)

    def centred_after(steps: int) -> bool:
        """Whether the centre lands outside the opposite region after ``steps``."""
        after = row_after_tilt(centre, cy, fy, steps * tilt_rad)
        return upper_edge <= after <= lower_edge

    if bottom_cut and top_cut:
        return VerticalFix(0, REGION_CUT_BOTH, centre, "the box fills the frame vertically; no tilt helps")
    if bottom_cut or top_cut:
        sign = 1 if bottom_cut else -1
        label = REGION_CUT_BOTTOM if bottom_cut else REGION_CUT_TOP
        steps = rounded(sign)
        while steps != 0:
            after = box_after_tilt(predicted, cy, fy, steps * tilt_rad)
            far_edge_in = (after[1] >= margin_px) if sign > 0 else (after[3] <= height - margin_px)
            if allowed(steps) and far_edge_in:
                return VerticalFix(steps, label, centre, "%s cut by the %s edge%s: %s by %d"
                                   % ("box", "lower" if sign > 0 else "upper",
                                      " after the step" if advance_m > 0 else "",
                                      "LOOK_DOWN" if sign > 0 else "LOOK_UP", abs(steps)))
            steps -= sign
        return VerticalFix(0, label, centre, "box cut by the %s edge, but no tilt within the limits keeps the %s edge in"
                           % ("lower" if sign > 0 else "upper", "upper" if sign > 0 else "lower"))
    if centre > lower_edge or centre < upper_edge:
        sign = 1 if centre > lower_edge else -1
        label = REGION_LOWER if sign > 0 else REGION_UPPER
        steps = rounded(sign)
        while steps != 0:
            after = box_after_tilt(predicted, cy, fy, steps * tilt_rad)
            if allowed(steps) and box_fits(after, height, margin_px) and centred_after(steps):
                return VerticalFix(steps, label, centre, "box centre in the %s region%s (row %.0f): %s by %d"
                                   % (label, " after the step" if advance_m > 0 else "", centre,
                                      "LOOK_DOWN" if sign > 0 else "LOOK_UP", abs(steps)))
            steps -= sign
        return VerticalFix(0, label, centre, "box centre in the %s region, but no tilt keeps the box whole" % label)
    return VerticalFix(0, REGION_CENTRE, centre, "box centre in the centre band (row %.0f)" % centre)


def region_compatible(cy: float, fy: float, height: int, tilt_rad: float, region: float) -> bool:
    """Whether the centre band is at least as wide as one tilt's rounding dead band (no hunting).

    After a tilt chosen by rounding, the box centre lies within
    ``fy * tan(tilt / 2)`` rows of the principal row; the band must contain
    that, or the servo would tilt back next frame.
    """
    half_band = fy * math.tan(0.5 * tilt_rad)
    return (cy - half_band >= height * region) and (cy + half_band <= height * (1.0 - region))



