#!/usr/bin/env python3
"""The Rooster axis-calibration experiment: what gets flown, and why.

Kept apart from the flight harness (:mod:`rooster_axis_calibration`) because
the design is the half that gets argued about and revised, while the harness
that flies it should not change at all. Separated it can be reviewed, diffed
and dry-run with no docker, no ROS and no aircraft anywhere near it.

A segment is ``(label, axes, hold_s, settle_s)``: the harness stops for
``settle_s``, publishes ``axes``, then holds them for ``hold_s``. Labels are
``<block>/<group>/<value>`` and the fit indexes everything by them, so they are
part of the data format rather than decoration.

Python 3.8-compatible.
"""
from __future__ import annotations

#: Hard ceiling on any commanded axis. Block (ii) is the only exemption: its 850
#: pre-load and 900/1000 samples exist to measure the top of the curve.
MAX_SAFE_AXIS, MAX_ANY_AXIS = 800.0, 1000.0

#: Two ``ros2 topic pub -1`` calls per segment, ~1.5 s each.
PUBLISH_OVERHEAD_S = 3.0

BREAKAWAY_VALUES = (550, 580, 610, 640, 670, 700, 750, 800)
STEADY_VALUES = (450, 500, 550, 600, 620, 650, 700, 800, 900, 1000)
YAW_VALUES = (80, 100, 120, 150, 200, 300, 400, 600, 800, 1000)


def _axes(x=0.0, y=0.0, r=0.0):
    """One axis triple. ``z`` is deliberately absent -- it is never ours."""
    return {"x": float(x), "y": float(y), "r": float(r)}


def block_standing_start():
    """Block (i): lowest axis value that moves a stationary aircraft.

    Per axis AND per sign, because the two directions are separate actuators
    until measurement says otherwise -- the only published numbers (x dead below
    ~620, y below ~700) were taken in one direction only.

    Returns:
        List of ``(label, axes, hold_s, settle_s)`` descriptors.
    """
    segments = []
    for axis in ("x", "y", "r"):
        for sign in (1, -1):
            for value in BREAKAWAY_VALUES:
                label = "i/%s%s/%d" % (axis, "+" if sign > 0 else "-", value)
                segments.append((label, _axes(**{axis: sign * value}), 3.0, 4.0))
    return segments


def block_steady_state():
    """Block (ii): forward gain while the aircraft is ALREADY moving.

    Every value is approached twice, from an 850 pre-load ("down") and a 650 one
    ("up"); the gap between those two curves *is* the standing-vs-moving
    hysteresis that made the 2026-08-18 curve under-deliver by 3x in flight.
    Each pre-load is its own zero-settle segment so the step lands with no stop
    in between. At the sweep's ends the step direction inverts -- what the fit
    uses is which pre-load a sample was approached from, not the step's sign.

    Returns:
        List of ``(label, axes, hold_s, settle_s)`` descriptors.
    """
    segments = []
    for approach, preload in (("down", 850), ("up", 650)):
        for value in STEADY_VALUES:
            label = "ii/%s/%d" % (approach, value)
            segments.append((label + "/preload", _axes(x=preload), 2.0, 4.0))
            segments.append((label, _axes(x=value), 5.0, 0.0))
    return segments


def block_combined():
    """Block (iii): more than one axis commanded at once.

    Nothing in the stack models this today: the dead-band offset is added per
    axis, so a diagonal pays it twice, and yaw's measured coupling law
    (``turn_coordination``) was never wired to the Rooster. The singles here are
    references taken in the same conditions as the combinations, so the ratio
    between them is not contaminated by block (i)'s standing start.

    Returns:
        List of ``(label, axes, hold_s, settle_s)`` descriptors.
    """
    segments = []
    for value in (700, 750, 800):
        segments.append(("iii/a/xy%d" % value, _axes(x=value, y=value), 5.0, 4.0))
    for value in (700, 800):
        segments.append(("iii/a/x%d" % value, _axes(x=value), 5.0, 4.0))
        segments.append(("iii/a/y%d" % value, _axes(y=value), 5.0, 4.0))
    for forward in (700, 800):
        for turn in (0, 150, 250, 400, 600):
            segments.append(("iii/b/x%d_r%d" % (forward, turn),
                             _axes(x=forward, r=turn), 5.0, 4.0))
    for forward, lateral, turn in ((700, 700, 250), (800, 800, 400)):
        segments.append(("iii/c/x%d_y%d_r%d" % (forward, lateral, turn),
                         _axes(forward, lateral, turn), 5.0, 4.0))
    for sign in (1, -1):
        for value in YAW_VALUES:
            segments.append(("iii/d/r%s%d" % ("+" if sign > 0 else "-", value),
                             _axes(r=sign * value), 4.0, 4.0))
    return segments


BLOCKS = {"i": block_standing_start, "ii": block_steady_state, "iii": block_combined}


def _limit(label, axis):
    """Cap for one axis of one segment.

    The 800 cap exists to keep the airframe from tilting into the safety
    monitor's 25 deg trip, so it binds the two translation axes. Yaw does not
    tilt the aircraft and its full scale is one of the numbers this experiment
    exists to measure (MISSION.md P5: "no calibrated inverse at all"), so the
    block (iii)(d) sweep is allowed to reach 1000 -- as is block (ii), whose
    850 pre-load and 900/1000 samples measure the top of the forward curve.
    """
    if axis == "r" or label.startswith("ii/"):
        return MAX_ANY_AXIS
    return MAX_SAFE_AXIS


def assert_within_limits(label, axes):
    """Refuse any axis past the cap that applies to it.

    Raises:
        ValueError: If a segment would command more deflection than it is
            allowed. Checked at the source, so nothing downstream can construct
            an over-cap segment at all.
    """
    for name, value in axes.items():
        limit = _limit(label, name)
        if abs(value) > limit:
            raise ValueError("segment %s asks for %s=%g, past the %g cap"
                             % (label, name, value, limit))


def build_plan(blocks):
    """Concatenate the requested blocks in order, capped-checked.

    Args:
        blocks: Block keys, e.g. ``["i", "ii", "iii"]``.

    Returns:
        The full segment list.

    Raises:
        ValueError: On an unknown block name, rather than silently flying less
            than was asked for.
    """
    plan = []
    for key in blocks:
        if key not in BLOCKS:
            raise ValueError("unknown block %r; known: %s" % (key, ", ".join(sorted(BLOCKS))))
        plan.extend(BLOCKS[key]())
    for label, axes, _, _ in plan:
        assert_within_limits(label, axes)
    return plan


def estimate_seconds(plan):
    """Wall-clock the plan should take, publish overhead included."""
    return sum(hold + settle + PUBLISH_OVERHEAD_S for _, _, hold, settle in plan)


def describe(blocks):
    """Print the plan and its duration. Commands nothing; touches nothing.

    Returns:
        ``(plan, total_seconds)``.
    """
    plan = build_plan(blocks)
    for label, axes, hold_s, settle_s in plan:
        print("%-22s x=%-7g y=%-7g r=%-7g  settle %.1fs  hold %.1fs"
              % (label, axes["x"], axes["y"], axes["r"], settle_s, hold_s))
    total = estimate_seconds(plan)
    print("\n%d segments, ~%.0f s (%.1f min) including ~%.0f s of publish overhead"
          % (len(plan), total, total / 60.0, PUBLISH_OVERHEAD_S * len(plan)))
    return plan, total
