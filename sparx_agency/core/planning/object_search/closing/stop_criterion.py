"""The stop criterion: how near its target the agent stands before it calls STOP, and by which measure.

The method decides that it has found its target on its own evidence (a
fresh, whole, confident box inside the terminal range). How near the target
it must then STAND is not the method's to decide: an ObjectNav benchmark
defines a success zone around each goal object, and two benchmarks that both
say "within 1 m" do not mean the same thing by it. So the distance and the
measure are a criterion the benchmark's adapter supplies from the benchmark's
own published definition (``TargetClosingSettings.stop_within_m`` and
``stop_measure``), and the closing walks on toward the target after its
decision until its own range to the target reads within it
(``TargetClosing._terminal``).

**What the agent measures.** Both definitions are distances from the place
the agent stands to the object's bounding box. The agent has no bounding
box; it has the points of the object its depth frames measured, and of
those two numbers stand for the box:

* the **planar range** to the nearest measured point (the near edge), and
* the **elevation** of the lowest measured point above the floor the agent
  stands on -- how far the box's underside is raised.

A box is its footprint extended over its height, so the distance from a
point on the floor to it is the planar distance to the footprint when the
box stands on the floor, and the hypotenuse of that distance and the
underside's height when it is raised. That is all a measure needs:

``planar``
    The horizontal distance to the object, whatever its height. HM3D
    ObjectNav: a goal's view points are the navigable floor within 1.0 m of
    the object's axis-aligned box *in the horizontal plane* (measured on the
    v1 validation goals: the farthest view point stands 1.00 m from the box
    for an object on the floor and for a television mounted 1.4 m up alike).

``euclidean``
    The 3-D distance from the floor under the agent to the object. MP3D
    ObjectNav: a goal's view points reach 1.0 m from the box for an object
    on the floor and less the higher its underside is (0.89 m at 0.3-0.6 m
    up, 0.74 m at 0.6-0.9 m, on the validation goals), as a 1.0 m ball about
    the floor point gives; no goal's underside is 1.0 m up or more.

Both readings err on the safe side: the measured near edge is never nearer
than the box's, and the lowest measured point never lower than its
underside, so a reading within the criterion puts the agent within it of the
box. (The caller may read an underside raised by less than a threshold as an
object on the floor -- ``TargetClosingSettings.stop_elevation_min_m``,
0.30 m, which shortens a 0.9 m reading by under 5 cm.) How each definition
was established, and what the other published methods do, is in
``tasks/planning/objnav_benchmark_runtime/habitat/goal_zone.py``.

Pure functions, no state. Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import math

#: The horizontal distance to the target, whatever its height.
PLANAR = "planar"
#: The 3-D distance from the floor under the agent to the target.
EUCLIDEAN = "euclidean"
#: The measures a stop criterion may name.
MEASURES = (PLANAR, EUCLIDEAN)


def stop_range_m(measure: str, planar_m: float, elevation_m: float) -> float:
    """The range a stop criterion of ``measure`` reads, metres.

    Args:
        measure: :data:`PLANAR` or :data:`EUCLIDEAN`.
        planar_m: Horizontal distance from the agent to the nearest measured
            point of the target.
        elevation_m: How far the target's lowest measured point stands above
            the floor under the agent; an object on the floor (or below it)
            is 0.

    Returns:
        ``planar_m`` under the planar measure; under the Euclidean one the
        hypotenuse of ``planar_m`` and the elevation.

    Raises:
        ValueError: An unknown measure, or a range that is negative or not finite.
    """
    if measure not in MEASURES:
        raise ValueError("stop measure must be one of %s, got %r" % (", ".join(MEASURES), measure))
    planar, elevation = float(planar_m), float(elevation_m)
    if not math.isfinite(planar) or planar < 0.0 or not math.isfinite(elevation):
        raise ValueError("stop range needs a finite, non-negative planar range and a finite elevation, got %r and %r"
                         % (planar_m, elevation_m))
    if measure == PLANAR:
        return planar
    return math.hypot(planar, max(0.0, elevation))

