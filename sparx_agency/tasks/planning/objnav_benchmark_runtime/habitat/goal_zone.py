"""The goal zone of a habitat-lab ObjectNav dataset: where its goal view points lie, and the STOP criterion that follows.

habitat-lab scores an ObjectNav episode by one number: the geodesic distance
from where the agent called STOP to the nearest *view point* of any goal
object of the category, which must be under ``SUCCESS_DISTANCE`` (0.1 m).
The view points are not computed at evaluation time; they are published with
the episodes, generated once per object by the dataset's authors. So what
"success" means geometrically -- how near the object, measured how, from
where -- is whatever that generator did, and it is not the same for the two
datasets this runtime evaluates. This module holds the generator's rule as a
small value (:class:`GoalZone`), each dataset's adapter states its own
(``hm3d/protocol.py``, ``mp3d/protocol.py``), and :func:`stop_criterion`
turns it into the method's STOP criterion
(``core/planning/object_search/closing/stop_criterion.py``).

**The generator** (habitat-lab's ObjectNav dataset code, as kept in
``Ram81/habitat-imitation-baselines`` -- ``object_nav_generator.build_goal``
-- and, for the HM3D lineage, ``naokiyokoyama/ovon`` --
``ObjectGoalGenerator._make_object_viewpoints``; read on 2026-11-16):

1. Candidate positions on a horizontal grid round the object's axis-aligned
   box, ``cell_size`` apart -- half the agent's radius: 0.09 m on MP3D and
   HM3D v1, 0.085 m on HM3D v2 -- out to ``max_distance`` = 1.0 m beyond the
   box on each side.
2. A candidate is kept when its distance to the object's box is at most
   ``max_distance`` (to its axis-aligned box and to its gravity-aligned
   oriented box, both), the floor under it is navigable, and the object's
   pixels show in the semantic frames rendered from it, turned toward the
   object at three camera pitches -- one tilt down, level, one tilt up
   (their summed coverage is stored as the view point's ``iou``). The kept
   candidate is then snapped onto the navigation mesh.
3. Success is a STOP within 0.1 m (geodesic) of a kept candidate.

**Which distance step 2 measures is the difference between the datasets**,
and it was measured here rather than assumed (2026-11-16, every validation
goal against the object's box as habitat-sim loads it; nothing of it is
used at run time):

* **HM3D** (v1: 432 goals of 20 scenes; v2: 417 goals whose annotation is
  the same object as in v1): the farthest view point of a goal stands
  1.00 m from the box *in the horizontal plane* -- for an object on the
  floor and for a television whose underside is 1.2-1.5 m up alike (its
  3-D distance from the floor point is then about 1.7 m). No view point of
  v1's 111,406 stands more than 1.01 m out. The published code says the
  same: the candidate is tested at the height of the box's centre.
  **Planar, 1.0 m.**
* **MP3D** (v1: 1,316 goals of 11 scenes, 291,810 view points): the reach
  shrinks as the object is raised -- 1.00 m in the plane for an object on
  the floor (median over goals), 0.89 m with the underside 0.3-0.6 m up,
  0.74 m at 0.6-0.9 m, 0.57 m at 0.9-1.0 m -- while the 3-D distance from
  the view point on the floor to the box stays at 1.00 m (1.11 m for the
  23 goals whose underside is over 0.9 m up); and no goal's underside is
  1.0 m up or more. A 1.0 m ball about the floor point, not a 1.0 m band
  in the plane. (The code that survives tests in the plane; the 2020
  dataset was evidently made otherwise. About 5 % of its view points stand
  beyond 1.0 m in the plane, by a few centimetres to a metre, where the
  generator moved them onto the navigation mesh.)
  **Euclidean from the floor, 1.0 m.**

**How the other published zero-shot methods stop** (their released code,
read the same day), all by a planar range to the nearest point of the
target's point cloud, on both datasets:

* VLFM: STOP when that range is under 0.9 m (``pointnav_stop_radius``).
* ASCENT: inside 1.0 m it steps straight on and STOPs at 0.6 m, or where a
  step no longer brings it nearer ("get close as possible").
* ApexNav: walks to the free cell nearest a point 0.5 m short of the object
  (0.70 m, then 0.85 m, when that has no route) and STOPs within 0.2 m of
  it; its evaluation configs also widen ``SUCCESS_DISTANCE`` to 0.2 m.

None of them measures in 3-D; on MP3D that leaves a planar 0.9 m STOP
outside the zone of any goal raised more than about 0.4 m, which is where
this method's STOPs on pictures and wall cabinets failed.

**The margin.** A STOP exactly at the zone's edge is within 0.1 m of a view
point only if the grid happens to have a candidate there. A decimetre inside
the edge it always has one: no place is farther than half the grid's
diagonal (0.064 m) from a candidate, and a candidate that near a place
0.9 m from the object is itself inside the zone. The rest of the decimetre
is for what the argument leaves out -- a candidate is a view point only if
it is navigable and sees the object, and the agent's range is a depth
measurement. So the STOP distance is the zone's distance less
:data:`STOP_MARGIN_M`: 0.9 m. It is VLFM's number, and the paired check of
that day on 180 single-storey MP3D episodes agrees with it. With the STOP
taken on the deciding frame, the 158 STOPs succeeded 40 times of 41 where
the range by MP3D's own measure read under 0.9 m, 46 of 51 between 0.9 and
1.0 m, and 48 of 66 beyond; and a criterion at the zone's edge itself
(1.0 m) against one at 0.9 m, the measure the same, lost three episodes and
won none under MP3D's measure and lost five under the planar one -- among
them STOPs 0.99 m from an object on the floor. The
protocol's success distance is checked, not subtracted: a run scored at a
wider radius (ApexNav's 0.2 m, which this runtime reports beside the
official one) stops where the official one does.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from sparx_agency.core.planning.object_search.closing.stop_criterion import MEASURES

#: How far inside its zone's edge the STOP is taken, metres (the module notes have the argument).
STOP_MARGIN_M = 0.1


@dataclass(frozen=True)
class GoalZone:
    """Where a dataset's generator put a goal's view points: the benchmark's success zone, as published.

    Attributes:
        distance_m: The generator's ``max_distance``: a view point lies
            within this of the object's bounding box.
        measure: How that distance is taken from the place the agent stands:
            ``planar`` (horizontal, whatever the object's height) or
            ``euclidean`` (3-D from the floor point).
        grid_m: The spacing of the candidate grid the view points were
            drawn from (half the agent's radius).

    Raises:
        ValueError: A distance or grid that is not positive, or an unknown measure.
    """

    distance_m: float
    measure: str
    grid_m: float

    def __post_init__(self) -> None:
        for name in ("distance_m", "grid_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError("GoalZone.%s must be a positive number, got %r" % (name, value))
        if self.measure not in MEASURES:
            raise ValueError("GoalZone.measure must be one of %s, got %r" % (", ".join(MEASURES), self.measure))


def stop_criterion(zone: GoalZone, success_distance_m: float) -> dict:
    """The ``TargetClosingSettings`` fields that put the method's STOP inside ``zone``.

    The zone's own measure, and its distance less :data:`STOP_MARGIN_M`: a
    STOP that far inside the zone's edge has a candidate of the zone within
    the success radius whichever way the candidate grid lies (see the module
    notes).

    Args:
        zone: The dataset's goal zone.
        success_distance_m: The protocol's ``SUCCESS_DISTANCE``: how near a
            view point a STOP must be. Checked against the grid; the STOP
            distance does not move with it.

    Returns:
        ``{"stop_within_m": ..., "stop_measure": ...}``.

    Raises:
        ValueError: The margin's argument does not hold for this zone -- a
            place can be farther from every candidate than the success
            radius, or than the margin -- or the margin leaves no distance
            to stop at.
    """
    success = float(success_distance_m)
    if not success > 0:
        raise ValueError("success distance must be positive, got %r" % (success_distance_m,))
    reach = zone.grid_m * math.sqrt(0.5)             # the farthest any place is from a candidate of the grid
    if reach >= success:
        raise ValueError("a %.3f m view-point grid leaves places %.3f m from every candidate, beyond the %.2f m "
                         "success distance: the STOP margin's argument does not hold" % (zone.grid_m, reach, success))
    if reach > STOP_MARGIN_M:
        raise ValueError("a %.3f m view-point grid leaves places %.3f m from every candidate, more than the %.2f m "
                         "STOP margin: the nearest candidate may lie outside the zone" % (zone.grid_m, reach, STOP_MARGIN_M))
    distance = round(zone.distance_m - STOP_MARGIN_M, 6)
    if not distance > 0:
        raise ValueError("a %.2f m margin leaves no STOP distance inside a %.2f m zone" % (STOP_MARGIN_M, zone.distance_m))
    return {"stop_within_m": distance, "stop_measure": zone.measure}
