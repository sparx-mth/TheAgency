"""The semantic sanity check on a target candidate: does the box make sense where it stands?

A detector's confidence is a score over the crop; it knows nothing about
the room the crop was taken in or how high the surface it fired on is. Two
of the 5x3 benchmark's failures (Allensville/2, Newfields/2) were a kitchen
counter front read as ``bed`` at 0.55-0.6 m -- inside a room the map had
already made a STRONG kitchen from its refrigerator and oven, with the
supported depth points 0.67-0.70 m above the agent's base where every real
bed of the same recording measured 0.11-0.52 m -- and STOPped on after two
frames from one spot.

:func:`suspect` names the conflict, or None:

* ``room:<label>`` -- the candidate's footprint lies in a room whose STRONG
  type is one the target is not searched for in
  (:data:`~room_priors.IMPLAUSIBLE_ROOMS`: a bed in a kitchen, a toilet in a
  living room). A weak label is a guess and never makes a candidate
  suspect; a room the target has already been confirmed in neither.
* ``height:<z>`` -- the measured height of the candidate's supported
  surface, above the agent's base, lies outside the band the class is seen
  at (:data:`CLASS_HEIGHT_BANDS`: a bed's or a toilet's surface is low; a
  counter top is not).

A suspect candidate is not refused -- the target may stand in a room the
partition merged with a kitchen, and a bed with a high mattress exists. It
is PENALISED: its confidence counts for ``context_penalty`` of itself
toward the takeover's start threshold, and the lock needs
``context_confirmation_frames`` consecutive frames taken from at least two
viewpoints ``context_baseline_m`` apart (``TargetClosingSettings``), so the
detector has to be right from more than one place before the episode ends
on its word.
"""
from __future__ import annotations

import math

from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_priors import (
    _ALIASES, implausible_room, target_key)


#: Class -> ``(lowest, highest)`` height, in metres above the agent's base, the measured
#: centroid of the class's supported depth points is seen at. Calibrated on the Allensville
#: recordings of 2026-10-05 (the agent's base is the Habitat navmesh height; a level camera
#: 0.88 m above it): toilets 0.04-0.47, beds 0.11-0.52 from any range, sofas -0.15-0.74,
#: chairs -0.19-0.70, plants 0.36-0.86 (one on the floor, one on a counter), the one
#: television 0.99. The bands are wide on the low side -- a box on a low object reads low --
#: and set on the high side where a counter, a table top or a shelf begins (0.65 m and up):
#: that is the confusion they are for. A class not listed has no band.
CLASS_HEIGHT_BANDS = {
    "bed": (-0.30, 0.60),
    "toilet": (-0.30, 0.60),
    "sofa": (-0.30, 0.80),
    "chair": (-0.30, 0.90),
    "potted plant": (-0.30, 1.60),
    "television": (0.20, 2.00),
}


def canonical(class_name):
    """The table key of a detector label (``couch`` -> ``sofa``), lower-cased."""
    name = str(class_name or "").strip().lower()
    return _ALIASES.get(name, name)


def height_conflict(class_name, xyz, base_z):
    """``height:<z above base>`` when the measured centroid height contradicts the class, else None."""
    band = CLASS_HEIGHT_BANDS.get(canonical(class_name))
    if band is None or xyz is None:
        return None
    above = float(xyz[2]) - float(base_z)
    if not math.isfinite(above):
        return None
    low, high = band
    if low <= above <= high:
        return None
    return "height:%.2f" % above


def room_conflict(target, label, strength, objects):
    """``room:<label>`` when a STRONG room type excludes the target and no target object was confirmed there."""
    if str(strength or "").lower() != "strong":
        return None
    if not implausible_room(target, label):
        return None
    accepts = getattr(target, "accepts", None)
    if callable(accepts) and any(accepts(name) for name in objects or ()):
        return None
    return "room:%s" % str(label).strip().lower()


def suspect(target, class_name, xyz, base_z, label=None, strength=None, objects=()):
    """Why a candidate box does not make sense where it stands, or None (the room first, then the height)."""
    if target_key(target) is None:
        return None
    why = room_conflict(target, label, strength, objects)
    if why is not None:
        return why
    return height_conflict(class_name, xyz, base_z)

