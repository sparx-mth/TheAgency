"""The geometric sanity check on a target candidate: is the box at a height its class is seen at?

A detector's confidence is a score over the crop; it knows nothing about how
high the surface it fired on is. Two of the 5x3 benchmark's failures
(Allensville/2, Newfields/2) were a kitchen counter front read as ``bed`` at
0.55-0.6 m, with the supported depth points 0.67-0.70 m above the agent's
base where every real bed of the same recording measured 0.11-0.52 m -- and
STOPped on after two frames from one spot.

:func:`suspect` names the conflict, or None:

* ``height:<z>`` -- the measured height of the candidate's supported
  surface, above the agent's base, lies outside the band the class is seen
  at (:data:`CLASS_HEIGHT_BANDS`: a bed's or a toilet's surface is low; a
  counter top is not).

Until 2026-11-15 a second conflict, ``room:<label>``, doubted a candidate
standing in a room whose STRONG type a hand-written table called implausible
for the target (a bed in a kitchen, a toilet in a living room). It went with
``room_priors.py``: which room a target may stand in is not this check's to
assume. What is left is geometry -- and still a hand-written, per-class
table (:data:`CLASS_HEIGHT_BANDS`), which is why ``context_check`` is a
setting recorded in every run's frozen configuration.

A suspect candidate is not refused -- a bed with a high mattress exists. It
is PENALISED: its confidence counts for ``context_penalty`` of itself
toward the takeover's start threshold, and the lock needs
``context_confirmation_frames`` consecutive frames taken from at least two
viewpoints ``context_baseline_m`` apart (``TargetClosingSettings``), so the
detector has to be right from more than one place before the episode ends
on its word.
"""
from __future__ import annotations

import math


#: Detector labels folded onto the band table's keys: a naming convention, not a judgement.
_ALIASES = {"couch": "sofa", "tv": "television", "tv monitor": "television", "monitor": "television",
            "armchair": "chair", "plant": "potted plant"}

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


def suspect(class_name, xyz, base_z):
    """Why a candidate box does not make sense where it stands, or None: its height against its class's band."""
    return height_conflict(class_name, xyz, base_z)

