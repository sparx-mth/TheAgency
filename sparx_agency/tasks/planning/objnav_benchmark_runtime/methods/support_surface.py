"""Support-surface resilience: a box of the surface the target stands on is spatial evidence, not a contradiction.

At close range a mounted or supported object leaves the frame before its
support does: the television on a dresser fills the top of the image and the
detector fires on the dresser; a plant on a side table becomes the table. The
target's class vanishes from the detections while the thing at the memory's
XYZ is still there -- and still consistent. This module answers one question
for the closing: is a box of *another* class, projected to ``surface_xyz``,
the support of the target the memory holds at ``target_xyz``? If so the lock
keeps its spatial evidence, the inspection goes on looking (the dynamic pitch
puts the camera back on the object) and the frame is never read as "saw
nothing" or as the map outvoting the lock.

The classes are the Gibson/HM3D context vocabulary's furniture surfaces plus
the common supports the detector names; the test is exact, case-insensitive
membership, and the geometry is strict: the surface's centroid must lie under
the target's footprint and at or below its height.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

#: Furniture that carries other objects. Exact labels; the detector vocabulary decides which ever fire.
SUPPORT_SURFACE_CLASSES = ("cabinet", "desk", "dining table", "table", "counter", "countertop", "dresser",
                           "nightstand", "shelf", "sink", "tv stand", "stand", "bench", "chest of drawers")


def is_support_surface(label: str, classes: Iterable[str] = SUPPORT_SURFACE_CLASSES) -> bool:
    """Whether ``label`` names a supporting surface (exact, case-insensitive)."""
    name = str(label).strip().lower()
    return any(name == str(cls).strip().lower() for cls in classes)


def supports(target_xyz: Sequence[float], surface_xyz: Sequence[float], radius_m: float,
             below_tolerance_m: float = 0.10) -> bool:
    """Whether a surface centroid at ``surface_xyz`` can be what the target at ``target_xyz`` stands on.

    Args:
        target_xyz: The memory's fused target centroid.
        surface_xyz: The projected centroid of the surface box.
        radius_m: Planar association radius -- the same one a re-sighting of
            the target itself must fall inside.
        below_tolerance_m: How far ABOVE the target's centroid the surface
            may still read (depth noise on a low television); a surface
            clearly above the target is not its support.
    """
    if radius_m <= 0 or not math.isfinite(radius_m):
        raise ValueError("Support association radius must be positive and finite")
    planar = math.dist((float(target_xyz[0]), float(target_xyz[1])), (float(surface_xyz[0]), float(surface_xyz[1])))
    return planar <= radius_m and float(surface_xyz[2]) <= float(target_xyz[2]) + float(below_tolerance_m)
