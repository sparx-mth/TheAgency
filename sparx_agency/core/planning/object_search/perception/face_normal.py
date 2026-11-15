"""The facing normal of a planar object, from the depth points a sighting measured on it.

A thin object mounted on a wall -- a picture, a wall cabinet's front, a towel
on its rail, a shower screen -- has its evaluator view points IN FRONT of it
(MP3D: within 0.26 m of a picture's width along the wall, median over the
single-storey set's 78 pictures, against 0.6-0.9 m for every other
category), because the dataset generator keeps a candidate only where the
object's pixels show with the camera turned to it, and a 2 cm picture shows
nothing from beside itself. The closing's standoff ring started from the
agent's own bearing, so a picture first seen at a grazing angle was walked
to and STOPped on from beside it: 8 of the 9 near-miss STOPs of the MP3D
baseline (2026-11-10) stood 0.2-0.4 m beyond their object's width, 45-70
degrees off its normal, inside the 1 m band and 0.1-0.5 m from every view
point.

The normal is in the sighting itself. The world-frame points of EVERY valid
depth pixel on the object's mask (``perception/projection.py``; the box's
central quarter without a mask) are a plane for such an object: the smallest
principal axis of their covariance is its normal, their spread along it is
its thickness. The whole mask, not the near-surface component the centroid
is measured on: the front of a sofa's seat or a chair's backrest is a plane
within the depth band, the whole sofa or chair is not. :func:`face_normal` returns the
planar unit normal pointed toward the camera when the points are planar
(thickness under ``max_thickness_m``, 5 cm), broad enough to fix a plane
(``min_breadth_m`` along the second axis -- a one-pixel sliver fixes none),
and the plane stands (``max_vertical`` on the normal's z: a table top is a
plane too, with nothing to stand in front of); None otherwise -- a chair, a
sofa, a bed, a plant are not planes and keep the ring as it was.

Read from the map instead, the free floor around the memory pointed toward
the EXPLORED side, which during an approach is the agent's own (the first
version, 2026-11-10: zsNo4HB9uLZ/000045 chose the identical standoff), and
moved as the map filled (pLe4wQe7qrG/000152: 31 re-selections in one lock).
A sighting's plane is what it is from the first frame.
"""
import math
from typing import Optional, Sequence, Tuple

import numpy as np

#: Fewer measured points than this fix no plane worth trusting.
MIN_POINTS = 30


def face_normal(points: np.ndarray, camera_xy: Sequence[float], max_thickness_m: float = 0.05,
                min_breadth_m: float = 0.05, max_vertical: float = 0.35) -> Optional[Tuple[Tuple[float, float], float]]:
    """The planar unit normal of a standing planar surface through ``points``, toward ``camera_xy``.

    Args:
        points: ``(N, 3)`` world-frame points (z up) the sighting measured on the object.
        camera_xy: Where the camera stood, planar; the normal points to its side of the plane.
        max_thickness_m: The points' standard deviation along the normal must not exceed this: a plane.
        min_breadth_m: The standard deviation along the plane's second axis must reach this: a surface, not a line.
        max_vertical: The normal's vertical component must not exceed this: a standing plane, not a top.

    Returns:
        ``((nx, ny), thickness_m)`` with ``(nx, ny)`` of unit length, or None when the points are too few,
        not a plane, a sliver, or a horizontal surface.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < MIN_POINTS or not np.all(np.isfinite(pts)):
        return None
    mean = pts.mean(axis=0)
    centred = pts - mean
    covariance = centred.T @ centred / float(len(pts))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)          # ascending
    thickness = math.sqrt(max(float(eigenvalues[0]), 0.0))
    breadth = math.sqrt(max(float(eigenvalues[1]), 0.0))
    normal = eigenvectors[:, 0]
    if thickness > max_thickness_m or breadth < min_breadth_m or abs(float(normal[2])) > max_vertical:
        return None
    planar = math.hypot(float(normal[0]), float(normal[1]))
    if planar <= 0.0:
        return None
    nx, ny = float(normal[0]) / planar, float(normal[1]) / planar
    if nx * (float(camera_xy[0]) - mean[0]) + ny * (float(camera_xy[1]) - mean[1]) < 0.0:
        nx, ny = -nx, -ny
    return (nx, ny), thickness
