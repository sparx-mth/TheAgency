"""Batched Sutherland-Hodgman clip of convex polygons against one half-space.

This is the primitive the slab clip is built from. It is written batched
because a building's collision geometry is millions of triangles: a Python
loop over triangles turns a two-minute job into an hour.

Polygons are carried as a padded ``(N, K, 3)`` array plus an ``(N,)`` vertex
count, so every polygon in a batch can be clipped with the same arithmetic
regardless of how many vertices it actually has. Clipping a convex polygon
against a half-space adds at most one vertex, so the output is padded to
``K + 1``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def clip_polygons_to_halfspace(
    polygons: np.ndarray,
    counts: np.ndarray,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clip convex polygons against an axis-aligned half-space.

    The kept half-space is ``coord >= bound`` when ``keep_greater`` is True and
    ``coord <= bound`` otherwise, where ``coord`` is component ``axis`` of each
    vertex.

    Args:
        polygons: ``(N, K, 3)`` float array of padded polygon vertices, in
            order around each polygon. Slots at or beyond the polygon's own
            vertex count are ignored and may hold anything.
        counts: ``(N,)`` integer vertex count per polygon. Zero means "already
            empty" and stays empty.
        axis: Which component to test, 0/1/2 for x/y/z.
        bound: The half-space boundary, in the same units as ``polygons``.
        keep_greater: Keep the side above ``bound`` rather than below it.

    Returns:
        ``(clipped, out_counts)`` where ``clipped`` is ``(N, K + 1, 3)`` and
        ``out_counts`` is ``(N,)``. Vertices past a polygon's count are zero.

    Raises:
        ValueError: If the inputs are not shaped as described.
    """
    polygons = np.asarray(polygons, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if polygons.ndim != 3 or polygons.shape[2] != 3:
        raise ValueError("polygons must be (N, K, 3), got %r" % (polygons.shape,))
    if counts.shape != (polygons.shape[0],):
        raise ValueError(
            "counts must be (N,) matching polygons, got %r vs %r"
            % (counts.shape, polygons.shape)
        )

    n_polys, n_slots = polygons.shape[0], polygons.shape[1]
    if n_polys == 0:
        return np.zeros((0, n_slots + 1, 3), dtype=np.float64), counts.copy()

    slot = np.arange(n_slots, dtype=np.int64)
    valid = slot[None, :] < counts[:, None]
    successor = np.where(slot[None, :] + 1 >= counts[:, None], 0, slot[None, :] + 1)

    start = polygons
    end = np.take_along_axis(polygons, successor[:, :, None].repeat(3, axis=2), axis=1)

    sign = 1.0 if keep_greater else -1.0
    dist_start = sign * (start[:, :, axis] - float(bound))
    dist_end = sign * (end[:, :, axis] - float(bound))
    inside_start = dist_start >= 0.0
    inside_end = dist_end >= 0.0

    denom = dist_start - dist_end
    ratio = np.divide(
        dist_start, denom, out=np.zeros_like(dist_start), where=denom != 0.0
    )
    crossing = start + ratio[:, :, None] * (end - start)

    # Per edge we emit the start vertex (if inside) and then the crossing
    # point (if the edge changes side); that ordering is what keeps the output
    # wound the same way as the input.
    emitted = np.empty((n_polys, n_slots, 2, 3), dtype=np.float64)
    emitted[:, :, 0, :] = start
    emitted[:, :, 1, :] = crossing
    keep = np.empty((n_polys, n_slots, 2), dtype=bool)
    keep[:, :, 0] = valid & inside_start
    keep[:, :, 1] = valid & (inside_start != inside_end)

    return _compact(emitted, keep, n_slots + 1)


def _compact(
    emitted: np.ndarray, keep: np.ndarray, out_slots: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather the kept candidate vertices into a dense padded array.

    Args:
        emitted: ``(N, K, 2, 3)`` candidate vertices in emission order.
        keep: ``(N, K, 2)`` boolean, which candidates survive.
        out_slots: Width of the padded output.

    Returns:
        ``(clipped, out_counts)``.
    """
    n_polys = emitted.shape[0]
    flat_points = emitted.reshape(n_polys, -1, 3)
    flat_keep = keep.reshape(n_polys, -1)

    out_counts = flat_keep.sum(axis=1).astype(np.int64)
    if int(out_counts.max(initial=0)) > out_slots:
        raise RuntimeError(
            "clip produced %d vertices, more than the %d a convex polygon can "
            "have; the input was probably not convex"
            % (int(out_counts.max()), out_slots)
        )

    position = np.cumsum(flat_keep, axis=1) - 1
    rows, cols = np.nonzero(flat_keep)
    clipped = np.zeros((n_polys, out_slots, 3), dtype=np.float64)
    clipped[rows, position[rows, cols]] = flat_points[rows, cols]
    return clipped, out_counts
