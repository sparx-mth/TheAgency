"""Instance boxes and masks from one per-pixel instance-id frame. numpy only; no simulator, no labels.

The semantic sensor renders an ``HxW`` integer frame whose value at a pixel is
the id of the annotated instance seen there (``0`` where nothing is annotated).
Everything a detector would have estimated about such a frame is exact here:
the tight axis-aligned box of every visible instance, how many pixels it
covers, and its pixel mask. The box convention is the detection server's --
``(x1, y1, x2, y2)`` in pixels, origin top-left, exclusive right/bottom edge,
so a one-pixel instance at column 5 is ``x1=5, x2=6`` -- which is what
``perception.projection.observed_objects`` checks against the frame size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class InstanceExtent:
    """Where one annotated instance appears in a frame.

    Attributes:
        instance: The annotation's instance id.
        xyxy: Tight pixel box ``(x1, y1, x2, y2)``, exclusive right/bottom.
        pixels: How many pixels carry the id (the mask's area, not the box's).
    """

    instance: int
    xyxy: Tuple[float, float, float, float]
    pixels: int


def instance_extents(ids, *, min_pixels=1, ignore=(0,)) -> Dict[int, InstanceExtent]:
    """The tight box and area of every instance visible in ``ids``.

    Args:
        ids: ``HxW`` integer frame of instance ids.
        min_pixels: An instance covering fewer pixels is not reported. The
            default reports every visible pixel; the oracle detector sets the
            floor it declares.
        ignore: Ids that are not instances (``0``, habitat-sim's "nothing").

    Returns:
        ``{instance id: InstanceExtent}`` in ascending id order.

    Raises:
        ValueError: ``ids`` is not a 2-D integer frame.
    """
    frame = np.asarray(ids)
    if frame.ndim != 2 or not np.issubdtype(frame.dtype, np.integer):
        raise ValueError("an instance-id frame must be a 2-D integer array, got %s %s" % (frame.shape, frame.dtype))
    if frame.size == 0:
        return {}
    present, counts = np.unique(frame, return_counts=True)
    extents = {}
    for instance, pixels in zip(present.tolist(), counts.tolist()):
        if instance in ignore or pixels < min_pixels:
            continue
        rows, cols = np.nonzero(frame == instance)
        extents[int(instance)] = InstanceExtent(
            int(instance),
            (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)),
            int(pixels))
    return extents


def instance_mask(ids, instance) -> np.ndarray:
    """The boolean ``HxW`` mask of one instance -- its own pixels, nothing of what occludes it."""
    frame = np.asarray(ids)
    if frame.ndim != 2:
        raise ValueError("an instance-id frame must be 2-D, got %s" % (frame.shape,))
    return frame == int(instance)
