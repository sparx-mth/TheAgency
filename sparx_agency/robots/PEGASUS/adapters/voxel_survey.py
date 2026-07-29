"""Sweep a whole Isaac Sim scene into a ground-truth 3D occupancy voxel grid.

One pass over the building at 10 cm, in C++, in seconds. Every 2D map the
flights plan against is then a *slice* of this rather than another sweep, so a
scene is surveyed once and any altitude is free.

Isaac ships the sweep as ``isaacsim.asset.gen.omap``, a PhysX-backed generator.
Doing it per-voxel from Python is not an alternative: a building at 10 cm is
tens of millions of voxels, and even the cheapest scene query is tens of
microseconds. Measured here, the C++ generator did 400x400x30 in 3.5 seconds.

Three things about that generator are not obvious and all three will waste an
afternoon:

* **Importing the binding before enabling the extension kills the process.**
  ``from isaacsim.asset.gen.omap.bindings import _omap`` on its own terminates
  Kit with exit code 0, no traceback, no crash dump -- indistinguishable from a
  clean shutdown. :func:`enable_omap` does it in the right order.
* **The bounds passed to ``set_transform`` are relative to the origin**, not
  absolute world coordinates. Confirmed by reading ``get_min_bound()`` back.
* **``get_buffer()`` is useless after ``generate3d()``** -- it returns a
  flattened 2D-sized buffer with no occupied cells in it. The 3D result comes
  out of ``get_occupied_positions()`` and ``get_free_positions()`` as lists of
  world-frame voxel centres, which is what this module rasterises.

The generator floods outward from its origin, so **the origin must be a point
inside the building** and everything it cannot reach stays UNKNOWN. That sounds
like it should separate the interior from the kilometre of open ground these
assets sit on, and it does not: the free space above the roof connects to
everything, so the flood escapes over the top and the surrounding field comes
back as perfectly good FREE space. Measured on ``office``, 867 m2 of building
became 4618 m2 of building-plus-car-park, with zero UNKNOWN voxels anywhere.
Separating them needs a ceiling test --
:func:`~sparx_agency.core.planning.environment.voxel_grid_3d.restrict_to_indoor`
-- which is one array reduction once the voxel column exists.

Must run inside a live Isaac Sim process with the timeline playing -- PhysX
returns nothing against a stopped simulation, silently.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from sparx_agency.core.planning.environment.voxel_grid_3d import (
    FREE, OCCUPIED, UNKNOWN, VoxelGrid3D,
)

DEFAULT_RESOLUTION_M = 0.1
DEFAULT_RADIUS_M = 40.0
DEFAULT_FLOOR_M = -0.3
DEFAULT_CEILING_M = 6.0
# Values handed to the generator. They only have to be distinct; the rasteriser
# reads positions rather than the buffer, so these never reach the output.
_OCCUPIED_VALUE, _FREE_VALUE, _UNKNOWN_VALUE = 1.0, 0.0, -1.0


def enable_omap():
    """Enable the occupancy-map extension and return its binding module.

    Must be called before the binding is imported anywhere. Importing
    ``_omap`` with the extension disabled terminates the process silently.

    Returns:
        The ``_omap`` binding module.

    Raises:
        RuntimeError: If the extension could not be enabled -- which is worth
            failing loudly on, because the alternative is a silent exit.
    """
    from isaacsim.core.utils.extensions import enable_extension

    if not enable_extension("isaacsim.asset.gen.omap"):
        raise RuntimeError(
            "could not enable isaacsim.asset.gen.omap. Importing its binding "
            "without it terminates Kit with exit code 0 and no traceback, so "
            "this is refused rather than attempted."
        )
    from isaacsim.asset.gen.omap.bindings import _omap

    return _omap


def survey_voxels(
    origin_xyz,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    radius_m: float = DEFAULT_RADIUS_M,
    floor_m: float = DEFAULT_FLOOR_M,
    ceiling_m: float = DEFAULT_CEILING_M,
    verbose: bool = True,
) -> Tuple[VoxelGrid3D, Dict]:
    """Sweep the loaded scene into a ground-truth voxel grid.

    Args:
        origin_xyz: A world point **inside the building and in free space** --
            the generator floods outward from it and everything it cannot reach
            stays UNKNOWN. A surveyed spawn point is the natural choice.
        resolution_m: Voxel edge length, metres.
        radius_m: Half-width of the swept area around the world origin, metres.
            These assets sit on a kilometre-wide ground plane, so the scene's
            own bounding box is not a useful limit.
        floor_m: Lowest world z to sweep, metres. Slightly below the floor so
            the floor surface itself is captured.
        ceiling_m: Highest world z to sweep, metres.
        verbose: Print progress and timings.

    Returns:
        ``(grid, metadata)``.

    Raises:
        RuntimeError: If the sweep found nothing, which almost always means the
            timeline was not playing or the origin was inside geometry.
    """
    import omni

    module = enable_omap()
    generator = module.Generator(omni.physx.get_physx_interface(),
                                 omni.usd.get_context().get_stage_id())
    generator.update_settings(resolution_m, _OCCUPIED_VALUE, _FREE_VALUE,
                              _UNKNOWN_VALUE)

    origin = tuple(float(v) for v in origin_xyz)
    # set_transform's bounds are RELATIVE to the origin, not absolute.
    lower = (-radius_m - origin[0], -radius_m - origin[1], floor_m - origin[2])
    upper = (radius_m - origin[0], radius_m - origin[1], ceiling_m - origin[2])

    if verbose:
        print(f"  sweeping +/-{radius_m:.0f} m, z {floor_m:.1f}..{ceiling_m:.1f} m "
              f"at {resolution_m:.2f} m from ({origin[0]:.1f}, {origin[1]:.1f}, "
              f"{origin[2]:.1f})...", flush=True)
    generator.set_transform(origin, lower, upper)
    started = time.time()
    generator.generate3d()
    elapsed = time.time() - started

    occupied = _positions(generator.get_occupied_positions())
    free = _positions(generator.get_free_positions())
    if verbose:
        print(f"  generate3d took {elapsed:.1f} s: {len(occupied)} occupied, "
              f"{len(free)} free voxels", flush=True)
    if len(occupied) == 0 and len(free) == 0:
        raise RuntimeError(
            "the sweep returned no voxels at all. The two causes are a stopped "
            "timeline (PhysX scene queries return nothing, silently) and an "
            "origin that is not in reachable free space."
        )

    grid = _rasterise(occupied, free, resolution_m,
                      (-radius_m, -radius_m, floor_m), (radius_m, radius_m, ceiling_m))
    metadata = {
        "resolution_m": resolution_m,
        "radius_m": radius_m,
        "floor_m": floor_m,
        "ceiling_m": ceiling_m,
        "origin_xyz": list(origin),
        "generate_seconds": round(elapsed, 2),
    }
    metadata.update(grid.stats())
    return grid, metadata


def _positions(points) -> np.ndarray:
    """Carb ``Float3`` list to an ``(N, 3)`` float64 array."""
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float64)


def _rasterise(occupied: np.ndarray, free: np.ndarray, resolution_m: float,
               lower, upper) -> VoxelGrid3D:
    """Scatter voxel-centre point lists into a dense ``[z, y, x]`` grid.

    The generator reports centres, so the index is an exact floor of the offset
    from the grid's lower corner -- no interpolation, no rounding policy to get
    wrong. Occupied is written after free so a voxel reported as both (they can
    share a boundary) reads as the obstacle.
    """
    lower = np.asarray(lower, dtype=np.float64)
    shape_xyz = np.maximum(
        np.ceil((np.asarray(upper, dtype=np.float64) - lower) / resolution_m).astype(int), 1)
    voxels = np.full((shape_xyz[2], shape_xyz[1], shape_xyz[0]), UNKNOWN, dtype=np.int8)

    for points, value in ((free, FREE), (occupied, OCCUPIED)):
        if len(points) == 0:
            continue
        index = np.floor((points - lower) / resolution_m).astype(np.int64)
        inside = np.all((index >= 0) & (index < shape_xyz), axis=1)
        index = index[inside]
        voxels[index[:, 2], index[:, 1], index[:, 0]] = value

    return VoxelGrid3D(voxels, resolution_m, lower, frame_id="world")


def trim_to_content(grid: VoxelGrid3D, margin_voxels: int = 2) -> VoxelGrid3D:
    """Crop a grid down to the part that was actually observed.

    A sweep is bounded by a generous radius around the world origin, so most of
    the result is UNKNOWN filler outside the building. Cropping it away shrinks
    the stored map several-fold and costs nothing -- the origin moves with it.

    Args:
        grid: The swept grid.
        margin_voxels: Voxels of UNKNOWN to keep around the content, so a
            planner sees a border rather than the array edge.

    Returns:
        A new, smaller :class:`VoxelGrid3D`. The original if nothing was
        observed.
    """
    known = np.argwhere(grid.known)
    if known.size == 0:
        return grid
    low = np.maximum(known.min(axis=0) - margin_voxels, 0)
    high = np.minimum(known.max(axis=0) + margin_voxels + 1, grid.voxels.shape)
    cropped = grid.voxels[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    origin = (grid.origin_x + low[2] * grid.resolution,
              grid.origin_y + low[1] * grid.resolution,
              grid.origin_z + low[0] * grid.resolution)
    return VoxelGrid3D(cropped, grid.resolution, origin, grid.frame_id)
