"""What a voxel grid costs, before you start the container and find out.

FALCON allocates its whole map on the first tick — ``data.resize(map_size_idx_.prod())``
in ``map_base_inl.h`` — as a dense flat array over the ``map`` box. Nothing grows
later and nothing is sparse: a cubic metre the aircraft never visits costs the
same as one it maps in detail. So the bill is knowable in advance, and it is
worth knowing, because the failure mode is a container that dies or thrashes on
startup rather than a leak you can watch develop.

Six arrays are sized to the full grid. The two ESDF scratch buffers are the
surprise: they are only ever used over the local update region, and they are more
than a third of the total.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from sparx_agency.tasks.planning.falcon_pegasus.mapsize.area import Box

# Bytes per voxel, per full-map-sized array, read off the type definitions.
# `FloatingPoint` is `double` (exploration_types.h), which is where most of it
# goes: a TSDF voxel is two of them and an ESDF voxel one.
BYTES_PER_VOXEL = (
    ("occupancy", 4.0, "OccupancyVoxel — one enum class"),
    ("tsdf", 16.0, "TSDFVoxel — value and weight, both double"),
    ("esdf", 8.0, "ESDFVoxel — one double"),
    ("esdf_scratch_1", 8.0, "ESDF temp buffer, only used locally"),
    ("esdf_scratch_2", 8.0, "ESDF temp buffer, only used locally"),
    ("frontier_flag", 0.125, "vector<bool>, one bit per voxel"),
)

TOTAL_BYTES_PER_VOXEL = sum(entry[1] for entry in BYTES_PER_VOXEL)

# Below this box volume FALCON picks its fine resolution, above it the coarse
# one (map_server.cpp). We set the resolution explicitly instead, but the
# threshold still matters: it is what a run silently crosses when someone shrinks
# the exploration box, and memory then moves by the cube of the resolution ratio.
VOLUME_RESOLUTION_THRESHOLD_M3 = 4000.0


@dataclass(frozen=True)
class GridCost:
    """The memory a voxel grid will occupy.

    Attributes:
        shape: Voxels on the x, y and z axes.
        resolution: Voxel edge in metres.
        total_bytes: Sum over all six full-map-sized arrays.
    """

    shape: Tuple[int, int, int]
    resolution: float
    total_bytes: float

    @property
    def voxels(self) -> int:
        """Total voxel count."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def megabytes(self) -> float:
        """Total in MB, base 1024."""
        return self.total_bytes / (1024.0 * 1024.0)

    def breakdown(self) -> Tuple[Tuple[str, float, str], ...]:
        """Per-array cost in bytes, in the same order as :data:`BYTES_PER_VOXEL`."""
        return tuple(
            (name, self.voxels * per_voxel, note)
            for name, per_voxel, note in BYTES_PER_VOXEL
        )


def grid_cost(box: Box, resolution: float) -> GridCost:
    """Compute what allocating ``box`` at ``resolution`` will cost.

    Args:
        box: The ``map`` box — the allocated extent, not the exploration box.
        resolution: Voxel edge in metres.

    Returns:
        The cost, including the per-array breakdown.

    Raises:
        ValueError: If ``resolution`` is not positive.
    """
    shape = box.grid_shape(resolution)
    voxels = shape[0] * shape[1] * shape[2]
    return GridCost(
        shape=shape,
        resolution=resolution,
        total_bytes=voxels * TOTAL_BYTES_PER_VOXEL,
    )


def implicit_resolution(box_volume_m3: float, fine: float = 0.1, coarse: float = 0.2) -> float:
    """The resolution FALCON would pick on its own, from the exploration volume.

    Only useful for warning that an explicit choice differs from it. Note the
    direction, which catches people out: a *smaller* exploration box can drop
    under the threshold and pick the *finer* resolution, costing eight times the
    memory to explore less space.

    Args:
        box_volume_m3: Volume of the exploration box.
        fine: Resolution used below the threshold.
        coarse: Resolution used at or above it.

    Returns:
        The resolution FALCON's own rule would choose.
    """
    return fine if box_volume_m3 < VOLUME_RESOLUTION_THRESHOLD_M3 else coarse
