"""A dense 3D occupancy voxel grid, with persistence and a 2D projection.

The ground-truth counterpart to :class:`OccupancyGrid2D`: where that says what
is in the way at one altitude, this says it for the whole building. It exists
for three jobs, and only the first is planning:

* it satisfies the :class:`~.voxelmap3d.VoxelMap3D` protocol, so the 3D planners
  can use it directly;
* the 2D maps every flight plans against are **derived** from it
  (:func:`project_to_occupancy_2d`), so a scene is surveyed once and every
  altitude is a slice rather than another sweep;
* it is the thing a human opens to check the survey is not nonsense.

Three states per voxel, and the third matters: FREE, OCCUPIED, and UNKNOWN for
"never observed". A survey reaches the inside of a building by flooding out from
a point in it, so everything beyond the walls stays UNKNOWN rather than being
mistaken for open air. Treating unknown as free is how a planner routes a drone
out through a wall.

Array layout is ``[z, y, x]``, matching ``tasks/planning/3D_planning``'s
existing voxel maps, so the two are interchangeable on disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 1


class VoxelGrid3D:
    """Dense 3D occupancy, indexed ``[z, y, x]``.

    Args:
        voxels: ``(depth, height, width)`` int8 of :data:`FREE` /
            :data:`OCCUPIED` / :data:`UNKNOWN`.
        resolution: Voxel edge length, metres.
        origin: World ``(x, y, z)`` of the **lower corner** of voxel
            ``[0, 0, 0]``.
        frame_id: Coordinate frame name.

    Raises:
        ValueError: If ``voxels`` is not 3D or ``resolution`` is not positive.
    """

    def __init__(self, voxels: np.ndarray, resolution: float, origin,
                 frame_id: str = "world"):
        voxels = np.asarray(voxels)
        if voxels.ndim != 3:
            raise ValueError(f"voxels must be 3D [z, y, x], got shape {voxels.shape}")
        if resolution <= 0.0:
            raise ValueError(f"resolution must be > 0, got {resolution}")

        self.voxels = voxels.astype(np.int8, copy=False)
        self.resolution = float(resolution)
        self.origin_x, self.origin_y, self.origin_z = (float(v) for v in origin)
        self.frame_id = str(frame_id)
        self.depth, self.height, self.width = self.voxels.shape
        self.clearance: Optional[np.ndarray] = None   # part of the VoxelMap3D protocol

    def __repr__(self) -> str:
        return (f"VoxelGrid3D({self.width}x{self.height}x{self.depth} @ "
                f"{self.resolution} m, origin=({self.origin_x:.1f}, "
                f"{self.origin_y:.1f}, {self.origin_z:.1f}))")

    @property
    def occupied(self) -> np.ndarray:
        """``(depth, height, width)`` boolean mask of occupied voxels."""
        return self.voxels == OCCUPIED

    @property
    def known(self) -> np.ndarray:
        """``(depth, height, width)`` boolean mask of observed voxels."""
        return self.voxels != UNKNOWN

    def world_to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """World metres to voxel indices ``(i, j, k)`` = ``(x, y, z)``.

        Note the order: the protocol asks for ``(i, j, k)`` while the array is
        indexed ``voxels[k, j, i]``.
        """
        return (int(np.floor((x - self.origin_x) / self.resolution)),
                int(np.floor((y - self.origin_y) / self.resolution)),
                int(np.floor((z - self.origin_z) / self.resolution)))

    def grid_to_world(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Voxel indices to the world coordinates of the voxel's **centre**."""
        return ((i + 0.5) * self.resolution + self.origin_x,
                (j + 0.5) * self.resolution + self.origin_y,
                (k + 0.5) * self.resolution + self.origin_z)

    def in_bounds(self, i: int, j: int, k: int) -> bool:
        """Whether ``(i, j, k)`` is inside the grid."""
        return (0 <= i < self.width and 0 <= j < self.height and 0 <= k < self.depth)

    def is_free(self, i: int, j: int, k: int) -> bool:
        """Whether voxel ``(i, j, k)`` is known to be free.

        Out of bounds is **not** free, and neither is UNKNOWN: a planner must
        never route through space the survey did not reach.
        """
        if not self.in_bounds(i, j, k):
            return False
        return bool(self.voxels[k, j, i] == FREE)

    def world_clearance(self, x: float, y: float, z: float) -> float:
        """Distance to the nearest obstacle at a world point, metres.

        Part of the :class:`VoxelMap3D` protocol. Returns 0 until a clearance
        field has been computed and assigned to :attr:`clearance`; a 3D
        Euclidean transform over a building-sized grid is expensive enough that
        it is not done unless something asks for it.
        """
        if self.clearance is None:
            return 0.0
        i, j, k = self.world_to_grid(x, y, z)
        if not self.in_bounds(i, j, k):
            return 0.0
        return float(self.clearance[k, j, i])

    def occupied_points(self) -> np.ndarray:
        """World-frame centres of every occupied voxel, ``(N, 3)`` float32.

        This is what gets written to a point cloud for viewing.
        """
        k, j, i = np.nonzero(self.occupied)
        return np.stack([
            (i + 0.5) * self.resolution + self.origin_x,
            (j + 0.5) * self.resolution + self.origin_y,
            (k + 0.5) * self.resolution + self.origin_z,
        ], axis=1).astype(np.float32)

    def stats(self) -> Dict[str, int]:
        """Voxel counts by state, for a one-line summary."""
        return {
            "voxels": int(self.voxels.size),
            "occupied": int((self.voxels == OCCUPIED).sum()),
            "free": int((self.voxels == FREE).sum()),
            "unknown": int((self.voxels == UNKNOWN).sum()),
        }


def project_to_occupancy_2d(grid: VoxelGrid3D, altitude_m: float,
                            half_height_m: float, values=None):
    """Slice a horizontal slab out of a voxel grid as a 2D occupancy grid.

    A cell is OCCUPIED if **any** voxel in the slab is, which is the right rule
    for ground truth: there is no sensor noise to average away, and a table leg
    that occupies one voxel of the band is exactly as solid as a wall.

    (``core/mapping/bev`` does a superficially similar projection and is
    deliberately not used here. Everything that makes it good -- the temporal
    Schmitt filter, the 3D neighbour confirmation, doorway protection, wall
    bridging -- exists to fight monocular-depth noise. Against a ground-truth
    grid those would only erode a map that is already correct.)

    Args:
        grid: The surveyed voxel grid.
        altitude_m: Centre of the slab, metres in the world frame.
        half_height_m: Half its thickness. Should cover the airframe's vertical
            extent plus how badly the autopilot holds height.
        values: Occupancy encoding for the output. Defaults to
            :class:`OccupancyValues`' FREE=0 / OCCUPIED=1 / UNKNOWN=-1.

    Returns:
        An :class:`OccupancyGrid2D` co-registered with ``grid`` in x and y.

    Raises:
        ValueError: If the slab falls entirely outside the grid's z range.
    """
    from .occupancy_grid2d import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues

    values = values or OccupancyValues()
    low = grid.world_to_grid(0.0, 0.0, altitude_m - half_height_m)[2]
    high = grid.world_to_grid(0.0, 0.0, altitude_m + half_height_m)[2]
    low, high = max(low, 0), min(high + 1, grid.depth)
    if low >= high:
        raise ValueError(
            f"the {altitude_m:.2f} +/- {half_height_m:.2f} m slab is outside the "
            f"grid's z range ({grid.origin_z:.2f} to "
            f"{grid.origin_z + grid.depth * grid.resolution:.2f} m)"
        )

    slab = grid.voxels[low:high]
    occupied = (slab == OCCUPIED).any(axis=0)
    known = (slab != UNKNOWN).any(axis=0)

    cells = np.where(occupied, values.occupied, values.free).astype(np.int16)
    cells[~known] = values.unknown
    return OccupancyGrid2D(
        cells,
        OccupancyGrid2DParams(resolution=grid.resolution, origin_x=grid.origin_x,
                              origin_y=grid.origin_y, frame_id=grid.frame_id),
        values=values,
    )


def landable_mask(grid: VoxelGrid3D, altitude_m: float, floor_z_m: float = 0.0,
                  floor_clearance_m: float = 0.25) -> np.ndarray:
    """Cells whose column is clear from just above the floor up to ``altitude_m``.

    Being able to fly somewhere and being able to *land* there are different
    questions: a cell can be wide open at 1.5 m with a desk at 0.7 m under it.
    An episode that ends there puts the aircraft on the desk, where it tips, and
    every later flight is refused on attitude. With a full voxel column the
    answer is exact rather than inferred from a downward ray.

    Args:
        grid: The surveyed voxel grid.
        altitude_m: Cruise altitude, metres.
        floor_z_m: World z of the floor.
        floor_clearance_m: How far above ``floor_z_m`` to start looking. The
            floor is itself a slab of occupied voxels -- and a conservatively
            voxelised one, so it is thicker than the surface it represents.
            Starting the scan inside it makes *every* cell unlandable, which is
            exactly what a too-small value here did.

    Returns:
        ``(height, width)`` boolean mask, co-registered with
        :func:`project_to_occupancy_2d`'s output.
    """
    low = grid.world_to_grid(0.0, 0.0, floor_z_m + floor_clearance_m)[2]
    high = grid.world_to_grid(0.0, 0.0, altitude_m)[2]
    low, high = max(low, 0), min(max(high, low + 1), grid.depth)
    column = grid.voxels[low:high]
    return ~(column == OCCUPIED).any(axis=0)


def indoor_mask(grid: VoxelGrid3D, altitude_m: float,
                max_ceiling_m: float = 8.0, gap_m: float = 0.3) -> np.ndarray:
    """Columns that have a ceiling over them: inside the building, not open sky.

    The sweep floods outward from a point in the building and marks what it
    cannot reach as UNKNOWN, which sounds like it should separate inside from
    outside on its own. It does not: the free space above the roof connects to
    everything, so the flood escapes over the top and the surrounding field
    comes back as perfectly good FREE space. Measured on ``office``, that turned
    867 m2 of building into 4618 m2 of building-plus-car-park.

    A ceiling test is what actually distinguishes them, and with a voxel column
    in hand it is one array reduction rather than a raycast per cell.

    Args:
        grid: The surveyed voxel grid.
        altitude_m: Flight altitude to look upward from, metres.
        max_ceiling_m: How far above that to keep looking. Beyond this there is
            no roof and the cell is outdoors.
        gap_m: Start looking this far above ``altitude_m``, so the aircraft's
            own slab is not mistaken for a ceiling.

    Returns:
        ``(height, width)`` boolean mask, co-registered with
        :func:`project_to_occupancy_2d`'s output.
    """
    low = grid.world_to_grid(0.0, 0.0, altitude_m + gap_m)[2]
    high = grid.world_to_grid(0.0, 0.0, altitude_m + max_ceiling_m)[2]
    low, high = max(low, 0), min(max(high, low + 1), grid.depth)
    return (grid.voxels[low:high] == OCCUPIED).any(axis=0)


def restrict_to_indoor(grid: VoxelGrid3D, altitude_m: float,
                       max_ceiling_m: float = 8.0) -> VoxelGrid3D:
    """Mark every column with no ceiling over it as UNKNOWN.

    Applied to the grid itself rather than to each derived 2D map, so the stored
    ground truth *is* the building: the point cloud, every altitude slice and
    every landability test all inherit it, and none of them can disagree.

    Args:
        grid: The swept voxel grid, modified in place and returned.
        altitude_m: Altitude to run the ceiling test from, metres.
        max_ceiling_m: See :func:`indoor_mask`.

    Returns:
        The same grid, with outdoor columns blanked.
    """
    outdoors = ~indoor_mask(grid, altitude_m, max_ceiling_m)
    grid.voxels[:, outdoors] = UNKNOWN
    return grid


def save_voxel_grid(path, grid: VoxelGrid3D, metadata: Optional[Dict] = None) -> Path:
    """Write a voxel grid and its provenance to a compressed ``.npz``.

    Args:
        path: Destination file. Parent directories are created.
        grid: The grid to store.
        metadata: JSON-serialisable provenance, returned verbatim on load.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        voxels=grid.voxels,
        resolution=np.float64(grid.resolution),
        origin=np.array([grid.origin_x, grid.origin_y, grid.origin_z], dtype=np.float64),
        frame_id=np.array(grid.frame_id),
        metadata=np.array(json.dumps(metadata or {})),
    )
    return path


def load_voxel_grid(path) -> Tuple[VoxelGrid3D, Dict]:
    """Read back a grid written by :func:`save_voxel_grid`.

    Args:
        path: The ``.npz`` file.

    Returns:
        ``(grid, metadata)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no voxel grid at {path}")
    with np.load(str(path), allow_pickle=False) as data:
        grid = VoxelGrid3D(data["voxels"], float(data["resolution"]),
                           data["origin"], str(data["frame_id"]))
        metadata = json.loads(str(data["metadata"]))
    return grid, metadata
