"""
Window views for reusing existing A* search code.

We do NOT re-implement A*.
Instead, we provide lightweight map "views" that:
- limit the effective map bounds to a local window
- translate local indices to global indices

The A* implementations only require:
2D:
    - in_bounds(x, y)
    - is_occupied(x, y)
    - is_unknown(x, y)

3D:
    - width, height, depth
    - is_free(i, j, k)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.environment.voxelmap3d import VoxelMap3D


Index2D = Tuple[int, int]
Index3D = Tuple[int, int, int]


@dataclass(frozen=True)
class WindowGridView2D:
    """
    A bounded view into an OccupancyGrid2D.

    Local coordinates are [0..w-1]x[0..h-1].
    They are mapped to global grid indices by adding (x0, y0).
    """
    base: OccupancyGrid2D
    x0: int
    y0: int
    w: int
    h: int

    @property
    def frame_id(self) -> str:
        return getattr(self.base, "frame_id", "map")

    def local_to_global(self, lx: int, ly: int) -> Index2D:
        return (lx + self.x0, ly + self.y0)

    def global_to_local(self, gx: int, gy: int) -> Index2D:
        return (gx - self.x0, gy - self.y0)

    def in_bounds(self, lx: int, ly: int) -> bool:
        if not (0 <= lx < self.w and 0 <= ly < self.h):
            return False
        gx, gy = self.local_to_global(lx, ly)
        return self.base.in_bounds(gx, gy)

    def is_occupied(self, lx: int, ly: int) -> bool:
        gx, gy = self.local_to_global(lx, ly)
        return self.base.is_occupied(gx, gy)

    def is_unknown(self, lx: int, ly: int) -> bool:
        gx, gy = self.local_to_global(lx, ly)
        return self.base.is_unknown(gx, gy)


@dataclass(frozen=True)
class WindowVoxelView3D:
    """
    A bounded view into a VoxelMap3D.

    Local indices are [0..W-1]x[0..H-1]x[0..D-1].
    They are mapped to global voxel indices by adding (i0, j0, k0).

    The global A* expects:
        - width/height/depth
        - is_free(i, j, k)
    """
    base: VoxelMap3D
    i0: int
    j0: int
    k0: int
    width: int
    height: int
    depth: int

    @property
    def resolution(self) -> float:
        return float(getattr(self.base, "resolution", 1.0))

    @property
    def origin_x(self) -> float:
        return float(getattr(self.base, "origin_x", 0.0))

    @property
    def origin_y(self) -> float:
        return float(getattr(self.base, "origin_y", 0.0))

    @property
    def origin_z(self) -> float:
        return float(getattr(self.base, "origin_z", 0.0))

    @property
    def frame_id(self) -> str:
        return getattr(self.base, "frame_id", "map")

    def local_to_global(self, li: int, lj: int, lk: int) -> Index3D:
        return (li + self.i0, lj + self.j0, lk + self.k0)

    def global_to_local(self, gi: int, gj: int, gk: int) -> Index3D:
        return (gi - self.i0, gj - self.j0, gk - self.k0)

    def is_free(self, li: int, lj: int, lk: int) -> bool:
        gi, gj, gk = self.local_to_global(li, lj, lk)
        return bool(self.base.is_free(gi, gj, gk))
