# =========================
# File: interactive_rrtstar/voxelmap.py
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import open3d as o3d

from logging_utils import pinfo, pok, pwarn


@dataclass
class VoxelMapFromPointCloud:
    """
    3D voxel occupancy map derived from point cloud surface voxels, with obstacle inflation
    and optional mesh raycasting checks.

    IMPORTANT FIX:
      - Adds is_free_world(x,y,z): mesh checks are evaluated at the REAL query point,
        not at voxel-center. This is what OMPL should use.
    """
    origin_x: float
    origin_y: float
    origin_z: float
    width: int
    height: int
    depth: int
    resolution: float
    frame_id: str = "map"

    occupancy: np.ndarray = None  # shape (depth, height, width), bool

    _ray_scene: Optional[o3d.t.geometry.RaycastingScene] = None
    _ray_mesh_id: Optional[int] = None

    robot_radius: float = 0.20
    enforce_inside_mesh: bool = True

    # Debug controls (high-signal throttled prints)
    debug_enabled: bool = True
    debug_max_print: int = 250
    debug_every_n: int = 2000
    debug_print_invalid_always: bool = False

    _debug_calls: int = 0
    _debug_printed: int = 0

    def world_to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        i = int(np.floor((x - self.origin_x) / self.resolution))
        j = int(np.floor((y - self.origin_y) / self.resolution))
        k = int(np.floor((z - self.origin_z) / self.resolution))
        return i, j, k

    def _in_bounds(self, i: int, j: int, k: int) -> bool:
        return (0 <= i < self.width) and (0 <= j < self.height) and (0 <= k < self.depth)

    # ---------- Mesh helpers (WORLD point) ----------
    def _mesh_distance_world(self, x: float, y: float, z: float) -> float:
        if self._ray_scene is None or self._ray_mesh_id is None:
            return float("inf")
        q = o3d.core.Tensor(np.array([[x, y, z]], dtype=np.float32), dtype=o3d.core.Dtype.Float32)
        d = self._ray_scene.compute_distance(q)
        return float(d.numpy().reshape(-1)[0])

    def _mesh_inside_world(self, x: float, y: float, z: float) -> bool:
        if not self.enforce_inside_mesh:
            return True
        if self._ray_scene is None or self._ray_mesh_id is None:
            return True
        q = o3d.core.Tensor(np.array([[x, y, z]], dtype=np.float32), dtype=o3d.core.Dtype.Float32)
        occ = self._ray_scene.compute_occupancy(q)
        return bool(occ.numpy().reshape(-1)[0] > 0.5)

    # ---------- Original grid check (kept for UI/debug) ----------
    def is_free(self, i: int, j: int, k: int) -> bool:
        """
        Grid-based check (voxel-center mesh checks). Kept for UI + quick debug prints.
        OMPL should use is_free_world().
        """
        if not self._in_bounds(i, j, k):
            return False
        if bool(self.occupancy[k, j, i]):
            return False

        cx = self.origin_x + (i + 0.5) * self.resolution
        cy = self.origin_y + (j + 0.5) * self.resolution
        cz = self.origin_z + (k + 0.5) * self.resolution

        inside = self._mesh_inside_world(cx, cy, cz)
        d = self._mesh_distance_world(cx, cy, cz)

        if not inside:
            return False
        if d < self.robot_radius:
            return False
        return True

    def is_free_world(self, x: float, y: float, z: float) -> bool:
        """
        WORLD validity check used by OMPL:
          1) Out-of-bounds => blocked
          2) Voxel occupancy at (i,j,k) => blocked
          3) Mesh inside check at (x,y,z) (if enforced)
          4) Mesh distance check at (x,y,z): must be >= robot_radius
        """
        self._debug_calls += 1
        do_throttled_print = (
            self.debug_enabled and
            self._debug_printed < self.debug_max_print and
            (self._debug_calls % max(1, self.debug_every_n) == 0)
        )

        i, j, k = self.world_to_grid(x, y, z)

        if not self._in_bounds(i, j, k):
            if self.debug_enabled and self.debug_print_invalid_always and self._debug_printed < self.debug_max_print:
                self._debug_printed += 1
                print(f"[VALIDITY world] OOB world=({x:.3f},{y:.3f},{z:.3f}) grid=({i},{j},{k}) -> BLOCKED")
            return False

        if bool(self.occupancy[k, j, i]):
            if self.debug_enabled and self.debug_print_invalid_always and self._debug_printed < self.debug_max_print:
                self._debug_printed += 1
                print(f"[VALIDITY world] OCC world=({x:.3f},{y:.3f},{z:.3f}) grid=({i},{j},{k}) -> BLOCKED")
            return False

        inside = self._mesh_inside_world(x, y, z)
        d = self._mesh_distance_world(x, y, z)

        ok = inside and (d >= self.robot_radius)

        if do_throttled_print:
            self._debug_printed += 1
            ray = "ON" if self._ray_scene is not None else "OFF"
            print(
                f"[VALIDITY world] world=({x:.3f},{y:.3f},{z:.3f}) grid=({i},{j},{k}) occ=False "
                f"ray={ray} enforce_inside={self.enforce_inside_mesh} inside={inside} "
                f"d_mesh={d:.3f} r={self.robot_radius:.3f} -> {'FREE' if ok else 'BLOCKED'}"
            )

        return ok

    def world_clearance(self, x: float, y: float, z: float) -> float:
        d = self._mesh_distance_world(x, y, z)
        if np.isfinite(d):
            return float(d)
        return 1e9

    @staticmethod
    def _inflate_occupancy(occ: np.ndarray, r_cells: int) -> np.ndarray:
        if r_cells <= 0:
            return occ

        depth, height, width = occ.shape
        inflated = occ.copy()

        offsets = []
        r2 = r_cells * r_cells
        for dz in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for dx in range(-r_cells, r_cells + 1):
                    if dx * dx + dy * dy + dz * dz <= r2:
                        offsets.append((dz, dy, dx))

        occ_idx = np.argwhere(occ)  # [k,j,i]
        for k, j, i in occ_idx:
            for dz, dy, dx in offsets:
                kk = k + dz
                jj = j + dy
                ii = i + dx
                if 0 <= kk < depth and 0 <= jj < height and 0 <= ii < width:
                    inflated[kk, jj, ii] = True

        return inflated

    @staticmethod
    def from_point_cloud_and_mesh(
        pcd: o3d.geometry.PointCloud,
        mesh: o3d.geometry.TriangleMesh,
        voxel_size: float,
        padding_m: float = 0.5,
        frame_id: str = "map",
        robot_radius: float = 0.20,
        inflation_margin_m: float = 0.05,
        enforce_inside_mesh: bool = True,
        debug_enabled: bool = True,
    ) -> "VoxelMapFromPointCloud":
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            raise ValueError("Empty point cloud")

        pmin = pts.min(axis=0) - padding_m
        pmax = pts.max(axis=0) + padding_m
        origin = pmin
        size = pmax - pmin

        width = int(np.ceil(size[0] / voxel_size))
        height = int(np.ceil(size[1] / voxel_size))
        depth = int(np.ceil(size[2] / voxel_size))

        occ = np.zeros((depth, height, width), dtype=bool)

        ijk = np.floor((pts - origin) / voxel_size).astype(np.int64)
        ijk[:, 0] = np.clip(ijk[:, 0], 0, width - 1)   # i
        ijk[:, 1] = np.clip(ijk[:, 1], 0, height - 1)  # j
        ijk[:, 2] = np.clip(ijk[:, 2], 0, depth - 1)   # k
        occ[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = True

        inflate_m = float(robot_radius + inflation_margin_m)
        r_cells = int(np.ceil(inflate_m / float(voxel_size)))
        pinfo(f"Inflating occupancy: inflate_m={inflate_m:.3f} -> r_cells={r_cells} (voxel={voxel_size:.3f})")
        occ_inflated = VoxelMapFromPointCloud._inflate_occupancy(occ, r_cells=r_cells)

        vm = VoxelMapFromPointCloud(
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            origin_z=float(origin[2]),
            width=width,
            height=height,
            depth=depth,
            resolution=float(voxel_size),
            frame_id=frame_id,
            occupancy=occ_inflated,
            robot_radius=float(robot_radius),
            enforce_inside_mesh=bool(enforce_inside_mesh),
            debug_enabled=bool(debug_enabled),
        )

        try:
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            mesh_id = scene.add_triangles(tmesh)
            vm._ray_scene = scene
            vm._ray_mesh_id = mesh_id
            pok("RaycastingScene ready (mesh distance + occupancy checks enabled).")
            pinfo(
                f"Raycast config: scene=ON enforce_inside_mesh={vm.enforce_inside_mesh} robot_radius={vm.robot_radius:.3f}"
            )
        except Exception as e:
            pwarn(f"Failed to init RaycastingScene. Falling back to voxel-only collision. Error: {e}")
            vm._ray_scene = None
            vm._ray_mesh_id = None
            vm.enforce_inside_mesh = False
            pwarn("Raycast OFF => voxel-only collision. Inside-mesh enforcement disabled.")
            pinfo(
                f"Raycast config: scene=OFF enforce_inside_mesh={vm.enforce_inside_mesh} robot_radius={vm.robot_radius:.3f}"
            )

        return vm
