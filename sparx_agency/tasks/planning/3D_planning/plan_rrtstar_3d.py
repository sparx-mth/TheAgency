#!/usr/bin/env python3
"""
3D RRT* planning inside a Gibson-tiny scene with Open3D visualization.

Flow:
  1) Pick START point (Shift+LeftClick) -> Q / close window
  2) Adjust START with keyboard -> Enter
  3) Pick GOAL point (Shift+LeftClick) -> Q / close window
  4) Adjust GOAL with keyboard -> Enter
  5) Final window: press P to plan (RRT*), display path as a THICK BLACK TUBE
     Optional: N/B/R to move a "HERE" marker along the path

Adjust window keys (reliable everywhere):
  Move:
    W/S : +Y / -Y
    A/D : -X / +X
    E/C : +Z / -Z
    Arrows: XY fallback
    PageUp/PageDown: Z fallback
  Step:
    + : increase step
    - : decrease step
  Confirm/Cancel:
    Enter : confirm
    Esc   : cancel

Important collision guarantees:
- We use BOTH:
  (A) Inflated voxel occupancy from point cloud (makes thin walls thicker).
  (B) Mesh raycasting checks (distance to mesh + occupancy-in-mesh) to prevent
      passing through walls/ceiling or leaving the interior (when mesh is watertight enough).

If the mesh is not watertight, occupancy may be imperfect; distance-to-mesh + inflated voxels
still strongly prevents "through-wall" paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import open3d as o3d

from sparx_agency.core.common.types import Pose3D, PlanStatus
from sparx_agency.core.planning.planners.rrtstar.algorithm import plan_rrtstar_3d
from sparx_agency.core.planning.planners.rrtstar.params import RRTStarOmpl3DParams


# =============================================================================
# Logging helpers
# =============================================================================

def pinfo(msg: str) -> None:
    print(f"[INFO] {msg}")


def pok(msg: str) -> None:
    print(f"[OK]   {msg}")


def pwarn(msg: str) -> None:
    print(f"[WARN] {msg}")


def perr(msg: str) -> None:
    print(f"[ERR]  {msg}")


# =============================================================================
# Thick path (tube) helper
# =============================================================================

def make_tube_from_polyline(
    pts: np.ndarray,
    radius: float,
    rgb: Tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    """
    Create a thick "tube" mesh along a polyline by stitching cylinders segment-by-segment.
    Works reliably across platforms (unlike LineSet line_width).
    """
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points for a tube")

    tube = o3d.geometry.TriangleMesh()
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for i in range(len(pts) - 1):
        p0 = pts[i].astype(np.float64)
        p1 = pts[i + 1].astype(np.float64)
        v = p1 - p0
        L = float(np.linalg.norm(v))
        if L < 1e-9:
            continue

        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=float(radius),
            height=L,
            resolution=24,
            split=4
        )
        cyl.compute_vertex_normals()
        cyl.paint_uniform_color(list(rgb))

        dir_vec = v / L
        axis = np.cross(z_axis, dir_vec)
        axis_norm = float(np.linalg.norm(axis))

        if axis_norm < 1e-9:
            # parallel to Z
            if dir_vec[2] < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
                cyl.rotate(R, center=np.array([0.0, 0.0, 0.0]))
        else:
            axis = axis / axis_norm
            angle = float(np.arccos(np.clip(np.dot(z_axis, dir_vec), -1.0, 1.0)))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            cyl.rotate(R, center=np.array([0.0, 0.0, 0.0]))

        mid = (p0 + p1) * 0.5
        cyl.translate(mid)

        tube += cyl

    tube.merge_close_vertices(1e-6)
    tube.compute_vertex_normals()
    return tube


# =============================================================================
# Voxel map from point cloud + inflation + mesh raycasting checks
# =============================================================================

@dataclass
class VoxelMapFromPointCloud:
    """
    Voxel occupancy map derived from point cloud surface voxels, with obstacle inflation
    and optional mesh raycasting checks to prevent "through wall" / "through ceiling" paths.

    occupancy[k, j, i] == True => occupied
    out-of-bounds => occupied ("hard walls")

    Extra safety:
      - inflated occupancy (thickens walls)
      - mesh distance / occupancy queries:
            distance_to_mesh(point) > robot_radius  AND  occupancy(point) == inside (if reliable)
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

    # Mesh raycasting (optional but strongly recommended)
    _ray_scene: Optional[o3d.t.geometry.RaycastingScene] = None
    _ray_mesh_id: Optional[int] = None

    # Safety settings for mesh checks
    robot_radius: float = 0.20
    enforce_inside_mesh: bool = True  # if mesh is watertight enough, keeps path inside building

    def world_to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        i = int(np.floor((x - self.origin_x) / self.resolution))
        j = int(np.floor((y - self.origin_y) / self.resolution))
        k = int(np.floor((z - self.origin_z) / self.resolution))
        return i, j, k

    def _in_bounds(self, i: int, j: int, k: int) -> bool:
        return (0 <= i < self.width) and (0 <= j < self.height) and (0 <= k < self.depth)

    def _mesh_distance(self, xyz: np.ndarray) -> float:
        """
        Unsigned distance to mesh surface (meters).
        Returns +inf if no raycasting scene is configured.
        """
        if self._ray_scene is None or self._ray_mesh_id is None:
            return float("inf")
        q = o3d.core.Tensor(xyz.reshape(1, 3), dtype=o3d.core.Dtype.Float32)
        d = self._ray_scene.compute_distance(q)  # [1]
        return float(d.numpy().reshape(-1)[0])

    def _mesh_inside(self, xyz: np.ndarray) -> bool:
        """
        Returns True if point is inside mesh volume (requires mesh to be reasonably watertight).
        If enforce_inside_mesh is False or no ray scene => returns True (no restriction).
        """
        if not self.enforce_inside_mesh:
            return True
        if self._ray_scene is None or self._ray_mesh_id is None:
            return True
        q = o3d.core.Tensor(xyz.reshape(1, 3), dtype=o3d.core.Dtype.Float32)
        occ = self._ray_scene.compute_occupancy(q)  # [1] values in {0,1}
        return bool(occ.numpy().reshape(-1)[0] > 0.5)

    def is_free(self, i: int, j: int, k: int) -> bool:
        """
        Grid-based query used by planner.

        We combine:
          (1) bounds + inflated voxel occupancy
          (2) mesh distance check (prevents "through wall/ceiling")
          (3) mesh inside check (prevents escaping the structure), when reliable
        """
        if not self._in_bounds(i, j, k):
            return False
        if bool(self.occupancy[k, j, i]):
            return False

        # Convert this voxel-center to world, then apply mesh safety checks.
        # Important: planner calls is_free on many points; this must remain reasonably fast.
        x = self.origin_x + (i + 0.5) * self.resolution
        y = self.origin_y + (j + 0.5) * self.resolution
        z = self.origin_z + (k + 0.5) * self.resolution
        xyz = np.array([x, y, z], dtype=np.float32)

        # must be inside the mesh (when enforced)
        if not self._mesh_inside(xyz):
            return False

        # must not be too close to any surface
        d = self._mesh_distance(xyz)
        if d < self.robot_radius:
            return False

        return True

    def world_clearance(self, x: float, y: float, z: float) -> float:
        """
        Clearance objective hook: we return distance-to-mesh if available, else a large number.
        """
        d = self._mesh_distance(np.array([x, y, z], dtype=np.float32))
        if np.isfinite(d):
            return float(d)
        return 1e9

    @staticmethod
    def _inflate_occupancy(occ: np.ndarray, r_cells: int) -> np.ndarray:
        """
        Inflate occupancy by r_cells in 3D using a ball-like neighborhood.

        This thickens walls (crucial for thin geometry / pointcloud-only voxelization).
        """
        if r_cells <= 0:
            return occ

        depth, height, width = occ.shape
        inflated = occ.copy()

        # Precompute offsets in a 3D ball
        offsets = []
        r2 = r_cells * r_cells
        for dz in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                for dx in range(-r_cells, r_cells + 1):
                    if dx * dx + dy * dy + dz * dz <= r2:
                        offsets.append((dz, dy, dx))

        occ_idx = np.argwhere(occ)  # (N,3) with [k,j,i]
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
    ) -> "VoxelMapFromPointCloud":
        """
        Build voxel occupancy from point cloud (surface voxels) + inflate,
        and attach a RaycastingScene built from the mesh for robust collision checks.
        """
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

        # Inflate obstacles by (robot_radius + margin)
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
        )

        # Raycasting scene from mesh
        try:
            tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            mesh_id = scene.add_triangles(tmesh)
            vm._ray_scene = scene
            vm._ray_mesh_id = mesh_id
            pok("RaycastingScene ready (mesh distance + occupancy checks enabled).")
        except Exception as e:
            pwarn(f"Failed to init RaycastingScene. Falling back to voxel-only collision. Error: {e}")
            vm._ray_scene = None
            vm._ray_mesh_id = None
            vm.enforce_inside_mesh = False

        return vm


# =============================================================================
# Gibson helpers
# =============================================================================

def load_gibson_mesh(root: Path, scene: str) -> o3d.geometry.TriangleMesh:
    scene_dir = root / scene
    mesh_path = scene_dir / "mesh_z_up.obj"
    if not mesh_path.exists():
        mesh_path = scene_dir / "mesh.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"No mesh.obj or mesh_z_up.obj in {scene_dir}")

    pinfo(f"Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"Failed loading mesh: {mesh_path}")
    mesh.compute_vertex_normals()
    pok(f"Mesh loaded: vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
    return mesh


def sample_point_cloud(mesh: o3d.geometry.TriangleMesh, n: int) -> o3d.geometry.PointCloud:
    pinfo(f"Sampling point cloud: n={n}")
    pcd = mesh.sample_points_uniformly(number_of_points=int(n))
    if len(pcd.points) == 0:
        raise ValueError("Sampled point cloud is empty")

    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    t = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    colors = np.stack([t, 0.3 * np.ones_like(t), 1.0 - t], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pok(f"Point cloud ready: points={len(pcd.points)} z_range=[{z.min():.2f}, {z.max():.2f}]")
    return pcd


# =============================================================================
# Picking + adjustment (keyboard)
# =============================================================================

def pick_single_point(pcd: o3d.geometry.PointCloud, title: str) -> np.ndarray:
    """
    Pick exactly one point using VisualizerWithEditing:
      Shift + LeftClick to pick
      Q or close window to finish
    """
    pinfo(title)
    pinfo("Use Shift+LeftClick to pick ONE point, then press Q or close the window.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1200, height=800)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    idx = vis.get_picked_points()
    if len(idx) < 1:
        raise ValueError("No point picked.")
    if len(idx) > 1:
        pwarn(f"Picked {len(idx)} points; using the first one.")

    pts = np.asarray(pcd.points)
    raw = pts[idx[0]].copy()
    pok(f"Picked idx={idx[0]} xyz=({raw[0]:.3f}, {raw[1]:.3f}, {raw[2]:.3f})")
    return raw


def adjust_point_with_keyboard(
    pcd: o3d.geometry.PointCloud,
    initial_point: np.ndarray,
    title: str,
    voxelmap: Optional[VoxelMapFromPointCloud] = None,
    step: float = 0.05,
    downsample_voxel: float = 0.05,
) -> np.ndarray:
    """
    Adjust a 3D point with keyboard in a lightweight viewer window.
    """
    pcd_view = pcd.voxel_down_sample(voxel_size=float(downsample_voxel)) if downsample_voxel and downsample_voxel > 0 else pcd

    point = initial_point.astype(np.float64).copy()
    step_size = float(step)
    confirmed = {"ok": False}

    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    marker.compute_vertex_normals()
    marker.paint_uniform_color([1.0, 0.1, 0.1])
    marker.translate(point)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    KEY = {
        "W": ord("W"),
        "S": ord("S"),
        "A": ord("A"),
        "D": ord("D"),
        "E": ord("E"),
        "C": ord("C"),
        "+": ord("+"),
        "-": ord("-"),
        "=": ord("="),
        "LEFT": 263,
        "RIGHT": 262,
        "UP": 265,
        "DOWN": 264,
        "PGUP": 266,
        "PGDN": 267,
        "ENTER": 257,
        "ESC": 256,
    }

    def print_state(prefix: str) -> None:
        if voxelmap is None:
            print(f"[{prefix}] xyz=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) step={step_size:.3f}")
        else:
            i, j, k = voxelmap.world_to_grid(float(point[0]), float(point[1]), float(point[2]))
            free = voxelmap.is_free(i, j, k)
            clr = voxelmap.world_clearance(float(point[0]), float(point[1]), float(point[2]))
            print(
                f"[{prefix}] xyz=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) "
                f"step={step_size:.3f} grid=({i},{j},{k}) free={free} clearance={clr:.3f}m"
            )

    def move(dx: float, dy: float, dz: float, vis: o3d.visualization.Visualizer):
        nonlocal point
        delta = np.array([dx, dy, dz], dtype=np.float64) * step_size
        point += delta
        marker.translate(delta, relative=True)

        vis.update_geometry(marker)
        vis.poll_events()
        vis.update_renderer()

        print_state("MOVE")
        return False

    def inc_step(vis):
        nonlocal step_size
        step_size *= 1.5
        print_state("STEP+")
        return False

    def dec_step(vis):
        nonlocal step_size
        step_size /= 1.5
        print_state("STEP-")
        return False

    def confirm(vis):
        confirmed["ok"] = True
        print_state("CONFIRMED")
        vis.close()
        return False

    def cancel(vis):
        print("[CANCELLED]")
        vis.close()
        return False

    print("\n--- Adjust point ---")
    print("Click inside the window once to focus it.")
    print("Move: W/S (+Y/-Y), A/D (-X/+X), E/C (+Z/-Z)")
    print("Fallback: Arrows (XY), PageUp/PageDown (Z)")
    print("Step: + / -")
    print("Enter=confirm, Esc=cancel\n")
    print_state("START")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1200, height=800)
    vis.add_geometry(pcd_view)
    vis.add_geometry(marker)
    vis.add_geometry(frame)

    # letters
    vis.register_key_callback(KEY["W"], lambda v: move(0, +1, 0, v))
    vis.register_key_callback(KEY["S"], lambda v: move(0, -1, 0, v))
    vis.register_key_callback(KEY["A"], lambda v: move(-1, 0, 0, v))
    vis.register_key_callback(KEY["D"], lambda v: move(+1, 0, 0, v))
    vis.register_key_callback(KEY["E"], lambda v: move(0, 0, +1, v))
    vis.register_key_callback(KEY["C"], lambda v: move(0, 0, -1, v))

    # arrows + page up/down
    vis.register_key_callback(KEY["UP"], lambda v: move(0, +1, 0, v))
    vis.register_key_callback(KEY["DOWN"], lambda v: move(0, -1, 0, v))
    vis.register_key_callback(KEY["LEFT"], lambda v: move(-1, 0, 0, v))
    vis.register_key_callback(KEY["RIGHT"], lambda v: move(+1, 0, 0, v))
    vis.register_key_callback(KEY["PGUP"], lambda v: move(0, 0, +1, v))
    vis.register_key_callback(KEY["PGDN"], lambda v: move(0, 0, -1, v))

    # step
    vis.register_key_callback(KEY["+"], inc_step)
    vis.register_key_callback(KEY["="], inc_step)
    vis.register_key_callback(KEY["-"], dec_step)

    # confirm/cancel
    vis.register_key_callback(KEY["ENTER"], confirm)
    vis.register_key_callback(KEY["ESC"], cancel)

    vis.run()
    vis.destroy_window()

    if not confirmed["ok"]:
        raise RuntimeError("Point adjustment cancelled")

    return point


def pick_and_adjust_point(
    pcd: o3d.geometry.PointCloud,
    voxelmap: Optional[VoxelMapFromPointCloud],
    which: str,
    step: float,
) -> np.ndarray:
    raw = pick_single_point(pcd, title=f"Pick {which} (Shift+Click) then Q/close")
    adj = adjust_point_with_keyboard(
        pcd,
        raw,
        title=f"Adjust {which} (WASD / arrows / Enter)",
        voxelmap=voxelmap,
        step=step,
        downsample_voxel=0.05,
    )
    pok(f"Final {which} xyz=({adj[0]:.3f}, {adj[1]:.3f}, {adj[2]:.3f})")
    return adj


# =============================================================================
# Final window: press P to plan, show thick black tube path
# =============================================================================

def run_final_window_plan_and_show(
    pcd: o3d.geometry.PointCloud,
    voxelmap: VoxelMapFromPointCloud,
    start: Pose3D,
    goal: Pose3D,
    params: RRTStarOmpl3DParams,
    path_radius: float,
) -> None:
    start_xyz = np.array([start.x, start.y, start.z], dtype=np.float64)
    goal_xyz = np.array([goal.x, goal.y, goal.z], dtype=np.float64)

    start_s = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
    start_s.compute_vertex_normals()
    start_s.paint_uniform_color([0.1, 0.9, 0.1])
    start_s.translate(start_xyz)

    goal_s = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
    goal_s.compute_vertex_normals()
    goal_s.paint_uniform_color([0.9, 0.1, 0.1])
    goal_s.translate(goal_xyz)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    state: Dict[str, Any] = {
        "planned": False,
        "path_pts": None,
        "path_tube": None,
        "marker": None,
        "idx": 0,
    }

    def print_grid_debug() -> None:
        si, sj, sk = voxelmap.world_to_grid(start.x, start.y, start.z)
        gi, gj, gk = voxelmap.world_to_grid(goal.x, goal.y, goal.z)
        pinfo(f"START grid=({si},{sj},{sk}) free={voxelmap.is_free(si,sj,sk)} clr={voxelmap.world_clearance(start.x,start.y,start.z):.3f}m")
        pinfo(f"GOAL  grid=({gi},{gj},{gk}) free={voxelmap.is_free(gi,gj,gk)} clr={voxelmap.world_clearance(goal.x,goal.y,goal.z):.3f}m")

    def add_path_to_vis(vis: o3d.visualization.Visualizer, path_pts: np.ndarray) -> None:
        # Thick black tube path
        tube = make_tube_from_polyline(path_pts, radius=float(path_radius), rgb=(0.0, 0.0, 0.0))
        vis.add_geometry(tube)

        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
        marker.compute_vertex_normals()
        marker.paint_uniform_color([0.2, 0.2, 1.0])
        marker.translate(path_pts[0])
        vis.add_geometry(marker)

        state["path_pts"] = path_pts
        state["path_tube"] = tube
        state["marker"] = marker
        state["idx"] = 0

        vis.update_renderer()
        pok(f"Path drawn as THICK black tube. radius={path_radius:.3f}m. Use N/B/R to move marker.")

    def on_plan(vis: o3d.visualization.Visualizer):
        if state["planned"]:
            pwarn("Already planned. Restart script to plan again.")
            return False

        print_grid_debug()
        pinfo("Planning with OMPL RRT* 3D ...")
        res = plan_rrtstar_3d(start=start, goal=goal, voxelmap=voxelmap, params=params)
        pinfo(f"Planner returned: status={res.status} message='{res.message}'")

        if res.status != PlanStatus.SUCCESS:
            perr("Planning failed. (Inspect start/goal in this window.)")
            state["planned"] = True
            return False

        path3d = res.artifacts["path3d"]
        path_pts = np.array([[p.x, p.y, p.z] for p in path3d.points], dtype=np.float64)
        if path_pts.shape[0] < 2:
            perr("Planner returned a too-short path.")
            state["planned"] = True
            return False

        # Extra post-check: ensure no waypoint is too close / outside (debug aid)
        # This does not change planning, but helps you catch if constraints are too weak.
        bad = 0
        for q in path_pts:
            i, j, k = voxelmap.world_to_grid(float(q[0]), float(q[1]), float(q[2]))
            if not voxelmap.is_free(i, j, k):
                bad += 1
        if bad > 0:
            pwarn(f"Path has {bad} waypoints that violate is_free(). Consider increasing robot_radius/inflation or tightening OMPL params.")

        add_path_to_vis(vis, path_pts)
        state["planned"] = True
        return False

    def print_marker_state():
        pts = state["path_pts"]
        i = state["idx"]
        x, y, z = pts[i]
        print(f"[HERE] waypoint {i+1}/{len(pts)} xyz=({x:.3f}, {y:.3f}, {z:.3f})")

    def move_marker_to(vis: o3d.visualization.Visualizer, i_new: int):
        if state["marker"] is None or state["path_pts"] is None:
            return False
        pts = state["path_pts"]
        i_new = int(np.clip(i_new, 0, len(pts) - 1))
        if i_new == state["idx"]:
            return False

        cur = pts[state["idx"]]
        nxt = pts[i_new]
        state["marker"].translate((nxt - cur), relative=True)
        state["idx"] = i_new

        vis.update_geometry(state["marker"])
        vis.update_renderer()
        print_marker_state()
        return False

    def on_next(vis):
        return move_marker_to(vis, state["idx"] + 1)

    def on_prev(vis):
        return move_marker_to(vis, state["idx"] - 1)

    def on_reset(vis):
        return move_marker_to(vis, 0)

    pinfo("Final window:")
    pinfo("  P = plan (RRT*) and draw thick black tube path")
    pinfo("  N/B/R = move HERE marker (after planning)")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="RRT* 3D Planner (press P to plan)", width=1280, height=860)

    vis.add_geometry(pcd)
    vis.add_geometry(start_s)
    vis.add_geometry(goal_s)
    vis.add_geometry(frame)

    vis.register_key_callback(ord("P"), on_plan)
    vis.register_key_callback(ord("N"), on_next)
    vis.register_key_callback(ord("B"), on_prev)
    vis.register_key_callback(ord("R"), on_reset)

    vis.run()
    vis.destroy_window()


# =============================================================================
# Main
# =============================================================================

def main():
    # ---- CONFIG ----
    ROOT = Path("gibson/extracted/gibson_tiny")
    SCENE = "Benevolence"

    POINTS = 1_500_000
    VOXEL = 0.12
    PADDING = 0.5

    # Selection/adjustment
    ADJUST_STEP = 0.05
    Z_LIFT = 0.30  # lift after confirmation to avoid being exactly on surfaces

    # Critical collision safety:
    ROBOT_RADIUS = 0.25          # <-- תגדיל אם עדיין "חותך" קירות/תקרה
    INFLATION_MARGIN = 0.08      # <-- עוד עובי מעבר לרדיוס, עוזר במיוחד לקירות דקים
    ENFORCE_INSIDE_MESH = True   # <-- שומר מסלול בתוך המבנה (אם המודל מספיק watertight)

    # Thick path visualization
    PATH_RADIUS = 0.04  # <-- עובי המסלול (tube radius)

    # Planner params: tighter collision sampling helps prevent "tunneling"
    params = RRTStarOmpl3DParams(
        timeout=3.0,
        use_clearance_objective=True,
        clearance_weight=12.0,
        min_clearance_for_keep=ROBOT_RADIUS,  # keep solutions that respect clearance
        interpolation_spacing=0.10,
        collision_check_resolution=0.01,
        longest_valid_segment_m=0.10,         # smaller segments => less chance to skip through thin geometry
    )

    pinfo(f"Scene={SCENE} root={ROOT.resolve()}")
    pinfo(f"Sampling points={POINTS} voxel={VOXEL:.3f} padding={PADDING:.2f} z_lift={Z_LIFT:.2f}")
    pinfo(f"Safety: ROBOT_RADIUS={ROBOT_RADIUS:.2f} INFLATION_MARGIN={INFLATION_MARGIN:.2f} enforce_inside={ENFORCE_INSIDE_MESH}")
    pinfo(f"OMPL: timeout={params.timeout}s interp={params.interpolation_spacing} longest_seg={params.longest_valid_segment_m}")

    mesh = load_gibson_mesh(ROOT, SCENE)
    pcd = sample_point_cloud(mesh, POINTS)

    pinfo("Building voxel occupancy + inflation + mesh raycasting checks...")
    voxelmap = VoxelMapFromPointCloud.from_point_cloud_and_mesh(
        pcd=pcd,
        mesh=mesh,
        voxel_size=VOXEL,
        padding_m=PADDING,
        frame_id="map",
        robot_radius=ROBOT_RADIUS,
        inflation_margin_m=INFLATION_MARGIN,
        enforce_inside_mesh=ENFORCE_INSIDE_MESH,
    )
    pok(f"VoxelMap built: size=({voxelmap.width},{voxelmap.height},{voxelmap.depth}) res={voxelmap.resolution}")
    pinfo(f"VoxelMap origin=({voxelmap.origin_x:.2f},{voxelmap.origin_y:.2f},{voxelmap.origin_z:.2f})")

    # Pick & adjust START
    p0 = pick_and_adjust_point(pcd, voxelmap=voxelmap, which="START", step=ADJUST_STEP)
    start = Pose3D(float(p0[0]), float(p0[1]), float(p0[2] + Z_LIFT))
    pok(f"Using START (lifted) xyz=({start.x:.3f},{start.y:.3f},{start.z:.3f})")

    # Pick & adjust GOAL
    p1 = pick_and_adjust_point(pcd, voxelmap=voxelmap, which="GOAL", step=ADJUST_STEP)
    goal = Pose3D(float(p1[0]), float(p1[1]), float(p1[2] + Z_LIFT))
    pok(f"Using GOAL  (lifted) xyz=({goal.x:.3f},{goal.y:.3f},{goal.z:.3f})")

    # Final window
    run_final_window_plan_and_show(
        pcd=pcd,
        voxelmap=voxelmap,
        start=start,
        goal=goal,
        params=params,
        path_radius=PATH_RADIUS,
    )


if __name__ == "__main__":
    main()
