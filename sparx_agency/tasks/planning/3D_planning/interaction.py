# =========================
# File: interactive_rrtstar/interaction.py
# =========================
from __future__ import annotations

from typing import Optional
import numpy as np
import open3d as o3d

from logging_utils import pinfo, pok


def pick_single_point(pcd: o3d.geometry.PointCloud, title: str) -> np.ndarray:
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
        pinfo(f"Picked {len(idx)} points; using the first one.")

    pts = np.asarray(pcd.points)
    raw = pts[idx[0]].copy()
    pok(f"Picked idx={idx[0]} xyz=({raw[0]:.3f}, {raw[1]:.3f}, {raw[2]:.3f})")
    return raw


def adjust_point_with_keyboard(
    pcd: o3d.geometry.PointCloud,
    initial_point: np.ndarray,
    title: str,
    voxelmap=None,
    step: float = 0.05,
    downsample_voxel: float = 0.05,
) -> np.ndarray:
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
            free_grid = voxelmap.is_free(i, j, k)
            free_world = voxelmap.is_free_world(float(point[0]), float(point[1]), float(point[2]))
            clr = voxelmap.world_clearance(float(point[0]), float(point[1]), float(point[2]))
            print(
                f"[{prefix}] xyz=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) "
                f"step={step_size:.3f} grid=({i},{j},{k}) free_grid={free_grid} free_world={free_world} clearance={clr:.3f}m"
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

    vis.register_key_callback(KEY["W"], lambda v: move(0, +1, 0, v))
    vis.register_key_callback(KEY["S"], lambda v: move(0, -1, 0, v))
    vis.register_key_callback(KEY["A"], lambda v: move(-1, 0, 0, v))
    vis.register_key_callback(KEY["D"], lambda v: move(+1, 0, 0, v))
    vis.register_key_callback(KEY["E"], lambda v: move(0, 0, +1, v))
    vis.register_key_callback(KEY["C"], lambda v: move(0, 0, -1, v))

    vis.register_key_callback(KEY["UP"], lambda v: move(0, +1, 0, v))
    vis.register_key_callback(KEY["DOWN"], lambda v: move(0, -1, 0, v))
    vis.register_key_callback(KEY["LEFT"], lambda v: move(-1, 0, 0, v))
    vis.register_key_callback(KEY["RIGHT"], lambda v: move(+1, 0, 0, v))
    vis.register_key_callback(KEY["PGUP"], lambda v: move(0, 0, +1, v))
    vis.register_key_callback(KEY["PGDN"], lambda v: move(0, 0, -1, v))

    vis.register_key_callback(KEY["+"], inc_step)
    vis.register_key_callback(KEY["="], inc_step)
    vis.register_key_callback(KEY["-"], dec_step)

    vis.register_key_callback(KEY["ENTER"], confirm)
    vis.register_key_callback(KEY["ESC"], cancel)

    vis.run()
    vis.destroy_window()

    if not confirmed["ok"]:
        raise RuntimeError("Point adjustment cancelled")

    return point


def pick_and_adjust_point(pcd, voxelmap, which: str, step: float) -> np.ndarray:
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
