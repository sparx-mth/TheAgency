from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import open3d as o3d

from logging_utils import pinfo, pok, pwarn, perr
from tube import make_tube_from_polyline
from sparx_agency.core.common.types import Pose3D, PlanStatus

# NOTE: update these imports/paths if your project stores these elsewhere.
from sparx_agency.core.planning.planners.bitstar.algorithm import (
    plan_bitstar_3d,
)
from sparx_agency.core.planning.planners.bitstar.params import (
    BITStarParams,
)
from sparx_agency.core.planning.planners.informed_rrtstar.algorithm import (
    plan_informed_rrtstar_3d,
)
from sparx_agency.core.planning.planners.informed_rrtstar.params import (
    InformedRRTStarParams,
)


def debug_validate_path_segments(voxelmap, path_pts: np.ndarray, step_m: float = 0.02, max_reports: int = 40) -> None:
    print(f"[POSTCHECK] Validating path with step={step_m:.3f}m over {len(path_pts)-1} segments ...")
    reports = 0

    for s in range(len(path_pts) - 1):
        a = path_pts[s].astype(np.float64)
        b = path_pts[s + 1].astype(np.float64)
        v = b - a
        L = float(np.linalg.norm(v))
        if L < 1e-9:
            continue

        n = max(1, int(np.ceil(L / step_m)))
        for t in range(n + 1):
            alpha = t / n
            p = a + alpha * v
            ok = voxelmap.is_free_world(float(p[0]), float(p[1]), float(p[2]))
            if not ok:
                i, j, k = voxelmap.world_to_grid(float(p[0]), float(p[1]), float(p[2]))
                reports += 1
                print(
                    f"[POSTCHECK] COLLISION seg={s} t={alpha:.3f} "
                    f"world=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) grid=({i},{j},{k})"
                )
                if reports >= max_reports:
                    print("[POSTCHECK] Reached max_reports; stopping.")
                    return

    if reports == 0:
        print("[POSTCHECK] OK: No collisions found in dense segment sampling.")


def run_final_window_plan_and_show(
    pcd: o3d.geometry.PointCloud,
    voxelmap,
    start: Pose3D,
    goal: Pose3D,
    planner_name: str,
    bitstar_params: BITStarParams,
    informed_params: InformedRRTStarParams,
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
        "planner": planner_name,  # "bitstar" | "informed_rrtstar"
    }

    def _planner_label() -> str:
        return "BIT*" if state["planner"] == "bitstar" else "Informed RRT*"

    def print_grid_debug() -> None:
        si, sj, sk = voxelmap.world_to_grid(start.x, start.y, start.z)
        gi, gj, gk = voxelmap.world_to_grid(goal.x, goal.y, goal.z)
        pinfo(f"START grid=({si},{sj},{sk}) free_grid={voxelmap.is_free(si,sj,sk)} free_world={voxelmap.is_free_world(start.x,start.y,start.z)} clr={voxelmap.world_clearance(start.x,start.y,start.z):.3f}m")
        pinfo(f"GOAL  grid=({gi},{gj},{gk}) free_grid={voxelmap.is_free(gi,gj,gk)} free_world={voxelmap.is_free_world(goal.x,goal.y,goal.z)} clr={voxelmap.world_clearance(goal.x,goal.y,goal.z):.3f}m")

    def add_path_to_vis(vis: o3d.visualization.Visualizer, path_pts: np.ndarray) -> None:
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

    def _plan() -> Optional[Dict[str, Any]]:
        """
        Returns PlanResult-like dict:
            {"status": ..., "message": ..., "artifacts": {"path3d": ...}}
        """
        if state["planner"] == "bitstar":
            return plan_bitstar_3d(start=start, goal=goal, voxelmap=voxelmap, params=bitstar_params)
        if state["planner"] == "informed_rrtstar":
            return plan_informed_rrtstar_3d(start=start, goal=goal, voxelmap=voxelmap, params=informed_params)
        raise ValueError(f"Unknown planner: {state['planner']}")

    def on_toggle_planner(vis: o3d.visualization.Visualizer):
        if state["planned"]:
            pwarn("Already planned. Restart script to change planner.")
            return False

        state["planner"] = "informed_rrtstar" if state["planner"] == "bitstar" else "bitstar"
        pok(f"Planner switched to: {_planner_label()}")
        return False

    def on_plan(vis: o3d.visualization.Visualizer):
        if state["planned"]:
            pwarn("Already planned. Restart script to plan again.")
            return False

        print_grid_debug()
        pinfo(f"Planning with OMPL {_planner_label()} 3D ...")

        try:
            res = _plan()
        except Exception as e:
            perr(f"Planner crashed with exception: {type(e).__name__}: {e}")
            state["planned"] = True
            return False

        pinfo(f"Planner returned: status={res.status} message='{res.message}'")

        if res.status != PlanStatus.SUCCESS:
            perr("Planning failed or rejected (approx solution).")
            state["planned"] = True
            return False

        path3d = res.artifacts["path3d"]
        path_pts = np.array([[p.x, p.y, p.z] for p in path3d.points], dtype=np.float64)
        if path_pts.shape[0] < 2:
            perr("Planner returned a too-short path.")
            state["planned"] = True
            return False

        # Strong postcheck (WORLD)
        bad = 0
        for q in path_pts:
            if not voxelmap.is_free_world(float(q[0]), float(q[1]), float(q[2])):
                bad += 1

        if bad > 0:
            pwarn(f"Path has {bad} waypoints that violate is_free_world(). Inspect logs/voxelmap.")
            debug_validate_path_segments(voxelmap, path_pts, step_m=0.02, max_reports=40)
        else:
            debug_validate_path_segments(voxelmap, path_pts, step_m=0.02, max_reports=20)

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
    pinfo("  P = plan and draw thick black tube path")
    pinfo("  T = toggle planner (before planning) [BIT* <-> Informed RRT*]")
    pinfo("  N/B/R = move HERE marker (after planning)")
    pinfo(f"  Current planner: {_planner_label()}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Planner (P=plan, T=toggle)", width=1280, height=860)

    vis.add_geometry(pcd)
    vis.add_geometry(start_s)
    vis.add_geometry(goal_s)
    vis.add_geometry(frame)

    vis.register_key_callback(ord("P"), on_plan)
    vis.register_key_callback(ord("T"), on_toggle_planner)
    vis.register_key_callback(ord("N"), on_next)
    vis.register_key_callback(ord("B"), on_prev)
    vis.register_key_callback(ord("R"), on_reset)

    vis.run()
    vis.destroy_window()
