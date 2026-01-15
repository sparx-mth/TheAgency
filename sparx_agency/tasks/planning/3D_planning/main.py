#!/usr/bin/env python3
"""
Interactive 3D planning inside a Gibson-tiny scene with Open3D visualization.

Now supports:
- BIT* (default)
- Informed RRT*

Usage:
    python3 -m interactive_rrtstar.main
    python3 -m interactive_rrtstar.main --planner bitstar
    python3 -m interactive_rrtstar.main --planner informed_rrtstar
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from sparx_agency.core.common.types import Pose3D

from logging_utils import pinfo, pok
from gibson_io import load_gibson_mesh, sample_point_cloud
from voxelmap import VoxelMapFromPointCloud
from interaction import pick_and_adjust_point
from final_window import run_final_window_plan_and_show


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--planner",
        choices=["bitstar", "informed_rrtstar"],
        default="bitstar",
        help="Planner algorithm (default: bitstar).",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    ROOT = Path("gibson/extracted/gibson_tiny")
    SCENE = "Corozal"

    POINTS = 1_500_000
    VOXEL = 0.12
    PADDING = 0.5

    ADJUST_STEP = 0.05
    Z_LIFT = 0.0

    ROBOT_RADIUS = 0.1
    INFLATION_MARGIN = 0.02
    ENFORCE_INSIDE_MESH = True

    PATH_RADIUS = 0.04

    # -----------------------------
    # Planner params
    # -----------------------------
    # NOTE: update these imports/paths if your project stores these classes elsewhere.
    from sparx_agency.core.planning.planners.bitstar.params import (
        BITStarParams,
    )
    from sparx_agency.core.planning.planners.informed_rrtstar.params import (
        InformedRRTStarParams,
    )

    bitstar_params = BITStarParams(
        timeout=30.0,
        use_clearance_objective=True,
        clearance_weight=0.01,
        min_clearance_for_keep=ROBOT_RADIUS,
        interpolation_spacing=0.10,

        collision_check_resolution=0.02,
        longest_valid_segment_m=0.25,

        # BIT* specifics
        samples_per_batch=1000,
        use_k_nearest=True,
        rewire_factor=2.0,

        debug_enabled=True,
    )

    informed_params = InformedRRTStarParams(
        timeout=200.0,
        use_clearance_objective=True,
        clearance_weight=12.0,
        min_clearance_for_keep=ROBOT_RADIUS,
        interpolation_spacing=0.10,

        collision_check_resolution=0.001,
        longest_valid_segment_m=0.05,

        # optional: set range if you want (None = auto)
        range_m=None,

        debug_enabled=True,
    )

    pinfo(f"Scene={SCENE} root={ROOT.resolve()}")
    pinfo(f"Sampling points={POINTS} voxel={VOXEL:.3f} padding={PADDING:.2f} z_lift={Z_LIFT:.2f}")
    pinfo(f"Safety: ROBOT_RADIUS={ROBOT_RADIUS:.2f} INFLATION_MARGIN={INFLATION_MARGIN:.2f} enforce_inside={ENFORCE_INSIDE_MESH}")
    pinfo(f"Planner default (CLI) = {args.planner}")

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
        debug_enabled=True,
    )
    pok(f"VoxelMap built: size=({voxelmap.width},{voxelmap.height},{voxelmap.depth}) res={voxelmap.resolution}")
    pinfo(f"VoxelMap origin=({voxelmap.origin_x:.2f},{voxelmap.origin_y:.2f},{voxelmap.origin_z:.2f})")

    # START
    p0 = pick_and_adjust_point(pcd, voxelmap=voxelmap, which="START", step=ADJUST_STEP)
    start = Pose3D(float(p0[0]), float(p0[1]), float(p0[2] + Z_LIFT))
    pok(f"Using START (lifted) xyz=({start.x:.3f},{start.y:.3f},{start.z:.3f})")

    # GOAL
    p1 = pick_and_adjust_point(pcd, voxelmap=voxelmap, which="GOAL", step=ADJUST_STEP)
    goal = Pose3D(float(p1[0]), float(p1[1]), float(p1[2] + Z_LIFT))
    pok(f"Using GOAL  (lifted) xyz=({goal.x:.3f},{goal.y:.3f},{goal.z:.3f})")

    run_final_window_plan_and_show(
        pcd=pcd,
        voxelmap=voxelmap,
        start=start,
        goal=goal,
        planner_name=args.planner,              # "bitstar" | "informed_rrtstar"
        bitstar_params=bitstar_params,
        informed_params=informed_params,
        path_radius=PATH_RADIUS,
    )


if __name__ == "__main__":
    main()
