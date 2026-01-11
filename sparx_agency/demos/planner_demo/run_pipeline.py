#!/usr/bin/env python3
"""
RRT* Planner Demo with Hermite Smoothing.

Usage:
    python run_pipeline.py
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project imports
from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import Costmap2D, CostmapParams
from sparx_agency.core.mapping.costmap.occupancy import occupancy_from_grayscale, OccupancyThresholds
from sparx_agency.core.mapping.costmap.distance_field import compute_clearance_field, DistanceFieldParams
from sparx_agency.core.mapping.costmap.inflation import inflate_occupancy, InflationParams
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.interfaces.smoother import SmootherRequest
from sparx_agency.core.planning.planners.rrtstar import RRTStarOmplPlanner, RRTStarOmplParams
from sparx_agency.core.planning.smoothers.hermite import HermiteSmoother, HermiteParams


def load_map(pgm_path: str, yaml_path: str, inflate_radius: float = 0.1) -> Costmap2D:
    """Load PGM map with YAML metadata."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    resolution = config['resolution']
    origin = config['origin']

    img = plt.imread(pgm_path)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    thresholds = OccupancyThresholds(
        occupied_if_below=249,
        free_if_above=250,
        unknown_as_occupied=True
    )
    occupancy = occupancy_from_grayscale(img, thresholds)

    if inflate_radius > 0:
        occupancy = inflate_occupancy(
            occupancy,
            resolution=resolution,
            params=InflationParams(radius_m=inflate_radius)
        )

    clearance = compute_clearance_field(
        occupancy,
        resolution=resolution,
        params=DistanceFieldParams()
    )

    params = CostmapParams(
        resolution=resolution,
        origin_x=origin[0],
        origin_y=origin[1],
        frame_id="map"
    )
    return Costmap2D(occupancy, params, clearance=clearance)


def visualize(costmap: Costmap2D, start: Pose2D, goal: Pose2D,
              waypoints, trajectory=None, save_path: str = "path_result.png"):
    """Visualize map with raw path and smoothed trajectory."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Show map
    extent = [costmap.origin_x, costmap.origin_x + costmap.width * costmap.resolution,
              costmap.origin_y, costmap.origin_y + costmap.height * costmap.resolution]
    ax.imshow(costmap.occupancy, cmap='gray_r', origin='lower', extent=extent, alpha=0.7)

    # Plot raw path (cyan)
    if waypoints:
        xs = [p.x for p in waypoints]
        ys = [p.y for p in waypoints]
        ax.plot(xs, ys, 'c--', linewidth=1.5, alpha=0.7, label='RRT* Path')
        ax.scatter(xs, ys, c='cyan', s=20, marker='o', edgecolors='blue',
                   linewidths=0.5, zorder=5)

    # Plot smoothed trajectory (orange)
    if trajectory:
        samples = trajectory.sample_by_time(dt=0.05)
        tx = [s.x for s in samples]
        ty = [s.y for s in samples]
        ax.plot(tx, ty, color='orange', linewidth=2.5, label='Hermite Smooth')

    # Mark start/goal
    ax.scatter(start.x, start.y, c='green', s=100, marker='o', label='Start', zorder=10)
    ax.scatter(goal.x, goal.y, c='red', s=100, marker='*', label='Goal', zorder=10)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('RRT* + Hermite Smoothing')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.show()


def main():
    MAP_DIR = Path(__file__).parent / "maps"
    PGM_FILE = MAP_DIR / "hospital_map_cropped.pgm"
    YAML_FILE = MAP_DIR / "hospital_map_cropped.yaml"

    START = Pose2D(x=-2.0, y=-2.5)
    GOAL = Pose2D(x=5.0, y=5.0)

    # Load map
    print(f"Loading map: {PGM_FILE}")
    costmap = load_map(str(PGM_FILE), str(YAML_FILE), inflate_radius=0.2)
    print(f"Map: {costmap.width}x{costmap.height}, res={costmap.resolution}m")

    # Plan with RRT*
    print(f"Planning: ({START.x}, {START.y}) → ({GOAL.x}, {GOAL.y})")
    planner = RRTStarOmplPlanner(params=RRTStarOmplParams(
        timeout=5.0,
        use_clearance_objective=True,
        clearance_weight=10.0,
        interpolation_spacing=3.0
    ))

    request = PlanRequest(start=START, goal=GOAL, frame_id="map")
    result = planner.plan(request, costmap)

    print(f"Status: {result.status}")

    if not result.ok:
        print(f"Planning failed: {result.message}")
        return

    waypoints = list(result.path.points)
    print(f"Path: {len(waypoints)} waypoints, length={result.path.length():.2f}m")

    # Smooth with Hermite splines
    print("Smoothing with Hermite splines...")
    smoother = HermiteSmoother(params=HermiteParams(
        dt=0.02,
        nominal_speed_xy=0.5,
        tangent_scale=0.5,
    ))

    smooth_request = SmootherRequest(path=result.path)
    trajectory = smoother.smooth(smooth_request)
    print(f"Trajectory: {trajectory.total_time:.2f}s")

    # Visualize
    visualize(costmap, START, GOAL, waypoints, trajectory, "path_result.png")


if __name__ == "__main__":
    main()