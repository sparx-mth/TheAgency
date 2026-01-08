#!/usr/bin/env python3
"""
RRT* Planner Demo - Uses project modules.

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

# RRT* - adjust path if needed
from sparx_agency.core.planning.planners.rrtstar import RRTStarOmplPlanner, RRTStarOmplParams


def load_map(pgm_path: str, yaml_path: str, inflate_radius: float = 0.1) -> Costmap2D:
    """Load PGM map with YAML metadata using project modules."""
    # Load YAML config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    resolution = config['resolution']
    origin = config['origin']  # [x, y, theta]

    # Load PGM image
    img = plt.imread(pgm_path)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Convert to occupancy: White (>=250) = free, Gray/Black = occupied
    thresholds = OccupancyThresholds(
        occupied_if_below=249,  # Gray and black are walls
        free_if_above=250,      # Only white is free
        unknown_as_occupied=True
    )
    occupancy = occupancy_from_grayscale(img, thresholds)

    # Optional: inflate obstacles for safety margin
    if inflate_radius > 0:
        occupancy = inflate_occupancy(
            occupancy,
            resolution=resolution,
            params=InflationParams(radius_m=inflate_radius)
        )

    # Compute clearance field for cost optimization
    clearance = compute_clearance_field(
        occupancy,
        resolution=resolution,
        params=DistanceFieldParams()
    )

    # Build Costmap2D
    params = CostmapParams(
        resolution=resolution,
        origin_x=origin[0],
        origin_y=origin[1],
        frame_id="map"
    )
    return Costmap2D(occupancy, params, clearance=clearance)


def visualize(costmap: Costmap2D, start: Pose2D, goal: Pose2D, waypoints, save_path: str = "path_result.png"):
    """Visualize map with path overlay."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Show map
    extent = [costmap.origin_x, costmap.origin_x + costmap.width * costmap.resolution,
              costmap.origin_y, costmap.origin_y + costmap.height * costmap.resolution]
    ax.imshow(costmap.occupancy, cmap='gray_r', origin='lower', extent=extent, alpha=0.7)

    # Plot path
    if waypoints:
        xs = [p.x for p in waypoints]
        ys = [p.y for p in waypoints]
        ax.plot(xs, ys, 'b-', linewidth=2, label='Path')
        ax.scatter(xs, ys, c='blue', s=10, zorder=5)

    # Mark start/goal
    ax.scatter(start.x, start.y, c='green', s=200, marker='o', label='Start', zorder=10)
    ax.scatter(goal.x, goal.y, c='red', s=200, marker='*', label='Goal', zorder=10)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('RRT* Path Planning Result')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def main():
    # === CONFIG ===
    MAP_DIR = Path(__file__).parent / "maps"
    PGM_FILE = MAP_DIR / "hospital_map_cropped.pgm"
    YAML_FILE = MAP_DIR / "hospital_map_cropped.yaml"

    # Hardcoded start/goal (adjust based on your map!)
    START = Pose2D(x=-5.0, y=-5.0)
    GOAL = Pose2D(x=5.0, y=5.0)

    # === LOAD MAP ===
    print(f"Loading map: {PGM_FILE}")
    costmap = load_map(str(PGM_FILE), str(YAML_FILE), inflate_radius=0.1)
    print(f"Map size: {costmap.width}x{costmap.height}, resolution: {costmap.resolution}m")

    # === PLAN using project's RRTStarPlanner ===
    print(f"Planning from ({START.x}, {START.y}) to ({GOAL.x}, {GOAL.y})...")

    planner = RRTStarOmplPlanner(params=RRTStarOmplParams(
        timeout=5.0,
        use_clearance_objective=True,
        clearance_weight=10.0,
        interpolation_spacing=0.2
    ))

    request = PlanRequest(start=START, goal=GOAL, frame_id="map")
    result = planner.plan(request, costmap)

    print(f"Status: {result.status}")
    print(f"Message: {result.message}")

    if result.path:
        waypoints = list(result.path.points)
        print(f"\n=== PATH FOUND ({len(waypoints)} waypoints) ===")
        for i, wp in enumerate(waypoints):
            print(f"  [{i:3d}] x={wp.x:8.3f}, y={wp.y:8.3f}")

        # === VISUALIZE ===
        visualize(costmap, START, GOAL, waypoints, "path_result.png")
    else:
        print("Planning failed!")


if __name__ == "__main__":
    main()