#!/usr/bin/env python3
"""
BIT* Path Visualization Script
==============================

Visualizes benchmark paths in the 3D Gibson environment using Open3D.

Usage:
    # View first 5 paths from latest results (default)
    python3 visualize_paths.py

    # View first 5 paths from specific file
    python3 visualize_paths.py results/bitstar_benchmark_XXXX.json

    # View all paths
    python3 visualize_paths.py --all

    # View specific path by ID (shows evolution)
    python3 visualize_paths.py --pair-id 42

    # View first N paths
    python3 visualize_paths.py --num-paths 10

    # View specific pairs
    python3 visualize_paths.py --pair-ids 0 5 10 15

Controls in visualization:
    - Mouse: Rotate/pan/zoom
    - Q: Quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent))

from logging_utils import pinfo, pok, pwarn, perr
from gibson_io import load_gibson_mesh, sample_point_cloud
from tube import make_tube_from_polyline

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[ERROR] Open3D not available. Cannot visualize.")

# Color palette for multiple paths (distinguishable colors)
PATH_COLORS = [
    (0.0, 0.4, 1.0),  # Blue
    (1.0, 0.3, 0.0),  # Orange
    (0.0, 0.8, 0.2),  # Green
    (0.8, 0.0, 0.8),  # Purple
    (1.0, 0.8, 0.0),  # Yellow
    (0.0, 0.8, 0.8),  # Cyan
    (1.0, 0.0, 0.4),  # Pink
    (0.6, 0.4, 0.2),  # Brown
    (0.4, 0.4, 0.4),  # Gray
    (0.2, 0.6, 0.4),  # Teal
]


def load_results(path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_path_color(index: int) -> Tuple[float, float, float]:
    """Get color for path at given index."""
    return PATH_COLORS[index % len(PATH_COLORS)]


def create_path_geometry(
        waypoints: List[List[float]],
        color: Tuple[float, float, float],
        radius: float = 0.04,
) -> o3d.geometry.TriangleMesh:
    """Create a tube mesh for a path."""
    pts = np.array(waypoints, dtype=np.float64)
    return make_tube_from_polyline(pts, radius=radius, rgb=color)


def create_endpoint_sphere(
        point: List[float],
        color: Tuple[float, float, float],
        radius: float = 0.12,
) -> o3d.geometry.TriangleMesh:
    """Create a sphere at a point."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(list(color))
    sphere.translate(point)
    return sphere


def create_label_at_point(
        point: List[float],
        text: str,
        offset: List[float] = [0, 0, 0.3],
) -> o3d.geometry.TriangleMesh:
    """Create a small marker above a point (since Open3D doesn't have text labels easily)."""
    # We'll just use a small cone pointing down as a marker
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.05, height=0.15)
    cone.compute_vertex_normals()
    cone.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow

    # Rotate to point down
    R = o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0])
    cone.rotate(R, center=[0, 0, 0])

    cone.translate([point[0] + offset[0], point[1] + offset[1], point[2] + offset[2]])
    return cone


def visualize_paths(
        results: Dict[str, Any],
        pair_ids: List[int],
        show_first_solution: bool = False,
        show_all_solutions: bool = False,
        path_radius: float = 0.04,
) -> None:
    """
    Visualize selected paths in the 3D environment.

    Args:
        results: Benchmark results dict
        pair_ids: List of pair IDs to visualize
        show_first_solution: If True, show first solution in addition to final
        show_all_solutions: If True, show all intermediate solutions
        path_radius: Radius of path tubes
    """
    if not OPEN3D_AVAILABLE:
        perr("Open3D not available")
        return

    config = results["config"]
    pairs = results["pairs"]

    # Load environment
    ROOT = Path("gibson/extracted/gibson_tiny")
    SCENE = config["scene"]

    pinfo(f"Loading environment: {SCENE}")
    mesh = load_gibson_mesh(ROOT, SCENE)
    pcd = sample_point_cloud(mesh, 500_000)  # Fewer points for faster visualization

    # Filter to requested pairs
    selected_pairs = []
    for p in pairs:
        if p["pair_id"] in pair_ids:
            if p["success"] and p["solutions"]:
                selected_pairs.append(p)
            else:
                pwarn(f"Pair {p['pair_id']} was not successful, skipping")

    if not selected_pairs:
        perr("No valid pairs to visualize")
        return

    pok(f"Visualizing {len(selected_pairs)} path(s)")

    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"BIT* Paths - {SCENE} ({len(selected_pairs)} paths)",
        width=1400,
        height=900
    )

    # Add point cloud
    vis.add_geometry(pcd)

    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(frame)

    # Add paths
    legend_info = []

    for idx, pair in enumerate(selected_pairs):
        color = get_path_color(idx)
        pair_id = pair["pair_id"]
        solutions = pair["solutions"]

        start = pair["start"]
        goal = pair["goal"]

        # Add start sphere (green)
        start_sphere = create_endpoint_sphere(start, (0.1, 0.9, 0.1), radius=0.10)
        vis.add_geometry(start_sphere)

        # Add goal sphere (red)
        goal_sphere = create_endpoint_sphere(goal, (0.9, 0.1, 0.1), radius=0.10)
        vis.add_geometry(goal_sphere)

        if show_all_solutions:
            # Show all solutions with decreasing opacity (approximated by color brightness)
            num_sols = len(solutions)
            for sol_idx, sol in enumerate(solutions):
                # Make earlier solutions dimmer
                brightness = 0.3 + 0.7 * (sol_idx / max(1, num_sols - 1))
                sol_color = tuple(c * brightness for c in color)

                waypoints = sol["waypoints"]
                if len(waypoints) >= 2:
                    tube = create_path_geometry(waypoints, sol_color, radius=path_radius * 0.7)
                    vis.add_geometry(tube)

            legend_info.append(f"Pair {pair_id}: {num_sols} solutions (color varies by iteration)")

        elif show_first_solution and len(solutions) > 1:
            # Show first solution (dimmer)
            first_sol = solutions[0]
            first_color = tuple(c * 0.4 for c in color)
            if len(first_sol["waypoints"]) >= 2:
                tube_first = create_path_geometry(
                    first_sol["waypoints"], first_color, radius=path_radius * 0.6
                )
                vis.add_geometry(tube_first)

            # Show final solution (bright)
            final_sol = solutions[-1]
            if len(final_sol["waypoints"]) >= 2:
                tube_final = create_path_geometry(
                    final_sol["waypoints"], color, radius=path_radius
                )
                vis.add_geometry(tube_final)

            improvement = (first_sol["path_length_m"] - final_sol["path_length_m"]) / first_sol["path_length_m"] * 100
            legend_info.append(
                f"Pair {pair_id}: {first_sol['path_length_m']:.1f}m → {final_sol['path_length_m']:.1f}m "
                f"({improvement:.1f}% improvement)"
            )
        else:
            # Show only final solution
            final_sol = solutions[-1]
            if len(final_sol["waypoints"]) >= 2:
                tube = create_path_geometry(final_sol["waypoints"], color, radius=path_radius)
                vis.add_geometry(tube)

            legend_info.append(
                f"Pair {pair_id}: {final_sol['path_length_m']:.2f}m, "
                f"{final_sol['num_waypoints']} waypoints"
            )

    # Print legend
    print("\n" + "=" * 60)
    print("PATH LEGEND")
    print("=" * 60)
    for idx, info in enumerate(legend_info):
        color = get_path_color(idx)
        color_name = f"RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})"
        print(f"  [{color_name}] {info}")
    print("=" * 60)
    print("\nControls:")
    print("  Mouse drag: Rotate")
    print("  Scroll: Zoom")
    print("  Shift + drag: Pan")
    print("  Q: Quit")
    print("=" * 60)

    # Run visualization
    vis.run()
    vis.destroy_window()


def visualize_single_pair_evolution(
        results: Dict[str, Any],
        pair_id: int,
        path_radius: float = 0.04,
) -> None:
    """
    Visualize how a single path evolved over time.
    Shows all intermediate solutions with colors from red (first) to green (final).
    """
    if not OPEN3D_AVAILABLE:
        perr("Open3D not available")
        return

    config = results["config"]
    pairs = results["pairs"]

    # Find the pair
    pair = None
    for p in pairs:
        if p["pair_id"] == pair_id:
            pair = p
            break

    if pair is None:
        perr(f"Pair {pair_id} not found")
        return

    if not pair["success"] or not pair["solutions"]:
        perr(f"Pair {pair_id} has no solutions")
        return

    solutions = pair["solutions"]
    pinfo(f"Pair {pair_id} has {len(solutions)} solution(s)")

    # Load environment
    ROOT = Path("gibson/extracted/gibson_tiny")
    SCENE = config["scene"]

    pinfo(f"Loading environment: {SCENE}")
    mesh = load_gibson_mesh(ROOT, SCENE)
    pcd = sample_point_cloud(mesh, 500_000)

    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Path Evolution - Pair {pair_id}",
        width=1400,
        height=900
    )

    vis.add_geometry(pcd)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(frame)

    # Add start/goal
    start = pair["start"]
    goal = pair["goal"]

    start_sphere = create_endpoint_sphere(start, (0.1, 0.9, 0.1), radius=0.12)
    goal_sphere = create_endpoint_sphere(goal, (0.9, 0.1, 0.1), radius=0.12)
    vis.add_geometry(start_sphere)
    vis.add_geometry(goal_sphere)

    # Add all solutions with color gradient (red -> yellow -> green)
    num_sols = len(solutions)

    print("\n" + "=" * 60)
    print(f"PATH EVOLUTION - PAIR {pair_id}")
    print("=" * 60)

    for idx, sol in enumerate(solutions):
        # Color gradient: red (first) -> green (last)
        t = idx / max(1, num_sols - 1)  # 0 to 1
        color = (1.0 - t, t, 0.2)  # Red to green

        waypoints = sol["waypoints"]
        if len(waypoints) >= 2:
            tube = create_path_geometry(waypoints, color, radius=path_radius)
            vis.add_geometry(tube)

        print(f"  Solution {idx + 1}/{num_sols}: "
              f"time={sol['time_from_start_s']:.3f}s, "
              f"length={sol['path_length_m']:.2f}m, "
              f"waypoints={sol['num_waypoints']}")

    if num_sols > 1:
        first_len = solutions[0]["path_length_m"]
        final_len = solutions[-1]["path_length_m"]
        improvement = (first_len - final_len) / first_len * 100
        print(f"\n  Total improvement: {first_len:.2f}m → {final_len:.2f}m ({improvement:.1f}%)")

    print("=" * 60)
    print("\nColor legend:")
    print("  Red = First solution")
    print("  Green = Final solution")
    print("  Gradient = Intermediate solutions")
    print("=" * 60)

    vis.run()
    vis.destroy_window()


def find_latest_results_file(results_dir: str = "results") -> Optional[Path]:
    """Find the most recent benchmark results file in the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    json_files = list(results_path.glob("bitstar_benchmark_*.json"))
    if not json_files:
        return None

    # Sort by modification time, newest first
    json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return json_files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize BIT* benchmark paths in 3D environment"
    )
    parser.add_argument("results_file", type=str, nargs="?", default=None,
                        help="Path to benchmark results JSON (default: latest in results/)")

    # Selection options (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true",
                       help="Show all paths")
    group.add_argument("--pair-id", type=int, default=None,
                       help="Show specific pair by ID (with evolution)")
    group.add_argument("--pair-ids", type=int, nargs="+", default=None,
                       help="Show specific pairs by IDs")
    group.add_argument("--num-paths", type=int, default=5,
                       help="Number of paths to show (default: 5)")

    # Display options
    parser.add_argument("--show-first", action="store_true",
                        help="Show first solution alongside final")
    parser.add_argument("--show-evolution", action="store_true",
                        help="Show all intermediate solutions")
    parser.add_argument("--path-radius", type=float, default=0.04,
                        help="Radius of path tubes (default: 0.04)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to search for results files (default: results)")

    return parser.parse_args()


def main():
    args = parse_args()

    if not OPEN3D_AVAILABLE:
        perr("Open3D is required for visualization")
        return

    # Determine results file
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = find_latest_results_file(args.results_dir)
        if results_path is None:
            perr(f"No benchmark results found in '{args.results_dir}/' directory")
            pinfo("Run benchmark first or specify a results file explicitly")
            return
        pinfo(f"Using latest results file: {results_path}")

    if not results_path.exists():
        perr(f"Results file not found: {results_path}")
        return

    pinfo(f"Loading results from: {results_path}")
    results = load_results(results_path)

    pairs = results["pairs"]
    successful_ids = [p["pair_id"] for p in pairs if p["success"]]

    pinfo(f"Found {len(successful_ids)} successful pairs out of {len(pairs)}")

    # Determine which pairs to show
    if args.pair_id is not None:
        # Show single pair with evolution
        visualize_single_pair_evolution(
            results,
            args.pair_id,
            path_radius=args.path_radius
        )
        return

    if args.all:
        pair_ids = successful_ids
    elif args.pair_ids:
        pair_ids = args.pair_ids
    else:
        # Default: first N successful pairs
        pair_ids = successful_ids[:args.num_paths]

    pinfo(f"Visualizing pairs: {pair_ids}")

    visualize_paths(
        results,
        pair_ids=pair_ids,
        show_first_solution=args.show_first,
        show_all_solutions=args.show_evolution,
        path_radius=args.path_radius,
    )


if __name__ == "__main__":
    main()