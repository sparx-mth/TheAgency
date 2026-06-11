#!/usr/bin/env python3
"""
BIT* Benchmark Script for Gibson Tiny - Benevolence Scene
=========================================================

SELF-CONTAINED VERSION - Does not require modifying algorithm.py

This script:
1. Samples 100 valid points (50 from floor 1, 50 from floor 3)
2. Creates pairs (floor1 -> floor3)
3. Runs BIT* with 30s timeout, tracking all intermediate solutions
4. Saves timing data, path lengths, and waypoints to JSON

Floor definitions for Benevolence:
- Floor 1: z in [-2.652, -0.352]
- Floor 2: z in [0.048, 2.298]
- Floor 3: z in [2.648, 5.698]

Usage:
    python3 benchmark_bitstar.py
    python3 benchmark_bitstar.py --num-pairs 50 --timeout 60
    python3 benchmark_bitstar.py --output results/my_benchmark.json
"""

from __future__ import annotations
try:
    from ompl import util as ou
    ou.setLogLevel(ou.LOG_WARN)  # Only warnings and errors
except:
    pass
import argparse
import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import sys

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from sparx_agency.core.common.types import Pose3D, Path3D, PlanStatus, PlanResult

from logging_utils import pinfo, pok, pwarn, perr
from gibson_io import load_gibson_mesh, sample_point_cloud
from voxelmap import VoxelMapFromPointCloud


# ============================================================================
# DATA STRUCTURES FOR BENCHMARK RESULTS
# ============================================================================

@dataclass
class SolutionSnapshot:
    """A single solution found during planning."""
    time_from_start_s: float  # Time since planning started
    path_length_m: float  # Total Euclidean length of path
    num_waypoints: int  # Number of waypoints in raw path
    waypoints: List[List[float]]  # [[x,y,z], ...] for later visualization
    cost: float = 0.0  # OMPL cost value


@dataclass
class PairResult:
    """Result for a single start-goal pair."""
    pair_id: int
    start: List[float]  # [x, y, z]
    goal: List[float]  # [x, y, z]
    euclidean_distance: float  # Direct line distance

    # Timing
    planning_start_time: float  # Unix timestamp
    total_planning_time_s: float

    # Solutions (chronological order)
    solutions: List[Dict[str, Any]] = field(default_factory=list)

    # Final status
    success: bool = False
    final_path_length_m: Optional[float] = None
    time_to_first_solution_s: Optional[float] = None
    num_solutions_found: int = 0
    error_message: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Configuration used for this benchmark run."""
    scene: str
    num_pairs: int
    timeout_s: float
    robot_radius: float
    inflation_margin: float
    voxel_size: float
    samples_per_batch: int
    use_k_nearest: bool
    rewire_factor: float
    floor1_z_range: Tuple[float, float]
    floor3_z_range: Tuple[float, float]
    safety_margin: float  # Extra margin for point sampling
    poll_interval_s: float  # How often to check for new solutions


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    config: Dict[str, Any]  # Serialized config
    run_timestamp: str
    total_run_time_s: float
    num_successful_pairs: int
    num_failed_pairs: int
    pairs: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_path_length(waypoints: List[List[float]]) -> float:
    """Compute total Euclidean path length."""
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(len(waypoints) - 1):
        p1 = np.array(waypoints[i])
        p2 = np.array(waypoints[i + 1])
        total += float(np.linalg.norm(p2 - p1))
    return total


def snapshot_to_dict(snapshot: SolutionSnapshot) -> Dict[str, Any]:
    """Convert SolutionSnapshot to dict for JSON serialization."""
    return {
        "time_from_start_s": snapshot.time_from_start_s,
        "path_length_m": snapshot.path_length_m,
        "num_waypoints": snapshot.num_waypoints,
        "waypoints": snapshot.waypoints,
        "cost": snapshot.cost,
    }


def config_to_dict(config: BenchmarkConfig) -> Dict[str, Any]:
    """Convert BenchmarkConfig to dict for JSON serialization."""
    return {
        "scene": config.scene,
        "num_pairs": config.num_pairs,
        "timeout_s": config.timeout_s,
        "robot_radius": config.robot_radius,
        "inflation_margin": config.inflation_margin,
        "voxel_size": config.voxel_size,
        "samples_per_batch": config.samples_per_batch,
        "use_k_nearest": config.use_k_nearest,
        "rewire_factor": config.rewire_factor,
        "floor1_z_range": list(config.floor1_z_range),
        "floor3_z_range": list(config.floor3_z_range),
        "safety_margin": config.safety_margin,
        "poll_interval_s": config.poll_interval_s,
    }


def pair_result_to_dict(pair: PairResult) -> Dict[str, Any]:
    """Convert PairResult to dict for JSON serialization."""
    return {
        "pair_id": pair.pair_id,
        "start": pair.start,
        "goal": pair.goal,
        "euclidean_distance": pair.euclidean_distance,
        "planning_start_time": pair.planning_start_time,
        "total_planning_time_s": pair.total_planning_time_s,
        "solutions": pair.solutions,
        "success": pair.success,
        "final_path_length_m": pair.final_path_length_m,
        "time_to_first_solution_s": pair.time_to_first_solution_s,
        "num_solutions_found": pair.num_solutions_found,
        "error_message": pair.error_message,
    }


# ============================================================================
# POINT SAMPLING
# ============================================================================

def sample_valid_point_in_z_range(
        voxelmap,
        z_min: float,
        z_max: float,
        robot_radius: float,
        safety_margin: float,
        max_attempts: int = 10000,
        rng: np.random.Generator = None,
) -> Optional[np.ndarray]:
    """
    Sample a random valid point within the given z range.

    Point must:
    1. Be inside the voxelmap bounds
    2. Pass is_free_world check (includes robot_radius clearance)
    3. Have additional safety_margin clearance from obstacles
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get world bounds from voxelmap
    x_min = voxelmap.origin_x
    x_max = voxelmap.origin_x + voxelmap.width * voxelmap.resolution
    y_min = voxelmap.origin_y
    y_max = voxelmap.origin_y + voxelmap.height * voxelmap.resolution

    # Clamp z range to voxelmap bounds
    z_min_clamped = max(z_min, voxelmap.origin_z)
    z_max_clamped = min(z_max, voxelmap.origin_z + voxelmap.depth * voxelmap.resolution)

    required_clearance = robot_radius + safety_margin

    for _ in range(max_attempts):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        z = rng.uniform(z_min_clamped, z_max_clamped)

        # Check basic validity
        if not voxelmap.is_free_world(x, y, z):
            continue

        # Check additional clearance
        clearance = voxelmap.world_clearance(x, y, z)
        if clearance >= required_clearance:
            return np.array([x, y, z], dtype=np.float64)

    return None


def sample_valid_points(
        voxelmap,
        z_range: Tuple[float, float],
        num_points: int,
        robot_radius: float,
        safety_margin: float,
        rng: np.random.Generator,
) -> List[np.ndarray]:
    """Sample multiple valid points in a z range."""
    points = []
    attempts = 0
    max_total_attempts = num_points * 1000

    while len(points) < num_points and attempts < max_total_attempts:
        pt = sample_valid_point_in_z_range(
            voxelmap, z_range[0], z_range[1],
            robot_radius, safety_margin,
            max_attempts=100, rng=rng
        )
        if pt is not None:
            points.append(pt)
            # REMOVED: progress print
        attempts += 1

    return points


"""
FIXED: plan_bitstar_with_tracking function for benchmark_bitstar_standalone.py

Replace the existing plan_bitstar_with_tracking function (lines ~263-426) with this version.

The original polling approach doesn't work because:
1. OMPL's internal state is not thread-safe for concurrent access
2. Calling ss.haveSolutionPath() and ss.getSolutionPath() from a background thread
   while ss.solve() is running in the main thread causes issues

This fix uses iterative solve() calls instead of a background polling thread.
"""


def plan_bitstar_with_tracking(
        start: Pose3D,
        goal: Pose3D,
        voxelmap,
        timeout: float,
        samples_per_batch: int,
        use_k_nearest: bool,
        rewire_factor: float,
        clearance_weight: float,
        robot_radius: float,
        poll_interval_s: float = 0.1,
) -> Tuple[bool, List[SolutionSnapshot], Optional[str]]:
    """
    Run BIT* and track all intermediate solutions using iterative solving.

    FIXED VERSION: Instead of polling from a background thread (which doesn't
    work due to OMPL thread-safety issues), this version calls solve()
    repeatedly with short intervals and checks for improved solutions
    between calls.

    Returns:
        (success, solutions_list, error_message)
    """
    # Import OMPL components
    try:
        from sparx_agency.core.planning.planners.common import (
            ob, og, OMPL_AVAILABLE, OMPL_ERROR,
            dist3d, make_clearance_objective_3d, setup_ompl_space_3d,
        )
    except ImportError as e:
        return False, [], f"Failed to import OMPL components: {e}"

    if not OMPL_AVAILABLE:
        return False, [], f"OMPL unavailable: {OMPL_ERROR}"

    # Check start/goal validity
    if not voxelmap.is_free_world(start.x, start.y, start.z):
        return False, [], "Start in collision"
    if not voxelmap.is_free_world(goal.x, goal.y, goal.z):
        return False, [], "Goal in collision"

    # Create a minimal params-like object for setup_ompl_space_3d
    @dataclass
    class MinimalParams:
        collision_check_resolution: float = 0.02
        longest_valid_segment_m: Optional[float] = 0.25

    minimal_params = MinimalParams()

    # Setup OMPL
    space, ss, si = setup_ompl_space_3d(voxelmap, minimal_params)

    # Set start and goal
    start_state = ob.State(space)
    goal_state = ob.State(space)
    start_state[0], start_state[1], start_state[2] = start.x, start.y, start.z
    goal_state[0], goal_state[1], goal_state[2] = goal.x, goal.y, goal.z
    ss.setStartAndGoalStates(start_state, goal_state)

    # Create BIT* planner
    planner = og.BITstar(si)
    planner.setSamplesPerBatch(samples_per_batch)
    planner.setUseKNearest(use_k_nearest)
    planner.setRewireFactor(rewire_factor)

    ss.setPlanner(planner)
    ss.setup()

    # Solution tracking state
    solutions: List[SolutionSnapshot] = []
    last_cost = float('inf')
    planning_start = time.perf_counter()
    timeout_remaining = timeout

    def extract_and_record_solution() -> bool:
        """Extract current solution if improved. Returns True if new solution recorded."""
        nonlocal last_cost

        try:
            if not ss.haveSolutionPath():
                return False

            # Get current cost
            pdef = ss.getProblemDefinition()
            opt_obj = pdef.getOptimizationObjective()
            solution_path = pdef.getSolutionPath()

            if solution_path is None:
                return False

            current_cost = solution_path.cost(opt_obj).value()

            # Only record if better (with small tolerance)
            if current_cost >= last_cost - 1e-9:
                return False

            last_cost = current_cost
            elapsed = time.perf_counter() - planning_start

            # Extract path
            path = ss.getSolutionPath()
            waypoints = []
            for i in range(path.getStateCount()):
                s = path.getState(i)
                waypoints.append([float(s[0]), float(s[1]), float(s[2])])

            if len(waypoints) < 2:
                return False

            path_length = compute_path_length(waypoints)

            solutions.append(SolutionSnapshot(
                time_from_start_s=elapsed,
                path_length_m=path_length,
                num_waypoints=len(waypoints),
                waypoints=waypoints,
                cost=current_cost,
            ))
            return True

        except Exception:
            return False

    # Iterative solving: call solve() repeatedly with short intervals
    solved = False
    while timeout_remaining > 0:
        # Solve for a short interval
        interval = min(poll_interval_s, timeout_remaining)
        result = ss.solve(interval)

        if result:
            solved = True
            # Check if we have a new/improved solution
            extract_and_record_solution()

        timeout_remaining -= interval
        elapsed = time.perf_counter() - planning_start

        # Safety check - break if we've exceeded timeout
        if elapsed >= timeout:
            break

    # Final extraction in case we missed the last improvement
    extract_and_record_solution()

    if not solved:
        return False, solutions, "BIT* found no solution"

    # Check for exact solution
    try:
        is_exact = bool(ss.haveExactSolutionPath())
    except:
        is_exact = True

    if not is_exact:
        return False, solutions, "BIT* found only approximate solution"

    return True, solutions, None

# ============================================================================
# MAIN BENCHMARK LOGIC
# ============================================================================

def run_benchmark(
        voxelmap,
        floor1_points: List[np.ndarray],
        floor3_points: List[np.ndarray],
        config: BenchmarkConfig,
) -> BenchmarkResults:
    """Run the full benchmark."""

    results = BenchmarkResults(
        config=config_to_dict(config),
        run_timestamp=datetime.now().isoformat(),
        total_run_time_s=0.0,
        num_successful_pairs=0,
        num_failed_pairs=0,
        pairs=[],
    )

    benchmark_start = time.perf_counter()
    num_pairs = min(len(floor1_points), len(floor3_points), config.num_pairs)

    for pair_id in range(num_pairs):
        start_pt = floor1_points[pair_id]
        goal_pt = floor3_points[pair_id]

        start = Pose3D(float(start_pt[0]), float(start_pt[1]), float(start_pt[2]))
        goal = Pose3D(float(goal_pt[0]), float(goal_pt[1]), float(goal_pt[2]))

        euclidean_dist = float(np.linalg.norm(goal_pt - start_pt))

        # Single-line progress (overwrites previous)
        print(f"\rPair {pair_id + 1}/{num_pairs}...", end="", flush=True)

        pair_result = PairResult(
            pair_id=pair_id,
            start=[start.x, start.y, start.z],
            goal=[goal.x, goal.y, goal.z],
            euclidean_distance=euclidean_dist,
            planning_start_time=time.time(),
            total_planning_time_s=0.0,
        )

        original_debug = voxelmap.debug_enabled
        voxelmap.debug_enabled = False

        try:
            plan_start = time.perf_counter()
            success, solutions, error_msg = plan_bitstar_with_tracking(
                start=start,
                goal=goal,
                voxelmap=voxelmap,
                timeout=config.timeout_s,
                samples_per_batch=config.samples_per_batch,
                use_k_nearest=config.use_k_nearest,
                rewire_factor=config.rewire_factor,
                clearance_weight=0.01,
                robot_radius=config.robot_radius,
                poll_interval_s=config.poll_interval_s,
            )
            plan_end = time.perf_counter()

            pair_result.total_planning_time_s = plan_end - plan_start
            pair_result.solutions = [snapshot_to_dict(s) for s in solutions]
            pair_result.success = success
            pair_result.num_solutions_found = len(solutions)

            if success and solutions:
                pair_result.time_to_first_solution_s = solutions[0].time_from_start_s
                pair_result.final_path_length_m = solutions[-1].path_length_m
                results.num_successful_pairs += 1
            else:
                pair_result.error_message = error_msg
                results.num_failed_pairs += 1

        except Exception as e:
            pair_result.error_message = f"Exception: {type(e).__name__}: {e}"
            results.num_failed_pairs += 1

        finally:
            voxelmap.debug_enabled = original_debug

        results.pairs.append(pair_result_to_dict(pair_result))

    results.total_run_time_s = time.perf_counter() - benchmark_start
    return results


def save_results(results: BenchmarkResults, output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "config": results.config,
        "run_timestamp": results.run_timestamp,
        "total_run_time_s": results.total_run_time_s,
        "num_successful_pairs": results.num_successful_pairs,
        "num_failed_pairs": results.num_failed_pairs,
        "pairs": results.pairs,
    }

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    pok(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BIT* Benchmark for Gibson Tiny Benevolence Scene"
    )
    parser.add_argument(
        "--num-pairs", type=int, default=1000,
        help="Number of start-goal pairs to test (default: 100)"
    )
    parser.add_argument(
        "--timeout", type=float, default=20.0,
        help="Planning timeout per pair in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: results/bitstar_benchmark_TIMESTAMP.json)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--samples-per-batch", type=int, default=1000,
        help="BIT* samples per batch (default: 1000)"
    )
    parser.add_argument(
        "--rewire-factor", type=float, default=2.0,
        help="BIT* rewire factor (default: 2.0)"
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0.1,
        help="How often to poll for new solutions in seconds (default: 0.1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration
    ROOT = Path("gibson/extracted/gibson_tiny")
    SCENE = "Shelbyville"

    FLOOR1_Z = (-6.084, -3.484)
    FLOOR3_Z = (0.016, 3.316)

    ROBOT_RADIUS = 0.1
    INFLATION_MARGIN = 0.02
    SAFETY_MARGIN = 0.05

    POINTS = 1_500_000
    VOXEL_SIZE = 0.12
    PADDING = 0.5

    pinfo(f"BIT* Benchmark: {args.num_pairs} pairs, {args.timeout}s timeout, seed={args.seed}")

    rng = np.random.default_rng(args.seed)

    # Setup (silent)
    mesh = load_gibson_mesh(ROOT, SCENE)
    pcd = sample_point_cloud(mesh, POINTS)
    voxelmap = VoxelMapFromPointCloud.from_point_cloud_and_mesh(
        pcd=pcd,
        mesh=mesh,
        voxel_size=VOXEL_SIZE,
        padding_m=PADDING,
        frame_id="map",
        robot_radius=ROBOT_RADIUS,
        inflation_margin_m=INFLATION_MARGIN,
        enforce_inside_mesh=True,
        debug_enabled=False,
    )

    # Sample points (silent)
    floor1_points = sample_valid_points(voxelmap, FLOOR1_Z, args.num_pairs, ROBOT_RADIUS, SAFETY_MARGIN, rng)
    floor3_points = sample_valid_points(voxelmap, FLOOR3_Z, args.num_pairs, ROBOT_RADIUS, SAFETY_MARGIN, rng)

    num_pairs = min(len(floor1_points), len(floor3_points), args.num_pairs)
    if num_pairs < args.num_pairs:
        pwarn(f"Using {num_pairs} pairs (sampling limited)")

    # Create config
    config = BenchmarkConfig(
        scene=SCENE,
        num_pairs=num_pairs,
        timeout_s=args.timeout,
        robot_radius=ROBOT_RADIUS,
        inflation_margin=INFLATION_MARGIN,
        voxel_size=VOXEL_SIZE,
        samples_per_batch=args.samples_per_batch,
        use_k_nearest=True,
        rewire_factor=args.rewire_factor,
        floor1_z_range=FLOOR1_Z,
        floor3_z_range=FLOOR3_Z,
        safety_margin=SAFETY_MARGIN,
        poll_interval_s=args.poll_interval,
    )

    # Run benchmark
    pinfo("Running...")
    results = run_benchmark(voxelmap, floor1_points, floor3_points, config)
    print()  # Newline after progress

    # Summary (2 lines max)
    pinfo(f"Complete: {results.num_successful_pairs}/{len(results.pairs)} OK in {results.total_run_time_s:.1f}s")

    if results.num_successful_pairs > 0:
        successful = [p for p in results.pairs if p["success"]]
        avg_first = np.mean([p["time_to_first_solution_s"] for p in successful])
        avg_sols = np.mean([p["num_solutions_found"] for p in successful])
        pinfo(f"Avg: first={avg_first:.2f}s, solutions/pair={avg_sols:.1f}")

    # Save
    output_path = Path(args.output) if args.output else Path(f"results/bitstar_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_results(results, output_path)


if __name__ == "__main__":
    main()