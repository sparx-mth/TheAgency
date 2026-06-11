#!/usr/bin/env python3
"""
BIT* Benchmark Analysis Script
==============================

Analyzes benchmark results and generates statistics and plots.

Usage:
    # Analyze latest results (default)
    python3 analyze_benchmark.py

    # Analyze specific file
    python3 analyze_benchmark.py results/bitstar_benchmark_XXXX.json

    # Save plots to files
    python3 analyze_benchmark.py --save-plots

    # Analyze specific pair
    python3 analyze_benchmark.py --pair-id 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Try to import matplotlib (optional for plots)
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not available. Plots will be skipped.")


def load_results(path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def print_header(title: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def print_stat(name: str, mean: float, std: float, unit: str = "") -> None:
    """Print a statistic with mean ± std."""
    unit_str = f" {unit}" if unit else ""
    print(f"  {name:40s}: {mean:8.3f} ± {std:6.3f}{unit_str}")


def analyze_basic_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute basic statistics from results."""
    pairs = results["pairs"]
    successful = [p for p in pairs if p["success"]]

    if not successful:
        print("[ERROR] No successful pairs to analyze!")
        return {}

    # Extract data - ONLY from successful pairs (failed pairs have None times, not 0)
    times_to_first = [p["time_to_first_solution_s"] for p in successful]
    final_lengths = [p["final_path_length_m"] for p in successful]
    first_lengths = [p["solutions"][0]["path_length_m"] for p in successful if p["solutions"]]
    num_solutions = [p["num_solutions_found"] for p in successful]
    euclidean_dists = [p["euclidean_distance"] for p in successful]

    # Extract waypoint counts from final solutions
    final_waypoints = [p["solutions"][-1]["num_waypoints"] for p in successful if p["solutions"]]

    # Compute improvements
    improvements_abs = []  # Absolute improvement in meters
    improvements_pct = []  # Percentage improvement
    for p in successful:
        if p["solutions"] and len(p["solutions"]) >= 1:
            first_len = p["solutions"][0]["path_length_m"]
            final_len = p["solutions"][-1]["path_length_m"]
            improvements_abs.append(first_len - final_len)
            if first_len > 0:
                improvements_pct.append((first_len - final_len) / first_len * 100)

    # Path efficiency (path length / euclidean distance)
    efficiencies = [final_lengths[i] / euclidean_dists[i]
                    for i in range(len(final_lengths)) if euclidean_dists[i] > 0]

    stats = {
        "num_total": len(pairs),
        "num_successful": len(successful),
        "num_failed": len(pairs) - len(successful),
        "success_rate": len(successful) / len(pairs) * 100,

        "time_to_first_mean": np.mean(times_to_first),
        "time_to_first_std": np.std(times_to_first),
        "time_to_first_min": np.min(times_to_first),
        "time_to_first_max": np.max(times_to_first),
        "time_to_first_median": np.median(times_to_first),

        "final_length_mean": np.mean(final_lengths),
        "final_length_std": np.std(final_lengths),
        "final_length_min": np.min(final_lengths),
        "final_length_max": np.max(final_lengths),
        "final_length_median": np.median(final_lengths),

        "first_length_mean": np.mean(first_lengths) if first_lengths else 0,
        "first_length_std": np.std(first_lengths) if first_lengths else 0,

        "improvement_abs_mean": np.mean(improvements_abs) if improvements_abs else 0,
        "improvement_abs_std": np.std(improvements_abs) if improvements_abs else 0,
        "improvement_pct_mean": np.mean(improvements_pct) if improvements_pct else 0,
        "improvement_pct_std": np.std(improvements_pct) if improvements_pct else 0,
        "improvement_pct_median": np.median(improvements_pct) if improvements_pct else 0,

        "num_solutions_mean": np.mean(num_solutions),
        "num_solutions_std": np.std(num_solutions),
        "num_solutions_median": np.median(num_solutions),

        "efficiency_mean": np.mean(efficiencies) if efficiencies else 0,
        "efficiency_std": np.std(efficiencies) if efficiencies else 0,

        "euclidean_dist_mean": np.mean(euclidean_dists),
        "euclidean_dist_std": np.std(euclidean_dists),

        # Waypoint statistics
        "waypoints_mean": np.mean(final_waypoints) if final_waypoints else 0,
        "waypoints_std": np.std(final_waypoints) if final_waypoints else 0,
        "waypoints_median": np.median(final_waypoints) if final_waypoints else 0,
        "waypoints_min": np.min(final_waypoints) if final_waypoints else 0,
        "waypoints_max": np.max(final_waypoints) if final_waypoints else 0,
    }

    return stats


def print_analysis(results: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """Print analysis results to console."""
    config = results["config"]

    print_header("BENCHMARK CONFIGURATION")
    print(f"  Scene:              {config['scene']}")
    print(f"  Num pairs:          {config['num_pairs']}")
    print(f"  Timeout per pair:   {config['timeout_s']}s")
    print(f"  Robot radius:       {config['robot_radius']}m")
    print(f"  Samples per batch:  {config['samples_per_batch']}")
    print(f"  Rewire factor:      {config['rewire_factor']}")

    print_header("OVERALL RESULTS")
    print(f"  Total pairs:        {stats['num_total']}")
    print(f"  Successful:         {stats['num_successful']}")
    print(f"  Failed:             {stats['num_failed']}")
    print(f"  Success rate:       {stats['success_rate']:.1f}%")
    print(f"  Total runtime:      {results['total_run_time_s']:.1f}s")

    print_header("TIME TO FIRST SOLUTION")
    print_stat("Mean", stats["time_to_first_mean"], stats["time_to_first_std"], "s")
    print(f"  {'Min':40s}: {stats['time_to_first_min']:8.3f} s")
    print(f"  {'Max':40s}: {stats['time_to_first_max']:8.3f} s")
    print(f"  {'Median':40s}: {stats['time_to_first_median']:8.3f} s")

    print_header("PATH LENGTH (FINAL)")
    print_stat("Mean", stats["final_length_mean"], stats["final_length_std"], "m")
    print(f"  {'Min':40s}: {stats['final_length_min']:8.3f} m")
    print(f"  {'Max':40s}: {stats['final_length_max']:8.3f} m")
    print(f"  {'Median':40s}: {stats['final_length_median']:8.3f} m")

    print_header("PATH LENGTH (FIRST SOLUTION)")
    print_stat("Mean", stats["first_length_mean"], stats["first_length_std"], "m")

    print_header("PATH IMPROVEMENT (First → Final)")
    print_stat("Absolute improvement", stats["improvement_abs_mean"], stats["improvement_abs_std"], "m")
    print_stat("Percentage improvement", stats["improvement_pct_mean"], stats["improvement_pct_std"], "%")
    print(f"  {'Median percentage':40s}: {stats['improvement_pct_median']:8.3f} %")

    print_header("WAYPOINTS (FINAL PATH)")
    print_stat("Mean", stats["waypoints_mean"], stats["waypoints_std"])
    print(f"  {'Min':40s}: {stats['waypoints_min']:8.0f}")
    print(f"  {'Max':40s}: {stats['waypoints_max']:8.0f}")
    print(f"  {'Median':40s}: {stats['waypoints_median']:8.0f}")

    print_header("SOLUTION QUALITY")
    print_stat("Solutions found per pair", stats["num_solutions_mean"], stats["num_solutions_std"])
    print(f"  {'Median solutions':40s}: {stats['num_solutions_median']:8.0f}")
    print_stat("Path efficiency (path/euclidean)", stats["efficiency_mean"], stats["efficiency_std"])
    print_stat("Euclidean distance", stats["euclidean_dist_mean"], stats["euclidean_dist_std"], "m")


def compute_improvement_over_time(results: Dict[str, Any],
                                  time_bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute path improvement as a function of time.

    Returns:
        time_points: Array of time points
        mean_improvement: Mean improvement at each time point (percentage)
        std_improvement: Std of improvement at each time point
    """
    pairs = results["pairs"]
    successful = [p for p in pairs if p["success"] and len(p["solutions"]) > 0]

    if not successful:
        return np.array([]), np.array([]), np.array([])

    timeout = results["config"]["timeout_s"]
    time_points = np.linspace(0, timeout, time_bins)

    # For each time point, compute the improvement achieved by that time
    improvements_at_time = [[] for _ in range(time_bins)]

    for pair in successful:
        solutions = pair["solutions"]
        if not solutions:
            continue

        first_length = solutions[0]["path_length_m"]

        # Build a timeline of path length for this pair
        for t_idx, t in enumerate(time_points):
            # Find the best solution at or before time t
            best_length = None
            for sol in solutions:
                if sol["time_from_start_s"] <= t:
                    best_length = sol["path_length_m"]

            if best_length is not None:
                improvement_pct = (first_length - best_length) / first_length * 100
                improvements_at_time[t_idx].append(improvement_pct)

    # Compute mean and std at each time point
    mean_improvement = np.array([
        np.mean(imps) if imps else np.nan for imps in improvements_at_time
    ])
    std_improvement = np.array([
        np.std(imps) if imps else np.nan for imps in improvements_at_time
    ])

    return time_points, mean_improvement, std_improvement


def add_stats_textbox(ax, mean_val, std_val, median_val, unit: str = "") -> None:
    """Add a statistics textbox to a plot."""
    unit_str = unit if unit else ""
    text = f"Mean: {mean_val:.2f}{unit_str}\nStd: {std_val:.2f}{unit_str}\nMedian: {median_val:.2f}{unit_str}"
    ax.text(0.97, 0.97, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))


def plot_analysis(results: Dict[str, Any], stats: Dict[str, Any],
                  save_path: Optional[Path] = None) -> None:
    """Generate analysis plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib not available, skipping plots")
        return

    pairs = results["pairs"]
    successful = [p for p in pairs if p["success"]]

    # Create 3x3 grid for 7 plots (last 2 cells empty or combined)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(f"BIT* Benchmark Analysis - {results['config']['scene']}\n"
                 f"({stats['num_successful']}/{stats['num_total']} successful, "
                 f"{stats['success_rate']:.1f}% success rate)", fontsize=14, fontweight='bold')

    # 1. Time to first solution histogram
    ax = axes[0, 0]
    times = [p["time_to_first_solution_s"] for p in successful]
    ax.hist(times, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(stats["time_to_first_mean"], color='crimson', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(stats["time_to_first_median"], color='darkorange', linestyle=':', linewidth=2, label='Median')
    ax.set_xlabel("Time to First Solution (s)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Time to First Solution Distribution", fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    add_stats_textbox(ax, stats["time_to_first_mean"], stats["time_to_first_std"],
                      stats["time_to_first_median"], "s")

    # 2. Final path length histogram
    ax = axes[0, 1]
    lengths = [p["final_path_length_m"] for p in successful]
    ax.hist(lengths, bins=25, color='seagreen', edgecolor='white', alpha=0.8)
    ax.axvline(stats["final_length_mean"], color='crimson', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(stats["final_length_median"], color='darkorange', linestyle=':', linewidth=2, label='Median')
    ax.set_xlabel("Final Path Length (m)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Final Path Length Distribution", fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    add_stats_textbox(ax, stats["final_length_mean"], stats["final_length_std"],
                      stats["final_length_median"], "m")

    # 3. Improvement percentage histogram
    ax = axes[0, 2]
    improvements = []
    for p in successful:
        if p["solutions"] and len(p["solutions"]) >= 1:
            first = p["solutions"][0]["path_length_m"]
            final = p["solutions"][-1]["path_length_m"]
            if first > 0:
                improvements.append((first - final) / first * 100)
    ax.hist(improvements, bins=25, color='coral', edgecolor='white', alpha=0.8)
    ax.axvline(stats["improvement_pct_mean"], color='crimson', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(stats["improvement_pct_median"], color='darkorange', linestyle=':', linewidth=2, label='Median')
    ax.set_xlabel("Path Improvement (%)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Path Improvement Distribution", fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    add_stats_textbox(ax, stats["improvement_pct_mean"], stats["improvement_pct_std"],
                      stats["improvement_pct_median"], "%")

    # 4. Improvement over time
    ax = axes[1, 0]
    time_points, mean_imp, std_imp = compute_improvement_over_time(results)
    if len(time_points) > 0:
        ax.plot(time_points, mean_imp, color='purple', linewidth=2.5, label='Mean')
        ax.fill_between(time_points, mean_imp - std_imp, mean_imp + std_imp,
                        color='purple', alpha=0.2, label='±1 Std')
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Improvement from First Solution (%)", fontsize=10)
        ax.set_title("Path Improvement Over Time", fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 5. Number of solutions per pair
    ax = axes[1, 1]
    num_sols = [p["num_solutions_found"] for p in successful]
    max_sols = max(num_sols) if num_sols else 1
    ax.hist(num_sols, bins=range(1, max_sols + 2), color='goldenrod',
            edgecolor='white', alpha=0.8, align='left')
    ax.axvline(stats["num_solutions_mean"], color='crimson', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(stats["num_solutions_median"], color='darkorange', linestyle=':', linewidth=2, label='Median')
    ax.set_xlabel("Number of Solutions Found", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Solutions Found per Pair", fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    add_stats_textbox(ax, stats["num_solutions_mean"], stats["num_solutions_std"],
                      stats["num_solutions_median"], "")

    # 6. Path length vs Euclidean distance
    ax = axes[1, 2]
    euclidean = [p["euclidean_distance"] for p in successful]
    final_len = [p["final_path_length_m"] for p in successful]
    ax.scatter(euclidean, final_len, alpha=0.5, c='teal', edgecolors='darkslategray',
               linewidth=0.3, s=30)

    # Add y=x reference line
    max_val = max(max(euclidean), max(final_len))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, linewidth=1.5, label='y=x (optimal)')

    ax.set_xlabel("Euclidean Distance (m)", fontsize=10)
    ax.set_ylabel("Final Path Length (m)", fontsize=10)
    ax.set_title("Path Length vs Euclidean Distance", fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Add efficiency stat
    ax.text(0.97, 0.03, f"Avg Efficiency: {stats['efficiency_mean']:.2f}x",
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))

    # 7. Number of waypoints histogram (NEW)
    ax = axes[2, 0]
    waypoints = [p["solutions"][-1]["num_waypoints"] for p in successful if p["solutions"]]
    if waypoints:
        ax.hist(waypoints, bins=25, color='mediumpurple', edgecolor='white', alpha=0.8)
        ax.axvline(stats["waypoints_mean"], color='crimson', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(stats["waypoints_median"], color='darkorange', linestyle=':', linewidth=2, label='Median')
        ax.set_xlabel("Number of Waypoints", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Waypoints in Final Path", fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        add_stats_textbox(ax, stats["waypoints_mean"], stats["waypoints_std"],
                          stats["waypoints_median"], "")

    # 8. Waypoints vs Path Length scatter (NEW)
    ax = axes[2, 1]
    if waypoints:
        path_lengths = [p["final_path_length_m"] for p in successful if p["solutions"]]
        ax.scatter(waypoints, path_lengths, alpha=0.5, c='darkcyan', edgecolors='darkslategray',
                   linewidth=0.3, s=30)
        ax.set_xlabel("Number of Waypoints", fontsize=10)
        ax.set_ylabel("Final Path Length (m)", fontsize=10)
        ax.set_title("Waypoints vs Path Length", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Add correlation coefficient
        if len(waypoints) > 1:
            corr = np.corrcoef(waypoints, path_lengths)[0, 1]
            ax.text(0.97, 0.03, f"Correlation: {corr:.3f}",
                    transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))

    # 9. Summary stats box (using last cell)
    ax = axes[2, 2]
    ax.axis('off')
    summary_text = (
        f"SUMMARY STATISTICS\n"
        f"{'─' * 30}\n\n"
        f"Time to First Solution:\n"
        f"  Mean:   {stats['time_to_first_mean']:.3f}s\n"
        f"  Std:    {stats['time_to_first_std']:.3f}s\n"
        f"  Median: {stats['time_to_first_median']:.3f}s\n\n"
        f"Final Path Length:\n"
        f"  Mean:   {stats['final_length_mean']:.2f}m\n"
        f"  Std:    {stats['final_length_std']:.2f}m\n"
        f"  Median: {stats['final_length_median']:.2f}m\n\n"
        f"Path Improvement:\n"
        f"  Mean:   {stats['improvement_pct_mean']:.1f}%\n"
        f"  Std:    {stats['improvement_pct_std']:.1f}%\n"
        f"  Median: {stats['improvement_pct_median']:.1f}%\n\n"
        f"Waypoints:\n"
        f"  Mean:   {stats['waypoints_mean']:.1f}\n"
        f"  Std:    {stats['waypoints_std']:.1f}\n"
        f"  Median: {stats['waypoints_median']:.1f}\n"
    )
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', edgecolor='gray', alpha=0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[OK] Plot saved to: {save_path}")
    else:
        plt.show()


def plot_individual_pair_improvement(results: Dict[str, Any], pair_id: int,
                                     save_path: Optional[Path] = None) -> None:
    """Plot path improvement over time for a specific pair."""
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib not available, skipping plots")
        return

    pairs = results["pairs"]
    pair = None
    for p in pairs:
        if p["pair_id"] == pair_id:
            pair = p
            break

    if pair is None:
        print(f"[ERROR] Pair {pair_id} not found")
        return

    if not pair["success"]:
        print(f"[ERROR] Pair {pair_id} was not successful")
        return

    solutions = pair["solutions"]
    if not solutions:
        print(f"[ERROR] Pair {pair_id} has no solutions")
        return

    times = [s["time_from_start_s"] for s in solutions]
    lengths = [s["path_length_m"] for s in solutions]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, lengths, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.scatter(times[0], lengths[0], s=150, c='green', zorder=5, label='First Solution')
    ax.scatter(times[-1], lengths[-1], s=150, c='red', zorder=5, label='Final Solution')

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Path Length (m)", fontsize=12)
    ax.set_title(f"Path Improvement for Pair {pair_id}\n"
                 f"Start: ({pair['start'][0]:.2f}, {pair['start'][1]:.2f}, {pair['start'][2]:.2f}) → "
                 f"Goal: ({pair['goal'][0]:.2f}, {pair['goal'][1]:.2f}, {pair['goal'][2]:.2f})",
                 fontsize=11)

    improvement = (lengths[0] - lengths[-1]) / lengths[0] * 100
    ax.text(0.95, 0.95, f"Improvement: {improvement:.1f}%\n"
                        f"First: {lengths[0]:.2f}m → Final: {lengths[-1]:.2f}m\n"
                        f"Solutions: {len(solutions)}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Plot saved to: {save_path}")
    else:
        plt.show()


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
    parser = argparse.ArgumentParser(description="Analyze BIT* benchmark results")
    parser.add_argument("results_file", type=str, nargs="?", default=None,
                        help="Path to benchmark results JSON (default: latest in results/)")
    parser.add_argument("--save-plots", action="store_true", default=True,
                        help="Save plots to files instead of showing")
    parser.add_argument("--output-dir", type=str, default="analysis_output",
                        help="Directory for saving plots (default: analysis_output)")
    parser.add_argument("--pair-id", type=int, default=None,
                        help="Show detailed analysis for a specific pair ID")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to search for results files (default: results)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine results file
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = find_latest_results_file(args.results_dir)
        if results_path is None:
            print(f"[ERROR] No benchmark results found in '{args.results_dir}/' directory")
            print(f"        Run benchmark first or specify a results file explicitly")
            return
        print(f"[INFO] Using latest results file: {results_path}")

    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    # Compute statistics
    stats = analyze_basic_stats(results)
    if not stats:
        return

    # Print analysis
    print_analysis(results, stats)

    # Generate plots
    if MATPLOTLIB_AVAILABLE:
        output_dir = Path(args.output_dir)

        if args.save_plots:
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"analysis_{results_path.stem}.png"
            plot_analysis(results, stats, save_path=plot_path)
        else:
            plot_analysis(results, stats)

        # Individual pair analysis
        if args.pair_id is not None:
            if args.save_plots:
                pair_plot_path = output_dir / f"pair_{args.pair_id}_{results_path.stem}.png"
                plot_individual_pair_improvement(results, args.pair_id, save_path=pair_plot_path)
            else:
                plot_individual_pair_improvement(results, args.pair_id)

    print("\n[DONE] Analysis complete!")


if __name__ == "__main__":
    main()