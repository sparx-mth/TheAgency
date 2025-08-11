"""
run_benchmarks.py

Simple script to run various benchmark configurations easily with rendering options.
"""

import subprocess
import sys
from pathlib import Path


def ask_render_options():
    """Ask user about rendering preferences."""
    print("\nRendering options:")
    print("1. No rendering (fastest)")
    print("2. Render with normal speed")
    print("3. Render with slow motion (0.1s delay)")
    print("4. Render with very slow motion (0.3s delay)")

    choice = input("Select rendering option (1-4) [default=1]: ").strip() or "1"

    if choice == "1":
        return False, 0.0
    elif choice == "2":
        return True, 0.0
    elif choice == "3":
        return True, 0.1
    elif choice == "4":
        return True, 0.3
    else:
        return False, 0.0


def run_quick_test():
    """Run a quick test with minimal configuration."""
    print("Running quick test benchmark...")

    render, delay = ask_render_options()

    cmd = [
        sys.executable, "benchmark_runner.py",
        "--map_count", "2",
        "--drone_counts", "1", "2",
        "--iterations", "3",
        "--agent_type", "frontier",
        "--camera_range", "10",
        "--camera_fov", "60",
        "--max_steps", "500",
        "--csv_name", "quick_test.csv",
        "--log_name", "quick_test.log"
    ]

    if render:
        cmd.append("--render")
        if delay > 0:
            cmd.extend(["--render_delay", str(delay)])

    subprocess.run(cmd)

    print("\nAnalyzing results...")
    cmd = [
        sys.executable, "analyze_results.py",
        "--csv_name", "quick_test.csv",
        "--output_name", "quick_test_analysis.png"
    ]
    subprocess.run(cmd)


def run_full_benchmark():
    """Run a full benchmark suite."""
    print("Running full benchmark suite...")

    render, delay = ask_render_options()

    if render:
        print("\nWarning: Full benchmark with rendering will take a long time!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            return

    # Test both agent types
    for agent_type in ["frontier", "random"]:
        print(f"\nTesting {agent_type} agent...")
        cmd = [
            sys.executable, "benchmark_runner.py",
            "--map_count", "10",
            "--drone_counts", "1", "2", "3",
            "--iterations", "30",
            "--agent_type", agent_type,
            "--camera_range", "10",
            "--camera_fov", "60",
            "--max_steps", "2000",
            "--csv_name", f"benchmark_{agent_type}.csv",
            "--log_name", f"benchmark_{agent_type}.log"
        ]

        if render:
            cmd.append("--render")
            if delay > 0:
                cmd.extend(["--render_delay", str(delay)])

        subprocess.run(cmd)

    print("\nAnalyzing combined results...")
    # Combine CSVs for analysis
    import pandas as pd

    dfs = []
    for agent_type in ["frontier", "random"]:
        csv_path = Path("logs") / f"benchmark_{agent_type}.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_path = Path("logs") / "benchmark_combined.csv"
        combined_df.to_csv(combined_path, index=False)

        cmd = [
            sys.executable, "analyze_results.py",
            "--csv_name", "benchmark_combined.csv",
            "--output_name", "benchmark_analysis.png",
            "--show_plot"
        ]
        subprocess.run(cmd)


def run_camera_comparison():
    """Compare different camera configurations."""
    print("Running camera configuration comparison...")

    render, delay = ask_render_options()

    configs = [
        ("narrow", 10, 30),
        ("standard", 10, 60),
        ("wide", 10, 90),
        ("long", 15, 45),
        ("short", 5, 60),
    ]

    for name, range_val, fov in configs:
        print(f"\nTesting {name} camera (range={range_val}, fov={fov})...")
        cmd = [
            sys.executable, "benchmark_runner.py",
            "--map_count", "5",
            "--drone_counts", "1", "2", "3",
            "--iterations", "10",
            "--agent_type", "frontier",
            "--camera_range", str(range_val),
            "--camera_fov", str(fov),
            "--max_steps", "1500",
            "--csv_name", f"camera_{name}.csv",
            "--log_name", f"camera_{name}.log"
        ]

        if render:
            cmd.append("--render")
            if delay > 0:
                cmd.extend(["--render_delay", str(delay)])

        subprocess.run(cmd)

    print("\nAnalyzing camera comparison results...")
    # Combine all camera CSVs
    import pandas as pd

    dfs = []
    for name, _, _ in configs:
        csv_path = Path("logs") / f"camera_{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['camera_config'] = name
            dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_path = Path("logs") / "camera_comparison.csv"
        combined_df.to_csv(combined_path, index=False)

        cmd = [
            sys.executable, "analyze_results.py",
            "--csv_name", "camera_comparison.csv",
            "--output_name", "camera_comparison_analysis.png",
            "--show_plot"
        ]
        subprocess.run(cmd)


def run_scalability_test():
    """Test scalability with increasing number of drones."""
    print("Running scalability test...")

    render, delay = ask_render_options()

    cmd = [
        sys.executable, "benchmark_runner.py",
        "--map_count", "5",
        "--drone_counts", "1", "2", "3", "4", "5", "6",
        "--iterations", "20",
        "--agent_type", "frontier",
        "--camera_range", "10",
        "--camera_fov", "60",
        "--max_steps", "3000",
        "--csv_name", "scalability_test.csv",
        "--log_name", "scalability_test.log"
    ]

    if render:
        cmd.append("--render")
        if delay > 0:
            cmd.extend(["--render_delay", str(delay)])

    subprocess.run(cmd)

    print("\nAnalyzing scalability results...")
    cmd = [
        sys.executable, "analyze_results.py",
        "--csv_name", "scalability_test.csv",
        "--output_name", "scalability_analysis.png",
        "--show_plot"
    ]
    subprocess.run(cmd)


def run_single_demo():
    """Run a single demonstration with custom parameters."""
    print("Single demonstration run with custom parameters")

    # Get parameters from user
    try:
        num_drones = int(input("Number of drones (1-6) [default=2]: ") or "2")
        num_drones = max(1, min(6, num_drones))

        print("\nAgent type:")
        print("1. Frontier (intelligent)")
        print("2. Random")
        agent_choice = input("Select agent (1-2) [default=1]: ") or "1"
        agent_type = "frontier" if agent_choice == "1" else "random"

        camera_range = int(input("Camera range (3-20) [default=10]: ") or "10")
        camera_range = max(3, min(20, camera_range))

        camera_fov = int(input("Camera FOV in degrees (15-120) [default=60]: ") or "60")
        camera_fov = max(15, min(120, camera_fov))

        max_steps = int(input("Max steps (100-5000) [default=1000]: ") or "1000")
        max_steps = max(100, min(5000, max_steps))

    except ValueError:
        print("Invalid input, using defaults")
        num_drones, agent_type = 2, "frontier"
        camera_range, camera_fov = 10, 60
        max_steps = 1000

    # Always render for demo
    print("\nRendering speed:")
    print("1. Normal speed")
    print("2. Slow motion (0.05s delay)")
    print("3. Very slow motion (0.1s delay)")
    speed_choice = input("Select speed (1-3) [default=1]: ") or "1"

    if speed_choice == "2":
        delay = 0.05
    elif speed_choice == "3":
        delay = 0.1
    else:
        delay = 0.0

    print(f"\nRunning demo with:")
    print(f"  Drones: {num_drones}")
    print(f"  Agent: {agent_type}")
    print(f"  Camera: range={camera_range}, FOV={camera_fov}°")
    print(f"  Max steps: {max_steps}")

    cmd = [
        sys.executable, "benchmark_runner.py",
        "--map_count", "1",
        "--drone_counts", str(num_drones),
        "--iterations", "1",
        "--agent_type", agent_type,
        "--camera_range", str(camera_range),
        "--camera_fov", str(camera_fov),
        "--max_steps", str(max_steps),
        "--csv_name", "demo_run.csv",
        "--log_name", "demo_run.log",
        "--render"
    ]

    if delay > 0:
        cmd.extend(["--render_delay", str(delay)])

    subprocess.run(cmd)

    # Show quick results
    try:
        import pandas as pd
        csv_path = Path("logs") / "demo_run.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                row = df.iloc[-1]  # Get last run
                print("\n" + "="*50)
                print("DEMO RESULTS")
                print("="*50)
                print(f"Completed: {'Yes' if row['completed'] else 'No'}")
                print(f"Progress: {row['progress']*100:.1f}%")
                if row['completed']:
                    print(f"Time: {row['time']:.1f} seconds")
                print(f"Steps: {row['steps']}")
                print(f"Total Reward: {row['total_reward']:.1f}")
                print(f"Collisions: {row['collisions']}")
    except Exception as e:
        print(f"Could not display results: {e}")


def main():
    """Main menu for running benchmarks."""
    print("\n" + "=" * 50)
    print("SLAM BENCHMARK SUITE")
    print("=" * 50)

    while True:
        print("\nSelect option:")
        print("1. Quick test (2 maps, 3 iterations)")
        print("2. Full benchmark (10 maps, 30 iterations, both agents)")
        print("3. Camera configuration comparison")
        print("4. Scalability test (1-6 drones)")
        print("5. Single demo run (interactive, with rendering)")
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            run_quick_test()
        elif choice == "2":
            run_full_benchmark()
        elif choice == "3":
            run_camera_comparison()
        elif choice == "4":
            run_scalability_test()
        elif choice == "5":
            run_single_demo()
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again")
            continue

        if choice in ["1", "2", "3", "4", "5"]:
            print("\n" + "=" * 50)
            print("Complete! Check the 'logs' directory for detailed results.")
            print("=" * 50)

            another = input("\nRun another benchmark? (y/n): ").strip().lower()
            if another != 'y':
                break


if __name__ == "__main__":
    main()