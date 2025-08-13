"""
run_complete_benchmark.py - UPDATED FOR DQN

Complete benchmark runner that executes the comparison and analysis in order.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dqn_model():
    """Check if DQN model exists."""
    model_path = "/home/user/nadav/TheAgency/src/rl/models/dqn/interrupted_model.zip"
    if not os.path.exists(model_path):
        print("\n" + "=" * 70)
        print("WARNING: DQN MODEL NOT FOUND")
        print("=" * 70)
        print(f"\nThe trained DQN model was not found at: {model_path}")
        print("\nYou have two options:")
        print("1. Train the DQN model first by running: python train_dqn_wrapper.py")
        print("2. Continue without DQN (only Random vs Frontier comparison)")
        print("\nNote: Training DQN takes significant time depending on your settings.")

        choice = input("\nContinue without DQN? (y/n): ").strip().lower()
        if choice != 'y':
            print("\nExiting. Please train DQN first with: python train_dqn_wrapper.py")
            return False
        else:
            print("\nContinuing with Random and Frontier agents only...")
    else:
        print("\n✓ DQN model found!")

    return True


def run_comparison():
    """Run the agent comparison benchmark."""
    print("\n" + "=" * 70)
    print("STEP 1: RUNNING AGENT COMPARISON BENCHMARK")
    print("=" * 70)

    cmd = [sys.executable, "compare_agents.py"]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print("\nError running comparison benchmark!")
        return False

    print("\n✓ Comparison benchmark completed successfully!")
    return True


def run_analysis():
    """Run the enhanced analysis."""
    print("\n" + "=" * 70)
    print("STEP 2: RUNNING ENHANCED ANALYSIS")
    print("=" * 70)

    cmd = [sys.executable, "analyze_agent_comparison.py"]
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print("\nError running analysis!")
        return False

    print("\n✓ Analysis completed successfully!")
    return True


def display_results():
    """Display summary of results."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)

    results_dir = Path("results")
    if results_dir.exists():
        print("\nGenerated files in 'results/' directory:")
        for file in sorted(results_dir.glob("*")):
            print(f"  • {file.name}")

    print("\nKey outputs:")
    print("  1. agent_comparison_results.csv - Raw benchmark data")
    print("  2. comparison_plots.png - Main comparison visualization")
    print("  3. learning_curves.png - Progress over time for each agent")
    print("  4. performance_heatmap.png - Normalized performance metrics")
    print("  5. radar_comparison.png - Multi-dimensional comparison")
    print("  6. statistical_analysis.png - Statistical tests and distributions")
    print("  7. summary_report.txt - Complete text report")

    # Try to display a quick summary
    summary_file = results_dir / "summary_report.txt"
    if summary_file.exists():
        print("\n" + "-" * 70)
        print("QUICK SUMMARY (from summary_report.txt):")
        print("-" * 70)

        with open(summary_file, 'r') as f:
            lines = f.readlines()

        # Find and print the conclusions section
        in_conclusions = False
        for line in lines:
            if "CONCLUSIONS" in line:
                in_conclusions = True
            elif in_conclusions and "=" * 40 in line:
                break
            elif in_conclusions:
                print(line.rstrip())


def main():
    """Main function to run complete benchmark."""
    print("\n" + "=" * 70)
    print("COMPLETE SLAM AGENT BENCHMARK SUITE")
    print("=" * 70)
    print("\nThis will:")
    print("1. Check for trained DQN model")
    print("2. Run comparison benchmark (Random vs Frontier vs DQN)")
    print("3. Generate comprehensive analysis and visualizations")
    print("4. Perform statistical significance tests")

    input("\nPress Enter to start...")

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Step 1: Check for DQN model
    if not check_dqn_model():
        return

    # Step 2: Run comparison
    if not run_comparison():
        print("\nBenchmark failed. Please check for errors.")
        return

    # Step 3: Run analysis
    if not run_analysis():
        print("\nAnalysis failed. Please check for errors.")
        return

    # Step 4: Display results
    display_results()

    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nYou can now:")
    print("  • Check the 'results/' directory for all outputs")
    print("  • View the PNG files for visualizations")
    print("  • Read summary_report.txt for detailed analysis")
    print("\nTo re-run with different settings, edit compare_agents.py")


if __name__ == "__main__":
    main()