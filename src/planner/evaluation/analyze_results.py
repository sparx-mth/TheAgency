import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze SLAM Simulation Results')
    parser.add_argument('--log_dir', type=str, default='outputs', help='Directory to save logs')
    parser.add_argument('--csv_name', type=str, default='agency_planner_slam_results1.csv', help='Name of CSV results file')
    parser.add_argument('--png_name', type=str, default='slam_visualization_grid1.png', help='Name of PNG output file')
    args = parser.parse_args()
    return args

def main():
    """
    This script loads simulation results from a CSV file and generates a summary of
    visualizations to evaluate SLAM performance across different maps and drone counts.

    Visualizations include:
    1. Bar plot of completion times per map and drone count.
    2. Box plot showing distribution of completion times by number of drones.
    3. Relative improvement in completion time as more drones are added.
    4. Mean and standard deviation of completion times per drone count.

    The script saves the output as a single image file: `outputs/slam_visualization_grid.png`.

    Intended usage:
    > python analyze_results.py

    Ensure the file `outputs/slam_results.csv` exists and contains the expected format:
    columns = [map, iteration, drones, time]
    """
    args = parse_args()

    # === Locate  data
    outputs_dir = Path(args.log_dir)
    assert outputs_dir.exists(), "Logs directory does not exist"
    assert outputs_dir.is_dir(), "Logs directory is not a directory"
    csv_path = outputs_dir / args.csv_name
    assert csv_path.exists(), f"CSV file {csv_path} does not exist"

    # === Load and clean data ===
    df = pd.read_csv(csv_path)

    # Remove maps where all runs failed
    valid_maps = df.groupby("map")["time"].apply(lambda x: x.notnull().any())
    df = df[df["map"].isin(valid_maps[valid_maps].index)]
    df = df[df["time"].notnull()]  # Drop remaining NaNs

    # Set style
    sns.set_theme(style="whitegrid")

    # Create figure and subplots grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("SLAM Simulation Results by Map and Drone Count", fontsize=16)

    # === GRAPH 1: Completion Time per Map by Number of Drones ===
    sns.barplot(data=df, x="map", y="time", hue="drones", errorbar="sd", ax=axes[0, 0])
    axes[0, 0].set_title("Completion Time per Map by Number of Drones")
    axes[0, 0].set_xlabel("Map Number")
    axes[0, 0].set_ylabel("Time (seconds)")
    axes[0, 0].legend(title="Drones")

    # === GRAPH 2: Boxplot of Completion Time by Drones ===
    sns.boxplot(data=df, x="drones", y="time", ax=axes[0, 1], showmeans=True,
                meanprops={"marker": "o", "color": "black"})
    axes[0, 1].set_title("Distribution of Completion Time by Drones")
    axes[0, 1].set_xlabel("Number of Drones")
    axes[0, 1].set_ylabel("Time (seconds)")

    # === GRAPH 3: Relative Improvement per Map ===
    improvement_data = []
    for map_id in sorted(df["map"].unique()):
        prev_mean = None
        for drone in sorted(df["drones"].unique()):
            mean_time = df[(df["map"] == map_id) & (df["drones"] == drone)]["time"].mean()
            improvement = ((prev_mean - mean_time) / prev_mean * 100) if prev_mean else None
            improvement_data.append((map_id, drone, improvement))
            prev_mean = mean_time

    improvement_df = pd.DataFrame(improvement_data, columns=["map", "drones", "relative_improvement"])

    sns.barplot(data=improvement_df[improvement_df["relative_improvement"].notnull()],
                x="map", y="relative_improvement", hue="drones", ax=axes[1, 0])
    axes[1, 0].set_title("Relative Improvement per Map by Adding Drones")
    axes[1, 0].set_xlabel("Map Number")
    axes[1, 0].set_ylabel("Improvement (%)")
    axes[1, 0].legend(title="Drones Added")

    # === GRAPH 4: Average Time per Drones (mean + std) ===
    avg_std = df.groupby("drones")["time"].agg(["mean", "std"]).reset_index()
    sns.barplot(data=avg_std, x="drones", y="mean", hue="drones", ax=axes[1, 1], errorbar="sd", legend=False)
    axes[1, 1].set_title("Average Completion Time per Drone Count")
    axes[1, 1].set_xlabel("Number of Drones")
    axes[1, 1].set_ylabel("Average Time (seconds)")

    # Final layout
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = outputs_dir / args.png_name
    plt.savefig(png_path)
    print(f"Saved: {png_path}")
    # plt.show()  # <-- Optional: use only if not using PyCharm backend

if __name__ == "__main__":
    main()