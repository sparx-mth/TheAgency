import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def plot_absolute_velocity_comparison(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))

    # Apply absolute value to compare pure speed magnitude
    gt_speed = df['GT_Vx'].abs()
    center_speed = df['Center_Vx'].abs()
    global_speed = df['Global_Vx'].abs()
    right_speed = df['Right_Vx'].abs()

    # 1. Ground Truth (Absolute)
    plt.plot(df['Frame'], gt_speed, label='Ground Truth Speed', color='black', linewidth=3, zorder=5)

    # 2. Estimated speeds
    plt.plot(df['Frame'], center_speed, marker='o', label='Center Zone Speed', linewidth=2, alpha=0.8)
    plt.plot(df['Frame'], global_speed, marker='s', label='Global Speed (WLS)', linewidth=2, alpha=0.8)
    plt.plot(df['Frame'], right_speed, marker='x', label='Right Edge Speed', linestyle='--', alpha=0.5)

    # Styling
    plt.title('Velocity Magnitude Comparison: Algorithm vs. Ground Truth', fontsize=16, pad=15)
    plt.xlabel('Frame Number')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot velocity comparison from CSV")
    
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file"
    )

    args = parser.parse_args()

    plot_absolute_velocity_comparison(args.csv)