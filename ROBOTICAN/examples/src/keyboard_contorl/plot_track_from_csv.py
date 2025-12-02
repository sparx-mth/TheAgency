#!/usr/bin/env python3
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def plot_track(csv_path: str, show_arrows: bool = True):
    df = pd.read_csv(csv_path)

    xs = []
    ys = []

    # Collect positions from S and S' in order
    # We’ll treat them as waypoints on the ground (x,y).
    for _, row in df.iterrows():
        # S (before action)
        if pd.notna(row["S_pos_x"]) and pd.notna(row["S_pos_y"]):
            xs.append(row["S_pos_x"])
            ys.append(row["S_pos_y"])

        # S' (after action)
        if pd.notna(row["Sp_pos_x"]) and pd.notna(row["Sp_pos_y"]):
            xs.append(row["Sp_pos_x"])
            ys.append(row["Sp_pos_y"])

    if len(xs) == 0:
        print("No positions found in CSV (S_pos_x/S_pos_y or Sp_pos_x/Sp_pos_y).")
        return

    plt.figure()
    plt.plot(xs, ys, "-o", linewidth=1, markersize=3)

    if show_arrows and len(xs) > 1:
        # Draw a few arrows along the path to show direction
        step = max(len(xs) // 20, 1)
        for i in range(0, len(xs) - 1, step):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            plt.arrow(
                xs[i],
                ys[i],
                dx,
                dy,
                head_width=0.3,
                length_includes_head=True,
            )

    plt.xlabel("x [m] (ground)")
    plt.ylabel("y [m] (ground)")
    plt.title(f"Ground track from {csv_path}")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="RL CSV file (gui_manual_log_*.csv)")
    args = parser.parse_args()
    plot_track(args.csv)


if __name__ == "__main__":
    main()
