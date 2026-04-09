from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_marker_csvs(data_dir: Path, glob_pattern: str):
    files = sorted(data_dir.glob(glob_pattern))
    all_dfs = []
    for i, f in enumerate(files, start=1):
        df = pd.read_csv(f)
        required = {
            "ts", "marker_id", "color",
            "gt_depth_geom_m", "da3_depth_m",
            "abs_err_m", "jitter_m",
            "roll_deg", "pitch_deg", "yaw_deg"
        }
        if not required.issubset(df.columns):
            print(f"Skipping {f.name}: not a marker CSV")
            continue
        df["run"] = i
        df["run_name"] = f.stem
        df["rel_time"] = df["ts"] - df["ts"].min()
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def save_plot(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path.home() / "Documents" / "depth_validator_csv"))
    parser.add_argument("--glob", default="da3_markers_*.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    df = load_marker_csvs(data_dir, args.glob)

    if df.empty:
        print("No marker CSVs found.")
        return

    frame_df = (
        df.groupby(["run", "run_name", "ts"], as_index=False)
        .agg(
            rel_time=("rel_time", "first"),
            mae=("abs_err_m", "mean"),
            rmse=("abs_err_m", lambda s: float(np.sqrt(np.mean(np.square(s.dropna())))) if len(s.dropna()) else np.nan),
            jitter=("jitter_m", "mean"),
            roll_deg=("roll_deg", "first"),
            pitch_deg=("pitch_deg", "first"),
            yaw_deg=("yaw_deg", "first"),
            n_markers=("marker_id", "count"),
        )
    )

    run_aggs = {
        "mean_mae": ("abs_err_m", "mean"),
        "mean_rmse": ("abs_err_m", lambda s: float(np.sqrt(np.mean(np.square(s.dropna())))) if len(s.dropna()) else np.nan),
        "mean_jitter": ("jitter_m", "mean"),
        "rows": ("marker_id", "count"),
    }
    if "detected" in df.columns:
        run_aggs["detection_rate"] = ("detected", "mean")

    run_stats = df.groupby(["run", "run_name"], as_index=False).agg(**run_aggs)
    print("\nRun stats:")
    print(run_stats.to_string(index=False))

    marker_aggs = {
        "mean_abs_err_m": ("abs_err_m", "mean"),
        "p95_abs_err_m": ("abs_err_m", lambda s: np.nanpercentile(s.dropna(), 95) if len(s.dropna()) else np.nan),
        "mean_jitter_m": ("jitter_m", "mean"),
        "mean_gt_depth_m": ("gt_depth_geom_m", "mean"),
    }
    if "pixel_err_px" in df.columns:
        marker_aggs["mean_pixel_err_px"] = ("pixel_err_px", "mean")
    if "detected" in df.columns:
        marker_aggs["detection_rate"] = ("detected", "mean")

    marker_stats = (
        df.groupby(["marker_id", "color"], as_index=False)
        .agg(**marker_aggs)
        .sort_values("mean_abs_err_m")
    )
    print("\nPer-marker stats:")
    print(marker_stats.to_string(index=False))

    fig = plt.figure(figsize=(12, 6))
    for run in frame_df["run"].unique():
        subset = frame_df[frame_df["run"] == run]
        plt.plot(subset["rel_time"], subset["mae"], label=f"Run {run}")
    plt.title("MAE vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("MAE (m)")
    plt.grid(True)
    plt.legend()
    save_plot(fig, data_dir / "markers_mae_over_time.png")

    fig = plt.figure(figsize=(12, 6))
    for run in frame_df["run"].unique():
        subset = frame_df[frame_df["run"] == run]
        plt.plot(subset["rel_time"], subset["jitter"], label=f"Run {run}")
    plt.title("Jitter vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Jitter (m)")
    plt.grid(True)
    plt.legend()
    save_plot(fig, data_dir / "markers_jitter_over_time.png")

    fig = plt.figure(figsize=(10, 6))
    for marker_id, subset in df.groupby("marker_id"):
        plt.scatter(subset["gt_depth_geom_m"], subset["abs_err_m"], s=8, alpha=0.3, label=marker_id)
    plt.title("Absolute Error vs GT Depth")
    plt.xlabel("GT Depth (m)")
    plt.ylabel("Absolute Error (m)")
    plt.grid(True)
    plt.legend()
    save_plot(fig, data_dir / "markers_err_vs_gt_depth.png")

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(df["yaw_deg"], df["abs_err_m"], s=8, alpha=0.25)
    plt.title("Absolute Error vs Yaw")
    plt.xlabel("Yaw (deg)")
    plt.ylabel("Absolute Error (m)")
    plt.grid(True)
    save_plot(fig, data_dir / "markers_err_vs_yaw.png")

    if "pixel_err_px" in df.columns:
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(df["pixel_err_px"], df["abs_err_m"], s=8, alpha=0.25)
        plt.title("Absolute Error vs Pixel Error")
        plt.xlabel("Pixel Error (px)")
        plt.ylabel("Absolute Error (m)")
        plt.grid(True)
        save_plot(fig, data_dir / "markers_err_vs_pixel_err.png")

    fig = plt.figure(figsize=(10, 6))
    plt.bar(marker_stats["marker_id"], marker_stats["mean_abs_err_m"])
    plt.title("Mean Absolute Error per Marker")
    plt.ylabel("Mean Abs Error (m)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y")
    save_plot(fig, data_dir / "markers_mae_per_marker.png")

if __name__ == "__main__":
    main()
