#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# this script compares two pose evaluation CSV files (from pose_eval_rosbag2.py)
"""
run the script:
python3 compare_pose_eval.py \
  --base /home/shirb/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/pose_eval_rosbag2_2026_01_26-13_29_42.csv \
  --essential /home/shirb/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/pose_eval_rosbag2_2026_01_26-13_29_42_essential.csv \
  --tol-ms 100 \
  --points 10 \
  --smooth-sec 0.5 \
  --outdir /home/shirb/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/compare_out
"""
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"t_sec", "err_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    df = df.sort_values("t_sec").reset_index(drop=True)
    df["t_sec"] = df["t_sec"].astype(float)
    df["err_m"] = df["err_m"].astype(float)
    return df


def auc_error(df: pd.DataFrame, t_col: str = "t_sec", e_col: str = "err_m") -> float:
    t = df[t_col].to_numpy()
    e = df[e_col].to_numpy()
    if len(df) < 2:
        return float("nan")
    return float(np.trapz(e, t))


def summarize_errors(df: pd.DataFrame, name: str, t_col="t_sec", e_col="err_m") -> dict:
    e = df[e_col].to_numpy()
    return {
        "name": name,
        "N": int(len(df)),
        "t_start": float(df[t_col].iloc[0]),
        "t_end": float(df[t_col].iloc[-1]),
        "duration_s": float(df[t_col].iloc[-1] - df[t_col].iloc[0]),
        "mean_err": float(np.mean(e)),
        "median_err": float(np.median(e)),
        "p90_err": float(np.percentile(e, 90)),
        "p95_err": float(np.percentile(e, 95)),
        "max_err": float(np.max(e)),
        "auc_err": auc_error(df, t_col=t_col, e_col=e_col),
    }


def pick_even_points(df_aligned: pd.DataFrame, n_points: int) -> pd.DataFrame:
    if len(df_aligned) <= n_points:
        return df_aligned.copy()
    tmin = df_aligned["t_sec"].min()
    tmax = df_aligned["t_sec"].max()
    targets = np.linspace(tmin, tmax, n_points)

    idx = []
    t = df_aligned["t_sec"].to_numpy()
    for tt in targets:
        i = int(np.argmin(np.abs(t - tt)))
        idx.append(i)
    idx = sorted(set(idx))
    return df_aligned.iloc[idx].reset_index(drop=True)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stem_no_ext(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to base csv (no essential)")
    ap.add_argument("--essential", required=True, help="path to essential csv")
    ap.add_argument("--tol-ms", type=float, default=50.0,
                    help="time alignment tolerance in milliseconds (merge_asof)")
    ap.add_argument("--points", type=int, default=10, help="how many comparison points to export")
    ap.add_argument("--smooth-sec", type=float, default=0.5,
                    help="rolling smoothing window in seconds for plotting")
    ap.add_argument("--outdir", default="pose_eval_compare_out",
                    help="output directory for CSV + PNG")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    base = load_csv(args.base)
    ess = load_csv(args.essential)

    tol = args.tol_ms / 1000.0

    base2 = base.rename(columns={"err_m": "err_base"})
    ess2 = ess.rename(columns={"err_m": "err_ess"})

    aligned = pd.merge_asof(
        base2.sort_values("t_sec"),
        ess2.sort_values("t_sec"),
        on="t_sec",
        direction="nearest",
        tolerance=tol,
    )
    aligned = aligned.dropna(subset=["err_ess"]).reset_index(drop=True)
    if len(aligned) == 0:
        raise RuntimeError(
            "No aligned samples found. Increase --tol-ms (e.g. 200) or check time ranges."
        )

    aligned["delta_err"] = aligned["err_base"] - aligned["err_ess"]  # + => essential better

    # Summaries (raw)
    s_base = summarize_errors(base, "BASE")
    s_ess = summarize_errors(ess, "ESSENTIAL")

    # Summaries (aligned fair subset)
    s_base_al = summarize_errors(aligned, "BASE(aligned)", t_col="t_sec", e_col="err_base")
    s_ess_al = summarize_errors(aligned, "ESSENTIAL(aligned)", t_col="t_sec", e_col="err_ess")

    delta = aligned["delta_err"].to_numpy()
    improvement_mean = float(np.mean(delta))
    improvement_median = float(np.median(delta))
    win_rate = float(np.mean(delta > 0.0))
    gain_pct_mean = float((s_base_al["mean_err"] - s_ess_al["mean_err"]) / max(s_base_al["mean_err"], 1e-9) * 100.0)

    improvement_row = {
        "name": "IMPROVEMENT(aligned)",
        "N": int(len(aligned)),
        "mean_delta_err": improvement_mean,
        "median_delta_err": improvement_median,
        "win_rate": win_rate,
        "mean_gain_percent": gain_pct_mean,
        "auc_base_aligned": s_base_al["auc_err"],
        "auc_essential_aligned": s_ess_al["auc_err"],
        "auc_gain_percent": float((s_base_al["auc_err"] - s_ess_al["auc_err"]) / max(s_base_al["auc_err"], 1e-9) * 100.0),
    }

    # ===== Save CSV outputs =====
    tag = f"{stem_no_ext(args.base)}__VS__{stem_no_ext(args.essential)}"
    summary_path = os.path.join(args.outdir, f"summary__{tag}.csv")
    aligned_path = os.path.join(args.outdir, f"aligned_series__{tag}.csv")
    snapshot_path = os.path.join(args.outdir, f"snapshot_{args.points}pts__{tag}.csv")

    summary_df = pd.DataFrame([s_base, s_ess, s_base_al, s_ess_al])
    # Add improvement as a separate block (different columns) => concat with NaNs ok
    improvement_df = pd.DataFrame([improvement_row])
    summary_out = pd.concat([summary_df, improvement_df], ignore_index=True, sort=False)
    summary_out.to_csv(summary_path, index=False)

    # Save full aligned series (this is the main apples-to-apples timeline)
    aligned_out = aligned[["t_sec", "err_base", "err_ess", "delta_err"]].copy()
    aligned_out.to_csv(aligned_path, index=False)

    # Save N-point snapshot across time
    sample = pick_even_points(aligned, args.points)
    snap = sample[["t_sec", "err_base", "err_ess", "delta_err"]].copy()
    snap["better"] = np.where(snap["delta_err"] > 0, "ESSENTIAL", np.where(snap["delta_err"] < 0, "BASE", "TIE"))
    snap.to_csv(snapshot_path, index=False)

    # ===== Plotting (and save PNG) =====
    dt = np.median(np.diff(aligned["t_sec"].to_numpy()))
    win = max(1, int(round(args.smooth_sec / max(dt, 1e-9))))

    aligned["err_base_s"] = aligned["err_base"].rolling(win, center=True, min_periods=1).mean()
    aligned["err_ess_s"] = aligned["err_ess"].rolling(win, center=True, min_periods=1).mean()
    aligned["delta_s"] = aligned["delta_err"].rolling(win, center=True, min_periods=1).mean()

    # 1) errors over time
    fig1 = plt.figure()
    plt.plot(aligned["t_sec"], aligned["err_base_s"], label="BASE (smoothed)")
    plt.plot(aligned["t_sec"], aligned["err_ess_s"], label="ESSENTIAL (smoothed)")
    plt.xlabel("t_sec")
    plt.ylabel("err_m")
    plt.title("Error over time (smoothed)")
    plt.legend()
    plt.grid(True)
    fig1_path = os.path.join(args.outdir, f"plot_error_over_time__{tag}.png")
    fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")

    # 2) delta over time
    fig2 = plt.figure()
    plt.plot(aligned["t_sec"], aligned["delta_s"])
    plt.axhline(0.0)
    plt.xlabel("t_sec")
    plt.ylabel("Δerr = base - essential (m)")
    plt.title("Improvement over time (positive = ESSENTIAL better)")
    plt.grid(True)
    fig2_path = os.path.join(args.outdir, f"plot_delta_over_time__{tag}.png")
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")

    # Print where files are
    print("\n=== Saved outputs ===")
    print(f"Summary CSV:          {summary_path}")
    print(f"Aligned series CSV:   {aligned_path}")
    print(f"{args.points}-pt CSV:        {snapshot_path}")
    print(f"Plot (errors):        {fig1_path}")
    print(f"Plot (delta):         {fig2_path}")

    # Also show plots interactively (optional, still useful when running locally)
    plt.show()


if __name__ == "__main__":
    main()
