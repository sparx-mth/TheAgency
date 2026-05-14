#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stat", choices=["median", "mean"], default="median")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.stat == "median":
        grouped = (
            df.groupby("gt_m", as_index=False)["small_raw_median"]
            .median()
            .rename(columns={"small_raw_median": "small_raw"})
        )
    else:
        grouped = (
            df.groupby("gt_m", as_index=False)["small_raw_mean"]
            .mean()
            .rename(columns={"small_raw_mean": "small_raw"})
        )

    # Sort by raw value because np.interp expects x increasing.
    grouped = grouped.sort_values("small_raw")

    raw = grouped["small_raw"].to_numpy(dtype=np.float32)
    gt = grouped["gt_m"].to_numpy(dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        raw=raw,
        meters=gt,
        stat=args.stat,
    )

    print("Saved LUT:", out_path)
    print("\nLUT:")
    for r, z in zip(raw, gt):
        print(f"  raw={r:.6f} -> meters={z:.3f}")


if __name__ == "__main__":
    main()