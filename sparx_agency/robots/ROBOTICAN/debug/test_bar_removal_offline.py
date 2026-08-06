#!/usr/bin/env python3
"""Offline test for the moving cage-crossbar fix (bar_inpainter.py's dynamic
detector + vertical-median fill), against real recorded Rooster frames.

Static-arc TELEA inpainting is NOT exercised here -- that mask is disabled
(see NEXT_SESSION_PLAN.md) and is a separate fix from the one this checks.

For every frame in --in-dir:
  - detects crossbar bands (_detect_dynamic_bar_rows)
  - fills them (_fill_band_vertical_median)
  - saves a before|after side-by-side image, with the detected band(s)
    outlined in red on the "before" half, to --out-dir
  - logs one CSV row per detected band (frame, row range, thickness,
    intensity/std) so real thickness/darkness stats can inform whether
    _BAR_MIN_PX/_BAR_MAX_PX/_BAR_DARK_MAX/_BAR_ROWSTD_MAX need retuning --
    those were calibrated against a screen-recorded RViz panel, not the
    real camera feed.

Usage:
    python3 test_bar_removal_offline.py \\
        --in-dir /home/user1/Pictures/2026_01_27___12_30_15 \\
        --out-dir /tmp/bar_removal_test_out
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sparx_agency.robots.ROBOTICAN.bar_inpainter import (
    _detect_dynamic_bar_rows, _fill_band_vertical_median,
)


def _annotate_bands(bgr: np.ndarray, bands: list) -> np.ndarray:
    """Return a copy of bgr with each detected band outlined in red."""
    out = bgr.copy()
    for start, end in bands:
        cv2.rectangle(out, (0, start), (out.shape[1] - 1, end), (0, 0, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="Directory of real RGB frames (*.jpg/*.png)")
    ap.add_argument("--out-dir", default="/tmp/bar_removal_test_out")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.png"))
    if not frames:
        print(f"[ERROR] no .jpg/.png frames found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    csv_path = out_dir / "bands.csv"
    total_bands = 0
    frames_with_band = 0
    thicknesses = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "row_start", "row_end", "thickness_px",
                          "mean_intensity", "row_std"])

        for path in frames:
            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f"[WARN] could not read {path}", file=sys.stderr)
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            bands = _detect_dynamic_bar_rows(gray)

            after = bgr.copy()
            for start, end in bands:
                _fill_band_vertical_median(after, start, end)
                thickness = end - start + 1
                writer.writerow([path.name, start, end, thickness,
                                  round(float(gray[start:end + 1].mean()), 2),
                                  round(float(gray[start:end + 1].std()), 2)])
                thicknesses.append(thickness)
            total_bands += len(bands)
            frames_with_band += 1 if bands else 0

            before_annotated = _annotate_bands(bgr, bands)
            sep = np.full((bgr.shape[0], 4, 3), (0, 255, 0), dtype=np.uint8)
            combined = np.hstack([before_annotated, sep, after])
            cv2.imwrite(str(out_dir / f"{path.stem}_before_after.jpg"), combined)

    print(f"frames processed:     {len(frames)}")
    print(f"frames with a band:   {frames_with_band}")
    print(f"total bands detected: {total_bands}")
    if thicknesses:
        print(f"band thickness (px):  min={min(thicknesses)} "
              f"max={max(thicknesses)} mean={np.mean(thicknesses):.1f}")
    print(f"before|after images -> {out_dir}")
    print(f"per-band CSV        -> {csv_path}")


if __name__ == "__main__":
    main()
