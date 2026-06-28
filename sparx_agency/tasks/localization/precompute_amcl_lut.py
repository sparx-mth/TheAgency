"""Precompute and cache the AMCL ray-cast LUT.

Run this offline (on laptop or any fast machine) before starting the AMCL node.
Copy the resulting amcl_lut.npy to Jetson alongside the map files.

Usage:
  python3 -m sparx_agency.tasks.localization.precompute_amcl_lut \\
    --map-dir /tmp/xtend_map/ \\
    --beams 64 --orientations 32 --max-range-m 8.0

Output:
  <map-dir>/amcl_lut.npy  — shape (H, W, orientations, beams), float32
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sparx_agency.tasks.localization.amcl import ray_cast_lut_vectorized


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-dir", required=True,
                    help="Directory containing occ_grid_int8.npy and occ_metadata.json")
    ap.add_argument("--beams", type=int, default=64,
                    help="Number of range beams (default 64)")
    ap.add_argument("--orientations", type=int, default=32,
                    help="Number of heading bins 0..2π (default 32)")
    ap.add_argument("--max-range-m", type=float, default=8.0,
                    help="Maximum sensor range in metres (default 8.0)")
    ap.add_argument("--beam-fov-deg", type=float, default=180.0,
                    help="Total horizontal FOV spanned by beams in degrees (default 180)")
    ap.add_argument("--step", type=float, default=1.0,
                    help="Ray-march step in grid cells (default 1.0 = one cell per step)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing amcl_lut.npy if present")
    args = ap.parse_args()

    d = Path(args.map_dir)
    grid_path = d / "occ_grid_int8.npy"
    meta_path = d / "occ_metadata.json"
    lut_path = d / "amcl_lut.npy"

    if not grid_path.exists():
        sys.exit(f"ERROR: {grid_path} not found. Build the occupancy map first.")
    if not meta_path.exists():
        sys.exit(f"ERROR: {meta_path} not found.")
    if lut_path.exists() and not args.overwrite:
        sys.exit(f"ERROR: {lut_path} already exists. Pass --overwrite to replace it.")

    raw = np.load(str(grid_path))
    with open(meta_path) as f:
        meta = json.load(f)
    m_per_cell = float(meta["resolution_m_per_cell"])

    binary = (raw == 100).astype(np.float32)
    orientations = np.linspace(0, 2 * np.pi, args.orientations, endpoint=False)
    half_fov = np.deg2rad(args.beam_fov_deg / 2.0)
    beam_angles = np.linspace(-half_fov, half_fov, args.beams)
    max_range_cells = args.max_range_m / m_per_cell

    lut_shape = binary.shape + (args.orientations, args.beams)
    expected_mb = int(np.prod(lut_shape)) * 4 / 1e6

    print(f"Map:          {binary.shape}, {m_per_cell * 100:.1f} cm/cell")
    print(f"LUT shape:    {lut_shape}  (~{expected_mb:.0f} MB)")
    print(f"Max range:    {args.max_range_m} m = {max_range_cells:.1f} cells")
    print(f"Step:         {args.step} cell(s)")
    print(f"Output:       {lut_path}")
    print()

    lut = ray_cast_lut_vectorized(binary, orientations, beam_angles, max_range_cells,
                                  step=args.step)
    np.save(str(lut_path), lut)
    print(f"\nDone — saved {lut.nbytes / 1e6:.0f} MB to {lut_path}")


if __name__ == "__main__":
    main()