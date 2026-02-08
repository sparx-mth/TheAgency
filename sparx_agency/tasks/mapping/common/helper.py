import os
from typing import Optional

import cv2
import numpy as np

def depth_compare_report(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    name_a: str = "full",
    name_b: str = "tiled",
) -> dict:
    """
    Compare two depth maps in meters.
    Returns dict of stats + prints a compact report.
    """
    assert depth_a.shape == depth_b.shape, f"shape mismatch {depth_a.shape} vs {depth_b.shape}"

    a = depth_a.astype(np.float32)
    b = depth_b.astype(np.float32)

    finite = np.isfinite(a) & np.isfinite(b)
    if valid_mask is not None:
        finite = finite & valid_mask

    if not np.any(finite):
        print("DEPTH COMPARE: no finite overlap")
        return {"ok": False}

    da = a[finite]
    db = b[finite]

    diff = db - da
    absd = np.abs(diff)
    reld = absd / (np.abs(da) + 1e-6)

    def pct(x, p):
        return float(np.percentile(x, p))

    rep = {
        "ok": True,
        "finite_pct": float(finite.mean() * 100.0),
        "a_p01_p50_p99": (pct(da, 1), pct(da, 50), pct(da, 99)),
        "b_p01_p50_p99": (pct(db, 1), pct(db, 50), pct(db, 99)),
        "absdiff_p50_p90_p99": (pct(absd, 50), pct(absd, 90), pct(absd, 99)),
        "reldiff_p50_p90_p99": (pct(reld, 50), pct(reld, 90), pct(reld, 99)),
        "mean_absdiff": float(absd.mean()),
        "mean_reldiff": float(reld.mean()),
        "median_absdiff": float(np.median(absd)),
        "median_reldiff": float(np.median(reld)),
    }

    print("DEPTH COMPARE:")
    print(f"  overlap finite: {rep['finite_pct']:.2f}%")
    print(f"  {name_a} p01/p50/p99: {rep['a_p01_p50_p99']}")
    print(f"  {name_b} p01/p50/p99: {rep['b_p01_p50_p99']}")
    print(f"  absdiff p50/p90/p99: {rep['absdiff_p50_p90_p99']}  (m)")
    print(f"  reldiff p50/p90/p99: {rep['reldiff_p50_p90_p99']}  (ratio)")
    print(f"  mean absdiff: {rep['mean_absdiff']:.4f} m  mean reldiff: {rep['mean_reldiff']:.4f}")
    return rep


def save_depth_diff_visuals(
    out_dir: str,
    depth_full: np.ndarray,
    depth_tiled: np.ndarray,
    max_abs_m: float = 2.0,
):
    """
    Saves abs-diff and rel-diff as debug images (u8).
    """
    os.makedirs(out_dir, exist_ok=True)

    a = depth_full.astype(np.float32)
    b = depth_tiled.astype(np.float32)

    finite = np.isfinite(a) & np.isfinite(b)

    absd = np.zeros_like(a, dtype=np.float32)
    reld = np.zeros_like(a, dtype=np.float32)

    absd[finite] = np.abs(b[finite] - a[finite])
    reld[finite] = absd[finite] / (np.abs(a[finite]) + 1e-6)

    # abs diff viz: 0..max_abs_m => 0..255 (brighter = worse)
    absd_clip = np.clip(absd, 0.0, max_abs_m)
    abs_u8 = (255.0 * (absd_clip / max_abs_m)).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "depth_absdiff_vis.png"), abs_u8)

    # rel diff viz: 0..0.5 (50%) => 0..255
    max_rel = 0.5
    reld_clip = np.clip(reld, 0.0, max_rel)
    rel_u8 = (255.0 * (reld_clip / max_rel)).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "depth_reldiff_vis.png"), rel_u8)
