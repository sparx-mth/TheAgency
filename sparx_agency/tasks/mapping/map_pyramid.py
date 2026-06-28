import argparse
import os
from typing import Optional

import cv2
import numpy as np

# import your model
from sparx_agency.core.mapping.depth.depth_anything_v2 import (
    DepthAnythingV2DepthModel,
    DepthAnythingV2Config,
)

#import os, cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

# ---------- helpers ----------
def save_u8(path: str, depth_m: np.ndarray, p01=1, p99=99) -> None:
    d = depth_m.astype(np.float32)
    m = np.isfinite(d)
    out = np.zeros(d.shape, dtype=np.uint8)
    if m.any():
        lo = float(np.percentile(d[m], p01))
        hi = float(np.percentile(d[m], p99))
        x = (d - lo) / max(hi - lo, 1e-6)
        x = np.clip(x, 0, 1)
        out[m] = (255.0 * (1.0 - x[m])).astype(np.uint8)  # near bright
    cv2.imwrite(path, out)

def depth_compare_report(A: np.ndarray, B: np.ndarray, name_a="A", name_b="B") -> Dict:
    a = A.astype(np.float32)
    b = B.astype(np.float32)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"finite_overlap": 0.0}
    aa = a[m]; bb = b[m]
    absd = np.abs(aa - bb)
    reld = absd / np.maximum(aa, 1e-6)
    rep = {
        "finite_overlap": float(m.mean()) * 100.0,
        f"{name_a}_p01_p50_p99": tuple(np.percentile(aa, [1,50,99]).tolist()),
        f"{name_b}_p01_p50_p99": tuple(np.percentile(bb, [1,50,99]).tolist()),
        "absdiff_p50_p90_p99": tuple(np.percentile(absd, [50,90,99]).tolist()),
        "reldiff_p50_p90_p99": tuple(np.percentile(reld, [50,90,99]).tolist()),
        "mean_absdiff": float(absd.mean()),
        "mean_reldiff": float(reld.mean()),
    }
    return rep

def blend_weights(h: int, w: int, border: int) -> np.ndarray:
    border = max(1, int(border))
    yy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    xx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    wy = np.clip(yy / border, 0.0, 1.0)
    wx = np.clip(xx / border, 0.0, 1.0)
    return np.outer(wy, wx).astype(np.float32)

def iter_tiles_2x2(W: int, H: int, overlap_x: int, overlap_y: int):
    # 2x2 tiles with overlap, covering whole image
    mid_x = W // 2
    mid_y = H // 2
    # tile extents (ensure overlap around midlines)
    x0a = 0
    x1a = min(W, mid_x + overlap_x)
    x0b = max(0, mid_x - overlap_x)
    x1b = W

    y0a = 0
    y1a = min(H, mid_y + overlap_y)
    y0b = max(0, mid_y - overlap_y)
    y1b = H

    # (x0,y0,x1,y1)
    return [
        (x0a, y0a, x1a, y1a),  # TL
        (x0b, y0a, x1b, y1a),  # TR
        (x0a, y0b, x1a, y1b),  # BL
        (x0b, y0b, x1b, y1b),  # BR
    ]

def resize_keep_aspect(rgb: np.ndarray, short_side: int) -> np.ndarray:
    H, W = rgb.shape[:2]
    if H < W:
        new_h = short_side
        new_w = int(round(W * (short_side / H)))
    else:
        new_w = short_side
        new_h = int(round(H * (short_side / W)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ---------- experiment ----------
@dataclass
class ExpCfg:
    overlap_div: int = 4              # overlap = tile_size/overlap_div
    model_w: Optional[int] = 518      # or None
    model_h: Optional[int] = 518      # or None
    low_short_side: int = 518         # low-res "one tile" short side
    fade_ratio: float = 0.5           # fade border = fade_ratio*overlap
    global_norm: bool = True          # if your wrapper supports norm_stats

def infer_depth_single(rgb: np.ndarray, depth_model, model_w: Optional[int], model_h: Optional[int],
                      norm_stats=None) -> np.ndarray:
    if model_w is not None and model_h is not None:
        rgb_in = cv2.resize(rgb, (model_w, model_h), interpolation=cv2.INTER_AREA)
        d = depth_model.infer_depth(rgb_in, norm_stats=norm_stats).astype(np.float32)
        d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return d
    else:
        # no forced reshape
        return depth_model.infer_depth(rgb, norm_stats=norm_stats).astype(np.float32)

def infer_depth_high4(rgb_full: np.ndarray, depth_model, cfg: ExpCfg, norm_stats=None) -> np.ndarray:
    H, W = rgb_full.shape[:2]
    # overlap derived from "tile size" = half-image in this 2x2 scheme
    tile_w = int(np.ceil(W / 2))
    tile_h = int(np.ceil(H / 2))
    overlap_x = tile_w // cfg.overlap_div
    overlap_y = tile_h // cfg.overlap_div
    fade = max(16, int(cfg.fade_ratio * min(overlap_x, overlap_y)))

    depth_sum = np.zeros((H, W), np.float32)
    w_sum = np.zeros((H, W), np.float32)

    for (x0, y0, x1, y1) in iter_tiles_2x2(W, H, overlap_x, overlap_y):
        tile = rgb_full[y0:y1, x0:x1]
        th, tw = tile.shape[:2]

        # run tile inference (optionally forcing model_w/h)
        d_tile = infer_depth_single(tile, depth_model, cfg.model_w, cfg.model_h, norm_stats=norm_stats)

        Wt = blend_weights(th, tw, border=fade)
        depth_sum[y0:y1, x0:x1] += d_tile * Wt
        w_sum[y0:y1, x0:x1] += Wt

    return depth_sum / (w_sum + 1e-6)

def infer_depth_low1(rgb_full: np.ndarray, depth_model, cfg: ExpCfg, norm_stats=None) -> np.ndarray:
    H, W = rgb_full.shape[:2]
    rgb_small = resize_keep_aspect(rgb_full, cfg.low_short_side)
    d_small = depth_model.infer_depth(rgb_small, norm_stats=norm_stats).astype(np.float32)
    d_full = cv2.resize(d_small, (W, H), interpolation=cv2.INTER_LINEAR)
    return d_full

def run_pyramid_experiment(rgb_full: np.ndarray, depth_model, out_dir: str,
                          grid: List[ExpCfg]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # baseline full (no forced resize)
    norm_stats_full = None
    if grid[0].global_norm and hasattr(depth_model, "infer_raw") and hasattr(depth_model, "raw_stats"):
        raw = depth_model.infer_raw(resize_keep_aspect(rgb_full, 518))
        norm_stats_full = depth_model.raw_stats(raw)

    depth_full = depth_model.infer_depth(rgb_full, norm_stats=norm_stats_full).astype(np.float32)
    save_u8(os.path.join(out_dir, "depth_full.png"), depth_full)

    # loop configs
    for i, cfg in enumerate(grid):
        tag = f"exp{i}_mw{cfg.model_w}_ov{cfg.overlap_div}_low{cfg.low_short_side}"

        # norm stats for this experiment (recommended: same norm for all passes inside exp)
        norm_stats = None
        if cfg.global_norm and hasattr(depth_model, "infer_raw") and hasattr(depth_model, "raw_stats"):
            raw = depth_model.infer_raw(resize_keep_aspect(rgb_full, cfg.low_short_side))
            norm_stats = depth_model.raw_stats(raw)

        d_high = infer_depth_high4(rgb_full, depth_model, cfg, norm_stats=norm_stats)
        d_low  = infer_depth_low1(rgb_full, depth_model, cfg, norm_stats=norm_stats)

        # simple fusion: low provides global scale, high provides detail
        # alpha can be tuned; start with 0.7 high
        alpha = 0.7
        d_pyr = alpha * d_high + (1.0 - alpha) * d_low

        # save
        save_u8(os.path.join(out_dir, f"{tag}_high4.png"), d_high)
        save_u8(os.path.join(out_dir, f"{tag}_low1.png"), d_low)
        save_u8(os.path.join(out_dir, f"{tag}_pyr.png"), d_pyr)

        # reports vs full
        r_high = depth_compare_report(depth_full, d_high, "full", "high4")
        r_low  = depth_compare_report(depth_full, d_low,  "full", "low1")
        r_pyr  = depth_compare_report(depth_full, d_pyr,  "full", "pyr")

        print("\n====", tag, "====")
        print("full vs high4:", r_high)
        print("full vs low1 :", r_low)
        print("full vs pyr  :", r_pyr)

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to RGB image")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile-w", type=int, default=640)
    ap.add_argument("--tile-h", type=int, default=384)
    ap.add_argument("--overlap-div", type=int, default=4)  # overlap = tile/overlap_div
    ap.add_argument("--model-w", type=int, default=518)
    ap.add_argument("--model-h", type=int, default=518)
    ap.add_argument("--max-absdiff-m", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth_model = DepthAnythingV2DepthModel(DepthAnythingV2Config())

    # A) full baseline (your current path): infer on full image once
    depth_full = depth_model.infer_depth(rgb, norm_stats=None).astype(np.float32)
    out_dir = args.out_dir
    grid = [
        ExpCfg(overlap_div=4, model_w=518, model_h=518, low_short_side=518, global_norm=True),
        ExpCfg(overlap_div=4, model_w=None, model_h=None, low_short_side=518, global_norm=True),
        ExpCfg(overlap_div=3, model_w=518, model_h=518, low_short_side=518, global_norm=True),
        ExpCfg(overlap_div=5, model_w=518, model_h=518, low_short_side=518, global_norm=True),
        # try a bit different low-res
        ExpCfg(overlap_div=4, model_w=518, model_h=518, low_short_side=384, global_norm=True),
    ]

    run_pyramid_experiment(rgb, depth_model, out_dir, grid)

if __name__ == "__main__":
    main()
