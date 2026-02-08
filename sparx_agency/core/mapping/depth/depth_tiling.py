# sparx_agency/core/mapping/depth/depth_tiling.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
import numpy as np
import cv2

from sparx_agency.core.common.types.perception import Intrinsics  # adjust import if needed


from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TileCfg:
    tile_w: int = 640
    tile_h: int = 384

    # overlap_div=4 => overlap is 1/4 of tile size
    overlap_div: int = 4

    # derived (filled in __post_init__)
    overlap_x: int = field(init=False)
    overlap_y: int = field(init=False)
    step_x: int = field(init=False)
    step_y: int = field(init=False)

    # model fixed input size (DepthAnything paper uses 518x518 crops; runtime can be different)
    model_w: Optional[int] = None
    model_h: Optional[int] = None

    # if your DepthAnything wrapper does global min/max normalization
    global_norm_from_full_pass: bool = True

    def __post_init__(self) -> None:
        if self.overlap_div <= 0:
            raise ValueError("overlap_div must be > 0")

        self.overlap_x = self.tile_w // self.overlap_div
        self.overlap_y = self.tile_h // self.overlap_div

        self.step_x = max(1, self.tile_w - self.overlap_x)
        self.step_y = max(1, self.tile_h - self.overlap_y)

        if self.model_w is not None and self.model_w <= 0:
            raise ValueError("model_w must be positive or None")
        if self.model_h is not None and self.model_h <= 0:
            raise ValueError("model_h must be positive or None")



def iter_tiles(W: int, H: int, cfg: TileCfg) -> Iterator[Tuple[int, int, int, int]]:
    step_x = max(1, cfg.tile_w - cfg.overlap_x)
    step_y = max(1, cfg.tile_h - cfg.overlap_y)

    y = 0
    while True:
        if y + cfg.tile_h >= H:
            y = max(0, H - cfg.tile_h)
        x = 0
        while True:
            if x + cfg.tile_w >= W:
                x = max(0, W - cfg.tile_w)
            x1 = min(W, x + cfg.tile_w)
            y1 = min(H, y + cfg.tile_h)
            yield (x, y, x1, y1)
            if x + cfg.tile_w >= W:
                break
            x += step_x
        if y + cfg.tile_h >= H:
            break
        y += step_y


def tile_intrinsics(intr: Intrinsics, x0: int, y0: int, tile_w: int, tile_h: int) -> Intrinsics:
    # crop-only intrinsics (same resolution)
    return Intrinsics(
        width=tile_w,
        height=tile_h,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.cx - float(x0),
        cy=intr.cy - float(y0),
    )


def resize_intrinsics(intr: Intrinsics, new_w: int, new_h: int) -> Intrinsics:
    sx = new_w / intr.width
    sy = new_h / intr.height
    return Intrinsics(
        width=new_w, height=new_h,
        fx=intr.fx * sx, fy=intr.fy * sy,
        cx=intr.cx * sx, cy=intr.cy * sy,
    )


def blend_weights(h: int, w: int, border: int = 32) -> np.ndarray:
    # Smooth ramp down toward edges
    # border: width (pixels) of the fading zone
    border = max(1, int(border))
    yy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    xx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    wy = np.clip(yy / border, 0.0, 1.0)
    wx = np.clip(xx / border, 0.0, 1.0)
    W = np.outer(wy, wx).astype(np.float32)
    return W


def infer_depth_tiled(
    rgb_full: np.ndarray,
    intr_full: Intrinsics,
    depth_model,
    cfg: TileCfg,
    global_norm_from_full_pass: bool = True,
) -> Tuple[np.ndarray, Intrinsics]:
    """
    Returns:
      depth_full (H x W) in meters (your inferred mapping),
      intr_out matching that depth (same as intr_full).
    Assumes rgb_full is RGB uint8.
    """
    H, W = rgb_full.shape[:2]

    # ---- global norm stats (critical for tiling with your current model) ----
    norm_stats = None
    if global_norm_from_full_pass:
        # Low-res full pass is enough to get stable d_min/d_max
        small_w = 640
        small_h = int(round(H * (small_w / W)))
        rgb_small = cv2.resize(rgb_full, (small_w, small_h), interpolation=cv2.INTER_AREA)
        raw_small = depth_model.infer_raw(rgb_small)
        norm_stats = depth_model.raw_stats(raw_small)

    depth_sum = np.zeros((H, W), dtype=np.float32)
    w_sum = np.zeros((H, W), dtype=np.float32)

    # Choose fade border proportional to overlap
    fade = int(0.5 * min(cfg.overlap_x, cfg.overlap_y))
    fade = max(16, fade)

    for x0, y0, x1, y1 in iter_tiles(W, H, cfg):
        tile_rgb = rgb_full[y0:y1, x0:x1]
        th, tw = tile_rgb.shape[:2]

        # tile intrinsics (cropped)
        intr_tile = tile_intrinsics(intr_full, x0, y0, tw, th)

        # optional resize for model
        if cfg.model_w is not None and cfg.model_h is not None:
            tile_rgb_in = cv2.resize(tile_rgb, (cfg.model_w, cfg.model_h), interpolation=cv2.INTER_AREA)
            intr_tile_in = resize_intrinsics(intr_tile, cfg.model_w, cfg.model_h)
        else:
            tile_rgb_in = tile_rgb
            intr_tile_in = intr_tile

        # infer depth tile with SHARED norm_stats
        depth_tile = depth_model.infer_depth(tile_rgb_in, norm_stats=norm_stats).astype(np.float32)

        # if resized, bring depth back to tile native size for stitching
        if depth_tile.shape[1] != tw or depth_tile.shape[0] != th:
            depth_tile = cv2.resize(depth_tile, (tw, th), interpolation=cv2.INTER_LINEAR)

        Wt = blend_weights(th, tw, border=fade)

        depth_sum[y0:y1, x0:x1] += depth_tile * Wt
        w_sum[y0:y1, x0:x1] += Wt

    depth_full = depth_sum / (w_sum + 1e-6)
    return depth_full, intr_full
