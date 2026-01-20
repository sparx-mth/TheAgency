import numpy as np
import cv2

def depth_to_vis_u8(depth_m: np.ndarray, clip_min=0.3, clip_max=50.0) -> np.ndarray:
    d = depth_m.copy().astype(np.float32)
    d[~np.isfinite(d)] = 0.0
    d = np.clip(d, clip_min, clip_max)
    # normalize to 0..255
    u8 = ((d - clip_min) / max(1e-6, (clip_max - clip_min)) * 255.0).astype(np.uint8)
    return u8

def make_depth_grid_vis(depth_m: np.ndarray, grid_w: int, grid_h: int,
                        clip_min=0.3, clip_max=50.0) -> np.ndarray:
    """
    heatmap of depth values based on grid_w, grid_h
    """
    H, W = depth_m.shape[:2]
    d = depth_m.astype(np.float32)
    d[~np.isfinite(d)] = 0.0

    # downsample with INTER_AREA ~= average
    small = cv2.resize(d, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

    # clip+normalize
    small = np.clip(small, clip_min, clip_max)
    small_u8 = ((small - clip_min) / max(1e-6, (clip_max - clip_min)) * 255.0).astype(np.uint8)

    big = cv2.resize(small_u8, (W, H), interpolation=cv2.INTER_NEAREST)

    # colormap
    colored = cv2.applyColorMap(big, cv2.COLORMAP_TURBO)

    for x in np.linspace(0, W, grid_w + 1).astype(int):
        cv2.line(colored, (x, 0), (x, H - 1), (50, 50, 50), 1)
    for y in np.linspace(0, H, grid_h + 1).astype(int):
        cv2.line(colored, (0, y), (W - 1, y), (50, 50, 50), 1)

    return colored
