import glob
import os
import sys
import cv2
import numpy as np
import time

from sparx_agency.robots.common.image_utils import create_hist_image_with_objects, estimate_floor_mask_from_bottom_band, \
    choose_near_bins, mask_by_bins, apply_mask_to_rgb, hist_to_bgr_image, compute_dynamic_delta_m

# Add project root to path so we can import sparx_agency
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from sparx_agency.core.mapping.depth import DepthAnythingV2DepthModel
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2Config
from sparx_agency.core.mapping.depth.depth_object_segmenter import DepthObjectSegmenter
from sparx_agency.robots.common import TicToc, get_objects_via_histogram


def get_depth_from_model(rgb_img, depth_model):
    # depth_model is now passed in to avoid reloading it every frame
    depth_img = depth_model.infer_depth(rgb_img).astype(np.float32)
    return depth_img


def make_depth_bin_overlays(
    depth_m: np.ndarray,
    min_dist: float = 0.3,
    max_dist: float = 5.0,
    bins: int = 60,
    invalid_value: float = 0.0,
    hist_size: tuple[int, int] = (400, 300),   # (W,H)
    alpha: float = 0.65,                       # overlay strength
) -> dict:
    """
    Build:
      1) histogram image (OpenCV)
      2) a "binned color" image: each depth-bin gets a unique color
      3) optional overlay for visualization

    Returns dict with:
      - color_bgr: (H,W,3) uint8
      - overlay_bgr: (H,W,3) uint8
      - hist_bgr: (hist_h, hist_w, 3) uint8
      - bin_edges: (bins+1,) float32
      - bin_colors_bgr: (bins,3) uint8  (color per bin)
      - bin_ids: (H,W) int32  (-1 for invalid/out-of-range)
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    H, W = depth.shape[:2]

    # Valid mask: finite, not invalid_value, and within range
    finite = np.isfinite(depth)
    not_invalid = depth != float(invalid_value)
    in_range = (depth >= min_dist) & (depth <= max_dist)
    valid = finite & not_invalid & in_range

    # Bin edges
    bin_edges = np.linspace(min_dist, max_dist, bins + 1, dtype=np.float32)

    # Bin IDs: -1 for invalid/out of range
    bin_ids = np.full((H, W), -1, dtype=np.int32)
    # digitize returns 1..bins where edges are [e0,e1)... last edge inclusive nuance
    # We'll clamp to 0..bins-1
    idx = np.digitize(depth[valid], bin_edges, right=False) - 1
    idx = np.clip(idx, 0, bins - 1)
    bin_ids[valid] = idx

    # Colors per bin (HSV hue sweep -> BGR)
    # Hue 0..179 in OpenCV. Spread across bins.
    hues = np.linspace(0, 179, bins, endpoint=False, dtype=np.uint8)
    hsv = np.stack([hues, np.full(bins, 220, np.uint8), np.full(bins, 255, np.uint8)], axis=1)  # (bins,3)
    hsv_img = hsv.reshape(1, bins, 3)
    bin_colors_bgr = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR).reshape(bins, 3)

    # Color image
    color_bgr = np.zeros((H, W, 3), dtype=np.uint8)
    valid_bins = bin_ids >= 0
    color_bgr[valid_bins] = bin_colors_bgr[bin_ids[valid_bins]]

    # Optional overlay on grayscale depth visualization (helps see structure)
    depth_norm = (np.clip(depth, min_dist, max_dist) - min_dist) / (max_dist - min_dist + 1e-6)
    depth_gray = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    depth_gray[~valid] = 0
    depth_gray_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
    overlay_bgr = cv2.addWeighted(depth_gray_bgr, 1.0 - alpha, color_bgr, alpha, 0.0)

    # Histogram (OpenCV drawing)
    hist_w, hist_h = hist_size
    hist_bgr = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    if np.any(valid):
        vals = depth[valid]
        counts, _ = np.histogram(vals, bins=bin_edges)
        counts = counts.astype(np.float32)
        cmax = float(counts.max()) if counts.max() > 0 else 1.0

        # Draw bars (filled) with the bin color
        for i in range(bins):
            x0 = int(i * hist_w / bins)
            x1 = int((i + 1) * hist_w / bins)
            bar_h = int((counts[i] / cmax) * (hist_h - 10))
            y0 = hist_h - 1
            y1 = hist_h - 1 - bar_h
            col = tuple(int(c) for c in bin_colors_bgr[i])
            cv2.rectangle(hist_bgr, (x0, y1), (x1 - 1, y0), col, thickness=-1)

        # Axis / labels (simple)
        cv2.putText(hist_bgr, f"{min_dist:.2f}m", (5, hist_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(hist_bgr, f"{max_dist:.2f}m", (hist_w - 90, hist_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return {
        "color_bgr": color_bgr,
        "overlay_bgr": overlay_bgr,
        "hist_bgr": hist_bgr,
        "bin_edges": bin_edges,
        "bin_colors_bgr": bin_colors_bgr,
        "bin_ids": bin_ids,
    }


def show_rgb_depth_hist_bins(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    min_dist: float = 0.3,
    max_dist: float = 5.0,
    bins: int = 60,
    window_name: str = "RGB | Depth-bins | Histogram",
    display_h: int = 250,
):
    """
    Shows: [RGB | binned-color overlay | histogram]
    Uses only cv2.imshow (no matplotlib).
    """
    out = make_depth_bin_overlays(depth_m, min_dist=min_dist, max_dist=max_dist, bins=bins)

    rgb = rgb_bgr.copy()
    bins_overlay = out["overlay_bgr"]
    hist = out["hist_bgr"]

    H, W = rgb.shape[:2]
    disp_w = int(display_h * (W / float(H)))

    rgb_r = cv2.resize(rgb, (disp_w, display_h))
    bins_r = cv2.resize(bins_overlay, (disp_w, display_h))
    hist_r = cv2.resize(hist, (disp_w, display_h))

    combined = np.hstack([rgb_r, bins_r, hist_r])
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


# --- Main Logic ---

def create_hist_image(depth_map, min_dist=0.1, max_dist=4.0, bins=50):
    h_hist, w_hist = 200, 400
    hist_img = np.zeros((h_hist, w_hist, 3), dtype=np.uint8)

    # Use the FULL depth map
    valid_pixels = depth_map[(depth_map >= min_dist) & (depth_map <= max_dist)]

    if len(valid_pixels) == 0:
        cv2.putText(hist_img, "No Data", (w_hist // 3, h_hist // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return hist_img

    hist, _ = np.histogram(valid_pixels, bins=bins, range=(min_dist, max_dist))

    if hist.max() > 0:
        # Normalize bars to the height of the hist_img
        hist_norm = (hist / hist.max() * (h_hist - 20)).astype(int)
        bin_w = w_hist // bins
        for i in range(bins):
            # Draw cyan bars for the histogram
            cv2.rectangle(hist_img, (i * bin_w, h_hist - hist_norm[i]),
                          ((i + 1) * bin_w, h_hist), (255, 255, 0), -1)

    return hist_img

# def _to_bgr(img: np.ndarray) -> np.ndarray:
#     """Ensure image is BGR uint8 for cv2.hstack."""
#     if img is None:
#         raise ValueError("One of the display images is None")
#
#     if img.ndim == 2:  # grayscale
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     elif img.ndim == 3 and img.shape[2] == 4:  # BGRA
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#
#     if img.dtype != np.uint8:
#         # Best-effort conversion for display
#         img = np.clip(img, 0, 255).astype(np.uint8)
#
#     return img

def _to_panel_img(x, target_h: int = 400) -> np.ndarray:
    """
    Convert x to a displayable BGR uint8 image.
    Supports:
      - BGR images (H,W,3)
      - grayscale images (H,W)
      - boolean masks (H,W) or list-of-lists
      - 1D histograms (N,)
      - scalars (float/int/bool) -> text tile
    """
    # 1) Scalars -> text tile
    if isinstance(x, (float, int, bool, np.number)):
        img = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        cv2.putText(img, str(x), (10, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    # 2) Convert to numpy
    arr = np.asarray(x)

    # 3) 1D -> histogram bars
    if arr.ndim == 1:
        hist = arr.astype(np.float32).reshape(-1)
        w = target_h
        h = target_h
        img = np.zeros((h, w, 3), dtype=np.uint8)
        mx = float(hist.max()) if hist.size and float(hist.max()) > 0 else 1.0
        hist_n = hist / mx
        n = hist_n.size
        for i in range(n):
            x1 = int(i * w / n)
            x2 = int((i + 1) * w / n)
            y2 = int((1.0 - float(hist_n[i])) * (h - 1))
            cv2.rectangle(img, (x1, y2), (max(x1 + 1, x2 - 1), h - 1), (255, 255, 255), -1)
        return img

    # 4) 2D -> grayscale/mask
    if arr.ndim == 2:
        # Boolean mask: show True=255, False=0
        if arr.dtype == bool:
            gray = (arr.astype(np.uint8) * 255)
        else:
            # Normalize other 2D arrays for display
            a = arr.astype(np.float32)
            mn = float(np.nanmin(a))
            mx = float(np.nanmax(a))
            if mx > mn:
                a = (a - mn) / (mx - mn)
            a = np.nan_to_num(a, nan=0.0)
            gray = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 5) 3D image
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            mn = float(np.nanmin(a))
            mx = float(np.nanmax(a))
            if mx > mn:
                a = (a - mn) / (mx - mn)
            a = np.nan_to_num(a, nan=0.0)
            arr = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
        return arr

    raise ValueError(f"Unsupported display type/shape: type={type(x)} shape={getattr(arr, 'shape', None)}")


def display_panel(display_dict: dict, window_prefix="Detection Pipeline", display_h=400):
    items = list(display_dict.items())
    titles = [k for k, _ in items]

    imgs = []
    for k, v in items:
        img = _to_panel_img(v, target_h=display_h)
        h, w = img.shape[:2]
        new_w = int(display_h * (w / float(h)))
        imgs.append(cv2.resize(img, (new_w, display_h), interpolation=cv2.INTER_AREA))

    combined = np.hstack(imgs)
    title = f"{window_prefix}: " + " | ".join(titles)
    cv2.imshow(title, combined)
    cv2.waitKey(0)

def overlay_mask(bgr, mask, alpha=0.5):
    mask_u8 = (np.asarray(mask).astype(np.uint8) * 255)
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(bgr, 1 - alpha, mask_bgr, alpha, 0)


def process_frame(image_path_or_array, depth_model):
    # 1. Load Image
    if isinstance(image_path_or_array, str):
        rgb_img = cv2.imread(image_path_or_array)
    else:
        rgb_img = image_path_or_array

    if rgb_img is None: return

    # 2. Inference
    with TicToc("Inference"):
        depth_map = get_depth_from_model(rgb_img, depth_model)

    # 3. Object Clustering (using your fixed function)
    with TicToc("Clustering"):
        min_d, max_d, bins = 0.3, 5.0, 60
        # objs_k = get_objects_via_depth_kmeans(depth_map, min_dist=0.3, max_dist=5.0, K=3)
        # objs_e = get_objects_via_depth_edges(depth_map, min_dist=0.3, max_dist=5.0)
        objs = get_objects_via_histogram(depth_map, min_dist=0.3, max_dist=5.0, bins=bins)
        # objs = merge_boxes(objs_k + objs_e, iou_thr=0.5)

        # 4. Panel 1: RGB + Overlay
    h, w = rgb_img.shape[:2]
    overlay = rgb_img.copy()
    for obj in objs:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, f"{obj['avg_depth']:.1f}m", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 5. Panel 2: Depth Visualization
    # # Normalize depth map to 0-255 for visibility
    # depth_viz = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # depth_viz = 255 - depth_viz
    #
    # depth_raw_view = (depth_map - 0.5) / (5.0 - 0.5)
    # depth_raw_view = np.clip(depth_raw_view, 0, 1)
    # depth_viz_grayscale = (depth_raw_view * 255).astype(np.uint8)
    # depth_viz_bgr = cv2.cvtColor(depth_viz_grayscale, cv2.COLOR_GRAY2BGR)
    # 6. Panel 3: Histogram

    hist_viz = create_hist_image_with_objects(depth_map, objs, min_dist=0.3, max_dist=5.0, bins=60)

def object_mask_from_floor(depth_m: np.ndarray,
                           floor_profile: np.ndarray,
                           delta_m: float) -> np.ndarray:
    H, W = depth_m.shape
    floor2d = floor_profile[:, None]  # (H,1) broadcast
    residual = floor2d - depth_m
    mask = residual > delta_m
    return mask.astype(np.uint8)  # 0/1


def close_objects_filter(depth_m: np.ndarray, bgr: np.ndarray):
    """
    Removes floor and keeps only objects ABOVE the floor using dynamic delta.
    """

    floor_mask, floor_profile = estimate_floor_mask_from_bottom_band(
        depth_m,
        min_dist=0.3,
        max_dist=5.0,
        bottom_band_frac=0.25,
        close_thresh_m=0.10,
    )

    # 2) Residuals ONLY from floor
    residual = floor_profile[:, None] - depth_m
    floor_residuals = residual[floor_mask > 0]

    # 3) Dynamic delta
    delta_m = compute_dynamic_delta_m(
        floor_residuals,
        base_m=0.03,
        k=4.0,
    )

    # 4) Object mask
    obj_mask = object_mask_from_floor(
        depth_m,
        floor_profile,
        delta_m,
    )

    # 5) Clean RGB
    rgb_clean = apply_mask_to_rgb(bgr, obj_mask, bg_color=(0, 0, 0))

    return {
        "delta_m": delta_m,
        "floor_mask": overlay_mask(bgr, floor_mask),
        "object_mask": overlay_mask(bgr, obj_mask),
        "rgb_clean": rgb_clean,
    }



if __name__ == "__main__":
    folder_path = "/home/user1/Pictures/2026_01_27___12_30_15/"
    imgs_list = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    
    # Init models once
    d_config = DepthAnythingV2Config()
    d_model = DepthAnythingV2DepthModel(d_config)

    for test_image in imgs_list:
        # process_frame(test_image, d_model)
        # with TicToc("Inference"):
        #     rgb_img = cv2.imread(test_image)
        #     depth_map = get_depth_from_model(rgb_img, d_model)
        #
        # show_rgb_depth_hist_bins(
        #     rgb_bgr=rgb_img,
        #     depth_m=depth_map,
        #     min_dist=0.3,
        #     max_dist=5.0,
        #     bins=150,
        # )
        with TicToc("Depth Extraction"):
            rgb_img = cv2.imread(test_image)
            depth_map = get_depth_from_model(rgb_img, d_model)

        with TicToc("Close Objects Filter"):
            display_dict = close_objects_filter(depth_map, rgb_img)

        rgb_clean = display_dict["rgb_clean"]
        delta_m = display_dict["delta_m"]

        print(f"[Close Objects Filter] delta = {delta_m:.3f} m")

        # with TicToc("Display Panel"):
        display_panel(display_dict)
