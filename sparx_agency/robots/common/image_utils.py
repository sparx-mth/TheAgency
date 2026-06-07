from __future__ import annotations

import glob
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image


class BadFrameGuard:
    """
    Stateful guard that drops frames that are black, frozen, or malformed.
    Create one instance per node/source; call should_pass(frame) before processing.
    """

    def __init__(
        self,
        mean_min: float = 2.0,
        std_min: float = 1.0,
        sample_step: int = 16,
        log_every: int = 30,
        prefix: str = "",
    ):
        self.mean_min = float(mean_min)
        self.std_min = float(std_min)
        self.sample_step = max(1, int(sample_step))
        self.log_every = max(1, int(log_every))
        self._prefix = f"[{prefix}] " if prefix else ""
        self.bad_count = 0
        self.good_count = 0

    def check(self, frame) -> tuple[bool, str]:
        """Return (is_bad, reason). Pure check — no side effects."""
        if frame is None:
            return True, "frame is None"
        if not isinstance(frame, np.ndarray):
            return True, f"not ndarray: {type(frame).__name__}"
        if frame.size == 0:
            return True, "zero-size ndarray"
        if frame.ndim != 3 or frame.shape[2] != 3:
            return True, f"unexpected shape: {frame.shape}"
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return True, f"invalid shape: {frame.shape}"
        small = frame[:: self.sample_step, :: self.sample_step]
        if not np.isfinite(small).all():
            return True, "contains non-finite values"
        mean_val = float(small.mean())
        std_val = float(small.std())
        if mean_val < self.mean_min:
            return True, f"too dark/empty: mean={mean_val:.3f}"
        if std_val < self.std_min:
            return True, f"too flat/empty: std={std_val:.3f}"
        return False, f"mean={mean_val:.3f}, std={std_val:.3f}"

    def should_pass(self, frame) -> bool:
        """Return True if frame is good; logs drops and recovery."""
        is_bad, reason = self.check(frame)
        if is_bad:
            self.bad_count += 1
            self.good_count = 0
            if self.bad_count == 1 or self.bad_count % self.log_every == 0:
                print(f"{self._prefix}[drop] bad frame #{self.bad_count}: {reason}")
            return False
        self.good_count += 1
        if self.bad_count > 0:
            print(f"{self._prefix}recovered after {self.bad_count} dropped frame(s)")
            self.bad_count = 0
        return True

def list_frames(frames_dir: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(frames_dir, ext)))
    return sorted(files)


def center_crop_frac(img: np.ndarray, frac: float):
    frac = max(0.05, min(1.0, float(frac)))
    H, W = img.shape[:2]
    cw, ch = int(W * frac), int(H * frac)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    return img[y0:y0 + ch, x0:x0 + cw], (x0, y0, x0 + cw, y0 + ch)


def apply_crop_and_flip(img: np.ndarray, crop_frac: float, flip180: bool):
    work = img
    crop_box = None
    if crop_frac < 1.0:
        work, crop_box = center_crop_frac(img, crop_frac)
    if flip180:
        work = cv2.rotate(work, cv2.ROTATE_180)
    return work, crop_box


def ros_image_to_rgb_np(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
        return arr if enc == "rgb8" else arr[:, :, ::-1]
    if enc in ("rgba8", "bgra8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        return arr if enc == "rgba8" else arr[:, :, ::-1]
    if enc == "mono8":
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w))
        return np.stack([gray, gray, gray], axis=-1)
    raise ValueError(f"Unsupported Image encoding: {msg.encoding}")


def numpy_to_image_msg(arr: np.ndarray, *, frame_id: str, stamp, encoding: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = False

    if encoding.lower() in ("rgb8", "bgr8"):
        msg.step = msg.width * 3
    elif encoding.lower() == "mono8":
        msg.step = msg.width
    elif encoding.lower() == "32fc1":
        msg.step = msg.width * 4
        arr = arr.astype(np.float32)

    msg.data = arr.tobytes()
    return msg

import numpy as np
import cv2

def create_hist_image_with_objects(
    depth_map: np.ndarray,
    objects: list[dict],
    min_dist: float = 0.3,
    max_dist: float = 5.0,
    bins: int = 60,
    out_w: int = 640,
    out_h: int = 400,
):
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0) & (depth >= min_dist) & (depth <= max_dist)
    vals = depth[valid]
    img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    if vals.size == 0:
        cv2.putText(img, "No valid depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return img

    hist, edges = np.histogram(vals, bins=bins, range=(min_dist, max_dist))
    hist = hist.astype(np.float32)
    hmax = float(hist.max() + 1e-6)

    # draw histogram polyline
    pts = []
    for i in range(bins):
        x = int((i / (bins - 1)) * (out_w - 1))
        y = int(out_h - 1 - (hist[i] / hmax) * (out_h - 40))  # leave top margin for text
        pts.append((x, y))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (230, 230, 230), 2)
    print("vals:", vals.size,
          "median:", np.median(vals),
          "hist bins:", hist.sum())

    # helper to map depth->x
    def depth_to_x(d):
        t = (d - min_dist) / (max_dist - min_dist + 1e-9)
        t = float(np.clip(t, 0.0, 1.0))
        return int(t * (out_w - 1))

    # nice distinct colors (BGR)
    palette = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 200, 80),
        (200, 80, 255), (80, 255, 255), (255, 80, 200), (180, 180, 80),
        (80, 180, 180), (180, 80, 180),
    ]

    H, W = depth.shape[:2]
    image_area = H * W
    valid_area = int(valid.sum())

    # overlay object spans
    for i, obj in enumerate(objects):
        if "depth_range" in obj:
            dmin, dmax = obj["depth_range"]
        else:
            d = float(obj.get("avg_depth", 0.0))
            dmin, dmax = d - 0.05, d + 0.05

        x1 = depth_to_x(dmin)
        x2 = depth_to_x(dmax)
        if x2 <= x1:
            continue

        color = palette[i % len(palette)]

        # translucent band
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, 0), (x2, out_h - 1), color, -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        # boundary lines
        cv2.line(img, (x1, 0), (x1, out_h - 1), color, 2)
        cv2.line(img, (x2, 0), (x2, out_h - 1), color, 2)

        area = int(obj.get("area", 0))
        frac_img = 100.0 * area / float(max(1, image_area))
        frac_valid = 100.0 * area / float(max(1, valid_area))
        label = f"#{i} {area}px ({frac_img:.1f}% img, {frac_valid:.1f}% valid) [{dmin:.2f},{dmax:.2f}]"

        # label near the top, staggered
        ty = 18 + 18 * (i % 10)
        cv2.putText(img, label, (10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # axis labels (optional minimal)
    cv2.putText(img, f"{min_dist:.2f}m", (5, out_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, f"{max_dist:.2f}m", (out_w - 70, out_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return img

def get_objects_via_histogram(
    depth_img: np.ndarray,
    min_dist: float = 0.3,
    max_dist: float = 5.0,
    window_bins: int = 1,
    bins: int = 60,
    peak_prom_frac: float = 0.08,     # how "strong" a histogram peak must be (fraction of max)
    valley_frac: float = 0.35,        # stop range expansion when histogram falls below this * peak height
    max_ranges: int = 6,              # limit how many depth bands we turn into objects
    min_area_px: int = 800,           # reject tiny blobs
    close_kernel: int = 9,            # helps merge chair slats / small holes
    open_kernel: int = 3,             # removes pepper noise
    close_iters: int = 1,
    open_iters: int = 1,
) -> list[dict]:
    """
    Segment candidate objects from a depth image using histogram peaks (no predefined depth range),
    then connected components -> bounding boxes.

    Assumptions:
      - depth_img is float depth in meters (or anything consistent with min_dist/max_dist).
      - invalid pixels are 0, NaN, or Inf.

    Returns:
      detected_objects: list of dicts:
        {
          "bbox": (x1, y1, x2, y2),
          "avg_depth": float,
          "area": int,
          "depth_range": (d_min, d_max),
        }
    """
    if depth_img is None:
        return []
    depth = np.asarray(depth_img).astype(np.float32)

    # 1) Valid mask
    valid = np.isfinite(depth) & (depth > 0) & (depth >= min_dist) & (depth <= max_dist)
    if valid.sum() < 100:  # not enough data
        return []

    vals = depth[valid]

    # 2) Histogram in depth-space
    hist, edges = np.histogram(vals, bins=bins, range=(min_dist, max_dist))
    # smooth the histogram a bit so peaks are stable
    k = 5
    kernel = np.ones(k, dtype=np.float32) / k
    hist_s = np.convolve(hist.astype(np.float32), kernel, mode="same")

    # 3) Find local maxima (simple peak picking)
    max_h = float(hist_s.max() + 1e-6)
    peaks = []
    for i in range(1, len(hist_s) - 1):
        if hist_s[i] >= hist_s[i - 1] and hist_s[i] >= hist_s[i + 1]:
            if hist_s[i] >= peak_prom_frac * max_h:
                peaks.append(i)

    if not peaks:
        return []

    # Prefer nearer peaks first (foreground-first), because background often dominates
    peaks = sorted(peaks, key=lambda idx: (edges[idx] + edges[idx + 1]) * 0.5)
    peaks = peaks[:max_ranges]

    # 4) For each peak, expand left/right until reaching a "valley"
    ranges = []
    for p in peaks:
        peak_h = hist_s[p]
        thr = valley_frac * peak_h

        l = p
        while l > 0 and hist_s[l] > thr:
            l -= 1
        r = p
        while r < len(hist_s) - 1 and hist_s[r] > thr:
            r += 1

        d_min = float(edges[l])
        d_max = float(edges[r + 1] if (r + 1) < len(edges) else edges[-1])
        if d_max - d_min <= 1e-6:
            continue
        ranges.append((d_min, d_max))

    # Merge overlapping/nearby ranges (common when histogram is noisy)
    ranges.sort()
    merged = []
    for a, b in ranges:
        if not merged:
            merged.append([a, b])
        else:
            pa, pb = merged[-1]
            if a <= pb + 0.02:  # 2cm gap merge (tweak if needed)
                merged[-1][1] = max(pb, b)
            else:
                merged.append([a, b])
    ranges = [(a, b) for a, b in merged][:max_ranges]

    # 5) Build objects by connected components inside each depth band
    detected_objects = []
    H, W = depth.shape[:2]

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))

    for d_min, d_max in ranges:
        band = valid & (depth >= d_min) & (depth <= d_max)
        mask = (band.astype(np.uint8) * 255)

        # Fill small holes / connect slats
        if close_kernel > 1 and close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iters)
        if open_kernel > 1 and open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=open_iters)

        # Connected components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num):  # 0 is background
            x, y, w_obj, h_obj, area = stats[i]
            if area < min_area_px:
                continue

            # Avg depth from original depth (only where this component is valid)
            comp = (labels == i) & valid
            if comp.sum() == 0:
                continue
            avg_depth = float(np.mean(depth[comp]))

            detected_objects.append({
                "bbox": (int(x), int(y), int(x + w_obj), int(y + h_obj)),
                "avg_depth": avg_depth,
                "area": int(area),
                "depth_range": (float(d_min), float(d_max)),
            })

    # 6) Optional: remove near-duplicate boxes (same thing detected in overlapping ranges)
    detected_objects.sort(key=lambda o: (o["bbox"][0], o["bbox"][1], -o["area"]))
    final = []
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        area = max(1, (x2 - x1) * (y2 - y1))

        keep = True
        for prev in final:
            px1, py1, px2, py2 = prev["bbox"]
            ix1, iy1 = max(x1, px1), max(y1, py1)
            ix2, iy2 = min(x2, px2), min(y2, py2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            parea = max(1, (px2 - px1) * (py2 - py1))
            iou = inter / float(area + parea - inter + 1e-6)
            if iou > 0.6:
                keep = False
                break

        if keep:
            final.append(obj)

    return final

def _finite_mask(depth_m: np.ndarray) -> np.ndarray:
    return np.isfinite(depth_m) & (depth_m > 0)

def estimate_floor_mask_from_bottom_band(
    depth_m: np.ndarray,
    min_dist: float = 0.3,
    max_dist: float = 5.0,
    bottom_band_frac: float = 0.22,
    x_trim_frac: float = 0.06,
    close_thresh_m: float = 0.08,
    smooth_ksize: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a floor mask using per-row robust depth from a bottom band.
    Returns:
      floor_mask: bool[H,W]
      floor_profile: float[H] depth per row (nan where unknown)
    """
    H, W = depth_m.shape[:2]
    d = depth_m.astype(np.float32).copy()

    # Clamp to working range
    d[~_finite_mask(d)] = np.nan
    d[(d < min_dist) | (d > max_dist)] = np.nan

    y0 = int(H * (1.0 - bottom_band_frac))
    x0 = int(W * x_trim_frac)
    x1 = int(W * (1.0 - x_trim_frac))

    floor_profile = np.full((H,), np.nan, dtype=np.float32)

    # Robust per-row estimate from trimmed x-range (median is robust)
    for y in range(y0, H):
        row = d[y, x0:x1]
        row = row[np.isfinite(row)]
        if row.size < 50:
            continue
        floor_profile[y] = np.median(row)

    # Smooth profile (ignore NaNs by simple interpolation)
    ys = np.arange(H)
    valid = np.isfinite(floor_profile)
    if valid.sum() >= 10:
        interp = np.interp(ys, ys[valid], floor_profile[valid]).astype(np.float32)
        if smooth_ksize and smooth_ksize >= 3:
            if smooth_ksize % 2 == 0:
                smooth_ksize += 1
            interp = cv2.GaussianBlur(interp.reshape(-1, 1), (1, smooth_ksize), 0).reshape(-1)
        floor_profile = interp
    else:
        # Not enough signal, return empty mask
        return np.zeros((H, W), dtype=bool), floor_profile

    # Floor pixels are those close to the floor profile for their row
    floor_mask = np.zeros((H, W), dtype=bool)
    for y in range(y0, H):
        fy = floor_profile[y]
        if not np.isfinite(fy):
            continue
        floor_mask[y, :] = np.isfinite(d[y, :]) & (np.abs(d[y, :] - fy) <= close_thresh_m)

    return floor_mask, floor_profile

def compute_dynamic_delta_m(
    floor_residuals: np.ndarray,
    base_m: float = 0.03,
    k: float = 4.0,
    clip=(0.03, 0.25),
) -> float:
    """
    Computes a dynamic depth delta (meters) based on floor residual statistics.

    floor_residuals = floor_depth - depth (only where floor_mask==1)
    """
    r = floor_residuals[np.isfinite(floor_residuals)]
    if r.size < 200:
        return float(np.clip(base_m, clip[0], clip[1]))

    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-6
    sigma = 1.4826 * mad  # robust std

    delta = base_m + k * sigma
    return float(np.clip(delta, clip[0], clip[1]))

def object_mask_from_floor(
    depth_m: np.ndarray,
    floor_profile: np.ndarray,
    delta_m: float,
    floor_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns binary mask of objects ABOVE the floor.
    """
    floor_2d = floor_profile[:, None]  # broadcast
    residual = floor_2d - depth_m

    mask = residual > delta_m

    if floor_mask is not None:
        mask[floor_mask > 0] = 0

    return mask.astype(np.uint8)
#
# def apply_mask_to_rgb(bgr: np.ndarray, mask: np.ndarray, bg_color=(0, 0, 0)):
#     out = np.zeros_like(bgr)
#     out[:] = bg_color
#     out[mask > 0] = bgr[mask > 0]
#     return out

def choose_near_bins(
    depth_m: np.ndarray,
    min_dist: float = 0.3,
    max_dist: float = 5.0,
    bins: int = 80,
    k_near: int = 2,
    min_bin_frac: float = 0.01,
    ignore_floor_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick K near (small depth) histogram bins that have enough mass.
    Returns:
      bin_edges: float[bins+1]
      hist: float[bins]
      chosen_bin_ids: int[k'] (k' <= k_near)
    """
    d = depth_m.astype(np.float32)
    mask = _finite_mask(d) & (d >= min_dist) & (d <= max_dist)
    if ignore_floor_mask is not None:
        mask = mask & (~ignore_floor_mask.astype(bool))

    vals = d[mask]
    if vals.size == 0:
        bin_edges = np.linspace(min_dist, max_dist, bins + 1, dtype=np.float32)
        return bin_edges, np.zeros((bins,), dtype=np.float32), np.array([], dtype=np.int32)

    hist, bin_edges = np.histogram(vals, bins=bins, range=(min_dist, max_dist))
    hist = hist.astype(np.float32)
    total = float(hist.sum()) + 1e-6

    # Candidate bins sorted by depth (near first)
    candidate_ids = np.arange(bins, dtype=np.int32)
    # Keep bins with enough mass
    good = (hist / total) >= float(min_bin_frac)
    candidate_ids = candidate_ids[good]

    # Near-first: already increasing id corresponds to increasing depth
    chosen = candidate_ids[: int(k_near)]
    return bin_edges.astype(np.float32), hist, chosen.astype(np.int32)


def mask_by_bins(
    depth_m: np.ndarray,
    bin_edges: np.ndarray,
    chosen_bin_ids: np.ndarray,
    keep_floor: bool = False,
    floor_mask: np.ndarray | None = None,
    out_background: int = 0,  # 0=black, 255=white
) -> np.ndarray:
    """
    Build a uint8 mask where pixels in chosen bins are 255, else background value.
    """
    H, W = depth_m.shape[:2]
    d = depth_m.astype(np.float32)
    keep = np.zeros((H, W), dtype=bool)

    for bid in chosen_bin_ids.tolist():
        lo = float(bin_edges[bid])
        hi = float(bin_edges[bid + 1])
        keep |= _finite_mask(d) & (d >= lo) & (d < hi)

    if keep_floor and floor_mask is not None:
        keep |= floor_mask.astype(bool)
    else:
        if floor_mask is not None:
            keep &= ~floor_mask.astype(bool)

    mask = np.full((H, W), out_background, dtype=np.uint8)
    mask[keep] = 255
    return mask


def apply_mask_to_rgb(rgb_bgr: np.ndarray, mask_255: np.ndarray, bg_color=(0, 0, 0)) -> np.ndarray:
    """
    Keep pixels where mask==255, paint rest bg_color.
    """
    out = np.zeros_like(rgb_bgr)
    out[:] = bg_color
    keep = mask_255.astype(np.uint8) == 255
    out[keep] = rgb_bgr[keep]
    return out


def hist_to_bgr_image(hist: np.ndarray,
                      height: int = 400,
                      width: int = 400,
                      normalize: bool = True) -> np.ndarray:
    """
    hist: shape (bins,)
    returns: BGR image (height, width, 3) showing histogram as vertical bars
    """
    hist = np.asarray(hist).astype(np.float32).reshape(-1)
    bins = hist.shape[0]

    # Normalize to [0, 1] for drawing
    if normalize:
        mx = float(hist.max()) if hist.size else 0.0
        hist_n = hist / mx if mx > 1e-9 else hist
    else:
        hist_n = hist

    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Bar width in pixels
    bar_w = max(1, width // bins)

    for i in range(bins):
        v = float(hist_n[i])
        bar_h = int(v * (height - 2))
        x1 = i * bar_w
        x2 = min(width - 1, x1 + bar_w - 1)
        y1 = height - 1
        y2 = max(0, height - 1 - bar_h)
        cv2.rectangle(img, (x1, y2), (x2, y1), (255, 255, 255), thickness=-1)

    return img

def robust_depth_from_bbox_hist(
    depth_m,
    bbox,
    min_depth=0.2,
    max_depth=10.0,
    bins=80,
    min_frac=0.05,
):
    x1, y1, x2, y2 = bbox
    patch = depth_m[y1:y2, x1:x2]

    valid = np.isfinite(patch) & (patch > min_depth) & (patch < max_depth)
    vals = patch[valid]
    if vals.size < 30:
        return None

    hist, edges = np.histogram(vals, bins=bins, range=(min_depth, max_depth))
    thresh = max(int(min_frac * vals.size), 10)

    # Candidate bins: enough support
    candidate_bins = np.where(hist >= thresh)[0]
    if candidate_bins.size == 0:
        return float(np.median(vals))

    # Choose the CLOSEST bin (smallest depth)
    i = int(candidate_bins[0])

    lo, hi = edges[i], edges[i + 1]

    # Tighten window: discard far tail inside bin
    win = vals[(vals >= lo) & (vals <= hi)]
    if win.size < 10:
        return float(np.median(vals))

    return float(np.median(win))


def uvz_to_xyz_camera(u, v, z, fx, fy, cx, cy):
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)



def pad_width_center(frame, target_width: int):
    h, w = frame.shape[:2]

    if target_width <= 0 or w == target_width:
        return frame

    if w > target_width:
        raise ValueError(f"Frame width {w} is larger than target_width {target_width}")

    pad_total = target_width - w
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return cv2.copyMakeBorder(
        frame,
        0,
        0,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

def center_crop_resize(
    frame,
    crop_width: int,
    crop_height: int,
    output_width: int,
    output_height: int,
):
    h, w = frame.shape[:2]

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("crop_width and crop_height must be positive")

    if output_width <= 0 or output_height <= 0:
        raise ValueError("output_width and output_height must be positive")

    if crop_width > w or crop_height > h:
        raise ValueError(
            f"Crop {crop_width}x{crop_height} is larger than frame {w}x{h}"
        )

    x0 = (w - crop_width) // 2
    y0 = (h - crop_height) // 2
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    cropped = frame[y0:y1, x0:x1]

    resized = cv2.resize(
        cropped,
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )

    return resized
