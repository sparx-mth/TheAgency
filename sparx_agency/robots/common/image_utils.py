from __future__ import annotations

import glob
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image

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


# 
def get_objects_by_quantized_surfaces(depth_map, min_dist=0.5, max_dist=4.0):
    # 1. Convert real meters to 0-255 grayscale
    # We use your 0.5m to 5.0m range
    depth_scaled = ((depth_map - 0.5) / (5.0 - 0.5) * 255).clip(0, 255).astype(np.uint8)

    # 2. QUANTIZE: This is the magic step.
    # By dividing by 10 and multiplying by 10, all pixels within
    # a 10-unit range (your smoothness range) get the SAME value.
    quantized = (depth_scaled // 10) * 10

    detected_objects = []

    # 3. Iterate through the possible depth levels
    # We only care about levels representing distances < 4.0m
    unique_levels = np.unique(quantized)

    for level in unique_levels:
        if level == 0 or level > 200: continue  # Skip extreme near/far noise

        # Create a mask for just this depth "slice"
        mask = (quantized == level).astype(np.uint8) * 255

        # 4. Use Morphology to bridge small gaps (like the chair slats)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 5. Connected Components: "If you are close to each other, you are the same object"
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 800 < area < (depth_map.shape[0] * depth_map.shape[1] * 0.3):
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                    stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

                bbox_area = w * h
                area_ratio = area / bbox_area
                image_area = depth_map.shape[1] * depth_map.shape[0]

                if area_ratio < 0.2:
                    continue

                if area_ratio > 0.6 and area > 0.4 * image_area:
                    continue

                detected_objects.append({
                    "bbox": (x, y, x + w, y + h),
                    "avg_depth": level  # Or calculate median from depth_map
                })

    return detected_objects

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


def get_objects_via_depth_kmeans(
    depth_map: np.ndarray,
    min_dist=0.3,
    max_dist=5.0,
    K=3,
    sample_max=200_000,     # sample pixels for speed
    min_area_px=800,
    close_kernel=15,
    close_iters=1,
    prob=None,              # keep for signature compatibility
):
    depth = np.asarray(depth_map, np.float32)
    H, W = depth.shape[:2]
    image_area = H * W

    valid = np.isfinite(depth) & (depth > 0) & (depth >= min_dist) & (depth <= max_dist)
    if valid.sum() < 200:
        return []

    vals = depth[valid].reshape(-1, 1)  # Nx1 float32

    # random subsample for kmeans
    n = vals.shape[0]
    if n > sample_max:
        idx = np.random.choice(n, sample_max, replace=False)
        train = vals[idx]
    else:
        train = vals

    # kmeans in 1D
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    attempts = 3
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels_t, centers = cv2.kmeans(train, K, None, criteria, attempts, flags)
    centers = centers.reshape(-1)  # K depths

    # Assign ALL valid pixels to nearest center (1D nearest mean)
    # (faster than re-running kmeans full)
    # Compute |d - center|
    diffs = np.abs(vals - centers.reshape(1, -1))  # NxK
    lbl_all = np.argmin(diffs, axis=1).astype(np.int32)

    # Build full label image for valid pixels
    label_img = -np.ones((H, W), dtype=np.int32)
    label_img[valid] = lbl_all

    # Background is usually the cluster with the MOST pixels
    counts = np.bincount(lbl_all, minlength=K)
    bg_label = int(np.argmax(counts))

    detected_objects = []
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))

    for k in range(K):
        if k == bg_label:
            continue  # skip background cluster

        mask = (label_img == k).astype(np.uint8) * 255

        # close holes / merge chair slats
        if close_kernel > 1 and close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iters)

        num, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < min_area_px:
                continue

            comp = (labels_cc == i) & valid
            if comp.sum() == 0:
                continue

            dvals = depth[comp]
            avg_depth = float(np.mean(dvals))
            d_min = float(np.percentile(dvals, 10))
            d_max = float(np.percentile(dvals, 90))
            spread = d_max - d_min
            if should_skip_component(x, y, w, h, area, H, W, avg_depth, spread):
                continue
            if spread > 0.9:  # start with 0.5, tune (0.3–0.8 depending on your depth scale)
                continue
            print(f" ========= Object {i}, spread: {spread:.2f}m, ========= \n touches border: x: {x}, y: {y}, w: {w}, h: {h}, W: {W}, H: {H}, margin: {1}")
            detected_objects.append({
                "bbox": (int(x), int(y), int(x + w), int(y + h)),
                "avg_depth": avg_depth,
                "area": int(area),
                "depth_range": (d_min, d_max),
                "cluster_center": float(centers[k]),
                "cluster_id": int(k),
            })
            print(detected_objects[-1])

    return detected_objects

def touches_border(x, y, w, h, W, H, margin=1):
    return (x <= margin) or (y <= margin) or (x + w >= W - margin) or (y + h >= H - margin)


def should_skip_component(x, y, w, h, area, H, W, avg_depth, spread,
                          border_margin=2,
                          min_area_px=800,
                          max_spread_m=0.45,
                          strip_frac=0.12,
                          border_area_frac=0.003):
    """
    Rejects background-ish components:
      - tiny noise
      - border-touching strips (floor/wall bands)
      - border-touching components that are too big
      - depth-incoherent components (large spread)
    """
    if area < min_area_px:
        return True

    touches = (x <= border_margin) or (y <= border_margin) or (x + w >= W - border_margin) or (y + h >= H - border_margin)
    image_area = H * W

    # 1) Depth coherence gate (kills gradients / planes)
    # Use absolute + relative guard
    if spread > max(max_spread_m, 0.25 * avg_depth):
        return True

    if touches:
        # 2) Strip gate: if it touches border and is "thin", it's usually floor/wall band
        if (h < strip_frac * H) or (w < strip_frac * W):
            return True

        # 3) Border big blob gate (more aggressive than before)
        if area > border_area_frac * image_area:
            return True

    return False


def get_objects_via_depth_edges(
    depth_map: np.ndarray,
    min_dist=0.3,
    max_dist=5.0,
    grad_thresh=0.06,      # meters-per-pixel-ish (tune 0.03–0.12)
    min_area_px=800,
    close_kernel=17,
    close_iters=1,
):
    depth = np.asarray(depth_map, np.float32)
    H, W = depth.shape[:2]

    valid = np.isfinite(depth) & (depth > 0) & (depth >= min_dist) & (depth <= max_dist)
    if valid.sum() < 200:
        return []

    # Fill invalid with nearby values to avoid crazy gradients
    d = depth.copy()
    d[~valid] = np.nan
    # simple inpaint-like fill: replace NaNs with median of valid
    med = float(np.nanmedian(d))
    d = np.nan_to_num(d, nan=med)

    # Smooth a bit to reduce speckle noise
    d_blur = cv2.GaussianBlur(d, (5, 5), 0)

    # Gradient magnitude
    gx = cv2.Sobel(d_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Edge mask where depth changes sharply
    edge = (mag > grad_thresh).astype(np.uint8) * 255

    # Close to connect boundaries and fill gaps
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k_close, iterations=close_iters)

    # Convert boundary-ish mask into regions: fill holes by closing + dilation
    edge = cv2.dilate(edge, k_close, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(edge, connectivity=8)

    objs = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area_px:
            continue

        # get depth stats inside bbox, using valid pixels only
        roi_valid = valid[y:y+h, x:x+w]
        roi_depth = depth[y:y+h, x:x+w]
        if roi_valid.sum() < 50:
            continue

        dvals = roi_depth[roi_valid]
        avg = float(np.mean(dvals))
        d10 = float(np.percentile(dvals, 10))
        d90 = float(np.percentile(dvals, 90))

        objs.append({
            "bbox": (int(x), int(y), int(x+w), int(y+h)),
            "avg_depth": avg,
            "area": int(area),
            "depth_range": (d10, d90),
            "source": "edges"
        })

    return objs


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    area_a = max(1, (ax2-ax1)*(ay2-ay1))
    area_b = max(1, (bx2-bx1)*(by2-by1))
    return inter / float(area_a + area_b - inter + 1e-6)

def merge_boxes(objs, iou_thr=0.5):
    objs = sorted(objs, key=lambda o: o.get("area", 0), reverse=True)
    kept = []
    for o in objs:
        if all(iou(o["bbox"], k["bbox"]) < iou_thr for k in kept):
            kept.append(o)
    return kept


def preprocess_depth(depth_map, min_dist=0.3, max_dist=5.0):
    d = np.asarray(depth_map, np.float32)
    valid = np.isfinite(d) & (d > 0) & (d >= min_dist) & (d <= max_dist)

    if valid.sum() < 200:
        return d, valid

    # clamp & fill invalid with median (prevents crazy gradients)
    med = float(np.median(d[valid]))
    d = np.clip(d, min_dist, max_dist)
    d2 = d.copy()
    d2[~valid] = med

    # bilateral filter preserves edges but smooths quantization/noise
    d2 = cv2.bilateralFilter(d2, d=7, sigmaColor=0.08, sigmaSpace=7)

    return d2, valid

def get_objects_via_depth_edges_adaptive(
    depth_map, min_dist=0.3, max_dist=5.0,
    min_area_px=600,
    close_kernel=17,
    close_iters=1,
    grad_percentile=92,     # 90–96 works well
):
    d, valid = preprocess_depth(depth_map, min_dist, max_dist)
    H, W = d.shape[:2]

    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    mag_v = mag[valid]
    if mag_v.size < 200:
        return []

    thr = float(np.percentile(mag_v, grad_percentile))
    edge = (mag > thr).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k, iterations=close_iters)
    edge = cv2.dilate(edge, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(edge, connectivity=8)

    objs = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area_px:
            continue

        roi_valid = valid[y:y+h, x:x+w]
        if roi_valid.sum() < 80:
            continue

        roi_depth = depth_map[y:y+h, x:x+w]
        dvals = roi_depth[roi_valid]
        avg = float(np.mean(dvals))
        d10 = float(np.percentile(dvals, 10))
        d90 = float(np.percentile(dvals, 90))

        objs.append({
            "bbox": (int(x), int(y), int(x+w), int(y+h)),
            "avg_depth": avg,
            "area": int(area),
            "depth_range": (d10, d90),
            "source": "edges_adapt",
            "grad_thr": thr,
        })

    return objs


def bbox_filters(x, y, w, h, W, H):
    # kill super-thin junk
    if w < 12 or h < 12:
        return True
    # kill extreme aspect ratios
    ar = max(w/h, h/w)
    if ar > 8.0:
        return True
    return False

