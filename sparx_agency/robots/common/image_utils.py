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


def get_objects_via_histogram(depth_img, min_dist=0.3, max_dist=5.0, bins=50):
    # 1. Mask the floor (bottom 50%)
    h, w = depth_img.shape
    work_depth = depth_img.copy()
    work_depth[h:, :] = 0

    # 2. Compute Histogram to find depth "peaks"
    # Only consider pixels within our valid range
    valid_pixels = work_depth[(work_depth >= min_dist) & (work_depth <= max_dist)]
    if len(valid_pixels) == 0:
        return []

    hist, bin_edges = np.histogram(valid_pixels, bins=bins, range=(min_dist, max_dist))

    # 3. Find peaks in the histogram (simple threshold)
    # A peak is any bin with a significant number of pixels
    peak_threshold = (h * w) * 0.01  # e.g., at least 1% of the frame
    peak_bins = np.where(hist > peak_threshold)[0]

    detected_objects = []

    # 4. For each peak, find the blobs
    for bin_idx in peak_bins:
        d_min = bin_edges[bin_idx]
        d_max = bin_edges[bin_idx + 1]

        # Create a mask for this specific depth peak
        mask = ((work_depth >= d_min) & (work_depth <= d_max)).astype(np.uint8) * 255

        # 5. Look for Blobs (Connected Components)
        # This is more efficient than findContours for simple rectangles
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 500: continue  # Ignore tiny noise blobs

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_obj = stats[i, cv2.CC_STAT_WIDTH]
            h_obj = stats[i, cv2.CC_STAT_HEIGHT]

            detected_objects.append({
                "bbox": (x, y, x + w_obj, y + h_obj),
                "avg_depth": (d_min + d_max) / 2.0,
                "area": area
            })

    return detected_objects


def get_objects_via_histogram_clustered(depth_img, min_dist=0.3, max_dist=5.0, bins=100):
    h, w = depth_img.shape
    work_depth = depth_img.copy()
    # Fixed floor mask: h*0.5 to actually mask the bottom half
    work_depth[int(h):, :] = 0

    valid_pixels = work_depth[(work_depth >= min_dist) & (work_depth <= max_dist)]
    if len(valid_pixels) == 0: return []

    hist, bin_edges = np.histogram(valid_pixels, bins=bins, range=(min_dist, max_dist))

    # 1. Identify "Active Bins" (those above threshold)
    peak_threshold = (h * w) * 0.005  # Dropped to 0.5%
    active_bins = np.where(hist > peak_threshold)[0]

    # 2. Group ADJACENT active bins into "Depth Islands"
    # This is the "Smoothness" logic: if depths are close, they are one object
    groups = []
    if len(active_bins) > 0:
        current_group = [active_bins[0]]
        for i in range(1, len(active_bins)):
            if active_bins[i] == active_bins[i - 1] + 1:
                current_group.append(active_bins[i])
            else:
                groups.append(current_group)
                current_group = [active_bins[i]]
        groups.append(current_group)

    detected_objects = []

    # 3. For each group (Surface), find the spatial blobs
    for group in groups:
        d_min = bin_edges[group[0]]
        d_max = bin_edges[group[-1] + 1]

        # This mask now contains the WHOLE suitcase "window"
        mask = ((work_depth >= d_min) & (work_depth <= d_max)).astype(np.uint8) * 255

        # Clean up the mask to merge the "smooth" pixels
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1500: continue

            x, y, w_obj, h_obj = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            detected_objects.append({
                "bbox": (x, y, x + w_obj, y + h_obj),
                "avg_depth": np.median(work_depth[labels == i]),
                "area": area
            })

    return detected_objects


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

                detected_objects.append({
                    "bbox": (x, y, x + w, y + h),
                    "avg_depth": level  # Or calculate median from depth_map
                })

    return detected_objects