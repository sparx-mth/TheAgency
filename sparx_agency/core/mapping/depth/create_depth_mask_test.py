import glob
import os
import sys
import cv2
import numpy as np
import time

from sparx_agency.robots.common.image_utils import get_objects_by_quantized_surfaces, \
     create_hist_image_with_objects, get_objects_via_depth_kmeans, \
    get_objects_via_depth_edges, merge_boxes

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
    # Normalize depth map to 0-255 for visibility
    depth_viz = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_viz = 255 - depth_viz

    depth_raw_view = (depth_map - 0.5) / (5.0 - 0.5)
    depth_raw_view = np.clip(depth_raw_view, 0, 1)
    depth_viz_grayscale = (depth_raw_view * 255).astype(np.uint8)
    depth_viz_bgr = cv2.cvtColor(depth_viz_grayscale, cv2.COLOR_GRAY2BGR)
    # 6. Panel 3: Histogram

    hist_viz = create_hist_image_with_objects(depth_map, objs, min_dist=0.3, max_dist=5.0, bins=60)

    # Resize all to match height for hstack
    display_h = 400
    aspect = w / h
    display_w = int(display_h * aspect)

    res_overlay = cv2.resize(overlay, (display_w, display_h))
    res_depth = cv2.resize(depth_viz_bgr, (display_w, display_h))
    res_hist = cv2.resize(hist_viz, (display_w, display_h))

    # Combine: [ RGB Overlay | Depth Map | Histogram ]
    combined = np.hstack([res_overlay, res_depth, res_hist])

    cv2.imshow("Detection Pipeline: RGB Overlay | Depth | Histogram", combined)
    cv2.waitKey(0)


if __name__ == "__main__":
    folder_path = "/home/user1/Pictures/OneDrive_1_1-22-2026/"
    imgs_list = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    
    # Init models once
    d_config = DepthAnythingV2Config()
    d_model = DepthAnythingV2DepthModel(d_config)
    
    # Approximate Focal Length:
    # Taking a standard assumption that FoV ~ 60 degrees -> f ~ 1.0 * w usually.
    # But let's stick to 500-1000 range defaults or update per image.
    # The segmenter default is 500. Let's try 700.
    # seg = DepthObjectSegmenter(focal_length_px=700.0, ransac_thresh=0.04, cluster_min_points=50)

    # Process first 5 images for test
    for test_image in imgs_list[:5]:
        process_frame(test_image, d_model)