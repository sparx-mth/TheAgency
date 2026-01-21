import glob
import os
import sys
import cv2
import numpy as np
import time

# Add project root to path so we can import sparx_agency
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from sparx_agency.core.mapping.depth import DepthAnythingV2DepthModel
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2Config
from sparx_agency.core.mapping.depth.depth_object_segmenter import DepthObjectSegmenter


class TicToc:
    def __init__(self, name=None):
        self.name = name
        self.tstart = None
        
    def __enter__(self):
        self.tstart = time.time()
        
    def __exit__(self, type, value, traceback):
        print(f"[{self.name}] took {time.time() - self.tstart:.4f}s")



def get_depth_from_model(rgb_img, depth_model):
    # depth_model is now passed in to avoid reloading it every frame
    depth_img = depth_model.infer_depth(rgb_img).astype(np.float32)
    return depth_img


# --- Main Logic ---

def process_frame(image_path_or_array, depth_model, segmenter, output_dir="debug_output"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Image
    if isinstance(image_path_or_array, str):
        basename = os.path.basename(image_path_or_array)
        rgb_img = cv2.imread(image_path_or_array)
    else:
        basename = "frame.jpg"
        rgb_img = image_path_or_array

    if rgb_img is None:
        print(f"Error: Could not load image {basename}")
        return

    h, w = rgb_img.shape[:2]
    print(f"Processing {basename} ({w}x{h})...")

    # 2. Call Depth Anything
    with TicToc("Depth Anything Inference"):
        depth_map = get_depth_from_model(rgb_img, depth_model)

    # 3. Segment Objects (3D RANSAC + BEV)
    with TicToc("3D Segmentation"):
        # We assume some generic focal length if unknown, e.g. w usually ~ 1.0 * f
        # Let's update segmenter focal length based on image width just in case
        # segmenter.focal_length = w * 0.8 
        objects = segmenter.segment_objects(depth_map, rgb_img)

    print(f"Found {len(objects)} objects.")

    # 4. Overlay results on RGB
    overlay = rgb_img.copy()
    
    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj['bbox']
        d_val = obj['avg_depth']
        
        # Color based on depth
        color = (0, 255, 0)
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"ID {i}: {d_val:.1f}m", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save Crop
        crop = rgb_img[y1:y2, x1:x2]
        if crop.size > 0:
            crop_name = f"{os.path.splitext(basename)[0]}_obj_{i:02d}.jpg"
            cv2.imwrite(os.path.join(output_dir, crop_name), crop)

    # 5. Save Debug Full Image
    combined = np.hstack([rgb_img, overlay])
    # Resize for saving if huge
    if combined.shape[1] > 2000:
        scale = 2000 / combined.shape[1]
        combined = cv2.resize(combined, (0,0), fx=scale, fy=scale)
        
    cv2.imwrite(os.path.join(output_dir, f"debug_{basename}"), combined)
    print(f"Saved debug images to {output_dir}/")


if __name__ == "__main__":
    folder_path = "/home/daphnaa/GIT/Depth-Anything-V2-original/assets/examples/2025_10_05___15_01_16"
    imgs_list = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    
    # Init models once
    d_config = DepthAnythingV2Config()
    d_model = DepthAnythingV2DepthModel(d_config)
    
    # Approximate Focal Length:
    # Taking a standard assumption that FoV ~ 60 degrees -> f ~ 1.0 * w usually.
    # But let's stick to 500-1000 range defaults or update per image.
    # The segmenter default is 500. Let's try 700.
    seg = DepthObjectSegmenter(focal_length_px=700.0, ransac_thresh=0.04, cluster_min_points=50)

    # Process first 5 images for test
    for test_image in imgs_list[:5]:
        process_frame(test_image, d_model, seg)