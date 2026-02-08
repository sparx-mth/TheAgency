import cv2
import glob
import os
import sys
import numpy as np
# Fix import to point directly to module
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2DepthModel, DepthAnythingV2Config
from sparx_agency.core.mapping.depth.clean_mask import DepthMasker, MaskingConfig

def main():
    # 1. Setup
    folder_path = "/home/user1/Pictures/2026_01_27___12_30_15/" # 2026_01_27___12_30_15
    imgs_list = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    
    # Output dir
    out_dir = "/home/user1/.gemini/antigravity/brain/a8f85755-0492-47cb-a95a-484446462003/test_results"
    os.makedirs(out_dir, exist_ok=True)
    
    if not imgs_list:
        print(f"No images found in {folder_path}")
        return

    # 2. Init Models
    print("Loading models...")
    d_model = DepthAnythingV2DepthModel(DepthAnythingV2Config(device="cuda"))
    masker = DepthMasker(MaskingConfig(
        ransac_iters=500,
        ransac_dist_thresh=0.03, 
        floor_max_angle=15.0,
        remove_walls=True,
        min_object_area=200,
        max_dist=6.0
    ))
    
    print("Models loaded. Starting processing...")

    for i, img_path in enumerate(imgs_list):
        print(f"Processing {os.path.basename(img_path)}...")
        rgb = cv2.imread(img_path)
        if rgb is None:
            continue
            
        # 3. Depth Inference
        depth_map = d_model.infer_depth(rgb).astype(np.float32)
        
        # 4. Masking
        result = masker.get_clean_mask(depth_map, rgb)
        
        # 5. Visualization
        d_min, d_max = depth_map.min(), depth_map.max()
        depth_vis = (depth_map - d_min) / (d_max - d_min + 1e-6)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        
        # Floor mask (Red overlay)
        floor_vis = rgb.copy()
        floor_vis[result["floor_mask"] == 255] = [0, 0, 255] # Red floor
        
        # Wall mask (Blue overlay)
        wall_vis = rgb.copy()
        wall_vis[result["wall_mask"] == 255] = [255, 0, 0] # Blue walls
        
        # Object mask (Green overlay)
        obj_vis = rgb.copy()
        obj_vis[result["object_mask"] == 255] = [0, 255, 0] # Green objects
        
        # Clean RGB (Objects on Black)
        clean_rgb = result["clean_rgb"]
        
        # Grid
        h, w = rgb.shape[:2]
        target_h = 300
        target_w = int(w * (target_h / h))
        
        def rz(img): return cv2.resize(img, (target_w, target_h))
        
        # Row 1: Original, Depth, Floor
        row1 = np.hstack([rz(rgb), rz(depth_vis), rz(floor_vis)])
        # Row 2: Walls, Objects, Clean RGB
        row2 = np.hstack([rz(wall_vis), rz(obj_vis), rz(clean_rgb)])
        
        grid = np.vstack([row1, row2])
        
        # Save
        base_name = os.path.basename(img_path)
        out_path = os.path.join(out_dir, f"result_{base_name}")
        cv2.imwrite(out_path, grid)
        print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
