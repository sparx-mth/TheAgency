import cv2
import numpy as np
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# Fix import path if running as script from root or submodule
# Assuming we run from root usually
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2DepthModel, DepthAnythingV2Config
from sparx_agency.core.mapping.depth.clean_mask import DepthMasker, MaskingConfig

@dataclass
class CostmapConfig:
    grid_res: float = 0.05       # 5 cm per pixel
    obstacle_height: float = 0.20 # 20 cm above floor is obstacle
    max_height: float = 2.0      # Ignore ceiling/very high objects
    max_range: float = 20.0      # Max depth range to consider
    map_size_m: float = 30.0     # 30x30 meter map (Large expo hall)
    
    # Camera intrinsics (approximate)
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0

class ImageToCostmap:
    def __init__(self, config: CostmapConfig = CostmapConfig()):
        self.cfg = config
        self.masker = DepthMasker() # Reusing for plane fitting logic

    def get_rotation_matrix(self, vec1, vec2):
        """ Get rotation matrix that rotates vec1 to vec2 """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        if np.any(v): # if not parallel
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            return rotation_matrix
        else:
            return np.eye(3) 

    def generate_costmap_from_points(self, points: np.ndarray) -> dict:
        """
        Generate costmap from 3D points (H*W, 3) or (N, 3).
        Expects points in Camera Frame (Y down, Z forward) or similar.
        Will fit floor and align.
        """
        # Ensure points is Nx3
        pts_flat = points.reshape(-1, 3)
        
        # Valid check
        valid_mask = (pts_flat[:, 2] > 0) & (pts_flat[:, 2] < self.cfg.max_range)
        valid_points = pts_flat[valid_mask]
        
        if len(valid_points) < 100:
             return {}

        # 1. Fit Floor Plane
        # Use bottom half heuristic for floor search if structured?
        # If unstructured points, difficult to use simple "bottom half" heuristic without Y sorting.
        # Let's simple use RANSAC with existing masker logic, but masker expects (H,W,3).
        # We'll adapt masker or just copy logic?
        # ImageToCostmap instance has self.masker.
        # Let's just use self.masker.fit_plane_ransac which now works on flat points?
        # NO, masker.fit_plane_ransac expects (points, mask). points can be any shape as long as mask matches?
        # Actually it flattens inside.
        
        # Heuristic: Filter points that are clearly too high to be floor (e.g. above camera if camera is looking straight?)
        # Let's blindly RANSAC on all valid points for robustness or subset?
        # Floor is usually the largest plane.
        
        plane, floor_mask_flat = self.masker.fit_plane_ransac(
            pts_flat, valid_mask, 
            orientation_axis=np.array([0, 1, 0]), 
            max_angle_deg=70.0 
        )
        
        if plane is None:
            # print("Failed to fit floor plane!")
            return {}
            
        # 2. Align World (same as before)
        normal = plane[:3]
        target_normal = np.array([0, 1, 0])
        R = self.get_rotation_matrix(normal, target_normal)
        
        pts_aligned = pts_flat @ R.T # Rotate all points (even invalid ones, to keep indexing simple? No need.)
        # Let's work with valid points for speed
        pts_aligned_valid = valid_points @ R.T
        
        # Floor mask on valid subset
        floor_mask_subset = floor_mask_flat[valid_mask]
        
        # 3. Translate Floor to Y=0
        floor_pts_aligned = pts_aligned_valid[floor_mask_subset]
        
        if len(floor_pts_aligned) > 0:
            floor_y = np.median(floor_pts_aligned[:, 1])
            pts_aligned_valid[:, 1] -= floor_y
        else:
            floor_y = 0.0
            
        camera_y = -floor_y
        if camera_y < 0:
             # print(f"Flipping Y axis")
             pts_aligned_valid[:, 1] *= -1
             
        if np.median(pts_aligned_valid[:, 2]) < 0:
             # print("Flipping Z axis")
             pts_aligned_valid[:, 2] *= -1
             pts_aligned_valid[:, 0] *= -1

        # --- Auto-Rotate Yaw using PCA ---
        obs_mask_rot = (pts_aligned_valid[:, 1] > self.cfg.obstacle_height) & (pts_aligned_valid[:, 1] < self.cfg.max_height)
        obs_pts = pts_aligned_valid[obs_mask_rot]
        
        if len(obs_pts) > 100:
            X = obs_pts[:, [0, 2]]
            mean = np.mean(X, axis=0)
            X_centered = X - mean
            cov = np.cov(X_centered, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            dominant_axis = evecs[:, 1]
            angle = np.arctan2(dominant_axis[0], dominant_axis[1])
            c, s = np.cos(-angle), np.sin(-angle)
            R_yaw = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
            pts_aligned_valid = pts_aligned_valid @ R_yaw.T

        # 4. Project to 2D Grid
        res = self.cfg.grid_res
        
        # Filter extremes
        valid_map_pts = (pts_aligned_valid[:, 2] > 0) & (pts_aligned_valid[:, 2] < self.cfg.map_size_m)
        if valid_map_pts.sum() == 0:
            return {}

        xs = pts_aligned_valid[valid_map_pts, 0]
        ys = pts_aligned_valid[valid_map_pts, 1]
        zs = pts_aligned_valid[valid_map_pts, 2]

        min_x, max_x = xs.min(), xs.max()
        min_z, max_z = zs.min(), zs.max()
        
        pad = 1.0 
        min_x -= pad
        max_x += pad
        min_z = max(0, min_z)
        max_z += pad
        
        map_w = int((max_x - min_x) / res)
        map_h = int((max_z - min_z) / res)
        
        if map_w <= 0 or map_h <= 0 or map_w > 10000 or map_h > 10000:
             return {}
            
        us = ((xs - min_x) / res).astype(np.int32)
        vs = ((zs - min_z) / res).astype(np.int32)
        
        valid_uv = (us >= 0) & (us < map_w) & (vs >= 0) & (vs < map_h)
        us = us[valid_uv]
        vs = vs[valid_uv]
        ys = ys[valid_uv]
        
        grid_acc = np.zeros((map_h, map_w, 3), dtype=np.int32)
        grid_flat = grid_acc.reshape(-1, 3)
        flat_idx = vs * map_w + us
        
        obstacle_h_thresh = 0.5
        is_obstacle = (ys > obstacle_h_thresh) & (ys < self.cfg.max_height)
        is_floor = (ys >= -0.15) & (ys <= obstacle_h_thresh)
        
        np.add.at(grid_flat[:, 0], flat_idx[is_obstacle], 1)
        np.add.at(grid_flat[:, 1], flat_idx[is_floor], 1)
        
        obs_count = grid_acc[:,:,0]
        floor_count = grid_acc[:,:,1]
        
        occupied = obs_count > 0
        free = (floor_count > 0) & (~occupied)
        
        costmap = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        costmap[free] = [0, 255, 0] 
        costmap[occupied] = [0, 0, 255] 
        
        kernel = np.ones((3,3), np.uint8)
        mask_occ = (costmap[:,:,2] == 255).astype(np.uint8)
        mask_occ = cv2.dilate(mask_occ, kernel, iterations=3)
        costmap[mask_occ == 1] = [0, 0, 255]
        
        rob_u = int((-min_x) / res)
        rob_v = int((-min_z) / res)
        if 0 <= rob_u < map_w and 0 <= rob_v < map_h:
            cv2.circle(costmap, (rob_u, rob_v), 5, (255, 255, 0), -1)

        # 5. Extract 3D Points
        # Recover boolean masks for valid_map_pts subset
        # is_obstacle and is_floor are indices into valid_map_pts which is subset of pts_aligned_valid
        
        obstacle_pts_3d = pts_aligned_valid[valid_map_pts][is_obstacle]
        floor_pts_3d = pts_aligned_valid[valid_map_pts][is_floor]

        return {
            "costmap": costmap,
            "obstacle_points": obstacle_pts_3d,
            "floor_points": floor_pts_3d, 
            "origin": (min_x, min_z), 
            "resolution": res
        }

    def generate_costmap(self, rgb: np.ndarray, depth: np.ndarray) -> dict:
        H, W = depth.shape
        points = self.masker.depth_to_points(depth) # (H, W, 3)
        return self.generate_costmap_from_points(points)

def main():
    # Setup
    img_path = "/home/user1/.gemini/antigravity/brain/a8f85755-0492-47cb-a95a-484446462003/uploaded_image_1769615314522.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    out_dir = "/home/user1/.gemini/antigravity/brain/a8f85755-0492-47cb-a95a-484446462003/costmap_results"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading models...")
    d_model = DepthAnythingV2DepthModel(DepthAnythingV2Config(device="cuda"))
    mapper = ImageToCostmap(CostmapConfig(max_range=20.0))
    
    print(f"Processing {os.path.basename(img_path)}...")
    rgb = cv2.imread(img_path)
    if rgb is None:
        print("Failed to read image")
        return
        
    # Resize for speed? Depth anything is flexible.
    # rgb = cv2.resize(rgb, (640, 480))
    
    # 1. Depth
    depth_map = d_model.infer_depth(rgb).astype(np.float32)
    print(f"Depth stats: min={depth_map.min():.3f}, max={depth_map.max():.3f}, mean={depth_map.mean():.3f}")
    
    # 2. Costmap
    result = mapper.generate_costmap(rgb, depth_map)
    
    if "costmap" in result:
        cmap = result["costmap"]
        
        # Save results
        # Flip vertically so "forward" is UP in the image
        cmap_flipped = cv2.flip(cmap, 0) 
        
        cv2.imwrite(os.path.join(out_dir, "depth.jpg"), (depth_map/depth_map.max()*255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_dir, "costmap.png"), cmap_flipped)
        print(f"Saved results to {out_dir}")
        
    else:
        print("Costmap generation failed.")

if __name__ == "__main__":
    main()
