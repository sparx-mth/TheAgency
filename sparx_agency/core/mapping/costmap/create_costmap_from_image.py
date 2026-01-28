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

    def generate_costmap(self, rgb: np.ndarray, depth: np.ndarray) -> dict:
        H, W = depth.shape
        points = self.masker.depth_to_points(depth) # (H, W, 3)
        valid_mask = (depth > 0) & (depth < self.cfg.max_range)
        
        # 1. Fit Floor Plane
        # Use bottom half heuristic for floor search
        search_mask = valid_mask.copy()
        search_mask[: int(H/3), :] = False
        
        # We expect floor normal roughly near Y axis (0, 1, 0)
        # But if camera is pitched down 45 deg, normal will be tilted.
        # Relax constraint to allow high-angle views (up to 70 deg pitch)
        plane, floor_mask = self.masker.fit_plane_ransac(
            points, search_mask, 
            orientation_axis=np.array([0, 1, 0]), 
            max_angle_deg=70.0 
        )
        
        if plane is None:
            print("Failed to fit floor plane!")
            return {}
        print(f"Floor Plane: {plane}")

        # 2. Align World
        normal = plane[:3]
        target_normal = np.array([0, 1, 0])
        
        R = self.get_rotation_matrix(normal, target_normal)
        
        # Apply rotation
        pts_flat = points[valid_mask]
        pts_aligned = pts_flat @ R.T
        
        # 3. Translate Floor to Y=0
        inlier_indices = floor_mask[valid_mask]
        floor_pts_aligned = pts_aligned[inlier_indices]
        
        if len(floor_pts_aligned) > 0:
            floor_y = np.median(floor_pts_aligned[:, 1])
            pts_aligned[:, 1] -= floor_y
        else:
            floor_y = 0.0
            
        # Ensure 'Up' is positive relative to floor
        # Camera is at 0 (before translation). After translation: -floor_y.
        # We assume Camera is ABOVE floor. So Camera Y must be positive.
        camera_y = -floor_y
        if camera_y < 0:
             print(f"Flipping Y axis (Camera y={camera_y:.2f} < 0)")
             pts_aligned[:, 1] *= -1
        else:
             print(f"Y axis orientation OK (Camera y={camera_y:.2f} >= 0)")
             
        # Check if Z is flipped (negative)
        # We expect points to be in front of the camera (positive Z usually, or at least forward).
        # We expect points to be in front of the camera (positive Z usually, or at least forward).
        # Typically "Forward" in map should be positive Z.
        if np.median(pts_aligned[:, 2]) < 0:
            print("Flipping Z axis to face forward")
            pts_aligned[:, 2] *= -1
            pts_aligned[:, 0] *= -1

        print(f"Aligned Points Stats (final):")
        print(f"  X: min={pts_aligned[:,0].min():.2f} max={pts_aligned[:,0].max():.2f}")
        print(f"  Y: min={pts_aligned[:,1].min():.2f} max={pts_aligned[:,1].max():.2f}")
        print(f"  Z: min={pts_aligned[:,2].min():.2f} max={pts_aligned[:,2].max():.2f}")

        # --- Auto-Rotate Yaw using PCA ---
        # Find dominant axis of obstacles (walls/stands)
        # Use points that are definitely obstacles
        obs_mask_rot = (pts_aligned[:, 1] > self.cfg.obstacle_height) & (pts_aligned[:, 1] < self.cfg.max_height)
        obs_pts = pts_aligned[obs_mask_rot]
        
        if len(obs_pts) > 100:
            # Flatten to 2D (X, Z)
            X = obs_pts[:, [0, 2]]
            # Center
            mean = np.mean(X, axis=0)
            X_centered = X - mean
            # Covariance
            cov = np.cov(X_centered, rowvar=False)
            # Eigenvalues/vectors
            evals, evecs = np.linalg.eigh(cov)
            # Dominant vector (largest eigenvalue) is last column of evecs
            dominant_axis = evecs[:, 1]
            
            # Angle of dominant axis wrt Z axis (0, 1)
            # We want to align dominant axis to Z axis (or X axis)
            angle = np.arctan2(dominant_axis[0], dominant_axis[1])
            print(f"Detected dominant structure angle: {np.degrees(angle):.1f} deg. Aligning...")
            
            # Rotation matrix around Y
            # We want to rotate by -angle
            c, s = np.cos(-angle), np.sin(-angle)
            R_yaw = np.array([
                [c, 0, -s],
                [0, 1, 0],
                [s, 0, c]
            ])
            pts_aligned = pts_aligned @ R_yaw.T
        else:
            print("Not enough obstacle points for PCA alignment.")

        # 4. Project to 2D Grid with Auto-Scaling
        res = self.cfg.grid_res
        
        # Filter extremes for map generation (clip to max range)
        valid_map_pts = (pts_aligned[:, 2] > 0) & (pts_aligned[:, 2] < self.cfg.map_size_m)
        if valid_map_pts.sum() == 0:
            print("No points in valid depth range!")
            return {}

        xs = pts_aligned[valid_map_pts, 0]
        ys = pts_aligned[valid_map_pts, 1]
        zs = pts_aligned[valid_map_pts, 2]

        min_x, max_x = xs.min(), xs.max()
        min_z, max_z = zs.min(), zs.max()
        
        print(f"Map Bounds: X=[{min_x:.2f}, {max_x:.2f}], Z=[{min_z:.2f}, {max_z:.2f}]")
        
        # Add padding
        pad = 1.0 # meters
        min_x -= pad
        max_x += pad
        min_z = max(0, min_z) # Start Z at 0? Or min_z? Let's stick to world 0 is camera.
        max_z += pad
        
        map_w = int((max_x - min_x) / res)
        map_h = int((max_z - min_z) / res)
        
        print(f"Map Grid Size: {map_w} x {map_h}")
        
        if map_w <= 0 or map_h <= 0:
            print("Invalid map size")
            return {}
            
        us = ((xs - min_x) / res).astype(np.int32)
        vs = ((zs - min_z) / res).astype(np.int32)
        
        # Clip
        valid_uv = (us >= 0) & (us < map_w) & (vs >= 0) & (vs < map_h)
        us = us[valid_uv]
        vs = vs[valid_uv]
        ys = ys[valid_uv]
        
        # Accumulate scores
        grid_acc = np.zeros((map_h, map_w, 3), dtype=np.int32)
        # Flatten grid to (N_cells, 3) to allow efficient indexing
        # grid_acc is (H, W, 3). Reshape to (H*W, 3) is safe view.
        grid_flat = grid_acc.reshape(-1, 3)
        flat_idx = vs * map_w + us
        
        # User requested 0.5m threshold
        obstacle_h_thresh = 0.5
        
        is_obstacle = (ys > obstacle_h_thresh) & (ys < self.cfg.max_height)
        is_floor = (ys >= -0.15) & (ys <= obstacle_h_thresh)
        
        # Add votes using unbuffered add.at on the view
        # column 0 is obstacle, 1 is floor
        np.add.at(grid_flat[:, 0], flat_idx[is_obstacle], 1)
        np.add.at(grid_flat[:, 1], flat_idx[is_floor], 1)
        
        obs_count = grid_acc[:,:,0]
        floor_count = grid_acc[:,:,1]
        
        print(f"Votes: Obstacle={obs_count.sum()}, Floor={floor_count.sum()}")
        print(f"Max votes in a cell: Obstacle={obs_count.max()}, Floor={floor_count.max()}")
        
        # Lower threshold to > 0 to catch sparse points
        occupied = obs_count > 0
        free = (floor_count > 0) & (~occupied)
        
        costmap = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        costmap[free] = [0, 255, 0] # Green
        costmap[occupied] = [0, 0, 255] # Red
        
        # Post-process for "schematic" look
        # Dilate obstacles to connect walls and make them solid blocks
        kernel = np.ones((3,3), np.uint8)
        mask_occ = (costmap[:,:,2] == 255).astype(np.uint8)
        # Just Dilate to make walls thicker
        mask_occ = cv2.dilate(mask_occ, kernel, iterations=3)
        costmap[mask_occ == 1] = [0, 0, 255]
        
        # Draw robot at (0,0) -> ( -min_x / res, -min_z / res )
        rob_u = int((-min_x) / res)
        rob_v = int((-min_z) / res)
        if 0 <= rob_u < map_w and 0 <= rob_v < map_h:
            cv2.circle(costmap, (rob_u, rob_v), 5, (255, 255, 0), -1)

        return {
            "costmap": costmap,
        }

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
