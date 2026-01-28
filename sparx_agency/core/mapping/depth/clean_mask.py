import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MaskingConfig:
    # Intrinsic parameters
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0
    
    # RANSAC params
    ransac_iters: int = 500
    ransac_dist_thresh: float = 0.03  # 3cm
    floor_max_angle: float = 15.0     # degrees deviation from vertical (Y-axis)
    
    # Wall params
    remove_walls: bool = True
    wall_min_angle: float = 75.0      # Wall normal must be horizontal (>75 deg deviation from Y)
    
    # Object filtering
    min_height_above_floor: float = 0.03 # 3cm
    max_dist: float = 6.0
    min_object_area: int = 200

class DepthMasker:
    def __init__(self, config: MaskingConfig = MaskingConfig()):
        self.cfg = config
        
    def depth_to_points(self, depth: np.ndarray) -> np.ndarray:
        H, W = depth.shape
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        
        # Pinhole model (X right, Y down, Z forward)
        X = (xs - self.cfg.cx) * depth / self.cfg.fx
        Y = (ys - self.cfg.cy) * depth / self.cfg.fy
        Z = depth
        
        points = np.stack([X, Y, Z], axis=-1)
        return points

    def fit_plane_ransac(self, points: np.ndarray, mask: np.ndarray, 
                         orientation_axis: np.ndarray = None, 
                         max_angle_deg: float = 15.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generic plane fitter.
        If orientation_axis is provided (e.g. [0,1,0]), enforces normal aligns with it.
        """
        valid_points = points[mask & (points[..., 2] > 0)]
        if valid_points.shape[0] < 100:
            return None, None
            
        pts_flat = valid_points.reshape(-1, 3)
        n_points = pts_flat.shape[0]
        
        best_plane = None
        best_inliers_count = -1
        
        max_cos_angle = np.cos(np.radians(max_angle_deg)) # For parallel alignment logic (dot product close to 1)
        # However, for walls, we might want "perpendicular to Y", etc.
        # Let's handle floor (parallel to Y normal) specifically via the angle check logic inside check.
        
        for _ in range(self.cfg.ransac_iters):
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = pts_flat[idx]
            
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-6:
                continue
            normal /= n_norm
            
            # Constraints
            if orientation_axis is not None:
                # Dot product of normal and axis.
                # If axis is [0,1,0] (down), floor normal should be [0,1,0] or [0,-1,0].
                # Abs dot product should be close to 1.
                dot = np.abs(np.dot(normal, orientation_axis))
                # angle = acos(dot)
                # We want angle < max_angle_deg
                # So dot > cos(max_angle)
                if dot < max_cos_angle:
                    # Orientation mismatch
                    continue

            d_val = -np.dot(normal, p1)
            
            # Check coherence with a random subset to speed up? (Skipping for simplicity)
            
            # Count inliers
            # dist = |ax+by+cz+d|
            # Vectorized on ALL points (expensive? standard ransac checks all)
            # We can check subset or all. Let's check a random subset of 500 points for speed estimation
            subset_idx = np.random.choice(n_points, min(500, n_points), replace=False)
            subset_pts = pts_flat[subset_idx]
            dists = np.abs(np.dot(subset_pts, normal) + d_val)
            inliers = np.sum(dists < self.cfg.ransac_dist_thresh)
            
            if inliers > best_inliers_count:
                best_inliers_count = inliers
                best_plane = np.array([normal[0], normal[1], normal[2], d_val])
                
        if best_plane is None:
            return None, None
            
        # Recompute final mask on ALL points
        a, b, c, d_val = best_plane
        # dist_map = np.abs(points[..., 0]*a + points[..., 1]*b + points[..., 2]*c + d_val)
        # Optimized dot product
        # points shape (H,W,3), plane (4,)
        # sum(points * plane[:3], axis=2) + plane[3]
        dot = np.sum(points * best_plane[:3], axis=2) + best_plane[3]
        dist_map = np.abs(dot)
        
        plane_mask = (dist_map < self.cfg.ransac_dist_thresh) & mask
        return best_plane, plane_mask

    def get_clean_mask(self, depth: np.ndarray, rgb: Optional[np.ndarray] = None) -> dict:
        H, W = depth.shape
        points = self.depth_to_points(depth)
        
        # Base valid mask
        valid_depth = (depth > 0) & np.isfinite(depth) & (depth < self.cfg.max_dist)
        
        # 1. Floor Fit
        # Search in bottom 2/3rds for robustness
        # Y axis is down. Floor normal should be ~[0, 1, 0]
        # Allow deviation
        search_mask = valid_depth.copy()
        search_mask[:int(H/3), :] = False 
        
        floor_plane, floor_mask = self.fit_plane_ransac(
            points, search_mask, 
            orientation_axis=np.array([0, 1, 0]), 
            max_angle_deg=self.cfg.floor_max_angle
        )
        
        if floor_plane is None:
            floor_mask = np.zeros((H,W), dtype=bool)

        # 2. Wall Detection
        # Search remaining valid points for large vertical planes
        # Vertical plane -> Normal perpendicular to Y -> dot(n, [0,1,0]) ~ 0
        # So deviation from [0,1,0] should be > (90 - threshold)
        # Or simply align with X [1,0,0] or Z [0,0,1]? Walls can be any yaw.
        # Just check that dot(n, [0,1,0]) < sin(angle)?
        # For fit_plane_ransac, we can pass nothing and filter later, or specific constraints.
        # Let's iterate: find large plane, if vertical -> classify as wall.
        
        wall_mask = np.zeros((H,W), dtype=bool)
        remaining_mask = valid_depth & (~floor_mask)
        
        if self.cfg.remove_walls:
            # Try to find up to 3 walls
            for _ in range(3):
                # Search
                plane, pmask = self.fit_plane_ransac(points, remaining_mask)
                if plane is None:
                    break
                    
                # Check orientation
                # If vertical, normal dot Y is small.
                normal = plane[:3]
                dot_y = np.abs(normal[1]) # dot with [0,1,0]
                
                # If almost vertical (dot_y small, e.g. < 0.2 (~78 deg from vertical))
                # 15 deg from horizontal allowed -> 75 deg from vertical
                # cos(75) = 0.25
                if dot_y < 0.35: # fairly lenient, allowing slanted walls
                    # It's a wall
                    wall_mask |= pmask
                    remaining_mask &= (~pmask)
                else:
                    # It's a slanted surface or object face?
                    # If it's really large and not floor (since we already removed floor), maybe remove it?
                    # For now, only remove explicit walls.
                    # Stop if found something dominant that isn't a wall? Or just continue masking?
                    # Let's remove it from search but not add to 'wall_mask' (so it stays as object candidate?)
                    # No, if it's a big plane and not floor/wall, it might be a table top. Keep it.
                    # But RANSAC finds the largest plane. If table top is largest, we might loop forever.
                    # Break loop.
                    break
        
        # 3. Object Extraction
        # Objects are valid depth, not floor, not walls.
        object_mask = valid_depth & (~floor_mask) & (~wall_mask)
        
        # Further filter: Must be *above* floor?
        if floor_plane is not None:
             # Signed distance: if normal points up (Y negative?), eqn is ax+by+cz+d=0
             # We need to know which side is "up".
             # In Y-down, floor normal [0,1,0] points DOWN.
             # "Above" the floor is smaller Y.
             # Check a point we know is high (e.g. camera at 0,0,0).
             # Plane distance at 0 is 'd'.
             # If d > 0, normal points away from origin.
             # This is tricky without explicit calibration.
             # Heuristic: object points should have HEIGHT relative to floor > threshold.
             # Height = distance to plane?
             # Yes.
             pass

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        object_mask_u8 = object_mask.astype(np.uint8) * 255
        object_mask_cleaned = cv2.morphologyEx(object_mask_u8, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(object_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(object_mask_cleaned)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.cfg.min_object_area:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)
                
        clean_rgb = None
        if rgb is not None:
            clean_rgb = np.zeros_like(rgb)
            clean_rgb[final_mask == 255] = rgb[final_mask == 255]

        return {
            "floor_mask": floor_mask.astype(np.uint8) * 255,
            "wall_mask": wall_mask.astype(np.uint8) * 255,
            "object_mask": final_mask,
            "clean_rgb": clean_rgb
        }
