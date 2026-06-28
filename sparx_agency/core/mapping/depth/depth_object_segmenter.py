from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional

class DepthObjectSegmenter:
    """
    Segments objects from RGB-D images using geometric heuristics:
    1. Back-projection to 3D point cloud.
    2. RANSAC plane fitting to remove the floor.
    3. Bird's Eye View (BEV) 2D projection of remaining points.
    4. Connected components on the BEV map to find object clusters.
    5. Projection of clusters back to 2D image bounding boxes.
    """

    def __init__(self,
                 focal_length_px: float = 500.0,
                 principal_point: Optional[Tuple[float, float]] = None,
                 ransac_thresh: float = 0.05,
                 cluster_min_points: int = 100,
                 bev_resolution: float = 0.05):
        """
        Args:
            focal_length_px: Estimated or real focal length in pixels.
            principal_point: (cx, cy). If None, center of image is used.
            ransac_thresh: Distance threshold for RANSAC plane fitting (meters).
            cluster_min_points: Minimum number of 3D points to constitute an object.
            bev_resolution: Size of one pixel in the BEV grid in meters.
        """
        self.focal_length = focal_length_px
        self.cy = principal_point[1] if principal_point else None
        self.cx = principal_point[0] if principal_point else None
        self.ransac_thresh = ransac_thresh
        self.cluster_min_points = cluster_min_points
        self.bev_res = bev_resolution

    def segment_objects(self, depth_map: np.ndarray, rgb_img: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Process a depth map to find objects.
        
        Args:
            depth_map: HxW float32 array, depth in meters.
            rgb_img: Optional HxW, used for dimensions or debugging.
            
        Returns:
            List of dicts: {'bbox': (x1,y1,x2,y2), 'avg_depth': float, 'confidence': float}
        """
        h, w = depth_map.shape
        if self.cx is None: self.cx = w / 2.0
        if self.cy is None: self.cy = h / 2.0

        # 1. Back-project to 3D (Z is depth)
        # We process only valid depth pixels to save time? 
        # Actually doing full array ops is often faster in numpy than masking indices for small imgs.
        # But for huge images, subsampling is smart. Let's do a subsample if large.
        scale = 1
        if w > 640:
            scale = 4  # Process at lower res for geometric segmentation speed
            depth_small = cv2.resize(depth_map, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
        else:
            scale = 1
            depth_small = depth_map

        h_s, w_s = depth_small.shape
        
        # Grid of coordinates
        xx, yy = np.meshgrid(np.arange(w_s), np.arange(h_s))
        
        # Adjust intrinsic for scale
        fx = self.focal_length / scale
        fy = self.focal_length / scale
        cx = self.cx / scale
        cy = self.cy / scale
        
        # Z = depth
        z_grid = depth_small
        
        # X = (u - cx) * Z / fx
        x_grid = (xx - cx) * z_grid / fx
        # Y = (v - cy) * Z / fy
        y_grid = (yy - cy) * z_grid / fy
        
        # Stack to (N, 3)
        points = np.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=1)
        
        # Filter invalid depths (e.g. 0 or very far)
        valid_mask = (points[:, 2] > 0.1) & (points[:, 2] < 20.0)
        points = points[valid_mask]
        
        print(f"DEBUG: Valid points: {len(points)} / {len(valid_mask)} (Z range: {points[:, 2].min():.2f} - {points[:, 2].max():.2f})")
        
        if len(points) < 100:
            print("DEBUG: Too few points.")
            return []

        if len(points) < 100:
            print("DEBUG: Too few points.")
            return []

        # 2. RANSAC for Floor
        # We only really want the MAIN floor.
        floor_model, inliers = self._find_floor_plane(points)
        
        if floor_model is None:
            print("DEBUG: No floor found.")
            object_points = points
            # Cannot rectify without floor, assuming camera is level
            rotated_points = points
        else:
            n_inliers = np.sum(inliers)
            ratio = n_inliers / len(points)
            print(f"DEBUG: Floor found. Model: {floor_model}, Inliers: {n_inliers} ({ratio:.1%})")
            
            # Remove floor
            object_points = points[~inliers]
            
            # RECTIFICATION: Rotate points so floor normal aligns with Y axis (0, 1, 0)
            # Floor model: ax + by + cz + d = 0 -> normal is (a,b,c)
            normal = floor_model[:3]
            # Ensure normal points "up" or "down" consistently?
            # We want it to align with [0, 1, 0] (Vertical down in CV or up?)
            # Usually camera Y is down. Let's align to [0, 1, 0].
            
            target_up = np.array([0.0, 1.0, 0.0])
            
            # Check direction. If normal is roughly [0, -1, 0], flip it?
            # Dot product
            if np.dot(normal, target_up) < 0:
                normal = -normal
                
            # Rotation that aligns 'normal' to 'target_up'
            R = self._get_rotation_matrix(normal, target_up)
            
            # Rotate OBJECT points
            # P_rot = R * P.T
            rotated_points = np.dot(object_points, R.T)


        print(f"DEBUG: Object points: {len(object_points)}")
        if len(object_points) < self.cluster_min_points:
            return []

        # 3. BEV Clustering
        # Project to X-Z plane (top down) in the ROTATED frame
        
        x_vals = rotated_points[:, 0]
        z_vals = rotated_points[:, 2]
        # Y vals are rotated_points[:, 1] -> Height above floor
        
        # We can also filter by Height now!
        # If we aligned floor to Y, then floor should be roughly at Y = -d (after rotation?)
        # Actually, with [a,b,c,0] rotation, the *offset* d changes or we just look at relative Y.
        # Let's just trust RANSAC removed the floor points.
        
        min_x, max_x = x_vals.min(), x_vals.max()
        min_z, max_z = z_vals.min(), z_vals.max()
        if len(object_points) < self.cluster_min_points:
            return []

        # 3. BEV Clustering
        # Project to X-Z plane (top down)
        # X is horizontal, Z is depth. Y is height (which we just removed floor from)
        
        x_vals = object_points[:, 0]
        z_vals = object_points[:, 2]
        
        min_x, max_x = x_vals.min(), x_vals.max()
        min_z, max_z = z_vals.min(), z_vals.max()
        
        bev_w = int((max_x - min_x) / self.bev_res) + 1
        bev_h = int((max_z - min_z) / self.bev_res) + 1
        
        # Safety clamp
        if bev_w > 2000 or bev_h > 2000:
            return [] # Too spread out/invalid
            
        bev_grid = np.zeros((bev_h, bev_w), dtype=np.uint8)
        
        # Map points to grid indices
        xi = ((x_vals - min_x) / self.bev_res).astype(np.int32)
        zi = ((z_vals - min_z) / self.bev_res).astype(np.int32)
        
        # Clip just in case
        xi = np.clip(xi, 0, bev_w - 1)
        zi = np.clip(zi, 0, bev_h - 1)
        
        # Mark occupancy
        bev_grid[zi, xi] = 255
        
        # Morphological close to join generic density
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bev_closed = cv2.morphologyEx(bev_grid, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bev_closed)
        
        results = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 5: # Ignore tiny noise in grid
                continue
                
            # Create mask for this cluster
            cluster_mask = (labels == i)
            
            # Find which 3D points fall into this cluster
            # This is the tricky "inverse" part.
            # We can re-check indices.
            
            # Points belonging to this cluster
            # (zi, xi) correspond to this label
            
            # Efficient way:
            # We know for every object_point 'k', it mapped to (zi[k], xi[k]).
            # We check if labels[zi[k], xi[k]] == i
            
            # Ensure indices valid for lookup
            # (zi and xi are parallel to object_points)
            point_labels = labels[zi, xi]
            mask_3d = (point_labels == i)
            
            cluster_pts = object_points[mask_3d]
            
            if len(cluster_pts) < self.cluster_min_points:
                continue
                
            # 4. Project back to 2D BBox
            # u = fx * X / Z + cx
            # v = fy * Y / Z + cy
            # Remember to un-scale if we scaled down!
            
            X = cluster_pts[:, 0]
            Y = cluster_pts[:, 1]
            Z = cluster_pts[:, 2]
            
            # Original intrinsics
            u = (self.focal_length * X / Z) + self.cx
            v = (self.focal_length * Y / Z) + self.cy
            
            x1, x2 = np.min(u), np.max(u)
            y1, y2 = np.min(v), np.max(v)
            
            # Clamp to image
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            results.append({
                "bbox": (x1, y1, x2, y2),
                "avg_depth": float(np.mean(Z)),
                "num_points": len(cluster_pts)
            })
            
        return results

    def _find_floor_plane(self, points: np.ndarray, iterations=50) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Simple RANSAC to find the largest plane.
        Returns: (plane_eq, inlier_mask)
        plane_eq is [a,b,c,d] where ax+by+cz+d=0
        """
        if len(points) < 50:
            return None, np.zeros(len(points), dtype=bool)
            
        best_inliers = np.zeros(len(points), dtype=bool)
        best_count = 0
        best_model = None
        
        n_points = len(points)
        
        # Heuristic: The floor is likely "below" (positive Y in many CV coords) 
        # but let's just find the dominant plane for now.
        
        for _ in range(iterations):
            # Sample 3 random points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p1
            
            # Normal
            normal = np.cross(v1, v2)
            n_norm = np.linalg.norm(normal)
            if n_norm == 0: continue
            normal = normal / n_norm
            
            # d
            d = -np.dot(normal, p1)
            
            # Distance of all points to plane
            # dist = |ax+by+cz+d| / 1 (since normal is normalized)
            dists = np.abs(np.dot(points, normal) + d)
            
            inliers = dists < self.ransac_thresh
            count = np.sum(inliers)
            
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_model = np.append(normal, d)
                
        # If best plane contains < 20% of points, maybe no floor?
        # Or maybe it is a wall?
        # Validating orientation could help (normal should be roughly vertical)
        # But for generic tasks, removing dominant plane is a good heuristic for "background/floor".
        
        return best_model, best_inliers

    def _get_rotation_matrix(self, vec1, vec2):
        """ Returns rotation matrix that maps vec1 to vec2 """
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0: return np.eye(3)
        
        vec1 = vec1 / n1
        vec2 = vec2 / n2
        
        v = np.cross(vec1, vec2)
        s = np.linalg.norm(v)
        c = np.dot(vec1, vec2)
        
        if s < 1e-6:
             # Parallel
             if c > 0:
                 return np.eye(3)
             else:
                 # Anti-parallel: return 180 deg rotation about X (if vec along Y) or random
                 # Just returning -I for simplicity (point reflection) -> actually technically rotation + reflection?
                 # Proper way: finding orthogonal axis.
                 # Safe fallback:
                 return -np.eye(3)

        v_skew = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        R = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / (s**2))
        return R
