"""
potential_mapper.py
====================
Self-contained depth→occupancy→potential field pipeline.

Architecture (spatial_logic.md):
  - Temporary map  M_temp: rebuilt from the current depth frame (no persistence).
  - Accumulated map M_acc = (1 - α) * M_acc + α * M_temp  (EMA in probability space).
  - Repulsive potential  U_rep, computed via ``PotentialFieldLayer``.
  - Gradient field ∇U_rep = np.gradient(U_rep) / resolution — flows AWAY from obstacles.

No ROS, no TRT, no HF: only numpy + cv2 (cv2 needed by PotentialFieldLayer).

Complexity notes are per call unless otherwise stated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Intrinsics  # re-exported from types.perception
from sparx_agency.core.mapping.costmap.potential_field_layer import PotentialFieldLayer
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import sigmoid, bresenham, update_ray_logodds, \
    fast_process_endpoints


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PotentialMapperConfig:
    resolution_m: float = 0.10
    size_m: float = 40.0
    alpha: float = 0.30
    occ_thresh: float = 0.7
    sigma_m: float = 0.25
    repulse_radius_m: float = 0.35
    inflation_radius_m: float = 0.01
    z_band: Tuple[float, float] = (0.05, 2.0)
    range_min_m: float = 0.2
    range_max_m: float = 15.0
    robot_clearance_m: float = 0.15
    stride: int = 2
    pitch_deg: float = 0.0
    height_m: float = 1.0
    zeta: float = 1.0
    min_wall_length_m: float = 0.3
    max_wall_gap_m: float = 0.15
    lo_occ: float = 2.0
    lo_free: float = -0.7
    lo_min: float = -3.5
    lo_max: float = 3.5
    att_radius_m: float = 2.5
    k_att: float = 1.0
    k_rep: float = 3.5
    ray_endpoint_stride: int = 1
    reference_scale: float = 1.0
    nav_memory_weight: float = 0.60
    unknown_decay: float = 0.995
    use_temp_for_navigation: bool = True


    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be > 0.")
        if self.size_m <= 0.0:
            raise ValueError("size_m must be > 0.")
        if self.sigma_m <= 0.0:
            raise ValueError("sigma_m must be > 0.")
        if self.range_min_m < 0.0:
            raise ValueError("range_min_m must be >= 0.")
        if self.range_max_m <= self.range_min_m:
            raise ValueError("range_max_m must be > range_min_m.")
        z_min, z_max = self.z_band
        if z_max <= z_min:
            raise ValueError("z_band[1] must be > z_band[0].")
        if not (0.0 <= self.nav_memory_weight <= 1.0):
            raise ValueError("nav_memory_weight must be in [0, 1].")
        if not (0.0 < self.unknown_decay <= 1.0):
            raise ValueError("unknown_decay must be in (0, 1].")
        if self.att_radius_m <= 0.0:
            raise ValueError("att_radius_m must be > 0.")

# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

class PotentialMapper:
    def __init__(self, cfg: Optional[PotentialMapperConfig] = None) -> None:
        self.cfg = cfg or PotentialMapperConfig()
        print("RUNNING POTENTIAL_MAPPER FROM:", __file__)
        n_cells = int(round(self.cfg.size_m / self.cfg.resolution_m))
        self._n = n_cells
        self._origin_fwd = 0.0
        self._origin_left = 0.5 * self.cfg.size_m

        self._M_acc: np.ndarray = np.zeros((n_cells, n_cells), dtype=np.float32)
        self._M_temp: np.ndarray = np.zeros((n_cells, n_cells), dtype=np.float32)
        self._M_nav: np.ndarray = np.zeros((n_cells, n_cells), dtype=np.float32)

        self._U_rep: np.ndarray = np.zeros((n_cells, n_cells), dtype=np.float32)
        self._D_obs: np.ndarray = np.full((n_cells, n_cells), np.inf, dtype=np.float32)
        self._grad: np.ndarray = np.zeros((n_cells, n_cells, 2), dtype=np.float32)
        self._grad_rep = None

        self._rays: Optional[np.ndarray] = None
        self._last_intrinsics: Optional[Intrinsics] = None

        self._goal_world: Optional[Tuple[float, float]] = None
        self._wall_segments: np.ndarray = np.array([])
        self._M_walls: np.ndarray = np.zeros((n_cells, n_cells), dtype=np.float32)

        self._U_att = np.zeros((self._n, self._n), dtype=np.float32)
        self._U_total = np.zeros((self._n, self._n), dtype=np.float32)
        self._grad_total = np.zeros((self._n, self._n, 2), dtype=np.float32)

        self._potential = PotentialFieldLayer(
            occ_thresh=self.cfg.occ_thresh,
            sigma_m=self.cfg.sigma_m,
            k_rep=3.5,
            repulse_radius_m=self.cfg.repulse_radius_m,
            inflation_radius_m=self.cfg.inflation_radius_m,
            u_max=1.0,
            unknown_as_obstacle=False,
        )

    def update(
        self,
        point_cloud: np.ndarray,
        *,
        delta_fwd_m: float = 0.0,
        delta_left_m: float = 0.0,
        delta_yaw_deg: float = 0.0,
    ) -> None:
        """
        Update maps from a 3D cloud in the current robot frame.
        """
        pts = point_cloud.reshape(-1, 3) if len(point_cloud.shape) == 3 else point_cloud
        pts_filtered = self._filter_cloud(pts)
        self._M_temp = self._build_temp_map(pts_filtered)

        self._M_acc = self._warp_probability_grid(
            self._M_acc,
            delta_fwd_m=delta_fwd_m,
            delta_left_m=delta_left_m,
            delta_yaw_deg=delta_yaw_deg,
        )
        self._M_acc *= self.cfg.unknown_decay

        mask = np.isfinite(self._M_temp)
        self._M_acc[mask] = (
            (1.0 - self.cfg.alpha) * self._M_acc[mask]
            + self.cfg.alpha * self._M_temp[mask]
        )

        self._M_nav = self._compose_navigation_map()
        self._M_walls = self._detect_walls_and_clean()

        nav_grid = np.maximum(self._M_nav, self._M_walls)
        # 2. Compute the repulsive potential
        u_rep_raw, d_obs = self._potential.compute_from_prob_grid(nav_grid, self.cfg.resolution_m)
        self._U_rep = u_rep_raw
        self._grad_rep = self.compute_gradient_from_potential(u_rep_raw)
        self._D_obs = d_obs

        # 3. Compute the attractive potential
        self._compute_attractive_potential()

        if self._goal_world is not None:
            gr, gc = self._goal_to_cell(*self._goal_world)
            print("U_att at goal cell =", float(self._U_att[gr, gc]))
            print("U_att min/max =", float(self._U_att.min()), float(self._U_att.max()))
        # 4. COMBINE AND SATURATE
        # Note: k_rep and k_att should be tuned so that at "danger" distance,
        # they sum to a value around 1.0 to 3.0 before tanh.
        u_combined = (self.cfg.k_rep * self._U_rep + self.cfg.k_att * self._U_att)
        self._U_total = u_combined.astype(np.float32)
        # 5. COMPUTE GRADIENT FROM THE SATURATED FIELD
        # We use negative gradient because we want to move TOWARDS lower potential
        # Single gradient: -∇U_total
        self._grad_total = self.compute_gradient_from_potential(self._U_total)
        self._grad = self._grad_total.copy()

    def compute_gradient_from_potential(self, potential: np.ndarray) -> np.ndarray:
        """Compute descent direction from potential field in [forward, left]."""
        g_row, g_col = np.gradient(potential, self.cfg.resolution_m)
        # row axis matches forward directly
        grad_fwd = -g_row
        grad_left = +g_col

        raw_grad = np.stack([grad_fwd, grad_left], axis=-1).astype(np.float32)

        # mag = np.linalg.norm(raw_grad, axis=-1) + 1e-8
        # max_force = 1.0
        # scale = np.minimum(1.0, max_force / mag)
        return raw_grad

    def get_nav_map(self) -> np.ndarray:
        """Return the fused occupancy grid used for potential-field generation."""
        return self._M_nav.copy()

    def _compose_navigation_map(self) -> np.ndarray:
        """
        Fuse current-frame evidence with accumulated memory.

        Fresh obstacles from M_temp dominate.
        Memory from M_acc fills in regions not currently observed.
        """
        if self.cfg.use_temp_for_navigation:
            return self._M_temp.copy()

        m_nav = (self.cfg.nav_memory_weight * self._M_acc).astype(np.float32)

        if not self.cfg.use_temp_for_navigation:
            return m_nav

        valid_temp = np.isfinite(self._M_temp)
        m_nav[valid_temp] = np.maximum(m_nav[valid_temp], self._M_temp[valid_temp]).astype(np.float32)
        return m_nav

    def _warp_probability_grid(
            self,
            grid: np.ndarray,
            *,
            delta_fwd_m: float,
            delta_left_m: float,
            delta_yaw_deg: float,
    ) -> np.ndarray:
        """
        Warp previous egocentric occupancy into the current robot frame.

        Positive delta_fwd_m means the robot moved forward.
        Positive delta_left_m means the robot moved left.
        Positive delta_yaw_deg means CCW yaw in the map plane.
        """
        if not np.any(np.isfinite(grid)):
            return grid.copy()

        try:
            import cv2
        except ImportError:
            return grid.copy()

        h, w = grid.shape
        center = ((w - 1) * 0.5, 0.0)

        shift_x = -delta_left_m / self.cfg.resolution_m
        shift_y = -delta_fwd_m / self.cfg.resolution_m

        rot = cv2.getRotationMatrix2D(center, -delta_yaw_deg, 1.0)
        rot[:, 2] += np.array([shift_x, shift_y], dtype=np.float32)

        warped = cv2.warpAffine(grid.astype(np.float32),
                                rot.astype(np.float32),
                                (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.0,)
        return warped.astype(np.float32)

    def reset(self) -> None:
        """Zero all internal maps and the navigation goal."""
        self._M_acc.fill(0.0)
        self._M_temp.fill(0.0)
        self._M_nav.fill(0.0)
        self._M_walls.fill(0.0)
        self._U_rep.fill(0.0)
        self._D_obs.fill(np.inf)
        self._grad.fill(0.0)
        self._U_total.fill(0.0)
        self._grad_total.fill(0.0)
        self._grad_rep = None
        self._goal_world = None
        self._wall_segments = np.array([])

    def get_repulsive_gradient(self) -> np.ndarray:
        return self._grad_rep.copy()

    def get_prob_map(self) -> np.ndarray:
        """Return the accumulated probability map M_acc.

        Returns:
            (H, W) float32 in [0, 1].

        Complexity: O(1).
        """
        return self._M_acc.copy()

    def get_temp_map(self) -> np.ndarray:
        """Return the last temporary (single-frame) probability map.

        Returns:
            (H, W) float32 in [0, 1].

        Complexity: O(1).
        """
        return self._M_temp.copy()

    def get_potential_map(self) -> np.ndarray:
        """Return the repulsive potential field U_rep.

        Returns:
            (H, W) float32 in [0, 1].

        Complexity: O(1).
        """
        return self._U_rep.copy()

    def get_gradient_field(self) -> np.ndarray:
        """Return the spatial gradient of U_rep.

        The gradient points AWAY from obstacles (toward lower potential), so
        ∇U_rep is the direction a robot should move to avoid obstacles.

        Returns:
            (H, W, 2) float32 where [..., 0] = grad_fwd, [..., 1] = grad_left.

        Complexity: O(1) — gradient is pre-computed in ``update()``.
        """
        return self._grad.copy()

    def set_goal(self, fwd: float, left: float) -> None:
        """Set navigation goal in world coordinates (metres)."""
        self._goal_world = (fwd, left)

    def get_total_gradient(self) -> np.ndarray:
        """Return the combined descent direction field (-∇U_total)."""
        return self._grad_total.copy()


    def get_distance_to_obstacle(self) -> np.ndarray:
        """Return the distance-to-nearest-obstacle map.

        Returns:
            (H, W) float32 in metres.

        Complexity: O(1).
        """
        return self._D_obs.copy()

    def _compute_attractive_potential(self) -> None:
        if self._goal_world is None:
            self._U_att.fill(0.0)
            return

        goal_fwd, goal_left = self._goal_world

        rows = np.arange(self._n, dtype=np.float32)
        cols = np.arange(self._n, dtype=np.float32)

        fwd_coords = rows * self.cfg.resolution_m + self._origin_fwd
        left_coords = self._origin_left - cols * self.cfg.resolution_m

        df = goal_fwd - fwd_coords[:, None]
        dl = goal_left - left_coords[None, :]

        dist = np.sqrt(df * df + dl * dl).astype(np.float32)
        # Implementation of the "Smooth" Combined Potential:
        # 1. Quadratic near the goal (smooth basin)
        # 2. Conic far from the goal (constant pull)
        d0 = self.cfg.att_radius_m  # The "transition" distance

        u_att = np.where(
            dist <= d0,
            0.5 * (dist ** 2),  # Quadratic basin
            d0 * dist - 0.5 * (d0 ** 2)  # Linear cone
        )

        self._U_att = u_att.astype(np.float32)

    def get_attractive_potential(self) -> np.ndarray:
        """Return the attractive potential field U_att."""
        self._compute_attractive_potential()
        return self._U_att.copy()

    def get_total_potential(self) -> np.ndarray:
        """Return total potential U_total."""
        return self._U_total.copy()

    def _goal_to_cell(self, goal_fwd_m: float, goal_left_m: float) -> tuple[int, int]:
        """
        Convert goal in (forward,left) meters into (row,col) indices.

        Row increases with forward.
        Col increases to the RIGHT in the grid indexing you used with gl:
          gl = (origin_left - left) / res
        """
        n = self._n
        res = self.cfg.resolution_m

        gr = int((goal_fwd_m - self._origin_fwd) / res)
        gc = int((self._origin_left - goal_left_m) / res)

        gr = max(0, min(n - 1, gr))
        gc = max(0, min(n - 1, gc))
        return gr, gc

    def rollout_trajectory_to_goal(
            self,
            start_fwd: float,
            start_left: float,
            step_m: float = 0.05,
            max_steps: int = 200,
            goal_tol_m: float = 0.25,
            obstacle_margin_m: float = 0.15,
            min_speed: float = 1e-3,
    ) -> np.ndarray:
        """
        Roll out a trajectory by following the total vector field.

        Returns:
            np.ndarray of shape (N, 2), each row = [fwd, left]
        """
        if self._goal_world is None:
            return np.array([[start_fwd, start_left]], dtype=np.float32)

        goal_fwd, goal_left = self._goal_world
        traj = []
        visited = []

        fwd = float(start_fwd)
        left = float(start_left)

        for _ in range(max_steps):
            traj.append([fwd, left])

            # goal reached
            dg = np.hypot(goal_fwd - fwd, goal_left - left)
            if dg <= goal_tol_m:
                break

            # obstacle / invalid area check
            if self.is_occupied_metric(fwd, left, margin_m=obstacle_margin_m):
                break

            # get local vector
            k1 = self.sample_total_vector_metric(fwd, left)
            speed1 = float(np.linalg.norm(k1))
            if speed1 < min_speed:
                break
            k1 = k1 / speed1

            mid_fwd = fwd + 0.5 * step_m * float(k1[0])
            mid_left = left + 0.5 * step_m * float(k1[1])

            k2 = self.sample_total_vector_metric(mid_fwd, mid_left)
            speed2 = float(np.linalg.norm(k2))
            if speed2 < min_speed:
                break
            k2 = k2 / speed2

            # integrate one step with midpoint direction
            fwd_next = fwd + step_m * float(k2[0])
            left_next = left + step_m * float(k2[1])

            # loop / oscillation check
            for pf, pl in visited[-15:]:
                if np.hypot(fwd_next - pf, left_next - pl) < 0.5 * step_m:
                    return np.array(traj, dtype=np.float32)

            visited.append((fwd, left))
            fwd, left = fwd_next, left_next

        return np.array(traj, dtype=np.float32)


    @property
    def grid_shape(self) -> Tuple[int, int]:
        """(H, W) integer grid dimensions."""
        return (self._n, self._n)

    @property
    def resolution_m(self) -> float:
        """Grid resolution in metres per cell."""
        return self.cfg.resolution_m

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backproject(self, depth_m: np.ndarray, intr: Intrinsics) -> np.ndarray:
        """Back-project using V3-style ray multiplication.
        
        Matches C++ snippet convention:
        - X: Left
        - Y: Up
        - Z: Forward
        """
        H, W = depth_m.shape
        s = max(1, int(self.cfg.stride))

        if self._rays is None or self._last_intrinsics != intr:
            self._rays = self._get_rays(intr, s)
            self._last_intrinsics = intr

        d = depth_m[::s, ::s]
        mask = np.isfinite(d) & (d > 0.0)
        pts_sampled = self._rays[mask] * d[mask][:, np.newaxis]

        # pts_sampled is in Optical frame: [X_right, Y_down, Z_fwd]
        xr, yd, zf = pts_sampled[:, 0], pts_sampled[:, 1], pts_sampled[:, 2]

        # Convert to Base frame (matching C++ snippet swap)
        # Left = -Right, Up = -Down, Forward = depth
        xl, yu, zf = -xr, -yd, zf
        
        # Apply Pitch (Rotation around Left-axis X) and Height
        # Forward (Z) and Up (Y) rotate.
        p = np.deg2rad(self.cfg.pitch_deg)
        cos_p, sin_p = np.cos(p), np.sin(p)
        
        # Looking down (pitch > 0) means Forward ray gets negative Up component
        # z_final (Forward) = z_fwd * cos(p) + y_up * sin(p)
        # y_final (Up)      = -z_fwd * sin(p) + y_up * cos(p) + height
        z_final = zf * cos_p + yu * sin_p
        y_final = -zf * sin_p + yu * cos_p + self.cfg.height_m
        x_final = xl

        # Return as [Left, Up, Forward] per C++ snippet
        return np.stack([x_final, y_final, z_final], axis=1).astype(np.float32)

    def _detect_walls_and_clean(self) -> np.ndarray:
        """Use HoughLinesP to extract straight segments and clear noise."""
        import cv2
        m_walls = np.zeros_like(self._M_acc)
        
        # Binarize
        binary = (self._M_acc >= self.cfg.occ_thresh).astype(np.uint8) * 255
        if np.count_nonzero(binary) < 5:
            self._wall_segments = np.array([])
            return m_walls

        # Use contour detection to trace obstacles
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segments = []
        min_len_cells = self.cfg.min_wall_length_m / self.cfg.resolution_m
        
        # Camera is at row 0, centre column (robot position)
        cam_col = self._n / 2.0
        cam_row = 0.0
        
        # We filter segments that are roughly parallel to the camera ray (flying pixels)
        # Cosine of 25 degrees (to be safe and catch slightly curved artifacts)
        cos_thresh = np.cos(np.deg2rad(25.0))
        
        # Create a fresh, mask that only contains the valid structure
        clean_binary = np.zeros_like(binary)
        
        for cnt in contours:
            # Approximate the boundary to extract straight line segments
            epsilon = 1.0
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
            pts = approx.reshape(-1, 2)
            n_pts = len(pts)
            
            if n_pts >= 2:
                for i in range(n_pts):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % n_pts]
                    
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = np.linalg.norm([dx, dy])
                    
                    if length < min_len_cells:
                        continue
                        
                    # Ray from camera to segment midpoint
                    mid_x = (p1[0] + p2[0]) / 2.0
                    mid_y = (p1[1] + p2[1]) / 2.0
                    rx = mid_x - cam_col
                    ry = mid_y - cam_row
                    r_len = np.linalg.norm([rx, ry])
                    
                    if r_len > 0 and length > 0:
                        dot = (dx * rx + dy * ry) / (length * r_len)
                        if abs(dot) > cos_thresh:
                            # This segment is pointing directly toward/away from the camera!
                            # It is a "rubber-sheet" flying-pixel artifact, e.g. the `/` or `\` in a doorway.
                            # We deliberately drop it to break the false bridge.
                            continue
                            
                    # Draw only the valid segment into the cleaned mask
                    cv2.line(clean_binary, (p1[0], p1[1]), (p2[0], p2[1]), 255, thickness=2)

        # Now apply HoughLinesP on the clean_binary mask to merge collinear segments correctly!
        max_gap = int(self.cfg.max_wall_gap_m / self.cfg.resolution_m)
        lines = cv2.HoughLinesP(
            clean_binary, 
            rho=1, 
            theta=np.pi/180, 
            threshold=int(min_len_cells), 
            minLineLength=int(min_len_cells), 
            maxLineGap=max_gap
        )

        if lines is not None:
            self._wall_segments = lines.squeeze(axis=1) # [N, 4]
            for line in self._wall_segments:
                x1, y1, x2, y2 = line
                # Draw the final merged straight walls into the probability map with thickness
                cv2.line(m_walls, (x1, y1), (x2, y2), 1.0, thickness=1)
        else:
            self._wall_segments = np.array([])

        return m_walls


    def _get_rays(self, intr: Intrinsics, stride: int) -> np.ndarray:
        """Precompute ray directions (X/fx, Y/fy, 1.0) for the grid."""
        ys = np.arange(0, intr.height, stride)
        xs = np.arange(0, intr.width, stride)
        xv, yv = np.meshgrid(xs, ys)
        
        rx = (xv - intr.cx) / intr.fx
        ry = (yv - intr.cy) / intr.fy
        rz = np.ones_like(rx)
        
        return np.stack([rx, ry, rz], axis=-1).astype(np.float32)

    def _filter_cloud(self, pts: np.ndarray) -> np.ndarray:
        """Filter points by range and height band.
        
        pts: [Left, Up, Forward]
        """
        if pts.shape[0] == 0:
            return pts

        xl, yu, zf = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(xl * xl + zf * zf)
        z_min, z_max = self.cfg.z_band # z_band is actually "Up" band

        mask = (
            np.isfinite(xl) & np.isfinite(yu) & np.isfinite(zf)
            & (r >= self.cfg.range_min_m) & (r <= self.cfg.range_max_m)
            & (yu >= z_min) & (yu <= z_max)
        )
        return pts[mask]

    def _build_temp_map(self, pts: np.ndarray) -> np.ndarray:
        n = self._n
        L = np.zeros((n, n), dtype=np.float32)
        seen = np.zeros((n, n), dtype=bool)

        if pts.shape[0] == 0:
            return np.full((n, n), np.nan, dtype=np.float32)

        # 1. Spatial Filtering
        zf, xl = pts[:, 2], pts[:, 0]
        mask_far = (xl ** 2 + zf ** 2) > (self.cfg.robot_clearance_m ** 2)
        zf, xl = zf[mask_far], xl[mask_far]

        if zf.size == 0:
            return np.full((n, n), np.nan, dtype=np.float32)

        # 2. Grid Projection
        gz = ((zf - self._origin_fwd) / self.cfg.resolution_m).astype(np.int32)
        gl = ((self._origin_left - xl) / self.cfg.resolution_m).astype(np.int32)

        # 3. Bounds & Uniqueness
        inb = (gz >= 0) & (gz < n) & (gl >= 0) & (gl < n)
        gz, gl = gz[inb], gl[inb]
        idx = np.unique(gz * n + gl)  # Only raycast once per unique cell
        gz, gl = (idx // n).astype(np.int32), (idx % n).astype(np.int32)

        # 4. Fast Numba Execution
        fast_process_endpoints(
            L, seen,
            0, n // 2,  # Robot origin
            gz, gl,
            self.cfg.lo_free, self.cfg.lo_occ,
            self.cfg.lo_min, self.cfg.lo_max
        )

        # 5. Conversion (keep the NaN for unknown space)
        M_temp = sigmoid(L).astype(np.float32)
        M_temp[~seen] = np.nan
        return M_temp

    def metric_to_cell(self, fwd: float, left: float) -> tuple[int, int] | None:
        r = int((fwd - self._origin_fwd) / self.cfg.resolution_m)
        c = int((self._origin_left - left) / self.cfg.resolution_m)

        if r < 0 or r >= self._n or c < 0 or c >= self._n:
            return None
        return r, c

    def sample_total_vector_metric(self, fwd: float, left: float) -> np.ndarray:
        """
        Sample the SAME total vector field used for debug visualization,
        using bilinear interpolation in metric coordinates.
        """
        if self._grad_total is None:
            return np.zeros(2, dtype=np.float32)

        rf = (fwd - self._origin_fwd) / self.cfg.resolution_m
        cf = (self._origin_left - left) / self.cfg.resolution_m

        if rf < 0 or cf < 0 or rf >= self._n - 1 or cf >= self._n - 1:
            return np.zeros(2, dtype=np.float32)

        r0 = int(np.floor(rf))
        c0 = int(np.floor(cf))
        r1 = r0 + 1
        c1 = c0 + 1

        dr = rf - r0
        dc = cf - c0

        v00 = self._grad_total[r0, c0]
        v01 = self._grad_total[r0, c1]
        v10 = self._grad_total[r1, c0]
        v11 = self._grad_total[r1, c1]

        v_top = (1.0 - dc) * v00 + dc * v01
        v_bot = (1.0 - dc) * v10 + dc * v11
        v = (1.0 - dr) * v_top + dr * v_bot

        return v.astype(np.float32)

    def is_occupied_metric(self, fwd: float, left: float, margin_m: float = 0.15) -> bool:
        rc = self.metric_to_cell(fwd, left)
        if rc is None:
            return True

        r, c = rc
        margin_cells = max(1, int(round(margin_m / self.cfg.resolution_m)))

        r0 = max(0, r - margin_cells)
        r1 = min(self._n, r + margin_cells + 1)
        c0 = max(0, c - margin_cells)
        c1 = min(self._n, c + margin_cells + 1)

        patch = self._M_nav[r0:r1, c0:c1]
        patch = np.nan_to_num(patch, nan=0.0)

        return np.any(patch > self.cfg.occ_thresh)


