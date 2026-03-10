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
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import sigmoid, bresenham, update_ray_logodds


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PotentialMapperConfig:
    """Configuration for ``PotentialMapper``.

    Args:
        resolution_m: Grid cell size in metres.
        size_m: Side length of the square map in metres.
        alpha: EMA blending factor in (0, 1].
            α = 1.0 means no decay (each frame completely replaces the past).
            α ≈ 0.3 gives smooth, slow-decaying memory.
        occ_thresh: Probability threshold above which a cell is treated as
            occupied when computing the potential field.
        sigma_m: Gaussian sigma (metres) for the repulsive potential falloff.
        inflation_radius_m: Hard inflation zone around obstacles (metres).
        z_band: (z_min, z_max) in the base frame; only points within this band
            are classified as obstacles for 2-D mapping.
        range_min_m: Minimum depth for valid cloud points.
        range_max_m: Maximum depth for valid cloud points.
        stride: Pixel sampling stride when back-projecting depth to a cloud.
            stride=2 gives 4× speedup over full-resolution.
    """
    resolution_m: float = 0.10
    size_m: float = 40.0
    alpha: float = 0.30
    occ_thresh: float = 0.55
    sigma_m: float = 0.2          # Sharper decay for tighter navigation
    inflation_radius_m: float = 0.20 # Reduced to allow passing through doorways
    z_band: Tuple[float, float] = (0.05, 2.0) # Lowered to 0.05 to catch low chairs
    range_min_m: float = 0.2          # Lowered to 0.2 for close objects
    range_max_m: float = 15.0

    stride: int = 2
    pitch_deg: float = 0.0
    height_m: float = 1.0
    zeta: float = 0.5           # Navigation gain (normalized attraction strength)
    
    # Wall Segmentation (Revision 5)
    min_wall_length_m: float = 0.3 # Min length to be a "wall" segment
    max_wall_gap_m: float = 0.15    # Max gap to merge collinear segments

    # Log-odds raycasting update
    lo_occ: float = 1.2  # occupied evidence
    lo_free: float = -0.7  # free evidence
    lo_min: float = -3.5
    lo_max: float = 3.5

    # Attraction weight
    k_att: float = 1.0
    # Repulsion weight (if you want it)
    k_rep: float = 1.0

    # Ray stride (optional): raycast only every Nth point endpoint for speed
    ray_endpoint_stride: int = 1




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


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------

class PotentialMapper:
    """Orchestrates the full RGB → depth → occupancy → potential pipeline.

    The mapper maintains two internal probability maps (float32 H×W):
      - ``_M_acc``: accumulated map updated with EMA each ``step()`` call.
      - ``_M_temp``: single-frame map rebuilt from scratch each ``step()`` call.

    After each ``step()``, ``_U_rep`` (H×W) and ``_grad`` (H×W×2) are recomputed.

    Usage::

        mapper = PotentialMapper(PotentialMapperConfig())
        depth_m = depth_model.infer_depth(rgb)
        mapper.step(depth_m, intrinsics)
        prob  = mapper.get_prob_map()       # (H, W) float32
        U     = mapper.get_potential_map()  # (H, W) float32
        grad  = mapper.get_gradient_field() # (H, W, 2) float32

    Args:
        cfg: Mapper configuration.
    """

    def __init__(self, cfg: Optional[PotentialMapperConfig] = None) -> None:
        self.cfg = cfg or PotentialMapperConfig()

        n_cells = int(round(self.cfg.size_m / self.cfg.resolution_m))
        self._n = n_cells  # grid is n×n
        self._origin = -0.5 * self.cfg.size_m  # world origin (m) for both axes

        # Float-space EMA maps
        self._M_acc: np.ndarray = np.zeros((n_cells, n_cells), dtype=np.float32)
        self._M_temp: np.ndarray = np.zeros((n_cells, n_cells), dtype=np.float32)

        # Derived outputs (computed lazily after step)
        self._U_rep: np.ndarray = np.zeros((n_cells, n_cells), dtype=np.float32)
        self._D_obs: np.ndarray = np.full((n_cells, n_cells), np.inf, dtype=np.float32)
        self._grad: np.ndarray = np.zeros((n_cells, n_cells, 2), dtype=np.float32)

        # Ray cache
        self._rays: Optional[np.ndarray] = None
        self._last_intrinsics: Optional[Intrinsics] = None

        self._goal_world: Optional[Tuple[float, float]] = None # (fwd, left) in metres
        
        # Wall segments (Revision 5)
        self._wall_segments: np.ndarray = np.array([]) # [N, 4] for (x1, y1, x2, y2) in grid coords
        self._M_walls: np.ndarray = np.zeros((n_cells, n_cells), dtype=np.float32)

        self._U_att = np.zeros((self._n, self._n), dtype=np.float32)
        self._U_total = np.zeros((self._n, self._n), dtype=np.float32)
        self._grad_total = np.zeros((self._n, self._n, 2), dtype=np.float32)

        # Potential layer
        self._potential = PotentialFieldLayer(
            occ_thresh=self.cfg.occ_thresh,
            sigma_m=self.cfg.sigma_m,
            k_rep=5.0, # Increased from 1.0 to 5.0 for stronger avoidance
            inflation_radius_m=self.cfg.inflation_radius_m,
            u_max=1.0,
            unknown_as_obstacle=False,
        )


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, depth_m: np.ndarray, intrinsics: Intrinsics) -> None:
        """Process one depth frame.

        1. Back-project ``depth_m`` → 3-D point cloud (base frame).
        2. Filter by range and height band.
        3. Bin XY hits into ``M_temp`` (probability per cell).
        4. Apply EMA: ``M_acc = (1-α)*M_acc + α*M_temp``.
        5. Recompute ``U_rep`` and ``∇U_rep`` from ``M_acc``.

        Args:
            depth_m: HxW float32 depth map in metres.
            intrinsics: Camera intrinsics with (fx, fy, cx, cy).

        Complexity: O(H*W/stride²) for backprojection +
                    O(n²) for potential field distance transform.
        """
        depth_m = np.asarray(depth_m, dtype=np.float32)

        pts_base = self._backproject(depth_m, intrinsics)
        pts_filtered = self._filter_cloud(pts_base)
        self._M_temp = self._build_temp_map(pts_filtered)

        # EMA accumulation
        self._M_acc = (1.0 - self.cfg.alpha) * self._M_acc + self.cfg.alpha * self._M_temp

        # 5. Wall Segmentation (Revision 5)
        self._M_walls = self._detect_walls_and_clean()

        # 6. Potential field from CLEANED wall map (prevents merging artifacts)
        # We blend the walls + the points to preserve both straight lines and chairs
        M_combined = self._M_acc
        U_rep, D = self._potential.compute_from_prob_grid(M_combined, self.cfg.resolution_m)
        self._U_rep = U_rep
        self._D_obs = D

        # build total potential + total descent direction
        self._compute_total_potential_and_gradient()

        # make the public gradient be the total one
        self._grad = self._grad_total.copy()



    def reset(self) -> None:
        """Zero all internal maps and the navigation goal."""
        self._M_acc.fill(0.0)
        self._M_temp.fill(0.0)
        self._M_walls.fill(0.0)
        self._U_rep.fill(0.0)
        self._D_obs.fill(np.inf)
        self._grad.fill(0.0)
        self._goal_world = None
        self._wall_segments = np.array([])



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

        Complexity: O(1) — gradient is pre-computed in ``step()``.
        """
        return self._grad.copy()

    def set_goal(self, fwd: float, left: float) -> None:
        """Set navigation goal in world coordinates (metres)."""
        self._goal_world = (fwd, left)

    def get_total_gradient(self) -> np.ndarray:
        """Compute combined gradient: Repulsive (Away) + Attractive (Toward Goal).
        
        Total Gradient = Repulsive_Grad_Away + zeta * (Goal - Position)
        This combines the avoidance vector with a pulling vector to the goal.
        """
        gv = self._grad.copy()
        if self._goal_world is None:
            return gv

        # current coordinates of every cell in the HxW grid
        rows = np.arange(self._n)
        cols = np.arange(self._n)
        
        # Row (Vertical) = Forward (gz)
        # origin is at -size/2
        fwd_coords = rows * self.cfg.resolution_m + self._origin
        
        # Col (Horizontal) = Left/Right (gl)
        # gl=0 is Left (origin_left), gl=N is Right (origin)
        origin_left = -self._origin
        left_coords = origin_left - cols * self.cfg.resolution_m
        
        target_fwd, target_left = self._goal_world
        
        # Conic Attraction: Normalized unit vector toward goal * zeta
        dist_fwd = target_fwd - fwd_coords[:, np.newaxis]
        dist_left = target_left - left_coords[np.newaxis, :]
        
        dist_total = np.sqrt(dist_fwd**2 + dist_left**2)
        # Avoid division by zero
        dist_total[dist_total == 0] = 1e-6
        
        att_fwd = self.cfg.zeta * (dist_fwd / dist_total)
        att_left = self.cfg.zeta * (dist_left / dist_total)
        
        gv[..., 0] += att_fwd
        gv[..., 1] += att_left
        
        return gv.astype(np.float32)


    def get_distance_to_obstacle(self) -> np.ndarray:
        """Return the distance-to-nearest-obstacle map.

        Returns:
            (H, W) float32 in metres.

        Complexity: O(1).
        """
        return self._D_obs.copy()

    def _compute_attractive_potential(self) -> None:
        """Compute U_att as Euclidean distance-to-goal on the grid."""
        if self._goal_world is None:
            self._U_att.fill(0.0)
            return

        goal_fwd, goal_left = self._goal_world

        # Cell center coordinates (fwd, left) for each (row, col)
        rows = np.arange(self._n, dtype=np.float32)
        cols = np.arange(self._n, dtype=np.float32)

        fwd_coords = rows * self.cfg.resolution_m + self._origin  # shape (n,)
        origin_left = -self._origin
        left_coords = origin_left - cols * self.cfg.resolution_m  # shape (n,)

        df = goal_fwd - fwd_coords[:, None]  # (n,1) broadcast to (n,n)
        dl = goal_left - left_coords[None, :]  # (1,n) broadcast to (n,n)

        self._U_att = np.sqrt(df * df + dl * dl).astype(np.float32)

    def _compute_total_potential_and_gradient(self) -> None:
        """
        U_total = k_rep * U_rep + k_att * U_att
        grad_total = -∇U_total  (NEGATIVE gradient = direction to move)
        """
        if self._goal_world is not None:
            r0 = c0 = self._n // 2
            gr, gc = self._goal_to_cell(*self._goal_world)
            print("U_rep start/goal", self._U_rep[r0, c0], self._U_rep[gr, gc],
                  "U_att start/goal", self._U_att[r0, c0], self._U_att[gr, gc],
                  "U_tot start/goal", self._U_total[r0, c0], self._U_total[gr, gc])
            print("grad_total at start", self._grad_total[r0, c0], "grad_rep at start", self._grad[r0, c0])
        self._compute_attractive_potential()

        self._U_total = (self.cfg.k_rep * self._U_rep + self.cfg.k_att * self._U_att).astype(np.float32)

        # np.gradient returns [dU/drow, dU/dcol]
        g_row, g_col = np.gradient(self._U_total, self.cfg.resolution_m)

        # Convert to (grad_fwd, grad_left) for a motion direction:
        # row corresponds to forward axis, col corresponds to left axis.
        # We want DESCENT direction => negative gradient.
        self._grad_total[..., 0] = (-g_row).astype(np.float32)  # toward decreasing potential
        self._grad_total[..., 1] = (-g_col).astype(np.float32)

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

        gr = int((goal_fwd_m - self._origin) / res)

        origin_left = -self._origin
        gc = int((origin_left - goal_left_m) / res)

        gr = max(0, min(n - 1, gr))
        gc = max(0, min(n - 1, gc))
        return gr, gc


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
        
        # Camera is at the center of the grid 
        cam_col = self._n / 2.0
        cam_row = self._n / 2.0
        
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
                    cv2.line(clean_binary, (p1[0], p1[1]), (p2[0], p2[1]), 255, thickness=8)

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
                cv2.line(m_walls, (x1, y1), (x2, y2), 1.0, thickness=8)
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
        """
        Build a temporary probability map using log-odds + raycasting.

        This preserves door openings because rays mark FREE space up to the hit.
        """
        n = self._n
        M_temp = np.zeros((n, n), dtype=np.float32)

        if pts.shape[0] == 0:
            return M_temp

        # Convert points -> grid endpoints (gz=row=fwd, gl=col=left)
        zf = pts[:, 2]  # Forward
        xl = pts[:, 0]  # Left

        gz = ((zf - self._origin) / self.cfg.resolution_m).astype(np.int32)
        origin_left = -self._origin
        gl = ((origin_left - xl) / self.cfg.resolution_m).astype(np.int32)

        inb = (gz >= 0) & (gz < n) & (gl >= 0) & (gl < n)
        gz = gz[inb]
        gl = gl[inb]
        if gz.size == 0:
            return M_temp

        # Optional endpoint downsampling for speed
        s = max(1, int(getattr(self.cfg, "ray_endpoint_stride", 1)))
        if s > 1:
            gz = gz[::s]
            gl = gl[::s]

        # Unique endpoints (speedup)
        idx = np.unique(gz * n + gl)
        gz = (idx // n).astype(np.int32)
        gl = (idx % n).astype(np.int32)

        # Log-odds temp grid
        L = np.zeros((n, n), dtype=np.float32)
        r0 = n // 2
        c0 = n // 2

        for r1, c1 in zip(gz, gl):
            update_ray_logodds(
                L, r0, c0, int(r1), int(c1),
                lo_free=self.cfg.lo_free,
                lo_occ=self.cfg.lo_occ
            )

        np.clip(L, self.cfg.lo_min, self.cfg.lo_max, out=L)

        # Convert log-odds -> probability
        M_temp = sigmoid(L).astype(np.float32)
        return M_temp



