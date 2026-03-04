import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None
    _cv2_import_error = e


class PotentialFieldLayer:
    """
    Repulsive potential field computed from an occupancy probability grid.

    Input:
      p_occ (H,W) float32 in [0,1], unknown can be NaN.

    Output:
      U_rep (H,W) float32 in [0,u_max]
      D_obs_m (H,W) float32 distance-to-nearest-obstacle in meters

    This is intended as a mapping layer (not goal-dependent).
    """

    def __init__(
        self,
        occ_thresh: float = 0.65,
        sigma_m: float = 0.6,
        k_rep: float = 1.0,
        inflation_radius_m: float = 0.35,
        u_max: float = 1.0,
        unknown_as_obstacle: bool = False,
    ):
        if cv2 is None:
            raise RuntimeError(
                f"PotentialFieldLayer requires OpenCV (cv2). Import error: {_cv2_import_error}"
            )

        self.occ_thresh = float(occ_thresh)
        self.sigma_m = float(sigma_m)
        self.k_rep = float(k_rep)
        self.inflation_radius_m = float(inflation_radius_m)
        self.u_max = float(u_max)
        self.unknown_as_obstacle = bool(unknown_as_obstacle)

        if self.sigma_m <= 0.0:
            raise ValueError("sigma_m must be > 0")

    def compute_from_prob_grid(self, p_occ: np.ndarray, resolution_m: float) -> tuple[np.ndarray, np.ndarray]:
        """
        p_occ: (H,W) float32 in [0,1], unknown can be NaN
        resolution_m: meters per cell
        """
        p = np.asarray(p_occ, dtype=np.float32)
        if p.ndim != 2:
            raise ValueError(f"Expected 2D grid, got shape={p.shape}")

        H, W = p.shape
        res = float(resolution_m)

        is_unknown = ~np.isfinite(p)
        is_occ = p >= self.occ_thresh

        if self.unknown_as_obstacle:
            is_occ = is_occ | is_unknown
        else:
            is_occ = is_occ & (~is_unknown)

        # distanceTransform expects:
        #   obstacles = 0 pixels
        #   free = non-zero pixels
        free_mask = (~is_occ).astype(np.uint8) * 255  # free=255, occ=0

        d_pix = cv2.distanceTransform(free_mask, distanceType=cv2.DIST_L2, maskSize=5)
        d_m = d_pix.astype(np.float32) * res

        # Repulsive potential (Gaussian falloff from obstacles)
        sigma2 = self.sigma_m * self.sigma_m
        with np.errstate(under='ignore', over='ignore', divide='ignore'):
            U = (self.k_rep * np.exp(-(d_m * d_m) / (2.0 * sigma2))).astype(np.float32)



        # Hard inflation near obstacles (optional)
        if self.inflation_radius_m > 0.0:
            U[d_m <= self.inflation_radius_m] = self.u_max

        np.clip(U, 0.0, self.u_max, out=U)
        return U, d_m