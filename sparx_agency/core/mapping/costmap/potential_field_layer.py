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
        repulse_radius_m: float = 1.0,
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

        self.repulse_radius_m = float(repulse_radius_m)
        if self.repulse_radius_m <= 0.0:
            raise ValueError("repulse_radius_m must be > 0")

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

        # Repulsive potential: Gaussian sum over all nearby obstacles.
        # Each free cell's value is a weighted sum of every wall within ~2.5σ,
        # so ∇U_rep naturally reflects ALL nearby walls at once — corridor
        # centres are local minima (opposing walls cancel), corners push
        # diagonally, single walls push perpendicular to themselves.
        obstacle = is_occ.astype(np.float32)
        sigma_px = max(self.sigma_m / res, 1e-3)
        U = cv2.GaussianBlur(obstacle, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)
        u_peak = float(U.max())
        if u_peak > 1e-6:
            U *= (self.u_max / u_peak)      # normalise to [0, u_max]
        return U.astype(np.float32), d_m


def occupancy_to_potential(
    occ_grid: np.ndarray,
    resolution_m: float = 0.10,
    *,
    sigma_m: float = 0.6,
    occ_thresh: float = 0.65,
    repulse_radius_m: float = 1.0,
    smooth: bool = True,
) -> tuple:
    """
    Convert an occupancy grid to a potential field + gradient.

    Accepts any of these formats:
      - float [0..1]   : probability grid (0=free, 1=occupied)
      - binary 0/1     : obstacle mask
      - int  [-1,0,100]: ROS OccupancyGrid convention (-1=unknown → treated as free)

    Args:
        occ_grid     : (H, W) array
        resolution_m : metres per cell
        sigma_m      : Gaussian spread of repulsion from obstacles
        occ_thresh   : probability above which a cell is considered occupied
        repulse_radius_m : max influence radius of obstacles
        smooth       : apply a light Gaussian blur to U_rep after computation

    Returns:
        U_rep     : (H, W) float32  repulsive potential in [0, 1]
        gradient  : (H, W, 2) float32  descent direction [fwd, left] per cell
        D_obs_m   : (H, W) float32  distance to nearest obstacle in metres
    """
    grid = np.asarray(occ_grid, dtype=np.float32)

    # Normalise ROS int format [-1, 0..100] → [NaN, 0..1]
    if grid.max() > 1.0:
        grid = np.where(grid < 0, np.nan, grid / 100.0)

    layer = PotentialFieldLayer(
        occ_thresh=occ_thresh,
        sigma_m=sigma_m,
        repulse_radius_m=repulse_radius_m,
    )
    U_rep, D_obs_m = layer.compute_from_prob_grid(grid, resolution_m)

    if smooth and cv2 is not None:
        U_rep = cv2.GaussianBlur(U_rep, (5, 5), 1.0)

    g_row, g_col = np.gradient(U_rep, resolution_m)
    gradient = np.stack([-g_row, g_col], axis=-1).astype(np.float32)

    return U_rep, gradient, D_obs_m