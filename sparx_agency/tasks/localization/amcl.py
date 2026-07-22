import math
from typing import Any, Tuple
import numpy as np
from numpy import ndarray, dtype, float64

SIGMA = 0.5


def init_belief(map_shape: tuple,
                orientations: np.ndarray,
                robot_loc_pred: np.ndarray,
                robot_orientation_pred: float,
                loc_uncertainty: tuple) -> np.ndarray:
    """
    Initialise belief as a Gaussian over (row, col, orientation).
    Vectorised — replaces the original triple Python loop.
    """
    map_lat, map_long = map_shape
    num_angles = len(orientations)

    orientation_pred_idx = int(np.argmin(np.abs(orientations - robot_orientation_pred)))
    sigma_spatial = np.asarray(loc_uncertainty, dtype=np.float64)
    sigma_angular = 1.0

    rows = np.arange(map_lat, dtype=np.float64)
    cols = np.arange(map_long, dtype=np.float64)
    wy = np.exp(-0.5 * ((rows - robot_loc_pred[0]) / sigma_spatial[0]) ** 2)
    wx = np.exp(-0.5 * ((cols - robot_loc_pred[1]) / sigma_spatial[1]) ** 2)
    # Matches original: norm([exp(-0.5*dy²), exp(-0.5*dx²)]) = sqrt(wy² + wx²)
    spatial_weight = np.sqrt(wy[:, None] ** 2 + wx[None, :] ** 2)  # (H, W)

    angle_idxs = np.arange(num_angles, dtype=np.float64)
    angular_dist = np.minimum(
        np.abs(angle_idxs - orientation_pred_idx),
        num_angles - np.abs(angle_idxs - orientation_pred_idx),
    )
    angular_weight = np.exp(-0.5 * (angular_dist / sigma_angular) ** 2)  # (K,)

    belief = spatial_weight[:, :, None] * angular_weight[None, None, :]
    total = belief.sum()
    if total > 0:
        belief /= total
    return belief


def motion_predict(
    prev_loc_grid: np.ndarray,
    prev_orientation: float,
    vx_ms: float,
    vy_ms: float,
    dt_sec: float,
    m_per_cell: float,
) -> Tuple[np.ndarray, float]:
    """Advance AMCL's own last estimate by optical flow velocity × dt.

    prev_loc_grid is [row, col] from the previous AMCL output — never an
    externally accumulated optical-flow position.  vx=forward, vy=lateral.
    """
    if dt_sec <= 0.0 or m_per_cell <= 0.0:
        return prev_loc_grid.copy(), prev_orientation
    c, s = math.cos(prev_orientation), math.sin(prev_orientation)
    dx_cells = (c * vx_ms - s * vy_ms) * dt_sec / m_per_cell  # col (x)
    dy_cells = (s * vx_ms + c * vy_ms) * dt_sec / m_per_cell  # row (y)
    return prev_loc_grid + np.array([dy_cells, dx_cells]), prev_orientation


def extract_local_window(arr: np.ndarray, center_grid: np.ndarray,
                         half_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    """Slice a (H, W, ...) array to a local window around center_grid.

    Returns (window_array, origin) where origin=[r0, c0] is the top-left
    corner in global grid coordinates.  Window may be smaller near borders.
    """
    H, W = arr.shape[0], arr.shape[1]
    r_c = int(round(float(center_grid[0])))
    c_c = int(round(float(center_grid[1])))
    r0 = max(0, r_c - half_cells)
    r1 = min(H, r_c + half_cells)
    c0 = max(0, c_c - half_cells)
    c1 = min(W, c_c + half_cells)
    return arr[r0:r1, c0:c1], np.array([r0, c0], dtype=np.float64)


def ray_cast_lut_pose(grid, orientations, beam_angles, max_range, step=0.1):
    """
    Calculates a ray-casting lookup table (LUT) for a grid.

    For large maps use ray_cast_lut_vectorized — it processes all free cells
    simultaneously per direction and is orders of magnitude faster.
    """
    if step <= 0:
        raise ValueError("Step size must be positive.")
    if max_range <= 0:
        raise ValueError("Max range must be positive.")

    m, n = grid.shape
    lut = np.ones((m, n, len(orientations), len(beam_angles)), dtype=np.float32) * np.inf

    for i in range(m):
        for j in range(n):
            if grid[i, j] == 1:
                continue
            for k, theta in enumerate(orientations):
                for b, rel in enumerate(beam_angles):
                    angle = theta + rel
                    dist = 0.0
                    while dist < max_range:
                        x = int(round(i + dist * math.cos(angle)))
                        y = int(round(j + dist * math.sin(angle)))
                        if x < 0 or y < 0 or x >= m or y >= n:
                            dist += max_range
                            break
                        if grid[x, y] == 1:
                            break
                        dist += step
                    lut[i, j, k, b] = dist
    return lut


def ray_cast_lut_vectorized(grid: np.ndarray, orientations: np.ndarray,
                             beam_angles: np.ndarray, max_range: float,
                             step: float = 1.0) -> np.ndarray:
    """Build LUT by vectorising ray-marching over all free cells simultaneously.

    Prefer over ray_cast_lut_pose for any non-trivial map.  Run offline via
    precompute_amcl_lut.py and cache the result alongside the map.
    """
    if step <= 0:
        raise ValueError("Step must be positive.")
    m, n = grid.shape
    lut = np.full((m, n, len(orientations), len(beam_angles)), max_range, dtype=np.float32)
    free_r, free_c = np.where(grid == 0)
    if len(free_r) == 0:
        return lut
    n_steps = int(max_range / step) + 1
    n_or = len(orientations)
    for k, theta in enumerate(orientations):
        for b, rel in enumerate(beam_angles):
            _ray_fill(lut, grid, free_r, free_c, theta + rel, max_range, step, n_steps, m, n, k, b)
        if (k + 1) % max(1, n_or // 4) == 0:
            print(f"  LUT: {k + 1}/{n_or} orientations done", flush=True)
    return lut


def _ray_fill(lut, grid, free_r, free_c, angle, max_range, step, n_steps, m, n, k, b):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dists = np.full(len(free_r), max_range, dtype=np.float32)
    active = np.ones(len(free_r), dtype=bool)
    for s in range(1, n_steps + 1):
        if not np.any(active):
            break
        d = float(s) * step
        ai = np.where(active)[0]
        xi = np.rint(free_r[ai] + d * cos_a).astype(np.int32)
        yj = np.rint(free_c[ai] + d * sin_a).astype(np.int32)
        in_bounds = (xi >= 0) & (xi < m) & (yj >= 0) & (yj < n)
        out = ~in_bounds
        dists[ai[out]] = d
        active[ai[out]] = False
        ib = np.where(in_bounds)[0]
        if len(ib):
            hit = grid[xi[ib], yj[ib]] == 1
            dists[ai[ib[hit]]] = d
            active[ai[ib[hit]]] = False
    lut[free_r, free_c, k, b] = dists


def range_likelihood_lut(z, lut, sigma):
    """
    Computes the likelihood values for a given range measurement 'z' using a look-up
    table 'lut' and a standard deviation 'sigma'.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    err = lut - z[None, None, None, :]
    return np.exp(-0.5 * np.sum((err / sigma) ** 2, axis=3))


def measurement_update_pose(bel, lut, z, sigma, occupancy):
    """
    Updates the belief of the pose based on a measurement using a likelihood function.
    """
    likelihood = range_likelihood_lut(z, lut, sigma)
    likelihood[occupancy == 1] = 0.0
    bel_new = bel * likelihood
    s = bel_new.sum()
    if s > 0:
        bel_new /= s
    return bel_new


def amcl_estimator(lut: np.ndarray,
                   orientations: np.ndarray,
                   robot_loc_prediction: np.ndarray,
                   robot_orientation_prediction: float,
                   world: np.ndarray,
                   z_measured_pose: np.ndarray,
                   prediction_uncertainty: tuple):
    """
    Estimate robot location and orientation using grid-based Bayesian localisation.

    Prior:   Gaussian centred on robot_loc_prediction (from motion_predict).
    Update:  Multiplied by range-measurement likelihood from LUT.
    Returns: (robot_loc_estimate [row,col], robot_orientation_estimate [rad])
    """
    belief = init_belief(
        map_shape=world.shape,
        orientations=orientations,
        robot_loc_pred=robot_loc_prediction,
        robot_orientation_pred=robot_orientation_prediction,
        loc_uncertainty=prediction_uncertainty,
    )
    belief = measurement_update_pose(belief, lut, z_measured_pose, sigma=SIGMA, occupancy=world)
    idx = np.unravel_index(np.argmax(belief), belief.shape)
    return np.array(idx[:2]), orientations[idx[2]]