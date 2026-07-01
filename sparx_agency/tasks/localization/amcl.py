import math
from typing import Any, Tuple
import numpy as np
from numpy import ndarray, dtype, float64, generic

SIGMA = 0.5

def init_belief(map_shape: tuple[int,int] ,
                orientations: ndarray[tuple[Any, ...], dtype[float64]],
                robot_loc_pred: ndarray[tuple[Any, ...], dtype[float64]],
                robot_orientation_pred: float,
                loc_uncertainty: tuple[int,int] ) -> ndarray[tuple[int, ...], dtype[float64]]:
    """
    Initializes the belief state of a robot's location and orientation on a discrete grid map, using
    a Gaussian distribution centered around a predicted position and orientation.

    Parameters:
        args: A namespace or object containing the necessary attributes such as `map_lat`, `map_long`,
            `num_angles`, `pred_offset_y`, and `pred_offset_x`. These attributes define the grid
            map's dimensions, angular discretization, and spatial prediction offset factors.
        orientations (ndarray[tuple[Any, ...], dtype[float64]]): Array of possible orientation
            angles for the robot, used to determine the angular prediction index.
        robot_loc_pred (ndarray[tuple[Any, ...], dtype[float64]]): Predicted [y, x] location of the
            robot on the grid.
        robot_orientation_pred (float): Predicted orientation of the robot in radians.

    Returns:
        ndarray[tuple[int, ...], dtype[float64]]: A 3-dimensional belief tensor of size
        (map_lat, map_long, num_angles), representing the robot's belief state over all
        possible positions and orientations. The tensor is normalized so that its sum is 1.
    """
    map_lat, map_long = map_shape
    num_angles = len(orientations)
    belief = np.zeros((map_lat, map_long, num_angles))

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


def ray_cast_lut_pose(grid, orientations, beam_angles, max_range, step=0.1):
    """
    Calculates a ray-casting lookup table (LUT) for a grid that approximates the distance from each cell to its nearest
    obstacle along specified orientations and beam angles.

    Parameters:
    grid: ndarray
        2D numpy array representing the occupancy grid. Cells with a value of 1 indicate obstacles, and cells
        with a value of 0 indicate free space.
    orientations: Sequence[float]
        A sequence of angles in radians specifying the orientations to consider for ray-casting.
    beam_angles: Sequence[float]
        A sequence of relative beam angles in radians to be used for ray-casting relative to each orientation.
    max_range: float
        The maximum distance to consider for a ray before it is terminated if no obstacle is encountered.
    step: float, optional
        The incremental step along the ray (in grid units). Default is 0.1.

    Returns:
    ndarray
        A 4D numpy array representing the ray-casting lookup table (LUT). The shape of the array is
        (grid_height, grid_width, len(orientations), len(beam_angles)), where each entry indicates the
        distance from the corresponding grid cell to the nearest obstacle along the associated orientation
        and beam angle. If no obstacle is encountered within `max_range`, the value will be set to infinity.
    Raises:
    ValueError
        If `step` is non-positive or if `max_range` is non-positive.
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


def range_likelihood_lut(z, lut, sigma):
    """
    Computes the likelihood values for a given range measurement 'z' using a look-up
    table 'lut' and a standard deviation 'sigma'. The function evaluates the
    Gaussian likelihood over a set of errors derived from the differences
    between the input measurement and the LUT values along the last axis.

    Parameters:
    z : np.ndarray
        Input range measurement array.
    lut : np.ndarray
        Look-up table containing reference values.
    sigma : np.ndarray
        Standard deviation for the Gaussian likelihood computation.

    Returns:
    np.ndarray
        The computed likelihood values as a NumPy array.
    Raises:
    ValueError
        If `sigma` is non-positive.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    err = lut - z[None, None, None, :]
    return np.exp(-0.5 * np.sum((err / sigma) ** 2, axis=3))


def measurement_update_pose(bel, lut, z, sigma, occupancy):
    """
    Updates the belief of the pose based on a measurement using a likelihood function.

    The function computes the updated belief by incorporating the measurement's
    likelihood into the prior belief and normalizing the result. The likelihood
    for each pose is determined using a precomputed lookup table (LUT) and a
    given measurement. Any poses marked as occupied are assigned a likelihood of 0.

    Parameters:
    ----------
    bel : numpy.ndarray
        The prior belief distribution for the pose.

    lut : numpy.ndarray
        Precomputed lookup table used to compute the measurement likelihood.

    z : float
        The actual measurement value.

    sigma : float
        The standard deviation used in the likelihood computation.

    occupancy : numpy.ndarray
        A binary mask marking positions that are occupied (1) or free (0).

    Returns:
    -------
    numpy.ndarray
        The updated belief distribution for the pose.
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
    Estimate the robot's current location and orientation using the AMCL algorithm.

    This function implements the Adaptive Monte Carlo Localization (AMCL) to estimate the
    robot's most probable location and orientation in the environment. The estimation process
    involves initializing a belief distribution, performing a measurement update based on
    the observed data, and determining the most likely pose of the robot.

    Parameters:
        lut (ndarray[tuple[Any, ...], dtype[float64]]): Lookup table connecting sensor
            observations to the probability distribution over possible robot poses.
        orientations (ndarray[tuple[Any, ...], dtype[float64]]): Array of possible
            robot orientations utilized in the environment.
        robot_loc_prediction (ndarray[tuple[Any, ...], dtype[Any]]): Prior prediction
            of the robot's location as derived by prediction models or sensors.
        robot_orientation_prediction (float): Prior prediction of the robot's orientation.
        world (ndarray[tuple[int, ...], dtype[float64]]): Environmental map represented
            as a grid where cells define occupancy probabilities.
        z_measured_pose (ndarray[tuple[Any, ...], dtype[float64]]): Pose information
            obtained from recent sensor measurements.
        prediction_uncertainty (tuple[int, int]): Tuple representing the uncertainty
            in the predicted location coordinates.

    Returns:
        robot_loc_estimate (ndarray[tuple[int, ...], dtype[int]]): Estimated
            location of the robot in the map grid.
        robot_orientation_estimate (float): Estimated orientation of the robot
            in radians or degrees depending on the input orientations.

    Raises:
        None
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

    robot_loc_estimate = np.array(idx[:2])
    robot_orientation_estimate = orientations[idx[2]]
    return robot_loc_estimate, robot_orientation_estimate