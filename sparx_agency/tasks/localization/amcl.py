import math
from typing import Any
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

    # Find the orientation index closest to prediction
    orientation_diffs = np.abs(orientations - robot_orientation_pred)
    orientation_pred_idx = np.argmin(orientation_diffs)

    # Apply gaussian around prediction
    sigma_spatial = np.array(loc_uncertainty)
    sigma_angular = 1.0

    for i in range(map_lat):
        for j in range(map_long):
            for k in range(num_angles):
                # Spatial distance
                offset = np.array([i - robot_loc_pred[0], j - robot_loc_pred[1]])
                # spatial_dist = np.sqrt((i - robot_loc_pred[0]) ** 2 + (j - robot_loc_pred[1]) ** 2)

                # Angular distance (handle wrapping)
                angular_dist = min(abs(k - orientation_pred_idx),
                                   num_angles - abs(k - orientation_pred_idx))

                # Gaussian weights
                spatial_weight = np.linalg.norm(np.exp(-0.5 * (offset / sigma_spatial) ** 2))
                # spatial_weight = np.exp(-0.5 * (spatial_dist / sigma_spatial) ** 2)
                angular_weight = np.exp(-0.5 * (angular_dist / sigma_angular) ** 2)

                belief[i, j, k] = spatial_weight * angular_weight

    belief /= belief.sum()
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


def amcl_estimator(lut: ndarray[tuple[Any, ...], dtype[float64]],
                   orientations: ndarray[tuple[Any, ...], dtype[float64]],
                   robot_loc_prediction: ndarray[tuple[Any, ...], dtype[Any]],
                   robot_orientation_prediction: float,
                   world: ndarray[tuple[int, ...], dtype[float64]],
                   z_measured_pose: ndarray[tuple[Any, ...], dtype[float64]],
                   prediction_uncertainty: tuple[int, int]):
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


    belief = init_belief(map_shape=world.shape, orientations=orientations, robot_loc_pred=robot_loc_prediction,
                         robot_orientation_pred=robot_orientation_prediction, loc_uncertainty=prediction_uncertainty)

    belief = measurement_update_pose(
        belief, lut, z_measured_pose, sigma=SIGMA, occupancy=world
    )

    idx = np.unravel_index(np.argmax(belief), belief.shape)

    robot_loc_estimate = np.array(idx[:2])
    robot_orientation_estimate = orientations[idx[2]]
    return robot_loc_estimate, robot_orientation_estimate