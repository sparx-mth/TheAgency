import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any
import numpy as np
from numpy import dtype, ndarray, float64

from tasks.sim.grid import generate_world, sample_robot_location, sample_robot_orientation

SENSOR_MAX_RANGE = 60
SENSOR_NOISE_SCALE = 0.0
SIGMA = 0.5
FOV_DEG = 120.0
NUM_ANGLES = 32
NUM_BEAMS = 64
MAP_LATERAL_SIZE = 16
MAP_LONGITUDE_SIZE = 64
NUM_SIMULATION_RUNS = 1000
PREDICTION_OFFSET_X = MAP_LONGITUDE_SIZE // 8
PREDICTION_OFFSET_Y = MAP_LATERAL_SIZE // 8


def make_orientations(num_angles: int) -> ndarray[tuple[Any, ...], dtype[float64]]:
    """
    Generates an array of evenly spaced orientation angles.

    This function computes a sequence of uniformly distributed angles between
    -negative pi and positive pi (exclusive of the endpoint). The input parameter determines the number of angles.
    It can be used for generating orientations in various applications like image processing or simulations.

    Args:
        num_angles (int): The number of orientation angles to generate.

    Returns:
        ndarray[tuple[Any, ...], dtype[float64]]: A numpy array containing the generated orientation angles.
    Raises:
    ValueError
        If `num_angles` is non-positive.
    """
    if num_angles <= 0:
        raise ValueError("Number of angles must be positive.")
    return np.linspace(-math.pi, math.pi, num_angles, endpoint=False)


def make_fov_angles(fov_rad: float, num_beams: int = 9) -> ndarray[tuple[Any, ...], dtype[float64]]:
    """
    Generate an array of field of view (FoV) angles.

    This function calculates and returns a sequence of evenly spaced angles
    within the specified field of view (FoV) range in radians. The center of
    the FoV is aligned with zero, and the angles are distributed symmetrically
    around it.

    Parameters:
    fov_rad: float
        The total field of view in radians.
    num_beams: int, optional
        The number of angles (beams) to generate, evenly distributed across
        the FoV range. Defaults to 9.

    Returns:
    ndarray
        A NumPy array containing the calculated angles.
    Raises:
    ValueError
        If `num_beams` is non-positive.
    """
    if num_beams <= 0:
        raise ValueError("Number of beams must be positive.")
    return np.linspace(-fov_rad / 2, fov_rad / 2, num_beams)


def ray_cast_from_pose(i, j, theta, grid, beam_angles, max_range, step: float=0.1, noise_scale: float = 0.0):
    """
    Performs ray casting from a given pose on a grid map.

    This function simulates the process of ray casting, allowing one to evaluate
    distances to obstacles in a grid-based environment. It loops through a set of
    beam angles, projecting rays until either an obstacle is encountered or the
    maximum range is exceeded. Optionally, it adds Gaussian noise to the simulated
    measurements for more realistic scenarios.

    Parameters:
        i (int): The x-coordinate of the pose in the grid map.
        j (int): The y-coordinate of the pose in the grid map.
        theta (float): The orientation of the pose in radians.
        grid (ndarray): 2D array representing the environment grid, where values
            of 1 represent obstacles and values of 0 represent free space.
        beam_angles (Iterable[float]): A list of relative beam angles (radians)
            to cast rays from the pose.
        max_range (float): The maximum sensing range for the rays.
        step (float): Step size along each ray for incremental checking; must
            be greater than 0. Defaults to 0.1.
        noise_scale (float): Standard deviation of the Gaussian noise added to
            measurements; defaults to 0.0.

    Returns:
        ndarray: A 1D array of distances measured along each ray, representing
            the distance to the nearest obstacle for each beam angle, up to
            the maximum range.

    Raises:
        ValueError: If the `step` size is not positive.
        ValueError: If `max_range` is not positive.
    """
    if step <= 0:
        raise ValueError("Step size must be positive.")
    if max_range <= 0:
        raise ValueError("Max range must be positive.")

    m, n = grid.shape
    distances = []
    for rel_angle in beam_angles:
        angle = theta + rel_angle
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

        distances.append(dist)
    z_measured_pose = np.array(distances)
    noise = np.random.normal(0, noise_scale, size=z_measured_pose.shape)
    z_measured_pose += noise
    return z_measured_pose


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


def init_belief(args,
                orientations: ndarray[tuple[Any, ...], dtype[float64]],
                robot_loc_pred: ndarray[tuple[Any, ...], dtype[float64]],
                robot_orientation_pred: float) -> ndarray[tuple[int, ...], dtype[float64]]:
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
    belief = np.zeros((args.map_lat, args.map_long, args.num_angles))

    # Find the orientation index closest to prediction
    orientation_diffs = np.abs(orientations - robot_orientation_pred)
    orientation_pred_idx = np.argmin(orientation_diffs)

    # Apply gaussian around prediction
    sigma_spatial = np.array([args.pred_offset_y * 2, args.pred_offset_x * 2])
    sigma_angular = 1.0

    for i in range(args.map_lat):
        for j in range(args.map_long):
            for k in range(args.num_angles):
                # Spatial distance
                offset = np.array([i - robot_loc_pred[0], j - robot_loc_pred[1]])
                # spatial_dist = np.sqrt((i - robot_loc_pred[0]) ** 2 + (j - robot_loc_pred[1]) ** 2)

                # Angular distance (handle wrapping)
                angular_dist = min(abs(k - orientation_pred_idx),
                                   args.num_angles - abs(k - orientation_pred_idx))

                # Gaussian weights
                spatial_weight = np.linalg.norm(np.exp(-0.5 * (offset / sigma_spatial) ** 2))
                # spatial_weight = np.exp(-0.5 * (spatial_dist / sigma_spatial) ** 2)
                angular_weight = np.exp(-0.5 * (angular_dist / sigma_angular) ** 2)

                belief[i, j, k] = spatial_weight * angular_weight

    belief /= belief.sum()
    return belief




def randomize_scenario(args, orientations: ndarray[tuple[Any, ...], dtype[float64]]) -> tuple[
    ndarray[tuple[Any, ...], dtype[Any]], float, ndarray[tuple[int, ...], dtype[float64]]]:
    """
    Randomizes the scenario by generating a world map, sampling a robot location, and a robot orientation.

    The function first generates a world map based on the specified map latitude and longitude dimensions. A ground
    truth robot location is sampled from the generated world map. The sampled location is adjusted to lie within
    specific boundaries of the map dimensions. The robot's orientation is then sampled within a small angular range.
    Finally, the function returns the adjusted robot location, the sampled robot orientation, and the modified world map.

    Parameters:
    args (Any): Configuration object containing parameters for map dimensions ('map_lat', 'map_long').
    orientations (ndarray): Array specifying possible orientations to sample from.

    Returns:
    tuple: A tuple containing the following elements:
        - ndarray: The ground truth robot location, adjusted to be within specific boundaries.
        - float: The sampled ground truth orientation of the robot.
        - ndarray: The generated world map with modifications reflecting the robot's location.
    """
    world = generate_world(args.map_lat, args.map_long)
    robot_loc_gt = sample_robot_location(world)
    robot_loc_gt = np.minimum(robot_loc_gt, np.array([args.map_lat // 2, args.map_long // 2]))
    robot_loc_gt = np.maximum(robot_loc_gt, np.array([args.map_lat // 4, args.map_long // 4]))
    world[robot_loc_gt[0], robot_loc_gt[1]] = 0
    robot_orientation_gt = sample_robot_orientation(orientations, min_angle=-math.pi / 16, max_angle=math.pi / 16)
    return robot_loc_gt, robot_orientation_gt, world




def main(args):
    logger = get_logger(__name__, args.log_level)
    logger.info('Starting the program...')
    logger.info('Program started at %s', time.asctime())
    logger.info(f'Arguments: {args}')

    correct_estimates = 0
    failed_estimates = 0
    num_runs = args.num_runs
    orientations = make_orientations(args.num_angles)
    beam_angles = make_fov_angles(fov_rad=math.radians(args.fov_deg), num_beams=args.num_beams)

    for run in range(num_runs):
        robot_loc_gt, robot_orientation_gt, world = randomize_scenario(args, orientations)

        # Generate a random prediction around the robot's current position
        prediction_offset = np.random.randint((- args.pred_offset_y, - args.pred_offset_x),
                                              (args.pred_offset_y, args.pred_offset_y), size=(2,))
        prediction = robot_loc_gt + prediction_offset


        # Create a lookup table for ray casting
        lut = ray_cast_lut_pose(
            world,
            orientations,
            beam_angles,
            max_range=args.max_range
        )

        # Read robot's depth sensor data
        z_measured_pose = ray_cast_from_pose(
            robot_loc_gt[0],
            robot_loc_gt[1],
            robot_orientation_gt,
            world,
            beam_angles,
            max_range=args.max_range,
            noise_scale=args.noise_scale
        )

        belief = init_belief(args, orientations, prediction, robot_orientation_gt)

        belief = measurement_update_pose(
            belief, lut, z_measured_pose, sigma=args.sigma, occupancy=world
        )

        idx = np.unravel_index(np.argmax(belief), belief.shape)

        logger.info(f"Prediction: {prediction.tolist()}")
        logger.info(f"MAP {[int(e) for e in idx[:2]]} θ (deg): {math.degrees(orientations[idx[2]])}")
        logger.info(f"GT {robot_loc_gt} θ (deg): {math.degrees(robot_orientation_gt):.2f}")
        correct = int(np.allclose(robot_loc_gt, np.array(idx[:2]))) and np.isclose(robot_orientation_gt,
                                                                                   orientations[idx[2]])
        logger.info(f"Correct: {correct}")
        correct_estimates += correct
        failed_estimates = run - correct_estimates + 1
        logger.info(
            f"Run {run + 1}/{num_runs}: Correct estimates: {correct_estimates}  Failed estimates: {failed_estimates}")
        logger.info(f"Accuracy: {(correct_estimates / (correct_estimates + failed_estimates)) * 100:.2f}%")

        # if not correct:
        #     logger.error(f"Prediction: {prediction.tolist()}")
        #     logger.error(f"MAP {[int(e) for e in idx[:2]]} θ (deg): {math.degrees(orientations[idx[2]])}")
        #     logger.error(f"GT {robot_loc_gt} θ (deg): {math.degrees(robot_orientation_gt):.2f}")
        #     show_world_map(world, location=robot_loc_gt, orientation=robot_orientation_gt, title='World Map with Robot GT')
        #     show_world_map(world, location=idx[:2], orientation=orientations[idx[2]],title='World Map with Robot Estimated')
        #     print()

    logger.info(f"Correct estimates: {correct_estimates}")
    logger.info(f"Failed estimates: {failed_estimates}")
    logger.info(f"Accuracy: {correct_estimates / num_runs * 100:.2f}%")

    # show_world_map(world, location=robot_loc_gt, title='World Map with Robot GT')
    # show_world_map(world, location=robot_map_estimate, title='World Map with Robot Estimated')


def get_logger(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path.cwd() / 'logs'
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f'{name}_{time.asctime()}.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='AMCL Location Estimation')
    parser.add_argument('--map-long', type=int, default=MAP_LONGITUDE_SIZE,
                        help=f'Longitudal size of the map (default: {MAP_LONGITUDE_SIZE})')
    parser.add_argument('--map-lat', type=int, default=MAP_LATERAL_SIZE,
                        help=f'Lateral size of the map (default: {MAP_LATERAL_SIZE})')
    parser.add_argument('--num-runs', type=int, default=NUM_SIMULATION_RUNS,
                        help='Number of simulation runs (default: 1000)')
    parser.add_argument('--max-range', type=float, default=SENSOR_MAX_RANGE,
                        help=f'Maximum range for ray casting (default: {SENSOR_MAX_RANGE})')
    parser.add_argument('--noise-scale', type=float, default=SENSOR_NOISE_SCALE,
                        help=f'Noise level added to ray casting (default: {SENSOR_NOISE_SCALE})')
    parser.add_argument('--sigma', type=float, default=SIGMA,
                        help=f'Sigma for likelihood calculation (default: {SIGMA})')
    parser.add_argument('--pred-offset-x', type=float, default=PREDICTION_OFFSET_X,
                        help=f'prediction error (default: {PREDICTION_OFFSET_X})')
    parser.add_argument('--pred-offset-y', type=float, default=PREDICTION_OFFSET_Y,
                        help=f'prediction error (default: {PREDICTION_OFFSET_Y})')
    parser.add_argument('--fov_deg', type=float, default=FOV_DEG,
                        help=f'Horizontal Field of View in degrees (default: {FOV_DEG})')
    parser.add_argument('--num-angles', type=int, default=NUM_ANGLES,
                        help='Number of angles for ray casting')
    parser.add_argument('--num-beams', type=int, default=NUM_BEAMS,
                        help='Number of beams to pack angles')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    main(args)
