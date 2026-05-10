import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any
import numpy as np
from numpy import dtype, ndarray, float64

from tasks.localization.amcl import ray_cast_lut_pose, amcl_estimator
from tasks.sim.grid import generate_world, sample_robot_location, sample_robot_orientation, ray_cast_from_pose

SENSOR_MAX_RANGE = 60
SENSOR_NOISE_SCALE = 0.0
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

        robot_loc_estimate, robot_orientation_estimate = amcl_estimator(lut, orientations, prediction,
                                                                        robot_orientation_gt, world, z_measured_pose,
                                                                        prediction_uncertainty=(args.pred_offset_y*2, args.pred_offset_x*2))

        logger.info(f"Prediction: {prediction.tolist()}")
        logger.info(f"MAP {robot_loc_estimate.tolist()} θ (deg): {math.degrees(robot_orientation_estimate)}")
        logger.info(f"GT {robot_loc_gt} θ (deg): {math.degrees(robot_orientation_gt):.2f}")
        correct = (int(np.allclose(robot_loc_gt, robot_loc_estimate))
                   and np.isclose(robot_orientation_gt, robot_orientation_estimate))
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
