import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype, ndarray, float64, complexfloating, floating

MAX_RANGE = 60
SIGMA = 0.5
FOV_DEG = 120.0
NUM_ANGLES = 32
NUM_BEAMS = 64
MAP_LATERAL_SIZE = 16
MAP_LONGITUDE_SIZE = 64
NUM_SIMULATION_RUNS = 1000
PREDICTION_OFFSET_X = MAP_LONGITUDE_SIZE // 8
PREDICTION_OFFSET_Y = MAP_LATERAL_SIZE // 8

def make_orientations(num_angles) -> ndarray[tuple[Any, ...], dtype[float64]] | ndarray[
    tuple[Any, ...], dtype[floating[Any]]] | ndarray[tuple[Any, ...], dtype[complexfloating[Any, Any]]] | ndarray[
                                         tuple[Any, ...], dtype[Any]]:
    return np.linspace(-math.pi, math.pi, num_angles, endpoint=False)

def make_fov_angles(fov_rad, num_beams=9) -> ndarray[tuple[Any, ...], dtype[float64]] | ndarray[
    tuple[Any, ...], dtype[floating[Any]]] | ndarray[tuple[Any, ...], dtype[complexfloating[Any, Any]]] | ndarray[
                                         tuple[Any, ...], dtype[Any]]:
    return np.linspace(-fov_rad / 2, fov_rad / 2, num_beams)

def ray_cast_from_pose(i, j, theta, grid, beam_angles, max_range, step=0.1):
    """
    Simulates depth reading from a sensor for a robot at (i, j) with orientation theta,
    casting rays along beam_angles relative to theta.
    Returns np.array of shape (len(beam_angles),)
    """
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
    return np.array(distances)

def ray_cast_lut_pose(grid, orientations, beam_angles, max_range, step=0.1):
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
    z:   (B,)
    lut: (H,W,K,B)
    """
    err = lut - z[None, None, None, :]
    return np.exp(-0.5 * np.sum((err / sigma) ** 2, axis=3))


def measurement_update_pose(bel, lut, z, sigma, occupancy):
    likelihood = range_likelihood_lut(z, lut, sigma)
    likelihood[occupancy == 1] = 0.0

    bel_new = bel * likelihood
    s = bel_new.sum()
    if s > 0:
        bel_new /= s
    return bel_new


def show_world_map(world, location=None, orientation=None, title=''):
    map = (1-world).T
    height, width = map.shape

    left = -1
    bottom = -2
    right = left + width - 1
    top = bottom + height - 1
    extent = [left, right, bottom, top]
    plt.imshow(map, cmap='gray', origin='lower', extent=extent)
    plt.title('World Map' if not plt.title else title)
    plt.grid(True)
    if location is not None:
        # Convert array indices to coordinate system
        x_coord = left + location[0]
        y_coord = bottom + location[1]

        plt.plot(x_coord, y_coord, 'rx', markersize=10, markeredgewidth=2)
        if orientation is not None:
            # Draw an arrow for orientation
            arrow_length = 0.8  # Adjust as needed
            dx = arrow_length * math.cos(orientation)
            dy = arrow_length * math.sin(orientation)
            plt.arrow(x_coord, y_coord, dy, dx, head_width=0.3, head_length=0.3, fc='blue', ec='blue')
    plt.show()


def generate_world(height=10, width=10,):

    world = np.zeros((height, width))
    rands = np.random.rand(height, width)
    world[rands < 0.3] = 1
    return world

def sample_robot_orientation(orientations) -> float:
    random_orientation_idx = np.random.randint(0, np.size(orientations))
    robot_orientation_gt = orientations[random_orientation_idx]
    return robot_orientation_gt


def sample_robot_location(world):
    free_cells = np.argwhere(world == 0)
    robot_map_gt = free_cells[np.random.randint(0, len(free_cells))]
    return robot_map_gt


def init_belief(args,
                orientations: ndarray[tuple[Any, ...], dtype[float64]],
                robot_loc_pred: ndarray[tuple[Any, ...], dtype[float64]],
                robot_orientation_pred: float) -> ndarray[tuple[int, ...], dtype[float64]]:
    belief = np.zeros((args.map_lat, args.map_long, args.num_angles))

    # Find the orientation index closest to prediction
    orientation_diffs = np.abs(orientations - robot_orientation_pred)
    orientation_pred_idx = np.argmin(orientation_diffs)

    # Apply gaussian around prediction
    sigma_spatial = np.array([args.pred_offset_y * 2,args.pred_offset_x * 2])
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
    parser.add_argument('--max-range', type=float, default=MAX_RANGE,
                        help=f'Maximum range for ray casting (default: {MAX_RANGE})')
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


def main(args):
    logger = get_logger(__name__, args.log_level)
    logger.info('Starting the program...')
    logger.info('Program started at %s', time.asctime())
    logger.info(f'Arguments: {args}')

    correct_estimates = 0
    failed_estimates = 0
    num_runs = args.num_runs
    orientations = make_orientations(args.num_angles)
    error_data = []
    for run in range(num_runs):
        world = generate_world(args.map_lat, args.map_long)
        robot_loc_gt = sample_robot_location(world)
        robot_loc_gt = np.minimum(robot_loc_gt, np.array([args.map_lat // 2, args.map_long // 2]))
        robot_loc_gt = np.maximum(robot_loc_gt, np.array([args.map_lat // 4, args.map_long // 4]))
        world[robot_loc_gt[0], robot_loc_gt[1]] = 0
        # robot_orientation_gt = sample_robot_orientation(orientations)
        # robot_orientation_gt = min(max(robot_orientation_gt, -math.pi / 16), math.pi / 16)
        robot_orientation_gt = math.radians(0)
        # show_world_map(world, location=robot_loc_gt, orientation=robot_orientation_gt, title='World Map with Robot GT')
        prediction_offset = np.random.randint((- args.pred_offset_y, - args.pred_offset_x), (args.pred_offset_y, args.pred_offset_y), size=(2,))
        prediction = robot_loc_gt + prediction_offset
        # prediction = np.maximum(0, np.minimum(prediction, np.array([args.map_lat - 1, args.map_long - 1])))
        belief = init_belief(args, orientations, prediction, robot_orientation_gt)

        beam_angles = make_fov_angles(fov_rad=math.radians(args.fov_deg), num_beams=args.num_beams)

        lut = ray_cast_lut_pose(
            world,
            orientations,
            beam_angles,
            max_range=args.max_range
        )

        z_measured_pose = ray_cast_from_pose(
            robot_loc_gt[0],
            robot_loc_gt[1],
            robot_orientation_gt,
            world,
            beam_angles,
            max_range=args.max_range
        )

        belief = measurement_update_pose(
            belief, lut, z_measured_pose, sigma=args.sigma, occupancy=world
        )

        idx = np.unravel_index(np.argmax(belief), belief.shape)

        logger.info(f"Prediction: {prediction.tolist()}")
        logger.info(f"MAP {[int(e) for e in idx[:2]]} θ (deg): {math.degrees(orientations[idx[2]])}")
        logger.info(f"GT {robot_loc_gt} θ (deg): {math.degrees(robot_orientation_gt):.2f}")
        correct = int(np.allclose(robot_loc_gt, np.array(idx[:2]))) and np.isclose(robot_orientation_gt, orientations[idx[2]])
        logger.info(f"Correct: {correct}")
        correct_estimates += correct
        failed_estimates = run - correct_estimates + 1
        logger.info(f"Run {run + 1}/{num_runs}: Correct estimates: {correct_estimates}  Failed estimates: {failed_estimates}")
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





if __name__ == "__main__":
    args = parse_args()
    main(args)
