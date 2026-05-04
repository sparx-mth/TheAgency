import time
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
import logging
import argparse

MAX_RANGE = 6.0
SIGMA = 0.5
NUM_ANGLES = 16
MAP_HEIGHT = 6
MAP_WIDTH = 3
NUM_SIMULATION_RUNS = 1000

def make_angles(num_angles):
    return np.linspace(-math.pi, math.pi, num_angles, endpoint=False)

def ray_cast_from_pos(i, j, grid, angles, max_range, step=0.1):
    """
    Returns LUT of shape (H, W, A)
    """
    H, W = grid.shape
    distances = []
    for k, theta in enumerate(angles):
                dist = 0.0
                while dist < max_range:
                    x = int(round(i + dist * math.cos(theta)))
                    y = int(round(j + dist * math.sin(theta)))

                    if x < 0 or y < 0 or x >= H or y >= W:
                        dist+=max_range
                        break
                    if grid[x, y] == 1:
                        break

                    dist += step

                distances.append(dist)
    return np.array(distances)

def ray_cast_lut(grid, angles, max_range, step=0.1):
    """
    Returns LUT of shape (H, W, A)
    """
    H, W = grid.shape
    A = len(angles)

    lut = np.zeros((H, W, A), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if grid[i, j] == 1:
                continue  # invalid pose

            for k, theta in enumerate(angles):
                dist = 0.0
                while dist < max_range:
                    x = int(round(i + dist * math.cos(theta)))
                    y = int(round(j + dist * math.sin(theta)))

                    if x < 0 or y < 0 or x >= H or y >= W:
                        dist+=max_range
                        break
                    if grid[x, y] == 1:
                        break

                    dist += step

                lut[i, j, k] = dist

    return lut

def range_likelihood_vectorized(z, z_hat, sigma):
    """
    z:      (A,)
    z_hat:  (H, W, A)
    returns likelihood map (H, W)
    """
    err = z_hat - z[None, None, :]
    return np.exp(-0.5 * np.sum((err / sigma) ** 2, axis=2))

def measurement_update_lut(bel, lut, z, sigma, occupancy):
    """
    bel: belief (H,W)
    lut: precomputed (H,W,A)
    z: observed ranges (A,)
    occupancy: grid map
    """
    likelihood = range_likelihood_vectorized(z, lut, sigma)
    likelihood[occupancy == 1] = 0.0

    bel_new = bel * likelihood
    s = bel_new.sum()
    if s > 0:
        bel_new /= s
    return bel_new


def show_world_map(world, location=None, title=''):
  plt.imshow(1-world, cmap='gray')
  plt.title('World Map' if not title else title)
  plt.axis('off')
  if location is not None:
    plt.plot(location[1], location[0], 'rx', markersize=10, markeredgewidth=2)
  plt.show()


def generate_world(height=10, width=10,):

    world = np.zeros((height, width))
    rands = np.random.rand(height, width)
    world[rands < 0.3] = 1
    return world


def sample_robot_location(world):
    free_cells = np.argwhere(world == 0)
    robot_map_gt = free_cells[np.random.randint(0, len(free_cells))]
    return robot_map_gt

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
    parser.add_argument('--map-height', type=int, default=MAP_HEIGHT,
                        help=f'Height of the map (default: {MAP_HEIGHT})')
    parser.add_argument('--map-width', type=int, default=MAP_WIDTH,
                        help=f'Width of the map (default: {MAP_WIDTH})')
    parser.add_argument('--num-runs', type=int, default=NUM_SIMULATION_RUNS,
                        help='Number of simulation runs (default: 1000)')
    parser.add_argument('--max-range', type=float, default=MAX_RANGE,
                        help=f'Maximum range for ray casting (default: {MAX_RANGE})')
    parser.add_argument('--sigma', type=float, default=SIGMA,
                        help=f'Sigma for likelihood calculation (default: {SIGMA})')
    parser.add_argument('--num-angles', type=int, default=NUM_ANGLES,
                        help='Number of angles for ray casting')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    return parser.parse_args()


def main(args):
    logger = get_logger(__name__, args.log_level)
    logger.info('Starting the program...')
    logger.info('Program started at %s', time.asctime())
    logger.info(f'Arguments: {args}')

    H, W = args.map_height, args.map_width
    correct_estimates = 0
    failed_estimates = 0
    num_runs = args.num_runs
    for _ in range(num_runs):
        world = generate_world(H, W)
        robot_loc = sample_robot_location(world)
        logger.info(f"Robot location: {robot_loc}")

        belief = np.ones((H, W)) / (H * W)
        angles = make_angles(args.num_angles)

        lut = ray_cast_lut(world, angles, max_range=args.max_range)

        z_measured = ray_cast_from_pos(robot_loc[0], robot_loc[1], world, angles, max_range=args.max_range)
        belief = measurement_update_lut(
            belief, lut, z_measured, sigma=args.sigma, occupancy=world
        )

        robot_map_estimate = np.unravel_index(np.argmax(belief), belief.shape)
        logger.info(f"MAP: {[int(e) for e in robot_map_estimate]}")
        logger.info(f"GT: {[int(e) for e in robot_loc]}")

        correct = int(np.allclose(robot_loc, robot_map_estimate))
        logger.info(f"Correct: {correct}")
        if correct:
            correct_estimates += 1
        else:
            failed_estimates += 1
            show_world_map(world, location=robot_loc, title='World Map with Robot GT')
            show_world_map(world, location=robot_map_estimate, title='World Map with Robot Estimated')
            print()

    logger.info(f"Correct estimates: {correct_estimates}")
    logger.info(f"Failed estimates: {failed_estimates}")
    logger.info(f"Accuracy: {correct_estimates / num_runs * 100:.2f}%")
    

    # show_world_map(world, location=robot_loc, title='World Map with Robot GT')
    # show_world_map(world, location=robot_map_estimate, title='World Map with Robot Estimated')


if __name__ == "__main__":
    args = parse_args()
    main(args)
