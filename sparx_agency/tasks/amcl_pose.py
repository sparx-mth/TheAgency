import time
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
import logging
import argparse

MAX_RANGE = 6.0
SIGMA = 0.5
FOV_DEG = 120.0
NUM_ANGLES = 32
NUM_BEAMS = 8
MAP_HEIGHT = 6
MAP_WIDTH = 3
NUM_SIMULATION_RUNS = 1000

def make_angles(num_angles):
    return np.linspace(-math.pi, math.pi, num_angles, endpoint=False)

def make_orientations(K):
    return np.linspace(-math.pi, math.pi, K, endpoint=False)


def make_fov_angles(fov_rad, num_beams=9):
    return np.linspace(-fov_rad / 2, fov_rad / 2, num_beams)

def ray_cast_from_pose(i, j, theta, grid, beam_angles, max_range, step=0.1):
    """
    Simulates depth reading from a sensor for a robot at (i, j) with orientation theta,
    casting rays along beam_angles relative to theta.
    Returns np.array of shape (len(beam_angles),)
    """
    H, W = grid.shape
    distances = []
    for rel_angle in beam_angles:
        angle = theta + rel_angle
        dist = 0.0
        while dist < max_range:
            x = int(round(i + dist * math.cos(angle)))
            y = int(round(j + dist * math.sin(angle)))

            if x < 0 or y < 0 or x >= H or y >= W:
                dist += max_range
                break
            if grid[x, y] == 1:
                break

            dist += step

        distances.append(dist)
    return np.array(distances)

def ray_cast_lut_pose(grid, orientations, beam_angles, max_range, step=0.1):
    H, W = grid.shape
    K = len(orientations)
    B = len(beam_angles)

    lut = np.ones((H, W, K, B), dtype=np.float32) * np.inf

    for i in range(H):
        for j in range(W):
            if grid[i, j] == 1:
                continue

            for k, theta in enumerate(orientations):
                for b, rel in enumerate(beam_angles):
                    angle = theta + rel
                    dist = 0.0

                    while dist < max_range:
                        x = int(round(i + dist * math.cos(angle)))
                        y = int(round(j + dist * math.sin(angle)))

                        if x < 0 or y < 0 or x >= H or y >= W:
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
  plt.imshow(1-world, cmap='gray')
  plt.title('World Map' if not title else title)
  plt.axis('off')
  if location is not None:
    plt.plot(location[1], location[0], 'rx', markersize=10, markeredgewidth=2)
    if orientation is not None:
        # Draw an arrow for orientation
        arrow_length = 0.8  # Adjust as needed
        dx = arrow_length * math.cos(orientation)
        dy = arrow_length * math.sin(orientation)
        # plt.arrow uses (x, y, dx, dy), where x,y are horizontal/vertical in plot coords
        # Our location is (row, col), so location[1] is x, location[0] is y
        plt.arrow(location[1], location[0], dy, dx, head_width=0.3, head_length=0.3, fc='blue', ec='blue')
  plt.show()


def generate_world(height=10, width=10,):

    world = np.zeros((height, width))
    rands = np.random.rand(height, width)
    world[rands < 0.3] = 1
    return world

def regenerate_robot_pose(H, W, K):
    world = generate_world(H, W)
    robot_map_gt = sample_robot_location(world)
    orientations = make_orientations(K)
    random_orientation_idx = np.random.randint(0, K)
    robot_orientation_gt = orientations[random_orientation_idx]
    return robot_map_gt, world, robot_orientation_gt

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

    H, W = args.map_height, args.map_width
    correct_estimates = 0
    failed_estimates = 0
    num_runs = args.num_runs
    for _ in range(num_runs):
        robot_loc, world, robot_orientation_gt = regenerate_robot_pose(H, W, args.num_angles)

        # u = np.random.randint(-1, 2, size=2)
        # prediction_pos = np.array(robot_map_gt) + np.array(u)
        # prediction_orientation = robot_orientation_gt + np.random.uniform(-0.1, 0.1)

        # print(f"Robot GT Position: {robot_loc}")
        # print(f"Robot GT Orientation (radians): {robot_orientation_gt:.2f}")
        # print(f"Robot GT Orientation (degrees): {math.degrees(robot_orientation_gt):.2f}")

        # print(f"Prediction Position: {prediction_pos}")
        # print(f"Prediction Orientation (radians): {prediction_orientation:.2f}")
        # print(f"Prediction Orientation (degrees): {math.degrees(prediction_orientation):.2f}")

        belief = np.ones((H, W, args.num_angles))
        belief /= belief.sum()

        orientations = make_orientations(args.num_angles)
        beam_angles = make_fov_angles(fov_rad=math.radians(args.fov_deg), num_beams=args.num_beams)

        lut = ray_cast_lut_pose(
            world,
            orientations,
            beam_angles,
            max_range=args.max_range
        )

        z_measured_pose = ray_cast_from_pose(
            robot_loc[0],
            robot_loc[1],
            robot_orientation_gt,
            world,
            beam_angles,
            max_range=args.max_range
        )

        belief = measurement_update_pose(
            belief, lut, z_measured_pose, sigma=args.sigma, occupancy=world
        )

        idx = np.unravel_index(np.argmax(belief), belief.shape)

        logger.info(f"MAP\n pos: {[int(e) for e in idx[:2]]}")
        logger.info(f" θ (deg): {math.degrees(orientations[idx[2]])}")

        logger.info(f"GT\n pos: {robot_loc}")
        logger.info(f" θ (deg): {math.degrees(robot_orientation_gt):.2f}")
        correct = int(np.allclose(robot_loc, np.array(idx[:2]))) and np.isclose(robot_orientation_gt, orientations[idx[2]])
        logger.info(f"Correct: {correct}")
        if correct:
            correct_estimates += 1
        else:
            failed_estimates += 1
            show_world_map(world, location=robot_loc, orientation=robot_orientation_gt, title='World Map with Robot GT')
            show_world_map(world, location=idx[:2], orientation=orientations[idx[2]],title='World Map with Robot Estimated')
            print()

    logger.info(f"Correct estimates: {correct_estimates}")
    logger.info(f"Failed estimates: {failed_estimates}")
    logger.info(f"Accuracy: {correct_estimates / num_runs * 100:.2f}%")

    # show_world_map(world, location=robot_loc, title='World Map with Robot GT')
    # show_world_map(world, location=robot_map_estimate, title='World Map with Robot Estimated')


if __name__ == "__main__":
    args = parse_args()
    main(args)
