import time
from pathlib import Path
from enum import Enum
import numpy as np
import math
import matplotlib.pyplot as plt
import logging


class Constants(Enum):
    MAX_RANGE = 6.0
    SIGMA = 0.5
    NUM_ANGLES = 32
    NUM_BEAMS = 12
    MAP_HEIGHT = 10
    MAP_WIDTH = 10
    FOV = 2* math.pi / 3

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
                dist=np.inf
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
                            dist = np.inf
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


def main():

    logger = get_logger(__name__, 'INFO')
    logger.info('Starting the program...')
    logger.info('Program started at %s', time.asctime())
    logger.info(Constants)

    H, W, NUM_ANGLES = Constants.MAP_HEIGHT.value, Constants.MAP_WIDTH.value, Constants.NUM_ANGLES.value
    FOV, NUM_BEAMS, MAX_RANGE = Constants.FOV.value,  Constants.NUM_BEAMS.value, Constants.MAX_RANGE.value
    correct_estimates = 0
    num_runs = 1000
    for _ in range(num_runs):
        robot_loc, world, robot_orientation_gt = regenerate_robot_pose(H, W, NUM_ANGLES)

        u = np.random.randint(-1, 2, size=2)
        # prediction_pos = np.array(robot_map_gt) + np.array(u)
        # prediction_orientation = robot_orientation_gt + np.random.uniform(-0.1, 0.1)

        print(f"Robot GT Position: {robot_loc}")
        print(f"Robot GT Orientation (radians): {robot_orientation_gt:.2f}")
        print(f"Robot GT Orientation (degrees): {math.degrees(robot_orientation_gt):.2f}")

        # print(f"Prediction Position: {prediction_pos}")
        # print(f"Prediction Orientation (radians): {prediction_orientation:.2f}")
        # print(f"Prediction Orientation (degrees): {math.degrees(prediction_orientation):.2f}")

        belief = np.ones((H, W, NUM_ANGLES))
        belief /= belief.sum()

        orientations = make_orientations(NUM_ANGLES)
        beam_angles = make_fov_angles(FOV, NUM_BEAMS)

        lut = ray_cast_lut_pose(
            world,
            orientations,
            beam_angles,
            max_range=MAX_RANGE
        )

        z_measured_pose = ray_cast_from_pose(
            robot_loc[0],
            robot_loc[1],
            robot_orientation_gt,
            world,
            beam_angles,
            max_range=MAX_RANGE
        )

        belief = measurement_update_pose(
            belief, lut, z_measured_pose, sigma=Constants.SIGMA.value, occupancy=world
        )

        idx = np.unravel_index(np.argmax(belief), belief.shape)

        logger.info(f"MAP\n pos: {[int(e) for e in idx[:2]]}")
        logger.info(f" θ (deg): {math.degrees(orientations[idx[2]])}")

        logger.info(f"GT\n pos: {robot_loc}")
        logger.info(f" θ (deg): {math.degrees(robot_orientation_gt):.2f}")
        correct = int(np.allclose(robot_loc, np.array(idx[:2]))) and np.isclose(robot_orientation_gt, orientations[idx[2]])
        logger.info(f"Correct: {correct}")
        correct_estimates += correct

    logger.info(f"Accuracy: {correct_estimates / num_runs * 100:.2f}%")

    # show_world_map(world, location=robot_loc, title='World Map with Robot GT')
    # show_world_map(world, location=robot_map_estimate, title='World Map with Robot Estimated')


if __name__ == "__main__":
    main()