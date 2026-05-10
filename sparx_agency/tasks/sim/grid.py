import math
from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, float64, signedinteger
from numpy._typing import _32Bit, _64Bit


def generate_world(height: int=10, width: int=10,) -> ndarray[tuple[int, ...], dtype[float64]]:
    """
    Generates a 2D numpy array representing a world with randomly placed obstacles.

    Each cell in the generated array represents a unit of the world. The value of each
    cell is either 0 (empty) or 1 (obstacle). Obstacles are randomly distributed with
    a 30% probability for each cell.

    Parameters:
        height (int): The height of the generated world in cells. Must be a positive integer.
        width (int): The width of the generated world in cells. Must be a positive integer.

    Returns:
        numpy.ndarray: A 2D array where 0 represents an empty space and 1 represents an obstacle.

    Raises:
        ValueError: If the provided height or width is less than or equal to 0.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive integers.")
    world = np.zeros((height, width))
    rands = np.random.rand(height, width)
    world[rands < 0.3] = 1
    return world


def sample_robot_orientation(orientations: ndarray[tuple[int, ...], dtype[float64]],
                             min_angle: float = -math.pi / 16,
                             max_angle: float = math.pi / 16) -> float:
    """
    Samples and bounds a robot's orientation based on random selection from a given array.

    This function selects a random orientation from the input array, ensuring
    that the resulting orientation is clamped within the specified range, defined
    by `min_angle` and `max_angle`. It returns the processed orientation value.

    Parameters:
    orientations (ndarray): An array of possible orientations for the robot.
    min_angle (float): The minimum angle bound for the orientation, defaulting
                       to -π/16.
    max_angle (float): The maximum angle bound for the orientation, defaulting
                       to π/16.

    Returns:
    float: A randomly selected and bounded orientation value from the input
           array.
    """
    random_orientation_idx = int(np.random.randint(0, np.size(orientations)))
    robot_orientation_gt = orientations[random_orientation_idx]
    robot_orientation_gt = min(max(robot_orientation_gt, min_angle), max_angle)
    return robot_orientation_gt


def sample_robot_location(world: ndarray[tuple[int, ...], dtype[float64]]) -> ndarray[
    tuple[Any, ...], dtype[signedinteger[_32Bit | _64Bit]]]:
    """
    Generates a random location for a robot in a given grid world.

    Creates a random location for a robot in a grid world based on available
    free cells. Only cells with a value of `0` in the `world` array are
    considered free.

    Parameters:
    world (ndarray[tuple[int, ...], dtype[float64]]): A 2D array
    representing the grid world where `0` indicates free cells and other
    values indicate obstacles or occupied cells.

    Returns:
    ndarray[tuple[Any, ...], dtype[signedinteger[_32Bit | _64Bit]]]:
    The randomly chosen free cell index representing the robot's
    location.
    """
    free_cells = np.argwhere(world == 0)
    robot_loc_gt = free_cells[np.random.randint(0, len(free_cells))]
    return robot_loc_gt


def show_world_map(world, location=None, orientation=None, title=''):
    """
    Displays a visual representation of a world display_map, optionally marking a specific
    location and orientation with an arrow. The display_map is rendered in grayscale, with
    the option to customize the title and display additional markers indicating
    specific points or directions.

    Parameters:
    world (ndarray): A 2D numpy array representing the world grid, where values
        indicate passable (0) or blocked (1) areas.
    location (tuple[int, int], optional): A tuple representing the coordinates
        (x, y) of a specific location in the grid. If provided, a red cross marks
        this location on the display_map.
    orientation (float, optional): Orientation in radians to represent the
        direction of the arrow drawn starting from the location marker. Only used
        if location is provided.
    title (str, optional): An optional string specifying the title of the display_map.
        Defaults to an empty string, which results in the title being set to
        "World Map".

    Raises:
    TypeError: If `world` is not a 2D array or if the parameters `location` or
        `orientation` violate expected data types.
    ValueError: If the `location` coordinates are out of bounds for the `world`.

    Notes:
    - The coordinate system for the location and orientation corresponds to the
      world display_map grid.
    - The orientation arrow length can be adjusted by modifying the internal
      `arrow_length` parameter.
    """

    LOCATION_MARKER = 'rx'
    LOCATION_MARKER_SIZE = 10
    LOCATION_MARKER_EDGE_WIDTH = 2
    ARROW_SCALE_FACTOR = 16
    ARROW_COLOR = 'blue'


    def validate_world(world):
        if not isinstance(world, np.ndarray) or world.ndim != 2:
            raise TypeError("world must be a 2D numpy array")

    def validate_location(location):
        if not isinstance(location, tuple) or len(location) != 2:
            raise TypeError("location must be a tuple of (x, y) coordinates")

    def validate_orientation(orientation):
        if not isinstance(orientation, float) or math.isnan(orientation):
            raise TypeError("orientation must be a float value")

    def draw_location(x_coord, y_coord):
        plt.plot(
            x_coord,
            y_coord,
            LOCATION_MARKER,
            markersize=LOCATION_MARKER_SIZE,
            markeredgewidth=LOCATION_MARKER_EDGE_WIDTH,
        )

    def draw_orientation_arrow(x_coord, y_coord, orientation, height, width):
        validate_orientation(orientation)

        arrow_length = max(height, width) // ARROW_SCALE_FACTOR
        dx = arrow_length * math.cos(orientation)
        dy = arrow_length * math.sin(orientation)

        plt.arrow(
            x_coord,
            y_coord,
            dy,
            dx,
            head_width=arrow_length // 2,
            head_length=arrow_length // 2,
            fc=ARROW_COLOR,
            ec=ARROW_COLOR,
        )

    validate_world(world)
    display_map = (1-world).T
    height, width = display_map.shape

    left = - width // 2
    bottom = - height // 2
    right = left + width
    top = bottom + height
    extent = [left, right, bottom, top]

    plt.imshow(display_map, cmap='gray', origin='lower', extent=extent)
    plt.title(title or 'World Map')
    plt.xticks(np.arange(left, right + 1, 2))
    plt.yticks(np.arange(bottom, top + 1, 2))
    plt.grid(True)

    if location is not None:
        validate_location(location)

        x_coord = left + location[0]
        y_coord = bottom + location[1]

        draw_location(x_coord, y_coord)

        if orientation is not None:
            draw_orientation_arrow(x_coord, y_coord, orientation, height, width)



    plt.show()




if __name__ == "__main__":
    world = generate_world(height=16, width=64,)
    loc = sample_robot_location(world)
    orientation = sample_robot_orientation(np.linspace(-math.pi, math.pi, 32, endpoint=False))
    show_world_map(world, location=(loc[0], loc[1]), orientation=orientation, title='World Map with Robot')


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
