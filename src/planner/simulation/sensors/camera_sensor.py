"""
Implements a realistic directional CameraSensor for SLAM drones.

This sensor uses ray-casting to simulate a camera's field of view, casting multiple rays
across the FOV angle and returning all visible cells. Handles realistic vision blocking
and can see through windows and open doors.

Used in directional vision-based SLAM simulations to mimic a real camera's behavior.
"""

from typing import Tuple, List, TYPE_CHECKING
from planner.simulation.sensors.base_sensor import BaseSensor
from planner.simulation.simulation_constants import FACING_TO_DELTA, WALL, DOOR_CLOSED, FACING_DIRECTION
if TYPE_CHECKING:
    from planner.simulation.grid_map_env import GridMapEnv


def _blocks_vision(cell_value: int) -> bool:
    """
    Determines what blocks vision completely.

    Realistic behavior:
    - Walls block vision completely
    - Closed doors block vision completely
    - Windows allow seeing through (you can see what's behind glass)
    - Open doors allow seeing through
    - Empty space doesn't block vision

    Args:
        cell_value (int): The value of the cell being checked.

    Returns:
        bool: True if this cell blocks vision to cells behind it.
    """
    return cell_value in {WALL, DOOR_CLOSED}


class CameraSensor(BaseSensor):
    """
    A sensor that simulates a realistic directional camera mounted on a SLAM drone.

    The camera uses ray-casting across a configurable field of view to detect all
    visible cells within range. Multiple rays are cast across the FOV angle to
    provide comprehensive coverage. Vision is blocked by obstacles such as walls
    and closed doors, while allowing sight through windows and open doors.

    Attributes:
        max_distance (int): Maximum sensing range in tiles.
        fov_angle_rad (float): Field of view angle in radians.
        num_rays (int): Number of rays cast across the FOV.
        ray_step (float): Step size for ray marching (0.1 for smooth rays).
    """
    def __init__(
        self,
        max_distance: int = 10,
        fov_angle_deg: int = 45,
        num_rays: int = 60
    ):
        """
        Initialize the camera sensor with specified parameters.

        Args:
            max_distance (int, optional): Maximum range of the camera in tiles. Defaults to 10.
            fov_angle_deg (int, optional): Field of view angle in degrees. Defaults to 45.
            num_rays (int, optional): Number of rays to cast across the FOV for coverage. Defaults to 60.
        """
        self.max_distance = max_distance
        self.fov_angle_rad = math.radians(fov_angle_deg)
        self.num_rays = num_rays
        self.ray_step = 0.1  # Step size for ray marching

    def sense(
        self,
        pos: Tuple[int, int],
        facing: FACING_DIRECTION,
        env: "GridMapEnv"
    ) -> List[Tuple[int, int, int]]:
        """
        Simulate camera sensing using ray-casting across the field of view.

        Casts multiple rays distributed evenly across the camera's field of view
        and returns all visible cells within range. Each ray stops when it hits
        an obstacle that blocks vision (walls, closed doors).

        Args:
            pos (Tuple[int, int]): Current (x, y) position of the drone.
            facing (FACING_DIRECTION): Current facing direction of the drone.
            env (GridMapEnv): The simulation environment to sense within.

        Returns:
            List[Tuple[int, int, int]]: List of (x, y, value) tuples representing
            all visible cells within the camera's field of view and range.
        """
        drone_x, drone_y = pos
        observations: Set[Tuple[int, int, int]] = set()

        # Get center direction angle (in radians)
        facing_dx, facing_dy = FACING_TO_DELTA[facing]
        center_angle = math.atan2(facing_dy, facing_dx)

        # Cast rays across the entire FOV
        for i in range(self.num_rays):
            if self.num_rays == 1:
                angle = center_angle
            else:
                # Distribute rays evenly across FOV
                angle = center_angle - self.fov_angle_rad / 2 + i * (self.fov_angle_rad / (self.num_rays - 1))
            ray_observations = self._cast_ray(drone_x, drone_y, angle, env)
            observations.update(ray_observations)
        return list(observations)

    def _cast_ray(
        self,
        x0: int,
        y0: int,
        angle: float,
        env: "GridMapEnv"
    ) -> List[Tuple[int, int, int]]:
        """
        Cast a single ray from the drone's position at the specified angle.

        Uses ray marching with small steps to ensure comprehensive cell coverage
        and smooth, realistic ray behavior. The ray continues until it hits an
        obstacle, reaches maximum distance, or goes out of bounds.

        Args:
            x0 (int): Starting x-coordinate (drone position).
            y0 (int): Starting y-coordinate (drone position).
            angle (float): Ray direction angle in radians.
            env (GridMapEnv): Environment to cast the ray through.

        Returns:
            List[Tuple[int, int, int]]: List of (x, y, value) tuples for all
            cells encountered along this ray before hitting an obstacle.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Start from center of drone's tile
        x, y = float(x0) + 0.5, float(y0) + 0.5
        ray_result = []
        visited_cells: Set[Tuple[int, int]] = set()

        # Maximum number of steps (prevent infinite loops)
        max_steps = int(self.max_distance / self.ray_step) + 50

        for step in range(max_steps):
            tile_x, tile_y = int(x), int(y)

            # Check bounds
            if not (0 <= tile_x < env.width and 0 <= tile_y < env.height):
                break

            # Only process each cell once per ray
            if (tile_x, tile_y) not in visited_cells:
                visited_cells.add((tile_x, tile_y))

                # Skip the drone's own position
                if tile_x != x0 or tile_y != y0:
                    cell_value = env.get_tile(tile_x, tile_y)
                    ray_result.append((tile_x, tile_y, cell_value))

                    # Check if this cell blocks further vision
                    if _blocks_vision(cell_value):
                        break

            # Step forward along the ray
            x += dx * self.ray_step
            y += dy * self.ray_step

            # Check if we've exceeded max distance
            dist = math.sqrt((x - (x0 + 0.5)) ** 2 + (y - (y0 + 0.5)) ** 2)
            if dist >= self.max_distance:
                break
        return ray_result
