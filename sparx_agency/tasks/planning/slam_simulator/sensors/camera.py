import numpy as np
from typing import List, Tuple
from sparx_agency.tasks.planning.slam_simulator.sensors.base import BaseSensor
from sparx_agency.tasks.planning.slam_simulator.constants import DIRECTIONS, DIRECTION_DELTAS, TileType

class CameraSensor(BaseSensor):
    def __init__(self, max_range: int = 10, fov_deg: float = 90, num_rays: int = 30):
        super().__init__(max_range)
        self.fov_deg = fov_deg
        self.num_rays = num_rays

    @property
    def name(self) -> str:
        return "camera"

    def sense(
        self,
        pos: Tuple[int, int],
        facing: str,
        true_map: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        observations = []
        seen = set()
        height, width = true_map.shape

        base_angle = DIRECTIONS.index(facing) * 90
        half_fov = self.fov_deg / 2
        angles = np.linspace(base_angle - half_fov, base_angle + half_fov, self.num_rays)

        for angle in angles:
            rad = np.radians(angle)
            dx = np.cos(rad)
            dy = -np.sin(rad)

            for dist in range(1, self.max_range + 1):
                x = int(pos[0] + dx * dist)
                y = int(pos[1] + dy * dist)

                if not (0 <= x < width and 0 <= y < height):
                    break

                if (x, y) not in seen:
                    seen.add((x, y))
                    tile = true_map[y, x]
                    observations.append((x, y, int(tile)))

                if tile in (TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS):
                    break

        return observations