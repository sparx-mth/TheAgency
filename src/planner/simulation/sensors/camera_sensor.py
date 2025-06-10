from src.planner.simulation.sensors.base_sensor import BaseSensor
from src.planner.simulation.simulation_constants import FACING_TO_DELTA, WALL, DOOR_CLOSED


class CameraSensor(BaseSensor):
    def __init__(self, max_distance):
        self.max_distance = max_distance

    def sense(self, pos, facing, env):
        """
        Simulates a forward-looking camera sensor that returns all tiles
        in a straight line in the current facing direction up to max_distance.

        Args:
            pos (tuple): Current (x, y) position.
            facing (str): Facing direction ('NORTH', 'EAST', etc.).
            env (object): Environment object with get_tile(x, y) method.

        Returns:
            list of (x, y, val): Cells directly ahead with their tile values.
        """
        dx, dy = FACING_TO_DELTA[facing]
        x, y = pos
        observations = []

        for i in range(1, self.max_distance + 1):
            nx, ny = x + i * dx, y + i * dy
            if not (0 <= nx < env.width and 0 <= ny < env.height):
                break
            val = env.get_tile(nx, ny)
            observations.append((nx, ny, val))

            # Stop if vision is blocked
            if val in {WALL, DOOR_CLOSED}:
                break

        return observations
