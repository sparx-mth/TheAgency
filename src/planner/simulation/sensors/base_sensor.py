from abc import ABC, abstractmethod


class BaseSensor(ABC):
    @abstractmethod
    def sense(self, pos, facing, env):
        """
        Perform sensing from the current position and facing direction.

        Args:
            pos (tuple): Current (x, y) position of the drone.
            facing (str): Current facing direction (e.g., 'NORTH').
            env (object): The environment instance for querying tiles.

        Returns:
            list of (x, y, val): All observed cells and their values.
        """
        pass
