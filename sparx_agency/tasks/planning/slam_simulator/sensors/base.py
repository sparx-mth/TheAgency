from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaseSensor(ABC):
    def __init__(self, max_range: int = 10):
        self.max_range = max_range

    @abstractmethod
    def sense(
        self,
        pos: Tuple[int, int],
        facing: str,
        true_map: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Return list of (x, y, tile_value) observations."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass