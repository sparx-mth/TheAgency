from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class DepthModel(ABC):
    """
    Depth model API (ROS-free).
    Input: RGB image (HxWx3), uint8 or float32.
    Output: depth (HxW) float32, meters or relative meters.
    """

    @abstractmethod
    def infer_depth(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError
