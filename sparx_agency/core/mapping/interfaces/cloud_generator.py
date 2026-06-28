from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from sparx_agency.core.common.types import Intrinsics


class CloudGenerator(ABC):
    """
    Depth->PointCloud API (ROS-free).
    Input depth (HxW float32) and intrinsics.
    Output Nx3 float32 points in camera frame by default.
    """

    @abstractmethod
    def depth_to_cloud_to_base_xyz(self, depth_m: np.ndarray, intr: Intrinsics) -> np.ndarray:
        raise NotImplementedError
