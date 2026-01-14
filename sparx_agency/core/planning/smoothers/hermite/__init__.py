"""Hermite spline trajectory smoothing (2D and 3D)."""
from .params import HermiteParams, HermiteParams3D
from .smoother import (
    HermiteSmoother,
    HermiteSmoother3D,
    SmootherRequest,
    SmootherRequest3D,
    BaseSmoother,
    BaseSmoother3D,
)

__all__ = [
    # 2D (original)
    "HermiteParams",
    "HermiteSmoother",
    "SmootherRequest",
    "BaseSmoother",
    # 3D (new)
    "HermiteParams3D",
    "HermiteSmoother3D",
    "SmootherRequest3D",
    "BaseSmoother3D",
]