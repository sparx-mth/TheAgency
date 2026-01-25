"""Common utilities for path planners."""
from .ompl_imports import ob, og, OMPL_AVAILABLE, OMPL_ERROR
from .utils_2d import (
    interpolate_path_2d,
    reduce_path_2d,
    make_clearance_objective_2d,
)
from .utils_3d import (
    dist3d,
    interpolate_path_3d,
    reduce_path_3d,
    make_clearance_objective_3d,
    get_voxelmap_dim,
    get_voxelmap_resolution,
    setup_ompl_space_3d,
)

__all__ = [
    # OMPL
    "ob",
    "og",
    "OMPL_AVAILABLE",
    "OMPL_ERROR",
    # 2D utilities
    "interpolate_path_2d",
    "reduce_path_2d",
    "make_clearance_objective_2d",
    # 3D utilities
    "dist3d",
    "interpolate_path_3d",
    "reduce_path_3d",
    "make_clearance_objective_3d",
    "get_voxelmap_dim",
    "get_voxelmap_resolution",
    "setup_ompl_space_3d",
]
