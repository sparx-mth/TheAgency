"""Common geometry stays OMPL-free; explicit OMPL exports load on demand."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ompl_imports import ob, og, OMPL_AVAILABLE, OMPL_ERROR

from .utils_2d import (
    interpolate_path_2d,
    split_long_segments_2d,
    decimate_min_spacing_2d,
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


def __getattr__(name):
    """Preserve the public OMPL aliases without loading bindings for A*."""
    if name in ("ob", "og", "OMPL_AVAILABLE", "OMPL_ERROR"):
        from importlib import import_module
        bindings = import_module(__name__ + ".ompl_imports")
        value = getattr(bindings, name)
        globals()[name] = value
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = [
    "ob", "og", "OMPL_AVAILABLE", "OMPL_ERROR",
    "interpolate_path_2d", "split_long_segments_2d", "decimate_min_spacing_2d",
    "reduce_path_2d", "make_clearance_objective_2d", "dist3d",
    "interpolate_path_3d", "reduce_path_3d", "make_clearance_objective_3d",
    "get_voxelmap_dim", "get_voxelmap_resolution", "setup_ompl_space_3d",
]
