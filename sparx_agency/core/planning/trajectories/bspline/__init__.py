"""FALCON's B-spline trajectory, evaluated locally.

See ``README.md`` next to this file for why the curve is carried rather than its
100 Hz samples.
"""
from sparx_agency.core.planning.trajectories.bspline.non_uniform_bspline import (
    NonUniformBspline,
)
from sparx_agency.core.planning.trajectories.bspline.projection import (
    ProjectionParams, TrajectoryProjector,
)
from sparx_agency.core.planning.trajectories.bspline.trajectory import BsplineTrajectory

__all__ = [
    "NonUniformBspline",
    "BsplineTrajectory",
    "ProjectionParams",
    "TrajectoryProjector",
]
