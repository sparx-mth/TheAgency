"""
Trajectory sampling utilities for safety checking.

This module provides functions for sampling trajectory points at fixed time
intervals and for finding the closest trajectory point to a given position.
These utilities support the trajectory safety checker in determining which
portion of a trajectory to validate.

Functions:
    sample_trajectory: Sample a trajectory at fixed time intervals.
    nearest_index_xyz: Find the trajectory point nearest to a 3D position.

Dependencies:
    - Trajectory, TrajectoryPoint from sparx_agency.core.common.types
"""

from __future__ import annotations

from math import inf
from typing import List, Sequence

from sparx_agency.core.common.types import Trajectory, TrajectoryPoint


def sample_trajectory(
    trajectory: Trajectory,
    dt: float,
    max_samples: int,
) -> List[TrajectoryPoint]:
    """
    Sample a trajectory at fixed time intervals.

    This function extracts discrete waypoints from a trajectory at regular
    time intervals, which can then be used for collision checking or
    visualization.

    Args:
        trajectory: The Trajectory object to sample. Must implement
            sample_by_time(dt) -> List[TrajectoryPoint].
        dt: Time step (seconds) between samples. Smaller values give
            finer resolution but more points to process.
        max_samples: Maximum number of samples to return. If the trajectory
            would produce more samples, it is truncated.

    Returns:
        A list of TrajectoryPoint objects sampled from the trajectory,
        with at most max_samples entries.

    Example:
        >>> trajectory = create_trajectory(...)  # Your trajectory
        >>> points = sample_trajectory(trajectory, dt=0.05, max_samples=200)
        >>> print(f"Sampled {len(points)} points")

    Notes:
        - The actual sampling logic is delegated to the Trajectory's
          sample_by_time method. This function mainly handles the
          max_samples truncation.
        - For a 5-second trajectory with dt=0.05, you'd get up to
          100 samples (if max_samples >= 100).
    """
    pts = trajectory.sample_by_time(dt)
    return pts[:max_samples] if len(pts) > max_samples else pts


def nearest_index_xyz(
    points: Sequence[TrajectoryPoint],
    x: float,
    y: float,
    z: float,
) -> int:
    """
    Find the index of the trajectory point closest to a 3D position.

    This function performs a linear search through the trajectory points
    to find the one with the minimum Euclidean distance to the query
    position (x, y, z). This is used to determine the robot's current
    "progress" along the trajectory.

    Args:
        points: Sequence of TrajectoryPoint objects to search.
        x: Query x-coordinate (meters).
        y: Query y-coordinate (meters).
        z: Query z-coordinate (meters).

    Returns:
        The index of the closest point in the sequence. Returns 0 if the
        sequence is empty (though this should be avoided).

    Example:
        >>> points = sample_trajectory(trajectory, dt=0.05, max_samples=200)
        >>> robot_x, robot_y, robot_z = 1.5, 2.0, 0.5
        >>> idx = nearest_index_xyz(points, robot_x, robot_y, robot_z)
        >>> closest_point = points[idx]
        >>> print(f"Robot is nearest to trajectory point {idx} at t={closest_point.t:.2f}s")

    Notes:
        - Uses squared Euclidean distance for efficiency (avoids sqrt).
        - Linear search is O(n) but sufficient for typical trajectory lengths
          (hundreds of points). For very long trajectories, consider spatial
          indexing structures.
        - If multiple points are equidistant, returns the first one found.
    """
    best_i = 0
    best_d2 = inf
    for i, p in enumerate(points):
        dx = p.x - x
        dy = p.y - y
        dz = p.z - z
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i