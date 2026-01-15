"""3D path planning utilities."""
from __future__ import annotations

from math import sqrt
from typing import List, Tuple, TYPE_CHECKING

from sparx_agency.core.common.types import Pose3D

from .ompl_imports import ob, og, OMPL_AVAILABLE

if TYPE_CHECKING:
    from ompl import base as ob
    from ompl import geometric as og


# =============================================================================
# Distance and Path Utilities
# =============================================================================

def dist3d(a: Pose3D, b: Pose3D) -> float:
    """Euclidean distance between two 3D poses."""
    return sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)


def interpolate_path_3d(points: List[Pose3D], spacing: float) -> List[Pose3D]:
    """Interpolate 3D path at uniform spacing."""
    if len(points) < 2 or spacing <= 0:
        return points

    result = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
        d = sqrt(dx * dx + dy * dy + dz * dz)

        if d > spacing:
            n_segments = int(d / spacing)
            for i in range(1, n_segments + 1):
                t = i / (n_segments + 1)
                result.append(Pose3D(a.x + t * dx, a.y + t * dy, a.z + t * dz))
        result.append(b)
    return result


def reduce_path_3d(si, voxelmap, states: List, min_clearance: float) -> List:
    """Adaptive waypoint reduction for 3D."""
    if len(states) < 3:
        return [si.cloneState(s) for s in states]

    kept = [si.cloneState(states[0])]
    for i in range(1, len(states) - 1):
        x, y, z = states[i][0], states[i][1], states[i][2]
        clearance = voxelmap.world_clearance(x, y, z)
        can_skip = si.checkMotion(kept[-1], states[i + 1])

        if clearance < min_clearance or not can_skip:
            kept.append(si.cloneState(states[i]))

    kept.append(si.cloneState(states[-1]))
    return kept


# =============================================================================
# Clearance Objective
# =============================================================================

def make_clearance_objective_3d(si, voxelmap, weight: float):
    """Create 3D clearance-based optimization objective."""
    if not OMPL_AVAILABLE:
        raise RuntimeError("OMPL not available")

    class ClearanceObjective3D(ob.StateCostIntegralObjective):
        def __init__(self, si, voxelmap, weight: float) -> None:
            super().__init__(si, True)
            self._voxelmap = voxelmap
            self._weight = weight

        def stateCost(self, state) -> ob.Cost:
            clearance = self._voxelmap.world_clearance(state[0], state[1], state[2])
            return ob.Cost(self._weight / (clearance + 1.0))

    return ClearanceObjective3D(si, voxelmap, weight)


# =============================================================================
# Voxelmap Dimension Helpers
# =============================================================================

def get_voxelmap_dim(voxelmap, primary: str, fallback: str) -> int:
    """Return integer dimension from voxelmap, supporting multiple naming conventions."""
    if hasattr(voxelmap, primary):
        return int(getattr(voxelmap, primary))
    if hasattr(voxelmap, fallback):
        return int(getattr(voxelmap, fallback))
    raise AttributeError(f"Voxelmap missing dimension attributes: '{primary}' or '{fallback}'")


def get_voxelmap_resolution(voxelmap) -> float:
    """Return voxel resolution (meters per cell) supporting common field names."""
    if hasattr(voxelmap, "resolution"):
        return float(getattr(voxelmap, "resolution"))
    if hasattr(voxelmap, "voxel_size"):
        return float(getattr(voxelmap, "voxel_size"))
    raise AttributeError("Voxelmap missing resolution attribute: 'resolution' or 'voxel_size'")


# =============================================================================
# OMPL Space Setup
# =============================================================================

def setup_ompl_space_3d(voxelmap, params) -> Tuple:
    """
    Create OMPL state space and SimpleSetup for 3D planning.
    
    Returns:
        Tuple of (space, simple_setup, space_information)
    """
    if not OMPL_AVAILABLE:
        raise RuntimeError("OMPL not available")

    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)

    # Get dimensions (support multiple attribute names)
    sx = get_voxelmap_dim(voxelmap, "size_x", "width")
    sy = get_voxelmap_dim(voxelmap, "size_y", "height")
    sz = get_voxelmap_dim(voxelmap, "size_z", "depth")
    res = get_voxelmap_resolution(voxelmap)

    ox = float(voxelmap.origin_x)
    oy = float(voxelmap.origin_y)
    oz = float(voxelmap.origin_z)

    # World bounds
    bounds.setLow(0, ox)
    bounds.setHigh(0, ox + sx * res)
    bounds.setLow(1, oy)
    bounds.setHigh(1, oy + sy * res)
    bounds.setLow(2, oz)
    bounds.setHigh(2, oz + sz * res)
    space.setBounds(bounds)

    # Longest valid segment fraction (in meters -> fraction of space diagonal)
    longest_valid_m = getattr(params, "longest_valid_segment_m", None)
    if longest_valid_m is not None:
        diag = sqrt((sx * res) ** 2 + (sy * res) ** 2 + (sz * res) ** 2)
        fraction = max(0.001, min(0.1, float(longest_valid_m) / diag))
        space.setLongestValidSegmentFraction(fraction)

    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    si.setStateValidityCheckingResolution(float(params.collision_check_resolution))

    def is_valid(state) -> bool:
        return bool(voxelmap.is_free_world(float(state[0]), float(state[1]), float(state[2])))

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
    return space, ss, si


__all__ = [
    "dist3d",
    "interpolate_path_3d",
    "reduce_path_3d",
    "make_clearance_objective_3d",
    "get_voxelmap_dim",
    "get_voxelmap_resolution",
    "setup_ompl_space_3d",
]
