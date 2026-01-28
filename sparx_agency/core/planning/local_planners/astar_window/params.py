"""
Parameters for windowed local A* replanning (indoor defaults).

These params are intentionally separate from the global A* params:
- global planning can expand large areas
- local replanning must be fast and constrained (small window + small expansion budget)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalAStarWindow2DParams:
    """
    Local A* params for 2D occupancy grids.

    Attributes:
        window_size_m: Side length of the local square window in meters.
        connectivity: 4 or 8.
        max_expansions: Hard cap for A* node expansions.
        allow_unknown: Whether to traverse unknown cells (usually False for safety).
        goal_lookahead_m: How far along the reference to place the local goal.
        min_goal_separation_m: Minimum distance between current position and local goal.
    """
    window_size_m: float = 6.0          # indoor: small window
    connectivity: int = 8
    max_expansions: int = 40_000
    allow_unknown: bool = False

    goal_lookahead_m: float = 3.0
    min_goal_separation_m: float = 0.8


@dataclass(frozen=True, slots=True)
class LocalAStarWindow3DParams:
    """
    Local A* params for 3D voxel maps.

    Attributes:
        window_size_xy_m: XY side length of the local window in meters.
        window_size_z_m: Z extent of the local window in meters.
        connectivity: 6, 18, or 26.
        max_expansions: Hard cap for A* node expansions.
        allow_unknown: Keep False unless your voxel map explicitly models unknown safely.
        goal_lookahead_m: How far along the reference to place the local goal (in XY).
        min_goal_separation_m: Minimum distance between current position and local goal.
    """
    window_size_xy_m: float = 6.0       # indoor
    window_size_z_m: float = 3.0        # indoor (e.g., one room height band)
    connectivity: int = 26
    max_expansions: int = 120_000
    allow_unknown: bool = False

    goal_lookahead_m: float = 3.0
    min_goal_separation_m: float = 0.8
