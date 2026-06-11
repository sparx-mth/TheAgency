# A* Window Local Planner (2D/3D)

This module provides **local replanning** using a **small planning window** around the robot
(for indoor environments). It reuses the existing global A* implementations:

- `core/planning/planners/astar/algorithm_2d.py::astar_grid_2d`
- `core/planning/planners/astar/algorithm_3d.py::astar_voxel_3d`

## Key idea

Instead of changing A*, we wrap the world map with a **Window View** object that:
- exposes the same minimal API required by the A* functions
- translates local window indices to global indices

## Outputs

Planners return `LocalPlanOutput` with a short-horizon **Path2D/Path3D** in artifacts.
(Converting to a time-parameterized Trajectory can be done later in the smart integration stage.)

## Indoor defaults

Small windows by default:
- 2D: ~6m x 6m
- 3D: ~6m x 6m x 3m
