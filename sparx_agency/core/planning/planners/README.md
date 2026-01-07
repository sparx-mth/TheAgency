# RRT* Planner

A **sampling-based path planner** that computes a **geometric path** (`Path2D`)
from a start pose to a goal pose in a 2D grid environment.

This module is **ROS-free** and produces **no velocities or timing**.
It is intended to be used as part of the core planning pipeline:

Planner → Path2D → Smoother → Trajectory → Tracker → ControlCommand


## What it does

Input:
- `start: Pose2D`
- `goal: Pose2D`
- `world`: 2D grid / costmap-like environment

Output:
- `PlanResult` containing a `Path2D` (sequence of `Pose2D`)

Failure cases:
- start or goal in obstacle
- no path found within planning budget


## Algorithm (high level)

1. **Sampling-based planning (RRT\*)**
   - Explores the free space using random sampling
   - Incrementally improves path quality via rewiring

2. **Clearance-aware preference (optional)**
   - Biases the solution away from obstacles when clearance information is available

3. **Adaptive waypoint reduction**
   - Removes redundant points in open space
   - Preserves points in narrow passages or when shortcuts are invalid

4. **Uniform interpolation**
   - Inserts points at roughly fixed world-space spacing
   - Improves stability for downstream smoothing and tracking


## Environment contract

The planner expects the `world` object to expose:

Required:
- `width`, `height` (cells)
- `resolution` (meters per cell)
- `origin_x`, `origin_y`
- `is_free(ix, iy) -> bool`

Optional:
- `clearance_at_world(x, y) -> float`
- or `clearance_at_cell(ix, iy) -> float`


## Output

- `Path2D`
  - geometric only (no dynamics)
  - suitable for smoothing and time-parameterization
- Debug information may be attached via `PlanResult.artifacts`
