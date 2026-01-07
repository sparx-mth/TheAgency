# Trajectory Smoothers

A **smoother** converts a geometric path (`Path2D`) into a time-parameterized trajectory (`Trajectory`).

Pipeline:
Planner -> Path2D -> Smoother -> Trajectory -> Tracker -> ControlCommand

## Contracts

### Input
- `Path2D`: sequence of 2D poses (`Pose2D`) with no velocity/dynamics.

### Output
- `Trajectory`: time-parameterized interface:
  - `total_time`
  - `start`, `end`
  - `sample(t)`
  - `sample_by_time(dt)`

### Discrete adapter
All smoothers in this package output discrete `TrajectoryPoint` samples and wrap them using:
- `DiscreteTrajectory` (`core/planning/smoothers/adapter.py`)

This keeps sampling logic centralized and avoids duplication across algorithms.

## Included smoothers

### 1) Minimum-snap (`minsnap`)
- Purpose: high-order smooth trajectories that penalize **snap** (4th derivative).
- Typical use: quadrotor motion planning, smooth accelerations, reduced control effort.
- Implementation:
  - filters near-duplicate points
  - allocates segment times (distance-based heuristic + safety margins)
  - calls `minsnap_trajectories` to build a polynomial trajectory
  - samples into `TrajectoryPoint` at `dt`

Notes:
- Time allocation is heuristic (not a hard constraint enforcement).
- Input is 2D; z is currently set to 0.

### 2) Bezier/Hermite (heading-aware) (`bezier`)
- Purpose: fast, simple smoothing with **G1 heading continuity**.
- Implementation:
  - builds cubic Hermite splines (x(u), y(u)) from waypoints + tangents
  - builds arc-length LUT
  - samples by time assuming a nominal constant speed
  - fills `yaw` from velocity direction, and sets `s` and `curvature`

Notes:
- Great for quick smoothing and stable tracking inputs.
- Not an optimization-based smoother; no constraint enforcement.

## Registry usage

The pipeline should not hardcode the smoother class. Use the registry:

```python
from core.planning.smoothers import SmootherRegistry
from core.planning.smoothers.register_defaults import register_default_smoothers

register_default_smoothers()

smoother = SmootherRegistry.create("minsnap")  # or "bezier"
traj = smoother.smooth(request)
