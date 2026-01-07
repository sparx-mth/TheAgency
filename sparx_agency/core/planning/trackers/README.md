# Trackers

This package contains **ROS-free** tracking algorithms that convert a **planned/smoothed trajectory**
into a **control command** (typically velocity) given the robot’s current state.

Goals:
- Keep tracking logic **pure core** (no ROS, no threads, no node lifecycle).
- Provide reusable trackers that can run in simulation, ROS, or any other runtime.
- Clean boundaries: trackers compute commands; integration layers apply/publish them.

---

## What is a Tracker?

A **Tracker** consumes:
- `State2D` / `State3D` (pose + twist): `core.common.types.motion`
- A `Trajectory` (time-parameterized): `core.common.types.planning.Trajectory`

And outputs:
- `ControlCommand` (ROS-free): `core.common.types.control.ControlCommand`

A tracker does **not**:
- manage control loops or timing threads
- publish commands to ROS topics
- arm/disarm, handle safety modes, or perform low-level motor control

Those belong in the **integration layer** (e.g., ROS node, simulator bridge, hardware driver).

---

## Available Trackers

### Pure Pursuit (`pure_pursuit`)
A classic geometric tracker that continuously steers toward a **lookahead point** on the trajectory.
This avoids stopping at intermediate waypoints and naturally produces smooth turns.

Key features in this implementation:
- Adaptive lookahead: base + speed-proportional, reduced on tight curves
- Speed profiling:
  - slow down near the goal
  - slow down on high curvature
  - optional obstacle clearance factor
- Smooth yaw-rate correction with deadband + low-pass filtering
- Optional vertical (altitude) P control for `State3D`

---

## Inputs / Outputs

### Inputs
- `State2D` / `State3D`
  - `state.pose.x, state.pose.y, state.pose.yaw`
  - for 3D: `state.pose.z`
- `Trajectory` (Protocol)
  - `trajectory.sample_by_time(dt)` returns `List[TrajectoryPoint]`

### Output
- `ControlCommand.velocity(vx, vy, vz, yaw_rate, **meta)`

**Important:** Pure Pursuit in this package outputs **BODY-frame** planar velocities:
- `x` = forward
- `y` = left (depending on your convention)
- `yaw_rate` = around +Z axis

Your integration layer must interpret that consistently when applying the command.

