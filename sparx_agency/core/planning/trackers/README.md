# Trackers

Trajectory tracking: convert `Trajectory` + `State3D` → `ControlCommand`.

## Pipeline

```
Planner → Path → Smoother → Trajectory → Tracker → ControlCommand
```

## Pure Pursuit

Classic geometric path tracking algorithm (Coulter 1992, CMU-RI-TR-92-01).

**Core formula:**
```
κ = 2·sin(α) / L_d
```
where:
- `α` = angle to lookahead point in robot frame
- `L_d` = lookahead distance
- `κ` = curvature of arc to follow

**For differential drive:** `ω = v · κ`

**For Ackermann steering:** `δ = arctan(L · κ)`

### Features

- **Classic geometry:** Arc-based steering, not point-to-point
- **Holonomic mode:** For omnidirectional robots (lateral velocity allowed)
- **Non-holonomic mode:** For differential drive (forward only + yaw rate)
- **Ackermann mode:** Computes steering angle given wheelbase
- **Adaptive lookahead:** `L_d = base + speed × gain`
- **Speed profiling:** Slows near goal and on curves

### Usage

```python
from sparx_agency.core.planning.trackers import PurePursuitTracker, PurePursuitParams
from sparx_agency.core.planning.interfaces import TrackerRequest

# Omnidirectional robot
tracker = PurePursuitTracker(PurePursuitParams(holonomic=True))

# Differential drive
tracker = PurePursuitTracker(PurePursuitParams(holonomic=False))

# Ackermann (car-like)
tracker = PurePursuitTracker(PurePursuitParams(holonomic=False, wheelbase=0.3))

# Control loop
result = tracker.step(TrackerRequest(state=current_state, trajectory=trajectory, t=0))

if result.done:
    stop()
elif result.failed:
    replan()
else:
    send(result.command)  # (vx, vy, vz, yaw_rate) in body frame
```

### Metadata Output

```python
result.metadata = {
    "alpha": 0.17,           # angle to lookahead (rad)
    "curvature": 0.51,       # arc curvature (1/m)
    "steering_angle": 0.15,  # Ackermann angle (rad), if wheelbase set
    "lookahead_dist": 0.66,  # computed L_d (m)
    ...
}
```

## Registry

Runtime tracker selection:

```python
from sparx_agency.core.planning.trackers import default_tracker_registry

tracker = default_tracker_registry().create("pure_pursuit")
```

## References

- Coulter, R.C. (1992). "Implementation of the Pure Pursuit Path Tracking Algorithm." CMU-RI-TR-92-01.