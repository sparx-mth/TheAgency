# multi_axis_follower

A continuous waypoint tracker for a platform that can drive **forward, sideways
and yaw at the same time**, holding altitude. It is the multi-axis sibling of
[`waypoint_follower`](../waypoint_follower) (which is strictly one-axis at a time:
pure X advance *or* pure yaw). Pick between them with the `~controller` rosparam
(`waypoint` | `multi_axis`) in the FALCON adapter — the legacy controller is
unchanged, so falling back is a one-line switch.

## Why a second controller

Localization and depth on this platform are **noisiest while yawing and while
standing still**, and **cleanest while flying forward**. The one-axis follower had
to stop and rotate in place to change heading, which is exactly the noisy regime.
Now that the platform accepts combined-axis commands, this tracker reaches a
waypoint with the **least yaw possible**:

- **Reach by translating.** The error to the waypoint is taken into the body
  frame and flown as forward + lateral velocity. Small offsets are absorbed by
  **crabbing (ROLL), with no rotation at all** — keeping the noisy yaw axis idle.
- **Yaw only past a deadband, with hysteresis.** Yaw engages above
  `yaw_engage_rad` and releases below `yaw_release_rad`, so the drone turns for
  large errors but not small ones, and never chatters on noise. When it does yaw,
  it **keeps translating** — never a stop-and-spin.
- **Never fly blind.** A `travel_cone_rad` clamp bounds how far off straight-ahead
  the translation may point; a steeper target is approached at the cone edge while
  yaw rotates it forward, so the forward camera always roughly sees where the
  drone is going.
- **Minimum force per axis.** Each axis command is either zero or at least the
  platform's minimum effective command (`min_vx` / `min_vy` / `min_wz`); a
  sub-threshold command the motors would ignore is never emitted. The snap is
  applied *after* slew so even a fast control rate never dribbles below the floor.
- **Station-keep, don't chase noise.** Once the final goal is captured the drone
  holds with a generous deadband (`hold_deadband`) and gentle, decisive nudges,
  since fighting the standstill noise only adds jitter.

Altitude is never commanded (`vz` is always 0); the platform holds height.

## Layout (single responsibility per file)

| file | responsibility |
|------|----------------|
| `params.py` | `MultiAxisFollowerParams` — all tuning, with validation. |
| `types.py` | `MultiAxisState` (IDLE/RUN/HOLD) and `MultiAxisCommand`. |
| `allocation.py` | Pure, stateless allocation math (body error, speed profile, travel-cone clamp, yaw hysteresis, the minimum-force deadband-with-snap). |
| `follower.py` | `MultiAxisFollower` — the stateful state machine and slew memory. |
| `predictor.py` | Forward-rollout of the follower against a holonomic first-order-lag plant (for the BEV viewer / planner). |

## Usage

```python
from sparx_agency.core.planning.trackers.multi_axis_follower import (
    MultiAxisFollower, MultiAxisFollowerParams,
)

f = MultiAxisFollower(MultiAxisFollowerParams())
f.set_path(waypoints, current_pose)            # list[Pose2D], drops passed waypoints
cmd = f.step(pose, dt)                          # -> MultiAxisCommand
twist.linear.x, twist.linear.y, twist.angular.z = cmd.vx, cmd.vy, cmd.wz
```

The public API (`set_path` / `step` / `reset` / `state` / `done` /
`required_axis`) matches the one-axis follower so an adapter can drive either.
`required_axis()` is always `None` (no per-axis handshake) and the command's
`freeze` is always `None` (sensors stay live; the tracker never stops to
re-measure).

> **Pose-estimator note.** When fed through `WindowedPoseEstimator`, pass the
> commanded `vy` (`set_command(vx, wz, vy=...)`) so the crab is propagated rather
> than dropped as drift. The FALCON adapter does this automatically for
> `controller:=multi_axis`.

## State machine

```
IDLE --(set_path)--> RUN --(final goal captured)--> HOLD
                      ^                               |
                      +-------(drifted far away)------+
```

## Tests

```
.venv/bin/python -m pytest \
  sparx_agency/core/planning/trackers/multi_axis_follower/tests/ -q
```

Covers the allocation math (deadband-with-snap, travel cone, yaw hysteresis,
approach-speed ramp) and closed-loop behaviour against a holonomic plant:
crab-without-yaw for small offsets, yaw for large ones, goal-behind turn-around,
fixed altitude, the minimum-force invariant (incl. decel), the station-keeping
deadband, and the predictor (incl. collision flag).
