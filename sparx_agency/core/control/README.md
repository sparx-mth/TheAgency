# control — from a planned trajectory to an actuator command

The layer below planning and above the autopilot. Pure numpy, ROS-free,
Python 3.8-clean.

## Why this is not `core/planning/trackers/`

The boundary is what each one's output *means*.

`core/planning/trackers/` produces **navigation** commands — a twist, a
velocity, a heading — for a vehicle whose autopilot will work out how to achieve
them. Those are the FALCON `controller:=` modes, and `reference_tracker_3d` is
the 3D one.

`core/control/` produces **airframe** commands — an acceleration, an attitude, a
thrust — by reasoning about the vehicle as a body with mass. It replaces part of
the autopilot rather than talking to it.

## The chain

```
trajectory (1-10 Hz)  ─┐
state (250 Hz)  ───────┴─►  trajectory_tracking  ──►  acceleration + heading
                                                            │
                                                        flatness
                                                            ▼
                                              attitude + specific thrust (m/s²)
                                                            │
                                                       thrust_model
                                                            ▼
                                              attitude + throttle (0..1)  ──► PX4
```

Everything to the left of PX4 runs at the state-estimate rate. PX4 keeps the
attitude loop (~250 Hz), the rate loop (~1 kHz on the gyro) and the mixer — the
three things that need a real-time clock and that nothing here should try to
own.

### `trajectory_tracking`

Feedforward from the plan plus feedback from the error, emitting a **world
acceleration and an absolute heading**.

Acceleration rather than velocity because a velocity setpoint keeps PX4's own
velocity loop in the chain, and that loop runs at tens of Hz off the same
position estimate this one uses — a stage of lag with no new information, and
the metre of tracking error the campaign was measuring.

Four terms: the plan's acceleration (a *lookup*, exact, no lag), position
feedback on a clamped error, velocity feedback against the measured velocity,
and an integral that only learns near the curve. Plus one that is easy to miss:
an **along-track catch-up**, which puts back the schedule that projection
deliberately throws away, acting only along the tangent so it cannot cut a
corner.

### `flatness`

A change of variables, not a controller — no gains, no state. A multirotor's
thrust all points one way, so "accelerate that way" and "point this way" are one
statement:

```
thrust axis = desired acceleration + gravity   → the tilt
thrust size = |that|                           → the throttle
heading     = free rotation about that axis    → the plan's yaw
```

Differentiate that once and the plan's **jerk** gives the rate the aircraft must
already be rotating at — the attitude feedforward, and the only reason the
B-spline is carried rather than its 100 Hz samples.

The heading convention is deliberate: the body x axis, projected onto the
horizontal, points exactly along the commanded heading. FALCON picks yaw to aim
the depth camera, and the camera looks along body x.

### `thrust_model`

One scalar — the specific thrust available at full throttle — measured in flight
rather than assumed, because it moves with battery voltage. A 10% error here is
a persistent 1 m/s² bias on the vertical axis that the position integrator
absorbs and hides, leaving every gain above tuned against a lie.

The measurement is specific force along the **thrust axis**, not vertical
acceleration: a tilted aircraft holding altitude is working harder than a level
one, and using the vertical component would make the estimate a function of how
hard the aircraft is cornering.

## Two findings worth keeping

**Nearest-point tracking is not a corner-cutting fix.** It is widely sold as
one. Measured against FALCON's trajectories with a realistic inner-loop lag it
is marginally *worse* through a bend, because the optimiser has already bounded
the curvature to something the airframe can fly. What it does fix is recovery
from a displacement in *time* — the aircraft holding while FALCON replans, then
resuming to find a time-indexed reference several seconds down the route. On an
L-shaped route with a 2 s hold: 0.32 m worst departure projected, 0.60 m
time-indexed.

**A reference lookahead is not free along-track pull.** It settles at a constant
position error of `lookahead × speed`, which the position gain turns into a
standing forward push that the damping term balances only by flying fast.
Measured: 0.15 s of lookahead flew 14% over the planned speed and overshot the
end of the trajectory. Hence the tangent-only catch-up term instead, and
`lookahead_s = 0`.

## Tests

```bash
pytest sparx_agency/core/control
```

The flatness rate feedforward is checked against a *numerical* differentiation
of the attitude it produces; the outer loop is flown closed-loop against a
first-order-lag airframe rather than inspected a tick at a time.
