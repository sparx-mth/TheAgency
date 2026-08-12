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

There are **two backends**, because there are two kinds of aircraft. What
differs is how much of the autopilot we replace; what does not differ is the
plan, which is why `reference/` sits above both.

```
trajectory (1-10 Hz)  ─┐
state          ────────┴─►  reference  ──►  where on the plan we are,
                                            and how far off it is
                                                    │
                        ┌───────────────────────────┴───────────────────────┐
                        ▼                                                   ▼
              trajectory_tracking                                   velocity_servo
              acceleration + heading                                body twist
                        │                                                   │
                    flatness                                        (vx, vy, vz, yaw_rate)
                        ▼                                                   │
          attitude + specific thrust (m/s²)                                 │
                        │                                                   │
                   thrust_model                                             │
                        ▼                                                   ▼
          attitude + throttle (0..1) ──► PX4              ──► an autopilot that
                                                              owns its velocity loop
```

**Left branch — we own the attitude loop.** PX4 keeps the attitude loop
(~250 Hz), the rate loop (~1 kHz on the gyro) and the mixer: the three things
that need a real-time clock and that nothing here should try to own. Everything
above runs at the state-estimate rate.

**Right branch — somebody else owns it.** A Gazebo model plugin, a hobby flight
controller in a velocity mode, an indoor platform whose vendor exposes nothing
lower. There is no attitude input port at all, so `flatness` and `thrust_model`
have nowhere to land and the chain terminates one stage earlier.

Picking a branch is a statement about the *vehicle*, not a tuning preference.
Ask what the lowest interface the autopilot will accept is; that answers it.

### `reference`

What the plan says, held in one place. The live trajectory, the one queued
behind it, when the swap happens, where on the curve the aircraft is, and the
gap from the plan split into along-track and cross-track.

It exists because the two backends must disagree about what to *command* and
must never disagree about what the plan *says*. Before it, the schedule logic
lived inside the acceleration tracker and the second backend would have had to
copy it — and a copy drifts.

### `velocity_servo`

The right-hand backend, and the only place in this package that models the
autopilot underneath rather than replacing it.

The temptation with a velocity-commanded aircraft is to send the plan's velocity
and add a proportional pull toward the plan's position. That is what the
previous stack did and it cannot work well, for a measurable reason: **the
autopilot underneath is slow.** Measured on the reference Gazebo airframe by
step response — 0.18 s of transport delay and a 0.51 s time constant
horizontally, against a DC gain of 1.00. Commanding `v_plan` therefore produces
an aircraft whose velocity matches the plan about 0.7 s late, which at a 0.6 m/s
cruise is 0.4 m of standing position error before any disturbance. Raising the
position gain to chase it is exactly wrong, because that same delay is what caps
the gain.

So the plant is inverted instead. A first-order lag is inverted by a lead:

```
plant:  tau * dv/dt + v = v_command
so:     v_command = v_wanted + tau * (dv/dt)_wanted
```

and `(dv/dt)_wanted` is the plan's **acceleration**, which the B-spline carries
analytically and exactly. That one term costs nothing — no gain to tune, no
derivative to filter, no extra state — and it is the reason the trajectory is
carried as a curve rather than as a stream of sampled points.

Measured twice — once against the model, once against the aircraft the model
describes. Keep the two apart when quoting them.

**Simulated** (`velocity_servo/tests/airframe.py`, the first-order-plus-delay
model seeded with the numbers above, L-shaped route):

| | mean gap | max gap | max cross-track |
|---|---|---|---|
| inverse-plant lead | 0.029 m | 0.082 m | **0.073 m** |
| P + velocity feedforward | 0.162 m | 0.463 m | **0.336 m** |

**Real airframe** (the SJTU Gazebo `/simple_drone`, playground world, real-time
factor 1.00, 0.6 m/s plan, a 1.43 m circle, mean over the run with 5 s of
start-from-hover transient discarded):

| | mean gap | mean cross-track | max yaw error |
|---|---|---|---|
| inverse-plant lead | 0.091 m | **0.054 m** | 7.6° |
| P + velocity feedforward | 0.257 m | **0.202 m** | 9.5° |

The columns are not interchangeable — the simulated row reports *worst*
cross-track and the real one reports *mean*, so compare within a table, not
across. What does compare is mean gap: 0.029 m modelled against 0.091 m flown.

The two agree on the **ordering** — the lead term wins on both, and wins by more
on cross-track than on gap — and agree on magnitude to within about 3x. The real
airframe being the worse of the two is expected rather than disappointing: the
model *is* the first-order-plus-delay system being inverted, so it can only show
the residual the inversion leaves behind, while the aircraft adds everything the
model omits — a nested PID cascade instead of a single lag, discrete 30 Hz
feedback, and whatever cross-axis coupling the plugin has. Treat the simulated
numbers as a floor on what the law can do, not as a prediction of a flight.

An earlier real-airframe campaign reported the error *growing* with the length
of the discarded window and concluded the plant model was missing a body-frame
rotation. It was measuring a **capsized** aircraft that could not move on any
axis while still reporting healthy odometry; see
`tasks/planning/falcon_sjtu/README.md`. The re-measurement above shows the error
falling as the transient is discarded, which is the correct signature.

On a *straight, constant-speed* leg the two configurations are identical, and
that is the honest result rather than a disappointing one: the plan's
acceleration is zero there, so the lead term has nothing to do. It earns its
keep exactly where an exploration route spends its time and where the error that
hits walls is generated.

Two further things this backend has that the acceleration one does not need:

* **A yaw servo.** The acceleration backend passes heading through, rate
  limited, and never closes a loop on it — correct, because the attitude command
  contains the heading and PX4's attitude loop makes it true. An airframe taking
  a yaw *rate* has no such loop above it, so feeding the plan's rate forward is
  open-loop integration and the heading walks away over a flight. On an
  exploration aircraft that matters more than it sounds: FALCON picks yaw to aim
  the depth camera at the frontier it means to observe next, so a heading 20°
  adrift is a map built of the wrong wall.
* **No velocity damping term.** Deliberately. The airframe already contains a
  velocity loop with unit DC gain; closing a second one around it double-counts
  the same feedback and drops the phase margin. Damping comes from the plant,
  which is what the plant model is for.

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

**Measure the plant; do not assume it.** `velocity_servo/plant.py` wants three
numbers per axis and all three come off a twenty-second experiment: command a
velocity step, log the odometry, read the steady-state ratio (DC gain), how long
nothing happened (delay) and how long the rest took to reach 63% (time
constant). A wrong time constant makes the lead term push the wrong amount and
reads exactly like a mistuned position gain, which is the most expensive way to
be wrong here.

## A third finding worth keeping

**A velocity setpoint is not automatically a stage of lag.** The
`trajectory_tracking` docstring argues against velocity setpoints because they
keep the autopilot's velocity loop in the chain, adding lag without adding
information. That argument is right *when you have the choice* — and it is what
the left-hand branch acts on. It is not an argument that a velocity-commanded
aircraft cannot be flown well: the lag is a known, measurable, first-order
system, and a lead term built from the plan's own acceleration cancels its
dominant pole. What the argument really says is that you should not accept the
loop **and then ignore it**, which is the mistake the previous stack made.

## Tests

```bash
pytest sparx_agency/core/control
```

The flatness rate feedforward is checked against a *numerical* differentiation
of the attitude it produces; both outer loops are flown closed-loop against a
first-order-plus-delay airframe rather than inspected a tick at a time. That
airframe (`velocity_servo/tests/airframe.py`) is the same model `plant.py`
describes, seeded with the same measured numbers — so a closed-loop result only
means what it claims while the inverted plant and the flown plant agree, and
`test_plant.py` pins them to each other so nothing can quietly drift.

Two tests exist specifically to stop a fixed bug coming back, and both were
verified to fail when the bug is reintroduced rather than merely passing today:

* past the end of a curve, an aircraft trailing directly behind the endpoint
  reports its gap as along-track **lag**, not as cross-track. `sample()` zeroes
  every derivative past the end, so taking the direction of travel from the
  sampled velocity left no direction at all and inverted the one number the
  docs call "the one that flies into walls" — exactly when the aircraft is
  furthest behind. The direction now comes from the velocity *curve*, which is
  defined everywhere;
* with projection disabled, `reference_time_s` tracks the elapsed time instead
  of reporting a stale `0.0` for the whole flight.

Anti-windup is gated **per axis** in both backends. `limit_acceleration` gives
the horizontal axes away first precisely so altitude survives, so a hard corner
saturates horizontally on nearly every tick — and a single shared saturation
flag froze the *vertical* integrator throughout, costing it the standing thrust
bias it exists to hold.
