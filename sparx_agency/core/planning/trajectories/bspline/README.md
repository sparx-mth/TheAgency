# bspline — FALCON's trajectory, evaluated on our side of the link

A faithful port of FALCON's `fast_planner::NonUniformBspline`, plus the
position-and-yaw bundle its `traj_server` builds from a `trajectory/Bspline`
message, plus a nearest-point projection it has no equivalent of.

Pure numpy. No ROS, no scipy, Python 3.8-clean — it is imported by the Noetic
container as well as by Isaac Sim's 3.12.

## Why hold the curve instead of its samples

FALCON already publishes the trajectory twice: once as the spline on
`/planning/bspline` when it replans, and once as 100 Hz samples of it on
`/planning/pos_cmd`. Taking the samples is less code, and it costs three things
a tracking controller needs:

| | from 100 Hz samples | from the curve |
|---|---|---|
| **jerk** | not in the message at all | third derivative, exact |
| **nearest point on the path** | impossible — you only have "now" | a search over the parameter |
| **the clock** | `traj_server`'s wall-clock timer | yours |

The third matters more than it looks. `traj_server` runs `ros::Time::now()` on a
100 Hz timer and its exploration node aborts outright under `use_sim_time`, so a
simulator running at anything other than 1× slides the reference along the curve
at a speed the airframe is not flying. Holding the curve makes that impossible
by construction.

## The three pieces

**`non_uniform_bspline.py`** — one curve. de Boor evaluation, exact derivative,
uniform or explicit knots. The indexing mirrors the C++ line for line on
purpose: this evaluates a curve the other side of a socket also evaluates, and a
disagreement between them is a tracking error nobody can attribute.

**`trajectory.py`** — the six curves `traj_server` builds: position and its
three derivatives, yaw and its one. Reproduces the endpoint behaviour too —
past the end the position holds and every derivative goes to zero, so an
overrun brakes to a hover on the last planned point instead of extrapolating a
polynomial into space FALCON never checked against the map.

**`projection.py`** — "which point of the curve am I nearest to", windowed
around the previous answer.

## Two asymmetries that look like bugs and are not

**The position curve carries explicit knots; the yaw curve carries an
interval.** FALCON's optimiser reparameterises the position curve to respect the
velocity limit, so its knots are genuinely unevenly spaced and must be
transmitted. Nothing reparameterises yaw. `from_falcon` reflects this.

**The yaw curve is always degree 3, regardless of the message's `order`
field.** That field describes the position curve. The C++ hardcodes 3 for yaw;
so does this.

## Projection: what it is actually for

The usual argument is that projection stops a lagging aircraft cutting corners.
**Measured on this stack, that argument does not hold.** Flown against
FALCON-shaped trajectories with a realistic inner-loop lag, tracking the nearest
point is not better than tracking the point at time *t* through a bend — it is
marginally worse. FALCON's optimiser has already made the curve dynamically
feasible, so its curvature is bounded by roughly the limits the airframe has,
and there is no runaway reference to be pulled away by.

Where it wins clearly is a displacement in **time** rather than in space.
FALCON condemns its own live trajectory whenever it finds an obstacle on it and
the aircraft holds until a replacement arrives — but the plan's clock keeps
running through the hold. On resuming, a time-indexed reference sits seconds
further down the route, around the corner and through whatever lies between, and
the aircraft is pulled straight at it. Projection resumes from where the
aircraft is and flies the part of the route it had not flown yet.

| L-shaped route, 2 s hold mid-flight | worst departure from the curve |
|---|---|
| projected | **0.32 m** |
| time-indexed | 0.60 m |

It also makes the diagnostics exact: off-path distance is measured to the curve
rather than inferred from decomposing an error vector, and schedule lag comes
out as a number in its own right.

The search is **local, not global**, because an exploration route crosses
itself. The globally nearest point on a curve that loops back through the same
room can be a leg flown a minute ago, and snapping to it flies the aircraft
backwards. `search_ahead_s` must exceed the furthest one control tick can move
the aircraft, or the projection falls progressively behind and the window drags
it back for the rest of the flight.

`lookahead_s` defaults to **zero and should stay there**. Leaning the reference
forward looks like a cheap way to keep some along-track pull; what it actually
does is settle at a constant position error of `lookahead_s × speed`, which the
position gain turns into a standing forward push. Measured, 0.15 s of it flew
14% over the planned speed and overshot the end of the trajectory. Schedule is
recovered by the tracker's along-track catch-up term instead, which acts only
along the tangent and therefore cannot cut a corner.

## Using it

```python
trajectory = BsplineTrajectory.from_falcon(order, knots, pos_pts, yaw_pts,
                                           yaw_dt, start_time_s, traj_id)
projector = TrajectoryProjector()
projector.reset()                                    # on every new trajectory
reference = trajectory.sample(projector.project(trajectory, position))
```

`reference` is a `TrajectoryPoint` carrying position, velocity, acceleration,
jerk, yaw and yaw rate — everything
`core/control/trajectory_tracking` needs and nothing it does not.

## Tests

```bash
pytest sparx_agency/core/planning/trajectories
```

The evaluator is checked against closed-form cubic B-spline basis functions
written independently of it, and every derivative against a central difference
of the curve above it — at knot-interval *midpoints*, because the jerk of a
cubic is piecewise constant and genuinely steps at each knot.
