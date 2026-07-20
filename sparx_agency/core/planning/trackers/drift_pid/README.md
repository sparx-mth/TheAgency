# drift_pid — the drift-cancelling path follower

A continuous multi-axis tracker for a heavy indoor drone flown on an AprilTag
pose. It exists because the platform has a specific, repeatable problem: it
**drifts**, and a proportional controller cannot cancel a drift — P only pushes
back in proportion to how far the drift has already carried you, so it settles at
a standing offset and stays there.

Measured on the closed-loop test in `tests/`, with a 0.035 m/s sideways push:

| | settled cross-track error |
|---|---|
| P only (`ki = 0`) | 0.096 m |
| P + drift integral | 0.044 m |

The residual lands on the cross-track **deadband** (0.04 m), and that is the
single most important thing to know when tuning: *the deadband is the floor on
how well the drone can hold its line.* Shrink it and the drone tracks tighter and
chases more localization noise; grow it and the drone is calmer and sloppier.

## The three drifts, and where each is cancelled

The drifts this controller was built for, in the order they actually bite:

1. **Sideways (left/right)** — the big one, present in forward flight and in
   turns alike. Cancelled by the cross-track loop on the lateral (ROLL) axis.
2. **Fore/aft** — shows up while turning or standing still. Cancelled by the
   along-track loop, but only in the regimes where it exists (see below).
3. **Yaw** — the rarest. Cancelled by the heading loop.

They usually arrive together, which is why all three loops run every tick.

## Two references, switched by regime

| Regime | Reference | Forward axis | Lateral axis | Yaw axis |
|---|---|---|---|---|
| `TRACK` | the **line** | cruise feed-forward | cross-track PID | heading PID |
| `TURN` | a latched **anchor on the line** | along-track PID (throttled) | cross-track PID × `lateral_turn_frac` | heading PID |
| `HOLD` | anchor / the goal | along-track PID | cross-track PID | heading hold |
| `ESCAPE` | — | the reflex owns every axis | | |

The regime switch is the design's core idea. While flying a leg, the reference is
the *line*, and the perpendicular foot moves with the drone — so along-track error
is structurally always zero and the forward PID has nothing to do (hence
`forward_track_frac = 0` by default; the cruise owns that axis). The moment the
drone stops flying forward — a turn, a map settle, the goal — the reference
switches to a **fixed anchor**, and *then* both translation PIDs do real
station-keeping. That is why fore/aft drift is cancelled exactly in the regimes
where the user observes it.

The anchor is latched onto the **trajectory**, not under the drone, so holding
still also finishes closing whatever cross-track error was open when the hold
began.

## What is control here, and what is planning

Deliberately split, because they were tangled before:

* **Control (this package).** Cancel drift. Respect the force envelope. Slow down
  when the pose is poor. Notice that a command is not reaching the world, and run
  a short reflex to break contact.
* **Planning (the A\* node).** Decide the route. Remember where the invisible
  obstacle was. Route around it.

The seam is one signal: when the reflexes are spent, the controller raises
`report_blocked` **once** and goes on holding the line it was given. It never
edits the route. The adapter forwards that to the planner, which injects a
virtual obstacle and replans.

## Modules

| File | Owns |
|---|---|
| `pid.py` | One-axis PID. The I term **is** the drift estimate (`AxisPid.drift`). |
| `envelope.py` | Per-axis caps, the combined multi-axis budget, slew, minimum force. |
| `confidence.py` | Localization quality → speed scale, gain scale, learn-or-freeze, hold. |
| `blockage.py` | "Commanded but not moving", per axis, debounced. |
| `escape.py` | The scripted reflexes. |
| `geometry.py` | Line projection, the lookahead carrot, heading error. |
| `follower.py` | The state machine that composes all of it. |
| `params.py` | Navigation dials + composition of the above. |

## Three things that will bite you if you change them

1. **Never learn drift from a coasted pose.** When no tag is in view the
   localization provider propagates the pose *by the commanded motion*. Such a
   pose always appears to obey commands perfectly, so integrating against it
   teaches nothing and a stuck detector fed by it can never fire. Both
   `confidence.py` and `blockage.py` gate on the `coasting` flag, and that flag
   comes from the provider's source string — **not** from the confidence number,
   because a coasted pose (≤ 0.25) and a genuine one-tag fix (≈ 0.21) are
   numerically indistinguishable.

2. **Confidence thresholds are not intuitive.** Pose confidence is a product of
   five degradations; a single visible tag is hard-capped near 0.21. A threshold
   like "0.5 = good" grounds the drone. `conf_full` defaults to 0.35.

3. **The order in `ForceEnvelope.apply` is load-bearing**: saturate → combined
   budget → slew (remembering the *unshaped* value) → minimum-force shape. Shaping
   last means the published command is never a dribble the motors ignore; keeping
   the pre-shape value in the slew memory means the snap-to-floor does not become
   the starting point of the next ramp. This mirrors
   `multi_axis_follower.follower._finalize` and the reasoning is the same.

## Tuning order

Start slow and raise, one dial at a time:

1. `cruise_speed` and `max_wz` until the drone flies the route at a comfortable pace.
2. `combined_effort` if it feels fast specifically when turning *while* flying.
3. `lateral_pid.kp` until it returns to the line without overshooting, then
   `lateral_pid.ki` until the standing offset disappears.
4. `deadband` last — it is the tracking floor, and shrinking it is what makes the
   drone chase noise.

Everything is exposed in `mission.yaml` under the `CONTROLLER 5` section.
