# drift_pid — the drift-cancelling path follower

A continuous multi-axis tracker for a heavy indoor drone flown on an AprilTag
pose. It exists because the platform has a specific, repeatable problem: it
**drifts**, and a proportional controller cannot cancel a drift — P only pushes
back in proportion to how far the drift has already carried you, so it settles at
a standing offset and stays there.

Measured on the closed-loop test in `tests/`, with a 0.035 m/s sideways push:

| | settled cross-track error (worst) |
|---|---|
| P only (`ki = 0`) | 0.085 m |
| P + drift integral | 0.040 m |
| P + I, pose delivered a full frame (100 ms) late | 0.039 m |

The last row is the latency lead doing its job: the tracking quality survives the
vision pipeline's real transport delay unchanged.

The residual lands on the cross-track **deadband** (0.03 m on a crisp pose), and
that is the single most important thing to know when tuning: *the deadband is the
floor on how well the drone can hold its line.* Shrink it and the drone tracks
tighter and chases more localization noise; grow it and the drone is calmer and
sloppier. The deadband is also **not constant**: it widens automatically with the
provider's own `pos_std` (see below), so the configured value is the floor for a
*good* pose, not a promise for a bad one.

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

## Turn coordination: the progress vector decides whether the yaw works

Measured on the deployed airframe (nav_debug run of 2026-07-21), the yaw the
platform *delivers* depends on the direction of the translation riding under it:

| translation under an active yaw | delivered rotation |
|---|---|
| forward (+0.08 m/s) | correct direction, immediate bite |
| none (in place) | ~11% of commanded, coasts |
| backward (−0.10 m/s) | degrades, then **reverses** — the drone turned *left* against a saturated right-yaw command |

So while the yaw axis is genuinely active (above the envelope's minimum-force
release point), every regime shapes its translation into a forward cone
(`turn_coordination` in the shared allocation module):

* **`vx` is floored** — at `turn_pitch_bias` in a TURN (the forward bite the
  rotation needs), at 0 everywhere else (never backward while yawing). A
  station-keeping correction that wants reverse is *deferred, not lost*: the
  position error stays open, and the loops close it once the yaw quietens —
  rotate first, translate after.
* **`|vy|` is capped at `tan(turn_side_cone_rad)·vx`** — YAW+ROLL is not the
  same manoeuvre as YAW+forward, so roll may season the progress vector but
  never dominate it. At cruise speed the cap sits above the lateral PID's own
  limit, so it only bites when the drone is slow and yawing hard — exactly
  where the coupling lives.

## The climb yield lives inside the control law

When the adapter's altitude hold runs a climb pulse it passes
`translation_scale` into `step()` rather than scaling the published twist: the
yield is folded in **before** the envelope, so slew memory, minimum-force
shaping, effort, the blockage monitor and the certainty log all see the command
that actually flies. (The old after-the-fact scaling had the controller
believing it commanded cruise while the drone got a fifth of it — which reads
as "the converter dropped my forward command" in the logs and poisons the
effectiveness estimate.) Yaw is never scaled: rotating costs no lift.

## The quality loop: the pose's own honesty signals, used every tick

Localization publishes more than a pose, and each extra signal drives exactly one
mechanism:

| signal | mechanism |
|---|---|
| `pos_std` (m) | **Adaptive deadbands.** Tracking deadbands widen by `std_deadband_gain` per metre of reported error above the crisp reference (`std_ref_m`), capped. An error smaller than the pose's own stated accuracy is noise — it is neither corrected nor learned from. The heading loop gets the same treatment via the provider's known yaw-std law (`0.02 + 0.20·(1−conf)²`), which is not published but is reconstructable from confidence. |
| `cmd_effectiveness` | **Earned speed** — a fresh flight starts near `eff_speed_floor × cruise` (the EMA starts unproven at 0.3) and speeds up over the first metres as the world confirms the commands work. Also **earns the latency lead** (below): commands that demonstrably do not move the drone must not move its pose estimate either. |
| `confidence` | Speed/gain schedule, integrator freeze, hold — as before. |
| `source == coast` | Lead forced to zero (a coasted pose is *already* command-propagated — leading it counts the same commands twice), integrators frozen, blockage evidence discarded. |
| arrival timing | **Fresh-measurement derivative.** The ~10 Hz pose is event-driven while the control loop is a timer, so some ticks re-see a held pose. Differentiating a held value gives 0 then a doubled spike; the D term instead banks stale time and differentiates over the true interval, detected from the quality age resetting. |

**Latency lead** (`latency_s`): the controller steers a pose advanced along the
last *commanded* velocity by the camera→control transport delay, so P/D react to
where the drone is, not where it was a frame ago. Three guards make it safe: it
is scaled by proven effectiveness, zeroed while coasting, and **never fed to the
blockage detector** — a stuck drone whose pose was advanced by its own commands
would look obedient, which is precisely the failure the detector exists to catch.

## The envelope's two asymmetries

- **Braking beats accelerating** (`decel_* ≥ accel_*`, validated). The ramp-up is
  deliberately gentle to keep the 2.5 Hz depth model fed with usable frames; the
  ramp-down must not inherit that gentleness, because taking thrust off is what
  obstacle response is made of.
- **Reverse is capped harder than forward** (`max_vx_back < max_vx`, validated).
  There is no rear camera; every backward metre is blind. Reverse exists to break
  contact, never to travel.

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
4. `latency_s` if corrections consistently overshoot then swing back — that
   signature usually means the assumed transport delay is wrong, not the gains.
5. `deadband` last — it is the tracking floor on a crisp pose, and shrinking it
   is what makes the drone chase noise.

Everything is exposed in `mission.yaml` under the `CONTROLLER 5` section. Keep
`use_pose_estimator: false` with this controller: it does its own latency lead,
and stacking the estimator's smoothing on top only adds lag.
