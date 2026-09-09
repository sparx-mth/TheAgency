# 010 - Flight anomalies from nav_debug_20260906_235809

**Branch:** `claude/drone-nav-trajectory-refactor-352f30`
**Status:** done (pending flight verification of the C++ half)
**Roadmap item:** FALCON/Sphera exploration tracking quality

## Goal

Six reported anomalies in run `nav_debug_20260906_235809` are fixed end to end in
the code that actually flew (the FALCON exploration follower and the 3D reference
tracker), each backed by a measurement from that run and each covered by a test.

## Why

The run's own numbers, measured over the 4598 **airborne** driving ticks (13% of
the flight is on the ground with the gate still reporting `driving`, so a
whole-flight denominator flatters every rate):

| | rate | what it is |
|---|---|---|
| motion commanded at a frozen reference | **18.1%** | the plan ended; the follower kept flying at its dead endpoint |
| backward body flight (`cmd.vx < -0.1`) | **9.5%** | blind flight — the only camera faces forward (hfov 135 deg) |
| reference behind the aircraft | 6.8% | the position loop brakes backward toward a passed point |
| `diverged` (> 2 m from the reference) | 6.5% | |

Backward flight is where the aircraft meets geometry. Distance to mapped
occupied cells, airborne ticks only:

| | p10 | p50 | within 0.3 m |
|---|---|---|---|
| commanding backward | **0.00 m** | 0.50 m | **33%** |
| everything else | 0.30 m | 0.73 m | 9% |

## Steps

- [x] Reuse, do not reinvent: `core/control/reference` already has `past_end`,
      `max_trajectory_age_s` and `decompose_error(offset, direction)`. Mirror that
      vocabulary in `reference_tracker_3d` rather than inventing a parallel one.
- [x] **Issue 1** — plan-state classifier: a reference whose commanded velocity has
      been zero for longer than a grace period is `past_end`. Fly to that endpoint
      (it is the last valid waypoint) and latch a hold there; hold where we are
      instead if the endpoint is implausibly far.
- [x] **Issue 4** — the same classifier covers a new trajectory that is degenerate
      from its first sample (9/145 in this run; 8/145 never move at all).
- [x] **Issue 6** — honest error split via an explicit direction of travel, plus an
      along-track retard clamp so an aircraft merely ahead of schedule is never
      braked backward onto a point it has passed.
- [x] **Issue 5** — turn-in-place gate in its own ROS-free module, gating on the
      angle between the commanded world velocity and the nose, **independent of
      `use_lateral`** (the existing align gate is nested inside `not use_lateral`
      and was therefore bypassed on this flight). Hysteresis, because a gate
      without it is documented to chatter. Plus a hard reverse clamp at the
      publish boundary so no code path can emit backward flight.
- [x] **Issue 3** — raise the follower's stale `max_yaw_rate_deg` default (45)
      to the 90 the launch already passes, and give the committed turn its own
      rate. Do **not** raise `course_slew_deg_s`.
- [x] **Issue 2** — make the existing `falcon_replan_from_pose` guard actually
      fire, and clear the tracker's latched hold on a trajectory change.
- [x] Tests for each, run by path.
- [x] CHANGELOG + LESSONS + README updates.

## Open questions

- Achieved yaw rate is unmeasurable from this run folder (the telemetry lane is a
  `/cmd_vel` echo, not measured velocity; ground truth lives in the ROS2 recorder
  half that was not collected). The committed-turn rate therefore ships at a
  conservative default and is flagged as needing a flight measurement.
- `course_slew_deg_s` stays at 45: LOOP_BUGS B31 and MISSION P17/P24 forbid raising
  it without a discriminator between a thrashing demand and a persistent one. The
  turn-in-place gate is arguably that discriminator, but proving it needs a flight.

## Notes

- 2026-09-09: Two of the nine cited frames do not show what the report describes.
  Frame 891 is not "a new trajectory with no reference" — the follower adopts every
  new `traj_id` within 0.092 s and never misses one; it is traj 15's 7.8 s frozen
  tail. Frame 1251's apparent "follower still on the old trajectory" is an as-of
  join artefact between two lanes recorded at different rates. The underlying
  defects at both frames are real, but they are issues 1 and 2, not issue 4 as
  described. Fixed the real defect at each.
- 2026-09-09: an adversarial review of the first implementation (5 lenses, every
  finding independently verified) confirmed 20 defects in it, 8 of which changed
  the code. The ones worth remembering:
  * The committed turn bypassed the course limiter but left `_course_cmd` behind,
    so at release `heading_err` was the whole gap the turn had just closed and the
    yaw loop commanded a **reversal** away from the target. A reviewer proved it by
    stubbing rospy and stepping the real node: a 180 deg demand released at
    +63 deg/s and inverted to -52 deg/s on the next tick, re-engaging the gate. The
    turn now drags the limiter with it.
  * `limit_lead` was applied to the full 3D error, so it capped the ALTITUDE
    correction against the horizontal direction of travel. Horizontal-only now.
  * Integral separation was defeated on the backward side: the band was tested
    against the lead-limited error, and 0.25 < 0.5 means an aircraft far ahead of
    its reference could charge the integrator. The band now judges the raw error.
  * The reverse clamp ran BEFORE the pulse shaper, whose brake pulse emits an
    opposite-signed tick — republishing exactly the backward command the clamp
    forbids. It now runs after.
  * All six new launch args were dropped at the `sphera_drone.launch` boundary.
  * My own first attempt at an endpoint dead-band re-latched the hold every tick,
    which is a drift ratchet — the pre-existing tracker tests caught it as a 7.4 m
    runaway. Replaced with a follower-side exemption bounded by `endpoint_settle_m`.
  Two of the tests written for the first implementation passed with the fix
  reverted; both were rewritten to close the loop and assert the latched state.

## Result

Shipped across two ROS-free core modules, one ROS-free node-side module, the follower,
the launch wiring, the campaign readback matrix and the replay tool.

| issue | where it landed |
|---|---|
| 1 safe stop | `reference_tracker_3d/plan_state.py` + `_hold_endpoint` in `tracker.py` |
| 2 replan origin | `patches/falcon_replan_from_pose.patch` (lead compensation) + `nav_stack.launch`; `tracker.on_new_trajectory()` |
| 3 yaw rate | follower `max_yaw_rate_deg` default 45 -> 90; `~turn_yaw_rate_deg` for the committed turn |
| 4 reference on new paths | the same classifier — a degenerate trajectory is `past_end`, not a live reference |
| 5 turn in place | `scripts/heading_gate.py`, unconditional, + the publish-boundary reverse clamp |
| 6 progression | `reference_tracker_3d/progress.py` (`split_error`, `limit_lead`) |

**Counterfactual replay of the recorded run through the new logic** (5567 airborne ticks;
measured velocity differentiated from pose, since the telemetry lane is a `/cmd_vel` echo —
so commands are not a faithful reproduction, but every gating decision is):

- a backward demand still arises on **10.2%** of ticks; **85% is answered by a turn**, 15%
  by the publish clamp, and **0% reaches the wire** (recorded flight: 9.5% published).
  Reported as a decomposition on purpose — "0%" alone cannot distinguish a working gate
  from a clamp hiding a broken one.
- finished plans classified and held on 26.6% of ticks; commanded speed there is p50 0.054,
  p90 0.729 m/s. The approach is **not** gentler than before — the win is that the aircraft
  arrives and stops, and refuses the approach beyond `endpoint_reach_m`.
- turning in place: 13.7% of ticks
- the replay found a defect the unit tests missed — 93 ticks flying 0.87 m/s toward a hold
  point over 3 m away, because the reach test only ran against the incoming reference and
  never against the latched hold. Fixed, with a regression test.

**Not done, and deliberately:** `course_slew_deg_s` is unchanged at 45 deg/s. LOOP_BUGS B31
and MISSION P17/P24 rule out raising it without a discriminator between a thrashing demand
and a persistent one, and the achieved-vs-commanded ratio that would have justified it turned
out to be unmeasurable from this run (see LESSONS 2026-09-09). The committed turn bypasses
that limiter instead, which is the narrow case P24 itself proposed.
