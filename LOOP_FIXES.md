# LOOP_FIXES — every change, its pre-registration, and its verdict

Companion to `LOOP_MISSION.md` (immutable) and `LOOP_BUGS.md` (the problems).
**Edit this file freely.**

## Index

| id | change | state |
|---|---|---|
| **F1** | Cold-start bring-up deadlock | APPLIED |
| **F2** | Instrumentation: record the plan, the tracking and the map | APPLIED |
| **F3** | v9.0: the velocity the servo closes on is 0.20 s late; make it 0.08 s | see entry |
| **F4** | `frac_of_box` over-reported by 6.5× | APPLIED |
| **F5** | A launch file that is not well-formed XML now fails in milliseconds | APPLIED |
| **F6** | Measure distance to the ROUTE, not to the reference point | APPLIED |
| **F7** | An import-time host path took the recorder down on every cycle | APPLIED |
| **F8** | Flag Sphera's bogus second publisher instead of recording it as truth | APPLIED |
| **F9** | Detect and clear Sphera's duplicate pawn | APPLIED |
| **F10** | Refuse to fly when the pose stream has latched onto the wrong pawn | APPLIED |
| **F11** | Hand the pose latch to the cluster that is actually moving | APPLIED |
| **F12** | Cross-check the rangefinder against the pose | APPLIED |
| **F13** | Two corrections to the comparison tool, both from evidence | APPLIED |
| **F14** | Rebuild the FALCON image with the two C++ patches | DONE |
| **F15** | The A* diagnostics shipped, verified, compiled, and printed nothing | see entry |
| **F16** | A forgotten pause cost ~15 hours of flying | see entry |
| **F17** | v10.0: stop compressing every slow trajectory by 1.667x | REVERTED |
| **F18** | v11.0: feed the course's own turn rate forward | STOPPED — inconclusive at n=1 |
| **F19** | v12.0: steer the nose only above 0.25 m/s commanded (course gate) | STOPPED n=1 — mechanism self-defeating |
| **F19b** | v13.0: the gate, plus hold the course demand across gaps < 2 s | PAUSED — confounded by B34 |
| **F20** | See the aircraft held against geometry (instrumentation) | LANDED |
| **F21** | Persistent hazard memory for the dozen cells that trap the aircraft | UNRESOLVED — n=2 reverses the n=1 signal |
| **F22** | v14.0: halve the periodic replan interval (3.0 s -> 1.5 s) | REVERTED at interim |
| **F23** | Give yaw back to the planner (`yaw_mode` course -> reference) | STOPPED — guard breached at n=1 |
| **F24** | Speed-scheduled yaw blend: planner's yaw when slow, travel when fast | NOT ADOPTED — inconclusive |
| **F25** | Give the B-spline optimiser more than 10 ms to converge | STOPPED — under-powered; instrument instead |
| **F26** | Let the FCU finish coming up before restarting Sphera (operational) | WITHDRAWN — wrong path, redundant |
| **F27** | Log NLopt's result code + per-solve budget re-read (image rebuild) | LANDED |
| **F28** | Lift control points out of obstacles before optimising (the B44 fix) | STOPPED — the solve undoes it |
| **F29** | Give the clearance cost a gradient inside obstacles | REVERTED — guard breached; mechanism vindicated |
| **F30** | The same escape at quarter strength | REVERTED — no workable dose; needs a signed ESDF |
| **F31** | Signed ESDF (the fix with nothing to tune) | DESIGNED — not implemented; see entry |
| **F32** | Log the selected viewpoint (ungated); re-arms F21 | LANDED |
| **F33** | Apply the blocked radius the launch file already derived (1.5 -> 2.75) | WANT NOT MET — primary mis-chosen |
| **F34** | Within-flight paired design, built end to end (B39's answer) | BUILT — not yet flown |
| **F35** | First paired run: F24's question asked properly | FLYING |

*Entries below are in the order they were written, newest first.*

---

## The rule this file exists to enforce

**A change is not "done" when it is written. It is done when it has a verdict.**
Run-to-run spread on this platform is large (coverage CV ~27 %, distance CV ~51 %). One good flight
proves nothing — a 40 % effect needs ~8 flights per arm, 25 % needs 19, 15 % needs ~52
(Finding G, `runs/AUTOLOOP_JOURNAL.md`). So every flight-behaviour change gets:

| field | meaning |
|---|---|
| **MECHANISM** | why it should work, in one sentence, before any data exists |
| **PRIMARY** | one low-variance metric, chosen *before* flying, plus the control's own value |
| **WANT** | the threshold that counts as success |
| **REVERT-IF** | the threshold that ends it — written at the same time as WANT |
| **GUARDS** | things that must not get worse (clearance, altitude excursions, frontier starvation) |
| **n** | flights per arm, interleaved A/B/A/B, never a baseline from hours earlier |
| **VERDICT** | ADOPTED / REVERTED / MORE-RUNS, with the numbers |

Anything flown without a pre-registration written first is an observation, not a result.
Harness/instrumentation changes that cannot alter flight behaviour are exempt and marked `[infra]`.

---

## Live queue — pre-registered, not yet flown

### The queue, in order, with why each must precede the next

Ordering is not preference — each item removes a confound the next one would otherwise inherit.

| # | change | rebuild? | why it is here and not later |
|---|---|---|---|
| **F3** | fixed-window velocity estimator | no | **flying now.** The servo's feedback is 0.20 s late on every flight; everything downstream is measured through it. |
| **F7** | `fsm_slow_traj_ratio_min` 0.60 → 1.0 | no | B12/B13. Removes a 1.667× mid-curve compression that fires on ~12 % of flight time **and** the yaw/position clock desync. A yaw experiment run before this measures a reference that structurally cannot finish its turn. One launch arg, already wired. |
| **F8** | direction-preserving saturation in the twist adapter | no | B2. The forward axis reaches 1.57 m/s and the lateral only 0.43 m/s, and each is clipped **independently**, so a saturating lateral silently rotates the commanded heading. Harmless while the nose points along travel; fatal the moment yaw is decoupled. |
| **F9** | `explore_course_rate_ff` 0 → 1.0 | no | Yaw is a pure P-loop, so a course slewing at 45 °/s needs a standing 45 ° error. Already wired, default off. Improves course mode on its own and calibrates the yaw loop before the mode changes. |
| **R1** | rebuild the FALCON image | **yes** | `sparx_map_stats.sh` (mapping-rate counters, the operator's explicit ask) + `sparx_astar_legible.sh` (two of A*'s three failure exits are silent, which blocks B6). Both compile-checked; both diagnostic-only. Batch any C++ control patch into the same build. |
| **F10** | `explore_yaw_mode:=reference` + `explore_yaw_dot_ff` | no | The operator's headline ask. Needs F7 (clock desync), F8 (lateral authority) and R1 (measurable) first. |

**A trap that was checked and does NOT apply to us.** The C++ read warned that switching to
`reference` yaw would silently disable the `body_vy = 0.0` line, the align gate and the turn-creep
floor, because all three sit inside `if heading_err is not None and not self.use_lateral:` and
`heading_err` is `None` in reference mode. That reasoning is correct for the launch file's own
default (`explore_use_lateral` false) but **the campaign overrides it to true** — verified two ways
on the live stack: `rosparam get /falcon_exploration_follower/use_lateral` returns `true`, and the
flight trace's own gate field records `use_lateral: True`. With lateral on, `not self.use_lateral`
is already false, so the block is skipped in *both* modes today and switching the mode changes only
the heading target. No follower restructure is needed. *Recorded because acting on the warning would
have meant a refactor with no purpose, and a wasted flight to gate it.*

### F18 — v11.0: feed the course's own turn rate forward  ·  PRE-REGISTERED

**Bug:** B1's near half. Yaw is driven by a **pure P-loop** on heading error
(`falcon_exploration_follower_node.py`, `yaw_kp = 1.0` rad/s per rad, `yaw_dot_ff_gain` 0.0 and
inactive in course mode). A proportional controller tracking a *ramping* reference carries a standing
error equal to the ramp rate divided by the gain — so holding a course that slews at R rad/s costs a
standing heading error of exactly **R / yaw_kp**, with no transient involved.

**The closed form matches the measurement, which is why this is worth flying.** At the 45 deg/s
course-slew ceiling that predicts a standing 45 deg; measured heading error is **p50 25.0 deg, p90
46.4 deg**. The p90 sits on the predicted ceiling and the median sits at the typical turn rate. The
node's own comment records the same numbers from a different angle (p50 18-28, p90 46-50).

**CHANGE.** `explore_course_rate_ff` 0.0 → **1.0**: feed `d(course)/dt`, which `_slew_course` already
computes and discards, forward into the yaw-rate command. One launch arg, already wired,
env-selectable. **Does not touch `course_slew_deg_s`**, so MISSION.md P17's prohibition on that knob
is respected.

**PRIMARY — `metrics.json` → `.tracking.hdg_err_deg.median`.** Measured baseline over 14 flights:
**mean 25.0 deg, sd 2.30, CV 9.2 %** — the lowest-variance tracking metric in the corpus, and the one
the closed form predicts. **n = 6 per arm** resolves a 15 % shift at 80 % power; the predicted effect
is far larger, so a null here would be genuinely informative rather than under-powered.
**WANT ≤ 0.70x** (25 deg → under 18). **REVERT-IF the 90 % interval on the ratio lies entirely above
0.95** — an interval rule, not a point rule, per the F3 lesson.

**Why not `hdg_err_deg.p90`:** CV 21 %, needing 31 flights per arm. The median is the standing error
the mechanism acts on; the p90 is dominated by transients it cannot fix.

**GUARDS.** Roll rate p99 must not rise above control (the operator has previously flagged aggressive
roll); `trace.tracking.moving.cross_track_m.p90` not worse by more than 15 % (a faster-turning nose
could destabilise translation); coverage ≥ 0.80x; stationary share ≤ 1.15x.

## FIRST CANDIDATE FLIGHT (2026-09-03, n=1) — the primary moved exactly as predicted, and the aircraft destabilised

| metric | candidate | recent baseline | |
|---|---|---|---|
| **heading error p50 — PRIMARY** | **12.0 deg** | 25.0 ± 2.3 | **0.48x — 5.6 sd below the mean** |
| turning | **240 deg/m** | 41–85 across 7 flights | **3–6x outside the entire range** |
| distance flown | **64 m** | 138–236 | |
| stationary share | **0.63** | 0.08–0.39 | |
| coverage | **467 m³** | ~1900 | |
| `collapse_signature` | **`['PARKED', 'CIRCLING']`** | — | |

**The mechanism is confirmed.** The closed form said the standing heading error is the course rate
divided by `yaw_kp`, and feeding that rate forward should remove it. It did: 25.0 deg → 12.0 deg on
the first flight, more than five standard deviations.

**And it appears to have re-created MISSION.md P17's yaw limit cycle** — the failure the previous
campaign identifies as its single largest win when it was fixed. Turning at 240 deg/m against a
41–85 range, `CIRCLING` in the collapse signature, a 231 s stall, and a quarter of the usual
coverage. This flight was environmentally clean (`dual_publisher_active: False`, zero pose or ranger
rejections), so the instability is not an artefact of B22.

**Physically coherent:** adding rate feed-forward to a loop whose plant has real lag is a standard
route to oscillation. The gain of 1.0 cancels the standing error exactly in the ideal case and
over-drives a plant that cannot turn instantly.

**ACTION: one confirmatory candidate flight, then stop.** The guards are breached by margins far
outside any observed flight, so this is almost certainly real — but a single flight is a single
flight, and the arms alternate, so confirmation costs ~22 minutes. **If it repeats, revert to
`course_rate_ff = 0` and re-open at a LOWER gain** (0.3–0.5) as its own pre-registration: the
heading-error improvement is large and genuinely wanted, and the question becomes how much of it can
be had below the stability limit — not whether to abandon it.

**Why this one is worth a flight when the last two were not.** F17 was reverted because its mechanism
was a *design decision*, not a defect. This is a genuine control deficiency with an analytic
prediction that the existing data already corroborates — the measured p90 sits exactly where the
theory puts the ceiling. It is also a prerequisite for the operator's yaw ask (B1): a heading loop
that cannot hold a moving reference will not hold FALCON's planned yaw either.

---

### F17 — v10.0: stop compressing every slow trajectory by 1.667x  ·  **STOPPED AT THE GATE, REVERTED**

**Bug:** B12/B13. `/fsm/slow_traj_target_yaw` and `/fsm/slow_traj_ratio_max` were never set, so the
C++ defaults apply and the computed rescale ratio always lands far below `ratio_min`. Verified on a
real flight: **11 fires, every one** `Rescale ratio 0.196|0.258|0.299 clamped to 0.600`.
`lengthenTime(0.60)` compresses only the middle knots, so peak mid-curve reference speed rises by up
to **1.667x** — on plans that were slow *deliberately*, because the yaw turn needed the time — and
nothing re-checks feasibility afterwards. It also desyncs yaw from position (B13), because the yaw
spline's knot span is not re-timed and only a scalar `yaw_dt` reaches `traj_server`.

**CHANGE.** `fsm_slow_traj_ratio_min` 0.60 → **1.0**, which makes `lengthenTime` a no-op. One launch
arg, already wired, env-selectable (`SPARX_SLOW_RATIO_MIN`).

**A TWO-STAGE DESIGN, because the variance differs 6x between the two questions.** Measured over 17
comparable flights:

| metric | CV | flights/arm for a 10 % shift |
|---|---|---|
| `trace.plans.mean_speed_mps.p90` | **3.7 %** | **2** |
| `trace.tracking.moving.along_track_m.p90` | 10.5 % | 17 |
| `trace.tracking.frac_reference_moving` | 14.2 % | 32 |
| `trace.route.moving.frac_outside_safe_distance` | 21.7 % | 74 |

**Stage 1 — MECHANISM GATE, n=3 per arm.** `trace.plans.mean_speed_mps.p90` is a property of the
*plan*, not of the flight, so it is almost noiseless (CV 3.7 %). If the rescale is really inflating
mid-curve speed, removing it must lower this. **WANT ≤ 0.95x** control. **If the gate does not move,
STOP** — the mechanism is not what B12 claims and no outcome measurement is worth taking.

**Stage 2 — OUTCOME, only if stage 1 passes, n=17 per arm.** PRIMARY
`trace.tracking.moving.along_track_m.p90` (baseline mean 1.644, sd 0.172): a reference that runs
1.667x too fast mid-curve inflates lateness specifically, and lateness is the half F3 showed the
controller can move. **WANT ≤ 0.90x. REVERT-IF ≥ 1.00x as an INTERVAL, not a point** — see the F3
lesson below.

**GUARDS.** Coverage ≥ 0.80x control; stationary share ≤ 1.15x; `frac_reference_moving` not below
control; zero new tilt-cut warnings.

## VERDICT (2026-09-03, n=3/4) — STOP. The gate failed and a guard broke, for the same reason.

| metric | A (no rescale) | B (control) | A/B |
|---|---|---|---|
| **plan peak speed — MECHANISM GATE** | 0.678 | 0.702 | **0.965** (WANT ≤ 0.95) |
| plan duration | 12.36 s | 11.58 s | **1.068** |
| plan length | 4.29 m | 4.97 m | **0.863** |
| distance flown | 208 m | 235 m | 0.885 |
| reference-moving fraction | 0.617 | 0.707 | 0.873 |
| stationary share | 0.151 | 0.103 | **1.470** |
| **coverage — GUARD (≥ 0.80)** | 1484 | 1988 | **0.747 — BREACHED** |

**The gate missed its bar (0.965 against ≤0.95), and the pre-registration says STOP when it does.**
Independently, the coverage guard is breached.

**And the six metrics tell one coherent story, which is why this is not noise.** Removing the
compression makes plans *longer in time* and *shorter in distance* — slower curves. The aircraft
then travels less, its reference is moving a smaller share of the time, it is stationary **47 % more
often**, and it maps a quarter less. That is a causal chain, not six independent noisy readings.

**I had the purpose of the rescale backwards.** B12 framed the 1.667x compression as a defect —
a plan sped up beyond what the aircraft can track. It is better read as a **deliberate trade**: FALCON
compresses slow trajectories precisely to stop the aircraft dawdling, and on this platform that trade
is *favourable*, because time spent parked is what limits coverage (Finding C/F). Removing it cost
coverage and bought **no tracking improvement at all** — along-track came out at 1.02.

**REVERTED** to `fsm_slow_traj_ratio_min = 0.60`, which is the shipped default; no code change is
needed to restore it.

**What the two-stage design saved.** The outcome stage was pre-registered at **17 flights per arm**.
The gate cost **3 flights per arm and answered it**. The B13 yaw/position clock desync that this
change would also have removed is real, but it must now be addressed some other way — the rescale
itself is earning its place.

**Two lessons from F3 applied here rather than repeated:**
1. **The primary matches the mechanism.** F3's primary mixed along-track and cross-track when the
   mechanism could only move one, and the effect was diluted to nothing. This one names the half the
   mechanism acts on.
2. **REVERT-IF is an interval rule.** F3's was a point-estimate rule, so a null result whose interval
   spanned 1.0 tripped a trigger meant to catch harm.

---

### F3 — v9.0: the velocity the servo closes on is 0.20 s late; make it 0.08 s
**Bug:** B3 / B11. **Pre-registered 2026-09-02, before any candidate flight existed.**

**MECHANISM.** `AxisVelocityServo` (`kp=90`, `ki=220`, `max_correction=350`) closes its loop on
`/R1/velocity_truth`. On **every** flight that stream is a differentiated position, not a
measurement: `SpheraPawnState.velocity` is all-zero in this vendor build, so
`rooster_ground_truth_localization.py` takes its documented fallback — confirmed in **11 of 11**
recent runs, and the log line itself says *"expect the ~0.25 s of filter lag the servo gains were
cut for"*. Measured on the baseline flight by cross-correlating the published estimate against the
true derivative over all 9069 samples: the peak sits at **0.20 s**. An integrator with `ki=220`
winding against feedback that late overshoots, which is exactly what the baseline shows.

**WHY THE OLD FILTER WAS NOT SIMPLY OVERKILL** (measured live, 2958 samples, off the message
*header* stamp — not arrival time, which was checked separately and matches): Sphera publishes state
at ~129 Hz with **2.3× dt jitter** (p90 18.1 ms against a 7.9 ms median). Differencing consecutive
samples divides a few ms of stamp error by ~8 ms, putting **p90 1.0 m/s of noise on a 0.5 m/s
signal**. The 0.25 s filter was doing real work. Lowering it alone — the obvious move — would have
traded lag for noise.

**THE CHANGE.** Difference over a **fixed 60 ms window** instead of consecutive samples, which
divides the same stamp error by 8× more time, and only then apply a **0.05 s** filter.
`velocity_window_s` (new, default 0.06) and `velocity_filter_tau_s` (0.25 → 0.05) in
`robots/ROBOTICAN/rooster_ground_truth_localization.py`. `velocity_window_s = 0` restores the old
estimator exactly.

**Verified offline before flying, by replaying the 2958 live samples through the actual
`_window_base` code path** (not a model of it): tick-to-tick noise **0.94×** the old estimator's —
slightly *better* — at **80 ms** of total lag against **250 ms**. It is strictly dominant on both
axes, which is why it is worth a flight; it is still an A/B because "strictly better offline" has
been wrong here before.

**PRIMARY — CORRECTED 2026-09-02 16:30, before any candidate flight had landed.**

> The original primary was `tracking.pos_err_m.p90` (control 2.15 m). **That metric is confounded
> and had to be replaced.** Checking the last eleven runs of the previous session against today's
> three showed the confound plainly:
>
> | | 2026-09-01 (n=11) | 2026-09-02 (n=3) |
> |---|---|---|
> | `pos_err_m.p90` | 0.64 – 2.33 m (*better*) | 2.15 – 2.47 m |
> | stationary fraction | 0.13 – **0.56** | 0.12 – 0.20 |
> | distance | 127 – 199 m | 204 – 233 m |
> | coverage gained | 743 – 1046 m³ | 1591 – **3894** m³ |
>
> **A parked aircraft next to a frozen reference scores a near-perfect tracking error.** The runs
> with the best `pos_err` are the ones that flew least and mapped a quarter as much. Optimising that
> number would reward the aircraft for standing still — the exact opposite of the mission. The
> correction is made *before* any candidate data exists, which is the only time a primary may be
> changed; the original value is left above so the change is auditable.
>
> (The 3–4× coverage difference between the two days is **not** explained by anything in this file —
> no flight-behaviour change had landed when `124337Z` flew. It is logged as an open question in
> `LOOP_BUGS.md` B20, not claimed as a result.)

**PRIMARY (outcome).** `trace.tracking.moving.position_error_m.p90` — tracking error measured only
on ticks where the reference is actually moving, which is the only regime in which "tracking" means
anything. Control value to be taken from the control arm's own flights (the metric did not exist
when `124337Z` flew). WANT ≤ **0.80×** the control median. REVERT-IF ≥ **1.00×**.

**CO-PRIMARY (the operator's question).** `trace.route.frac_outside_safe_distance` — share of the
flight spent further from FALCON's planned curve than the planner's own 0.55 m margin. First
measured value **0.427**. WANT ≤ 0.80× control. REVERT-IF ≥ 1.00× control.

**GUARD ADDED WITH THE CORRECTION.** `motion.frac_time_below_stop_speed` must not rise above
1.15× the control median. Without it, any change that makes the aircraft stop more would win the
primary by the same confound this correction exists to remove.

**GATE (mechanism) — the outcome is not believed unless this moves too.**
`actuation.x.achieved_over_commanded.p90`, control **1.749**. WANT ≤ **1.45**.
Secondary: `motion.speed.max`, control **1.647 m/s** against a 1.0 m/s follower cap; WANT ≤ 1.40.

**GUARDS — any one failing reverts regardless of the primary.**
- `coverage.gained_m3` median ≥ 0.80× the control arm's median.
- Altitude excursions above 2 m no more frequent than the control arm's.
- `clearance.aircraft_frac_inside_inflation` not worse than control by more than 0.05.
- Zero tilt-cut warnings (`"cutting drive until it is back under"`), as in the control.

**n.** 4 flights per arm, interleaved A/B/A/B, selected purely by `SPARX_VEL_WINDOW` /
`SPARX_VEL_TAU` so no repo file is edited between flights. `controller_rev` records the arm
(`v9.0-velwindow` vs `v2.1d-legacyvel`). Reassess at n=4; extend if the interval spans the bar.

**EXCLUSION, pre-registered with the metric correction and applied identically to both arms.**
A flight whose stationary share exceeds **0.50** is excluded as **planner-locked**, and the count and
identity of every exclusion is reported per arm. This is not a quality filter — it removes a
*different failure*. FALCON's terminal A\*-plan-fail lock (B6) parks the aircraft for minutes and
nothing in the control stack causes or cures it; the first candidate flight (`20260902_132444Z`)
stalled **205.6 s**, was stationary **76 %** of the time, flew 49 m and mapped 233 m³, against a
contemporaneous 233 m / 3894 m³. Such a flight sits on the curve it never left and scores a
near-perfect route error, so leaving it in lets whichever arm happened to lock more often "win".
**If the arms lock at different rates, that is the finding** and no tracking conclusion may be drawn
from the comparison.

**Tooling that enforces all of the above:** `tools/falcon_campaign/compare_arms.py` — deduplicates by
telemetry hash (Finding J), reads `metrics.json` rather than the copy embedded in `summary.json`
(which goes stale the moment a metric is added), applies the exclusion, and reports a **seeded
bootstrap interval on the ratio**, flagging explicitly when it spans 1.0 rather than rounding a point
estimate into a verdict.

**Analyzer correction that had to land first.** `actuation.*.achieved_*` was computed from
`velocity.vx` — *the very stream this change alters* — so judging this experiment by it would have
been circular. `analyze.py` now differences the recorded ground-truth pose over a fixed 0.15 s
window (`_fill_ground_velocity`) and every achieved-vs-commanded figure reads that instead. Re-running
the baseline through the corrected analyzer moved `actuation.x.gain` 0.493 → **0.531** and dropped
"axis x calibration off" out of the top-five findings. **Control values above are the corrected
ones.** Historical runs carry the old definition and are not comparable on these fields.

**MECHANISM CONFIRMED IN FLIGHT (2026-09-02) — the outcome is still pending.**

Cross-correlating the published velocity estimate against the true derivative of the recorded pose,
per flight, over ~8700 samples each:

| flight | arm | measured estimator lag | \|r\| at peak |
|---|---|---|---|
| `130620Z` | legacy, pre-change | **0.20 s** | 0.965 |
| `142446Z` | B control | **0.20 s** | 0.984 |
| `143742Z` | B control | **0.20 s** | 0.977 |
| `133813Z` | A candidate | **0.05 s** | 0.977 |
| `144746Z` | A candidate | **0.05 s** | 0.979 |

A **4× reduction**, perfectly separated between arms, at the measurement's finest resolution (50 ms,
the recorder's own cadence — so 0.05 s is a ceiling, and the predicted 80 ms is consistent with it).
The correlation is **as high in the candidate arm as in the control**, confirming the offline
prediction that the fixed window would not pay for its speed in noise.

This is the "verify the change reached the running system" step, and it passes cleanly. **It is not
the verdict** — the pre-registered primary is a tracking outcome, and at n=2 per arm the comparison
still reports TOO FEW. A mechanism that moves is a necessary condition, not a result; the campaign
has already recorded experiments whose mechanism moved decisively and whose outcome did not follow
(v3.0, v8.0).

**STANDING AT n=3 PER ARM (2026-09-02 18:31) — no verdict, and that is the correct reading.**

| metric | A (cand.) | B (ctrl) | A/B | 90 % interval |
|---|---|---|---|---|
| route: share outside the 0.55 m margin | 0.328 | 0.406 | **0.81** | [0.60, 1.04] spans 1 |
| route: distance to the curve, p50 | 0.389 | 0.455 | **0.86** | [0.72, 1.08] spans 1 |
| tracking error p90, moving only *(primary)* | 1.774 | 1.857 | **0.96** | [0.90, 1.29] spans 1 |
| correction saturation | 0.112 | 0.193 | **0.58** | [0.54, 1.52] spans 1 |
| distance flown *(guard)* | 250 m | 245 m | 1.02 | spans 1 |
| stationary share *(guard)* | 0.121 | 0.109 | 1.11 | spans 1 |
| coverage *(guard)* | 1841 | 1889 | 0.97 | spans 1 |

**Every tracking metric favours the candidate and not one of them is significant.** Every interval
spans 1.0, which is exactly what Finding G predicts at n=3 — that sample size can only resolve
effects of roughly 60 %. No guard has moved against the change either: distance and coverage are
flat, stationary share is up 11 % but with an interval from 0.26 to 2.00.

The largest point move is **correction saturation, 0.58x** — the metric nearest the mechanism, and
the one the tracker's own design rule is stated in (B25). It is also the lowest-variance of the set.
That is a reason to keep flying this experiment, not a reason to call it.

**Continuing to n>=8 per arm**, which Finding G puts at the threshold for resolving a 40 % effect;
several of these point estimates are smaller than that, so a null result at n=8 would itself be
informative rather than a failure to measure.

**THE DECOMPOSITION AT n=4/5 — the useful part, ahead of any verdict.**

Splitting the tracking error into its two halves separates what this change does from what it does
not:

| metric | A | B | A/B |
|---|---|---|---|
| **along-track** (lateness) p90 | 1.517 | 1.648 | **0.92** |
| **cross-track** (off the route) p90 | 1.010 | 1.009 | **1.00** |
| correction saturation | 0.133 | 0.181 | **0.73** |
| position error p90 *(primary, mixes both)* | 1.866 | 1.879 | 0.99 |

**The candidate improves lateness and leaves being-off-route untouched**, which is exactly what the
mechanism predicts: less feedback lag lets the servo hold the commanded *speed*, and speed is what
sets timing. Cross-track is set by path geometry and clearance, and no amount of velocity-feedback
bandwidth changes where the curve goes.

That also explains why the **primary is ~1.0**: `position_error` mixes the two halves, so a real
improvement in one is diluted by no change in the other. **Choosing a metric that mixes two
mechanisms was a flaw in my pre-registration** — the primary should have been along-track, which is
what a feedback-lag fix can move. The bar stays as written; this is recorded as a lesson for the next
pre-registration, not used to retro-fit a success.

**For the operator's goals this matters:** cross-track is what hits walls, lateness is not. So this
change, even if it lands, is not the collision fix — and the next experiment should target
cross-track directly.

**A caution about reading any of it yet.** At n=4/4 the primary's interval was [0.92, 1.16], which
*excluded* the 0.80 success bar. One more control flight moved it to [0.78, 1.11], which *includes*
it. A single flight flipping whether the pre-registered bar is reachable is precisely why Finding G
puts the requirement at ~8 per arm for a 40 % effect, and why nothing is being called here.

## VERDICT (2026-09-03, n=10 candidate / 8 control — the pre-registered size)

| metric | A | B | A/B | 90 % interval |
|---|---|---|---|---|
| **tracking error p90 — PRIMARY** | 1.973 | 1.890 | **1.044** | [0.88, 1.14] |
| route: share outside the margin — CO-PRIMARY | 0.341 | 0.384 | 0.887 | [0.72, 1.08] |
| along-track (the half the mechanism acts on) | 1.604 | 1.646 | 0.975 | [0.87, 1.07] |
| cross-track (the half it cannot) | 1.043 | 1.029 | 1.014 | [0.82, 1.15] |
| correction saturation | 0.152 | 0.172 | 0.886 | [0.71, 1.04] |
| reference-moving fraction | 0.699 | 0.663 | 1.056 | [0.94, 1.17] |
| stationary share *(guard)* | 0.110 | 0.132 | 0.829 | spans 1 |
| distance *(guard)* | 240 m | 242 m | 0.989 | spans 1 |
| coverage *(guard)* | 1874 | 1834 | 1.022 | spans 1 |

**OUTCOME: FAILED.** The primary is **1.044** against a WANT of ≤0.80 — the interval [0.88, 1.14]
excludes the success bar outright. The co-primary at 0.887 also misses its 0.80 bar. **Every one of
the nine metrics has an interval spanning 1.0**, so the honest summary is *no measurable difference
on any outcome*, in either direction.

**MECHANISM: CONFIRMED, decisively.** Estimator lag **0.20 s → 0.05 s**, perfectly separated between
arms across all 18 flights, with correlation as high in the candidate arm as the control — the
predicted "less lag at no cost in noise" is exactly what happened.

**DECISION: KEEP, and this is a deviation from my own revert rule, recorded as one.**

`REVERT-IF ≥ 1.00x` is nominally met (1.044). But **I specified it on a point estimate rather than an
interval**, so a null result trips a trigger written to catch harm. The interval spans 1.0; there is
no evidence of harm on any of nine metrics, and four guards are flat or favourable. Against that sits
a measured 4x improvement in feedback lag at no noise cost.

Reverting would restore a demonstrably worse velocity estimate to buy nothing, and every later
control experiment — the yaw work above all — runs on this feedback. **The operator may overrule
this; `SPARX_VEL_WINDOW=0 SPARX_VEL_TAU=0.25` restores the old estimator exactly.**

**WHAT THIS EXPERIMENT ACTUALLY BOUGHT**, which is worth more than the null:
1. **Lateness and being-off-route are separable, and only one is a control problem.** Along-track
   0.975 versus cross-track 1.014, exactly as the mechanism predicts. Cross-track — the half that
   hits walls — does not respond to feedback bandwidth at all, and 3/4 of it is inherited from the
   plan (B26).
2. **Tracking is not limited by feedback lag.** A 4x mechanical improvement moved no outcome. That
   retires a whole class of controller-side hypotheses and is why B25's remaining explanation
   (the plan asking for more than the plant sustains) is now the live one.
3. **Two design flaws, both corrected in F17 rather than repeated:** a primary that mixed two
   mechanisms when the change could only move one, and a revert rule stated on a point estimate.



---

---

## F5 — A launch file that is not well-formed XML now fails in milliseconds  ·  `[infra]`  ·  APPLIED 2026-09-02

**What happened.** A comment I added to `nav_stack.launch` contained `--` ("… of R/yaw_kp -- at the
45 deg/s slew …"), which XML forbids inside a comment. `bash -n`-style checks do not see launch
files, and `roslaunch` only discovers it **inside the container, after Sphera has been restarted and
the aircraft armed** — so the whole cycle was spent to learn it (`RLException: Invalid roslaunch XML
syntax … line 1288, column 44`, cycle 0 arm A, 2026-09-02 16:16).

**Change.** `bringup.assert_launch_xml()` parses every `adapter/launch/*.launch` with
`ElementTree` and raises `BringupError` before anything is spent. Called as the second step of
`full_bringup()`, right after `stop_twist_adapter()`. Put in `bringup.py` rather than in the loop
driver so it protects every caller, and so it needs no driver restart to take effect.

**Also fixed:** the two `--` sequences in the comments I had added (the second was in the
`slow_traj` block and had not yet been reached). All seven launch files now parse.

**Rule taken from it, beyond the fix:** a config format the flight path depends on gets a
pre-flight validator. The Python modules already had `py_compile`; the launch XML had nothing, and
it is loaded three minutes and one Sphera restart into every cycle.

---

## F2 — Instrumentation: record the plan, the tracking and the map  ·  `[infra]`  ·  APPLIED 2026-09-02

**Bug:** B10, plus the fact that every tracking number the campaign has ever quoted came from a
0.5 Hz log line (~490 samples of one scalar per flight).

**What already existed and was thrown away.** The follower publishes its *complete* per-tick
internals on `/nav_debug/control_trace` — the reference it was given, the error split into
along-track lag and cross-track, the feed-forward/damping/correction breakdown, every gate and
reflex flag — and `nav_debug_record` already defaults to `true`. Nothing collected it. FALCON
likewise publishes the B-spline itself (`trajectory/Bspline`: control points, knots, **yaw points**),
uncollected.

**Change.**
- `tools/falcon_campaign/probe_flight_trace.py` (new) — runs inside the `falcon` container for the
  flight, writes `flight_trace.jsonl`: the control trace, the B-spline, `pos_cmd` at 20 Hz, odom at
  10 Hz, replan events, coverage, frontier counts, and `/voxel_mapping/map_stats` once the image
  carries it. Subscribe-only. Verified live against a flight in progress: 9415 control ticks, 149
  B-splines, 8439 references, 581 coverage samples in one 430 s flight — against the 23 coverage
  samples the old path produced.
- `tools/falcon_campaign/trace_metrics.py` (new) + wired into `analyze.py`, so every flight now
  reports the error decomposition, the share of ticks whose position correction was **railed at its
  own limit**, which envelope limits bound the command, the plan's own duration/length/speed
  distribution, and the mapping-rate series.
- `patches/sparx_map_stats.sh` (new, in the Dockerfile) — extends the full-grid walk
  `MapServer::publishMapCoverage()` already does at 2 Hz with free/occupied counts, the 2D
  footprint, and 1×1×1 m cell coverage, published as JSON on `/voxel_mapping/map_stats`. Costs a few
  increments and two bitmaps, not a second pass. **Compile-checked** against the real image
  (`g++ -fsyntax-only` with the workspace's own include paths) — not yet in a built image.

**What it immediately showed, on the first flight that carried it:**
- Cross-track error **p50 0.55 m, p90 1.91 m, max 10.7 m** against a 0.55 m planner margin. The
  operator's "we get very far from the route" is confirmed and quantified for the first time.
- The position-correction term is **railed at its 1.0 m/s limit on 20 % of ticks**, and the
  tracker's own 1.5 m/s horizontal cap binds on **44 %** — while the plans it is following average
  only **0.39 m/s** (p50 over 149 trajectories, duration p50 10.4 s, length p50 4.5 m). The
  controller is demanding roughly four times the planned speed and saturating.
- The reference is **not moving on 31 %** of ticks.

**The mapping-rate deliverable, working end to end** (flight `133813Z`, 612 coverage samples against
the 23 the old polling produced — a 27x resolution gain):

| minute | volume m³ | new m³ | share of the flight's total |
|---|---|---|---|
| 0 | 575 | **369** | 16.4 % |
| 1 | 944 | 369 | 16.4 % |
| 2 | 1204 | 260 | 11.6 % |
| 3 | 1581 | 377 | 16.8 % |
| 4 | 2009 | **428** | 19.0 % |
| 5 | 2126 | 116 | 5.2 % |
| 6 | 2369 | 243 | 10.8 % |
| 7 | 2455 | **86** | 3.8 % |

**2249 m³ in 468 s = 0.21 s per m³, 288 m³/min overall, and the discovery rate decays 4.2×** from the
first minute to the last — the revisiting curve the operator asked for, measurable per flight for the
first time. Once the rebuilt image lands, the same series exists for voxel counts, 2D footprint area
and 1×1×1 m cells, at 2 Hz rather than 1.

**`[infra]`, so no A/B** — it changes no command, no gain and no timing; the probe is subscribe-only
and the C++ addition publishes a new topic without touching `map_coverage`.

---

## F4 — `frac_of_box` over-reported by 6.5×  ·  `[infra]`  ·  APPLIED 2026-09-02

**Bug:** B19. `EXPLORABLE_VOLUME_M3` was a hand-copied **4915.0**, from a 32 × 32 × 4.8 m box that
has not been the map since the bounds were widened. The real box
(`sphera_jail.yaml`: x[-14.6, 91.0], y[-42.4, 20.4], z[-1, 3.8]) is **31 832 m³**.

Now derived from the map yaml at import (`config._explorable_box()`), together with a new
`EXPLORABLE_AREA_M2` (6632 m²) for the 2D mapping-rate work. A hand-copied denominator cannot
survive a map change; this one cannot drift.

**Consequence for the mental model:** the baseline flight's coverage was **9.7 %** of the box, not
the 62.6 % printed. There is far more left to explore than the campaign has been assuming, and no
historical `frac_of_box` figure should be quoted.

---

## F6 — Measure distance to the ROUTE, not to the reference point  ·  `[infra]`  ·  APPLIED 2026-09-02

**The flaw.** `cross_track_error_m` splits the position error against the reference *point* and its
velocity direction (`reference_tracker_3d.tracker._split_error`). On a curved plan, an aircraft
exactly **on** the route but behind schedule still reports a cross-track of roughly
`curvature × lag`. Every "how far are we from the route" number the campaign has quoted is that
quantity, not the distance to the route.

**The fix.** `trace_metrics.curve_distance()` evaluates FALCON's published B-spline by de Boor and
returns the true shortest distance from the aircraft to the curve — possible only because F2 now
captures the control points and knots. Reported per flight as `trace.route`.

**Measured on the first flight that carried both** (n=3544 odom samples against 149 plans):

| | point metric (old) | curve distance (true) |
|---|---|---|
| p50 | 0.555 m | **0.465 m** |
| p90 | 1.939 m | **1.253 m** |
| further than 1 m | 24.1 % | **15.9 %** |
| further than 2 m | 8.6 % | **2.5 %** |

So the old metric overstated the tail by 2–3×. **The operator's complaint survives the correction:**
p50 0.47 m off the route, and **42.7 % of the flight is spent further from the route than the
planner's own 0.55 m clearance margin** — i.e. in geometry the plan never promised was free. That is
a direct, quantified explanation for hitting walls.

**Also refuted, in passing.** The obvious hypothesis — that the big excursions are replan
discontinuities (the "ghost start" of FALCON's `exploration_fsm.cpp:401-415`) — is **wrong**. A new
trajectory *reduces* cross-track by a median 0.075 m, and the 200 worst ticks occur a median
**4.2 s** after the last new plan, against a p50 inter-plan gap of 2.5 s. The aircraft drifts off as
a plan **ages**, and a replan is what rescues it. Do not spend a flight on replan-start blending
until something else supports it.

**Effect on the live F3 experiment:** none. F3's primary was pre-registered as
`tracking.pos_err_m.p90` before this existed and **stays** that — changing a primary mid-experiment
is exactly what pre-registration forbids. `trace.route.distance_to_curve_m.p90` and
`frac_outside_safe_distance` become the primary from the next experiment onward.

---

## Governance note — this loop does NOT commit

`LOOP_MISSION.md` §7 ends with "`git commit` locally as work lands", inherited from the previous
campaign's standing order. **That is superseded by `CLAUDE.md`**, which says version control is the
operator's and forbids `git commit`/`git push` on my own initiative. `LOOP_MISSION.md` is immutable
by instruction, so the line stays there; this note is the correction. **Nothing in this loop
commits.** The working tree carries every change described in this file, and the operator commits
when they choose.

---

## F7 — An import-time host path took the recorder down on every cycle  ·  `[infra]`  ·  APPLIED 2026-09-02

**What I broke.** F4 made `config.EXPLORABLE_VOLUME_M3` derive itself from the map yaml — read at
**import time**, through `REPO_ROOT`, which is a **host** path. `recorder.py` imports that module
*inside the `it` container*, where the repo is mounted at `/home/rooster/sparx_agency` and
`/home/user1/GIT/TheAgency/...` does not exist. So the recorder died on `FileNotFoundError` the
moment it started, and the cycle flew, mapped, landed and wrote `ended: completed` with an **empty
`truth.jsonl`**.

**What saved the flight anyway.** `flight_trace.jsonl` (F2) runs in the *falcon* container and was
untouched, so the run still carries its tracking, route and mapping metrics — 615 mapping samples
and a full control trace. The two recorders now cover for each other, which was not a designed
redundancy but is worth keeping. And Finding J's guard did its job: the campaign refused to copy the
previous flight's telemetry into the folder rather than silently reporting a duplicate.

**Fix.** `_explorable_box()` resolves the map path relative to `__file__`, not `REPO_ROOT`, so it is
correct from wherever it is imported, and it now returns `(None, None)` rather than raising if the
file is unreadable. Verified by importing `config` *inside* the `it` container and getting
31 832 m³ back.

**Guard, because the fix alone would not have caught it.** `bringup.assert_recorder_imports()` runs
`python3 -c 'import recorder'` inside `it`, in `IT_ENV` — exactly the environment `start_recorder`
uses — before the cycle commits to anything. `py_compile` cannot catch this class: it never executes
the imports. Verified both ways: it passes on the fixed tree, and a deliberately failing import
returns non-zero and would raise.

**The rule.** `config.py` is imported in at least three environments — the host, the ROS 2 Foxy
`it` container and the ROS 2 Humble dev container. **Nothing in it may touch the filesystem or the
network at import time through a path that is only valid in one of them.** Prefer `__file__`-relative
resolution, and fail soft for anything that is only a reporting scale.

---

## F8 — Flag Sphera's bogus second publisher instead of recording it as truth  ·  `[infra]`  ·  APPLIED 2026-09-02

**Bug:** B22. `recorder.py` now stamps every row with `truth.suspect`, and `analyze.py` prefers the
gated `attitude.*` stream over the raw `truth.roll/pitch` it used to read first — the bogus pawn
reports **zero attitude**, which was pulling tilt metrics toward zero (`truth.roll` p50 0.00° against
`attitude.roll` p50 1.84° on the affected flight).

Flagged rather than filtered, because `recorder.py`'s stated contract is that truth is recorded raw:
a correction cannot be un-applied once it turns out to be wrong. The flag uses the exact identity
`truth.x == -localization.x`, which holds for the real aircraft and fails by tens of metres for the
other publisher — verified against both cases and against a missing-field case.

**How it was found:** a lag check on a candidate flight returned |r| = 0.002 where every other flight
gave 0.98. Chasing that — rather than dismissing it as noise — turned up the 90 m inter-sample jumps.
**It also forced a re-check of F3's own derivation**, whose noise measurement could have been
contaminated by the same defect; re-partitioning that capture showed it was 100 % real aircraft, so
F3 stands.

---

## F9 — Detect and clear Sphera's duplicate pawn  ·  `[infra]`  ·  APPLIED 2026-09-02

**Bug:** B22/B23. `bringup.duplicate_state_publishers()` reads the publisher count on `/R1/state` and
`/R1/sphera/state`; `ensure_sphera()` restarts Sphera once when either exceeds one, and
`health_report` records `state_publishers` on every cycle so the analysis can condition on it.

**Deliberately advisory, not a veto.** It is *not* part of `health.ok`: one cycle took off and flew
normally with the duplicate active, so refusing to fly would ground the campaign over a defect
flights can survive.

**REVISED the same hour — the restart trigger was refuted and removed.** `ensure_sphera()` initially
restarted Sphera on detecting the duplicate, on the assumption that a restart clears it. It does not:
the duplicate has now survived roughly **ten consecutive restarts**, including full
`drone_simulator` recreation plus GUI re-entry. The trigger was pure cost and, worse, it created the
impression the condition was handled. Detection and per-cycle reporting are kept; the restart is
gone. **The stack survives the defect instead** — F11 for the latch, F10 for the safety net.

**Why it matters beyond tidiness:** three of five consecutive cycles ended `hover never settled` while
the duplicate was active, against zero before it appeared. The rangefinder alternates between the two
pawns and the altitude hold commands full climb and full descent several times a second.

**Not attempted:** making `rooster_unit`'s ranger guard latch the way
`rooster_ground_truth_localization._keep()` does — requiring N *consecutive* rejects before adopting a
new level instead of re-seeding on the rejected sample, which is what lets a strictly alternating pair
through. That is the real fix, it is a change to altitude hold (the hardest-won behaviour in this
stack), and it belongs in its own pre-registered A/B.

**What the investigation produced, which is worth more than the fix:** half of Finding K — the
campaign's top unsolved item, an altitude runaway reserved for the operator across two sessions —
now has a root cause. See B23.

---

## F10 — Refuse to fly when the pose stream has latched onto the wrong pawn  ·  **SAFETY**  ·  APPLIED 2026-09-02

**Bug:** B24. The worst consequence of B22's duplicate publisher is not a corrupted recording.
`rooster_ground_truth_localization._keep()` latches onto the **first** pose it sees and rejects later
ones that jump. With two pawns publishing, that is a coin flip — and when the impostor wins, every
message from the real aircraft is rejected as a jump for the rest of the run. **No warning is
printed**, because nothing looks anomalous from inside the gate (`re-latching pose` count: **0**).

**What that flight actually did** (`20260902_135459Z`): `localization.z` pinned at **0.118 m** for
453 s while the rangefinder read **1.29 m**. The aircraft was airborne. The entire stack — FALCON's
map, the follower, the servo — believed it was parked on the ground at the impostor's fixed position.
The follower drove the horizontal axes to **900 counts, the platform ceiling**, trying to close an
error that could never close. Recorded distance: **1.6 cm**. Coverage: 28 m³.

**That is an aircraft flying blind at full stick**, and it is exactly the condition under which the
operator's "it hits walls" complaint would be at its worst — with no telemetry that would ever show it.

**The check.** Once the aircraft is hovering — the only moment the two must agree, since on the ground
both read near zero and nothing can be told apart — compare `localization.z` against the rangefinder.
More than **0.6 m** apart and the cycle aborts before FALCON is handed control. The `finally` block
lands and disarms as on any other abort. Unreadable values are **not** treated as a failure — a
missing topic must not ground the campaign.

**CORRECTED TWICE. The final gate asks a different question from either earlier version.**

The test is no longer "do the pose and the rangefinder agree". It is **"is the pose stuck?"** — the
only condition that is actually unsafe — with ranger disagreement required as a second condition so a
genuinely motionless-but-correct pose cannot trip it. A real aircraft, even hovering, jitters by
centimetres; the impostor's pose spans tens of micrometres.

**Validated over five recorded flights, window by window** (349–356 windows each, the gate fires once
per cycle):

| flight | required | would abort in |
|---|---|---|
| pose latched to the impostor | **abort** | **88 %** of windows |
| healthy | pass | **0 %** |
| duplicate active, flew fine | pass | **0 %** |
| Finding K #1, a real altitude excursion | pass | **0 %** |
| Finding K #2, ranger impostor-dominated | pass | 3.1 % |

The residual 3 % is on the flight whose *rangefinder* was impostor-dominated, and aborting there is
defensible — that flight had a genuine runaway.

**The two false-positive modes it had to be designed out of, both caught in flight, not in review:**

*(1) A single ranger sample.* With the rangefinder alternating between two pawns, one reading is a coin
flip — a ~50 % false-positive rate. On `20260902_172137Z` it read a pose of **2.385 m**, matching the
independently measured hover of 2.43 m, against a ranger sample of **0.131 m**, and landed a flight
that had just come up correctly with every other fix working.

*(2) Many ranger samples, but blaming the pose for the disagreement.* On `20260902_173415Z`
**all 20** ranger samples read 0.131 m while the aircraft hovered at 1.21 m and the pose correctly
read 1.159 m. The ranger stream was entirely impostor; the pose was the good one. Aborting was
exactly backwards, and the abort message blamed the wrong stream.

**The lesson, and the reason this is written up rather than quietly patched:** a guard added to catch
an intermittent defect was twice written as if the streams were clean. Any check reading a stream
this defect touches must be built from a *sample*, and must not assume which of two disagreeing
streams is the broken one.

**Cheap, specific, and it catches the dangerous case rather than the noisy one.** The duplicate
publisher itself is survivable (a flight with 794 rejections flew 262 m normally); the *latch* is not.

---

## F11 — Hand the pose latch to the cluster that is actually moving  ·  APPLIED 2026-09-02  ·  partially validated

**Bug:** B24. `_keep()`'s existing escape hatch — adopt a new pose after `relatch_after_rejects = 25`
**consecutive** rejects — **cannot fire when two publishers interleave**, because every impostor
message is accepted (it matches the latch) and resets the streak. Replaying a real contaminated
flight through the actual code confirmed it: the latch stayed on the impostor for all 8749 poses,
with and without a streak-based candidate test I wrote first and then discarded.

**What replaced it.** A symmetric cluster test over a 3 s window: hand the latch over only when the
**accepted** cluster has not moved and the **rejected** one has. Across the contaminated flight the
impostor's position spanned 2.9e-05 m and the real aircraft's spanned 44 m, so 0.01 m separates them
by ~300× while sitting well below a hovering drone's own centimetre jitter. Both halves must hold
over a full window, so it cannot fire on a transient.

**Costs nothing on a healthy stack:** with one publisher there are no rejects, the rejected window
stays empty, and the test returns False before looking at anything.

**Replay result — a real improvement, and an honest limit.** The gate goes from accepting **100 %
impostor poses, span 0.00 m** to recovering the aircraft's **69 m of real motion**, with 27 % of
accepted poses still from the impostor (those that happen to fall within the 5 m jump gate while the
drone is near that spot).

**I did NOT tune `max_jump_m` against this replay, although the sweep was tempting.** Lowering it
made the replayed contamination *worse* (27 % → 62 %), which is the wrong direction for a physically
tighter gate — and that is the tell that **the replay is not faithful**: it is built from the
recorder's 20 Hz sample-and-hold, which picks whichever message was latest at each tick and does not
reproduce the true 129 Hz message order. The qualitative result (100 % impostor → aircraft recovered)
survives that limitation; the exact percentages and any tuning derived from them do not. Live data
decides `max_jump_m`, not this model.

**The safety net remains F10**, which aborts at hover if the latch is still wrong — belt and braces,
because this fix is validated only in part.

---

## F12 — Cross-check the rangefinder against the pose  ·  APPLIED 2026-09-02  ·  the yield fix

**Bug:** B23. This is what actually stops the aircraft taking off. `/R1/state` carries the duplicate
publisher too, so the rangefinder alternates sample by sample between the real value and the
impostor's (measured live: **0.131 m / 3.415 m**), and the terrain-relative altitude hold commands
full climb then full descent several times a second. **Five of eight consecutive cycles ended
"hover never settled."**

The existing rate guard cannot help: it rejects an implausible step and then **re-seeds on the
rejected value**, so a strictly alternating pair passes alternately.

**The fix.** `RoosterUnit` now subscribes the gated localization stream and drops any ranger sample
disagreeing with `localization.z` by more than **0.6 m**. The pose is an *independent* reference —
the localization node runs its own gate — so this is a genuine cross-check rather than a smoother.

**Measured before enabling, over three real flights including one with a true altitude excursion:**
`|z − ranger|` is p50 **0.019 m**, p90 **0.063**, p99 **0.217**, with only **0.37 %** of airborne
samples past 0.6 m. Replaying the actual gate code over those flights drops **0.4 %** of samples on
the healthy one and **0.1 %** on one that had the duplicate active and flew fine. The impostor
differs by 1.1–3.3 m and is always rejected.

**It fails OPEN in every uncertain case** — no pose yet, a pose older than 1 s, the gate disabled, or
more than 50 consecutive rejections (which means the *reference* is wrong, not the ranger). A filter
that can silently starve the altitude loop is more dangerous than the defect it removes. On the
flight whose pose had latched to the impostor it drops 90 % and then fails open in batches, which is
the designed degradation — and F10 aborts that cycle regardless.

**Also fixed in the same pass:** the streak relatch in the localization gate. Live logs showed F11's
cluster test correctly moving the latch onto the real aircraft and the **old** 25-consecutive-reject
path dragging it back to the impostor **2.5 s later** — the impostor's ~83 % share reaches 25
consecutive rejects within seconds. That path now requires the candidate to have moved as well.
*Neither the replay nor reasoning found this; running it did.*

---

## F13 — Two corrections to the comparison tool, both from evidence  ·  `[infra]`  ·  APPLIED 2026-09-02

**(a) Exposure to the duplicate pawn no longer excludes a flight — the mitigations changed the
facts.** F8 added an automatic exclusion for any flight flown while B22's duplicate publisher was
live, on the reasoning that such a flight is a different sampling regime. That was right *before* the
mitigations. `20260902_142446Z` then flew with **958 impostor poses** and produced a completely
ordinary flight — 250 m, 1889 m³, stationary 0.11, no altitude runaway — because the pose latch held
(0 bad relatches, and the new streak-refusal fired **once**, exactly as designed).

So exposure is no longer evidence of harm, and excluding on it would have thrown away every flight
while the defect persists — which is all of them. Now **reported per arm, excluded only with
`--drop-degraded`**. The planner-locked and stationary guards still catch the flights that actually
went wrong, which is the right level to filter at.

*Loosening an exclusion needs at least as much evidence as adding one, which is why it is written up
rather than just changed.*

**(b) The bootstrap interval no longer lies at small n.** At n=1 per arm, resampling a single value
always returns that value, so the interval printed as `[0.90, 0.90]` — **zero width, reading as
certainty, meaning the exact opposite**. Below three flights per arm it now prints
`n=1/1 TOO FEW`. Three is only the floor below which the arithmetic itself misleads; Finding G's real
requirement is far higher — ~8 flights per arm for a 40 % effect, ~52 for 15 %.

**Current F3 standing, for the record and NOT a verdict:** candidate/control point ratios are
`moving pos_err p90` **0.90×**, `route outside margin` **0.60×**, distance **1.05×**, coverage
**1.08×** — all in the wanted direction, all at **n=1 per arm**, all uninformative. Recorded so the
direction is not silently rediscovered as a surprise, and flagged so it is not mistaken for a result.

---

## F14 — Rebuild the FALCON image with the two C++ patches  ·  DONE 2026-09-02

Both patches were written and **compile-checked against the real image** days before this build
(`g++ -fsyntax-only` with the workspace's own include paths), and neither collides with an existing
patch — verified by listing which functions each touches: `sparx_map_stats.sh` edits only
`publishMapCoverage`, while the pre-existing scripts edit `publishOccupancyGrid*`, the resolution
parsing and the depth-overflow sizing.

**`sparx_map_stats.sh`** — the operator's explicit mapping-rate ask. Adds free/occupied voxel counts,
2D footprint area and 1×1×1 m cell coverage to the full-grid walk `publishMapCoverage()` already does
at 2 Hz, published as JSON on a new topic. `map_coverage` itself is untouched, so nothing downstream
changes. The recorder probe and `trace_metrics` already consume it; until this build lands, that
data is simply absent and **every flight flown without it loses those numbers permanently**.

**`sparx_astar_legible.sh`** — three throttled log lines, no control flow touched. Two of A*'s three
failure exits are silent upstream, so a 1 ms search timeout and a genuinely enclosed aircraft produce
the identical `[FSM] Plan fail`. They call for opposite fixes, and A*-plan-fail flooding is the
largest single sink of flight time on this platform (B6). No fix for it can be chosen while the
failure mode is unknown.

**Timing and risk.** Done during a deliberate `LOOP_PAUSE` rather than alongside a flight, because
`catkin_make -j8` would compete with a real-time control loop. The previous image is tagged
`falcon-ros:pre-sparx-instrumentation` as a rollback point. Both patches are diagnostic-only, so
straddling the F3 experiment is acceptable — and F3 has only n=1/2 usable flights so far, so there is
little to disturb. The build boundary is recorded here so no later comparison spans it unknowingly.

**VERIFIED IN THE COMPILED BINARIES**, not just in the build log — the campaign's own rule is that
applying cleanly is not the same as compiling, and compiling is not the same as being in the shipped
artefact. `strings` on the built libraries:

| check | library | found |
|---|---|---|
| `/voxel_mapping/map_stats` topic | `libvoxel_mapping.so` | yes |
| `known_cells_1m3` output field | `libvoxel_mapping.so` | yes |
| `sparx-astar` failure markers | `libpathfinding.so` | **3 of 3** |

Rollback: `docker tag falcon-ros:pre-sparx-instrumentation falcon-ros:noetic`.

**CONFIRMED LIVE 2026-09-02 18:10** — the falcon container came up on the new image and
`/voxel_mapping/map_stats` is publishing every declared field:
`known_voxels`, `free_voxels`, `occupied_voxels`, `known_volume_m3`, `occupied_volume_m3`,
**`known_area_m2`** (the operator's 2D half), `occupied_area_m2`, **`known_cells_1m3`** (the
1x1x1 m question), each with its box denominator. Sampled at the very start of a flight, so the
counts read zero -- which is the correct value there and confirms the publisher runs from the first
map tick rather than only once geometry exists.

---

## F15 — The A* diagnostics shipped, verified, compiled, and printed nothing  ·  FIXED 2026-09-02

**A patch that passed every check I had and still did nothing.** `sparx_astar_legible.sh` (F14)
applied cleanly, its own verification passed, `strings` found three markers in the built
`libpathfinding.so`, and the image shipped. Then three flights logged **63 `[FSM] Plan fail`, 25
`No path ... using default A*` and 19 `... using coarse A*` — and zero `sparx-astar` lines.**

**Why.** `astar.cpp` has **six** `search*` overloads, each with the same three failure exits, so each
anchor string occurs six times. My patch used `str.replace(old, new, 1)` — **first occurrence only**
— and every marker landed in `search(start, end, MODE, bbox_min, bbox_max)` at line 93. The planner
calls the two-argument `search(start, end)` at line 265 (`exploration_manager.cpp:1169`). The patch
instrumented an overload nothing exercises.

**Fixed:** all six overloads instrumented (18 markers), each message carrying `__FUNCTION__` so the
overload is identifiable — which is more useful than the original design anyway. The script's own
verification now requires **18** markers rather than 3, so an anchor drifting silently is caught at
build time instead of after a flight.

**The lesson is about the shape of the verification, not the typo.** Every check I ran confirmed the
patch was *present*; none confirmed it was *reachable*. "Applying cleanly is not compiling, and
compiling is not shipping" was already in the campaign's rules — this adds **"and shipping is not
executing."**

**And the `strings` check could never have caught it, which is worth knowing for the next patch.**
The rebuilt image has **18** markers in the source and **3** in `libpathfinding.so`. That is correct,
not a regression: all six overloads emit the *same* three format strings, so the linker keeps one
copy of each and `__FUNCTION__` separates them at run time. So a binary-level count of 3 is
consistent with one overload instrumented **or** all six — exactly the ambiguity that let the broken
version pass. **Count markers in the source, not in the binary**, and treat a `strings` hit as
evidence of presence only.

For a diagnostic patch the only real test is that it produces output on a flight that should trigger
it. That is now a required step, not an optional one.

---

## F16 — A forgotten pause cost ~15 hours of flying  ·  FIXED 2026-09-03

**What happened.** I set `runs/LOOP_PAUSE` at 19:20 on 2026-09-02 to rebuild the FALCON image, and
the session ended before the build started. The loop sat idle from then until **10:18 on 09-03** —
about **15 hours**, roughly 80 cycles of flying, and the rebuild it was waiting for never ran.

**This is a repeat, and that is the part worth recording.** `LOOP_MISSION.md` §7 carries the rule
because a forgotten sentinel cost the previous campaign 13.5 hours on 2026-08-18. That campaign's
`supervisor.sh` answered it with a self-expiring pause (`CAMPAIGN_PAUSE_MAX_AGE_S`, 30 min). **I wrote
my own driver and did not carry that across** — I reproduced the mechanism and dropped its guard in
the same file.

**Fix.** `runs/loop_driver_v3.sh`: `LOOP_PAUSE` self-expires after `LOOP_PAUSE_MAX_AGE_S`
(default **45 min** — long enough for a full image rebuild, short enough to bound the loss at one
build's worth of cycles). The driver logs loudly when it expires one, and its state line now shows
the sentinel's age counting toward expiry so a hold is visible rather than silent.

Swapped in per the campaign's own procedure: stop by PID, verify the PID is gone, then start — never
a racing `ps | grep`. Verified working on the first pass: the 15-hour-old sentinel was recognised and
cleared immediately on start-up.

**The general lesson.** When replacing a piece of unattended infrastructure, the *guards* in the old
one are the part most worth reading — they each encode an incident. I ported the loop and left the
scar tissue behind.

---

## F1 — Cold-start bring-up deadlock  ·  `[infra]`  ·  APPLIED 2026-09-02

**Bug:** B8. Every battery and armable probe is a `docker exec` into the `it` container, but
`full_bringup()` called `ensure_sphera()` *before* `start_containers()`. With `it` stopped — the
state after any long idle — `battery_fraction()` returned `None` forever, `_fresh_drone_ready()`
could never become true, and bring-up burned all four Sphera GUI re-entry attempts, **each one
force-removing a healthy `R1`**, before failing the cycle.

**Why it hid for a whole campaign:** the stack was never cold. `it` survives from the previous
cycle, so the ordering was never exercised. It surfaced on the first cycle after a 14-hour idle.

**Change:**
- `tools/falcon_campaign/bringup.py` — new `ensure_it_container()`, which `docker start`s a stopped
  `it` and raises `BringupError` only if the container does not exist at all (only Sphera creates
  it). Called as the **first** step of `full_bringup()`, before `ensure_sphera()`, and reused by
  `start_containers()` in place of its old unconditional raise.
- `tools/sphera_battery_watchdog.py` — `_wait_for_fresh_r1()` now starts a stopped `it` before
  polling, so the standalone recovery path (`--once`, which the loop driver calls after every failed
  cycle) has the same fix.

**Verification:** the cycle that exposed this was rescued live by starting `it` by hand — bring-up
went from a four-attempt failure loop straight to
`health: ... "armable": true, "ok": true` and `hover settled at 1.22 m`. `[infra]`, so no flight A/B
is required; the criterion is that a cold start reaches hover, and it did.

**Not changed, deliberately:** `POST_PLAY_VERIFY_TIMEOUT_SEC = 30.0` in the watchdog. The failure
message blamed a slow/misplaced click and a timeout bump was the tempting fix — but the actual cause
was the unreadable battery, and raising a timeout on a guess is the exact mistake recorded in
`MISSION.md` ("attempt 1 raised a timeout on a guess and hit a mechanism never involved"). If R1
genuinely proves slow to appear once `it` is reliably up, that will be a separate, evidenced change.

---

## Inherited state — what is already deployed and why

Carried in from the previous session (`runs/AUTOLOOP_JOURNAL.md`), configuration `v2.1d`. Listed so
this loop does not re-adopt or re-revert any of it by accident.

**Deployed and kept:**
- Measured power-law feedforward from `rooster_axis_curve.py`; lateral axis live at cap 600; gentle
  lateral slew (400/600).
- Altitude ranger-plausibility filter (`rooster_unit.py`, `altitude_hold_max_ranger_rate = 3.0`) —
  rejects rangefinder samples implying >3 m/s of vertical motion. Removes one runaway trigger; does
  **not** solve B7.
- `raycast_max 8.0`, `cluster_min 50`, `safe_distance 0.55`, bspline distance weight 150,
  course slew 45 °/s, tilt 35/27 with hysteresis, `tracker_pos_kp 1.0`, pinned hold 4 s,
  escape cooldown 4.0, tour commit off, `max_vel 0.8`, follower cap 1.0,
  `blocked_region_radius 1.5`.

**Reverted, with the reason — do not retry blindly:**

| rev | change | why it was reverted |
|---|---|---|
| v3.0 | obstacle inflation 0.40 → 0.30 | mechanism worked (A* "no path" fell ~10×) and still lost: coverage 0.92×, clearance guard 0.70×. Routing was never the binding constraint. |
| v4.0 | reference yaw + `yaw_dot` feed-forward | **deadlocked**, coverage 0.33× on both flights. Mechanism never established — see B1/B2 before retrying. |
| v5.0 | `accel_lead_s` 0.25 → 0.60 | along-track p90 0.565/0.525 vs a 0.370 bar. Finding F: the error is not made in transients. |
| v6.0 | wider altitude band | stationary time rose 0.32 → 0.41/0.52, distance 0.89×. Vertical authority spent climbing, not travelling. |
| v7.0/v7.1 | parked yaw scan (0.5, then 0.25 rad/s) | at n=5 interleaved: 0.536 vs 0.527 — nothing. Plus a tail hazard absent from the control: 2 of 8 scan flights ran away in altitude to ~3.6 m vs 0 of 4 controls. |
| v8.0 | `blocked_region_radius` 1.5 → 2.75 | re-strikes fell as designed, but it sterilised the map: 4 of 5 flights emptied the frontier set vs 1 of 4 controls. |

The plumbing from each reverted change is still wired and asserted (`yaw_mode`, `yaw_dot_ff_gain`,
`SPARX_PARK_SCAN`, `SPARX_BLOCKED_RADIUS`), so a future attempt solves the real problem instead of
rebuilding scaffolding.

---

## F19 — gate course steering on commanded speed (`course_steer_min_speed` 0.05 → 0.25 m/s)

**PRE-REGISTERED 2026-09-03, before any candidate flight existed.**

**MECHANISM (measured, B31, not assumed).** The follower aims the nose along the *direction of the
commanded velocity*. The direction of a small vector is ill-conditioned, and the command is small
often: **48 % of control ticks sit below 0.25 m/s**, and in that regime **45–55 % of all direction
changes exceed the 45 deg/s course-slew ceiling**, against 8 % above 1 m/s. The limiter absorbs it by
saturating on **71 %** of ticks, so the nose permanently chases a demand carrying no information.
FALCON's own B-spline is smooth (1 % over the ceiling) — this is entirely generated downstream.

The node already has exactly the right gate — `world_speed > course_min_speed` — set to **0.05 m/s**,
an order of magnitude below where the jitter lives. F19 raises only that gate.

**WHY A SEPARATE PARAMETER.** `course_min_speed` also gates the parked yaw-scan, which wants to stay
low. A new `course_steer_min_speed` (negative → falls back to `course_min_speed`, so the default is
byte-for-byte current behaviour) keeps the experiment to one mechanism. Documented at the definition.

**PRIMARY:** `tracking.hdg_err_deg.median`. Chosen because it is the metric the mechanism acts on,
and it is the best-characterised endpoint in the campaign (CV 9.2 %, B28).

**WANT:** primary ≤ **0.85 ×** control, bootstrap interval excluding 1.0.

**REVERT-IF** (any one):
- `motion.turning_deg_per_m` > **100** — the limit-cycle guard, added because F18 breached exactly
  this and the pre-registration had no tripwire for it;
- `coverage.gained_m3` < 0.80 × control;
- `trace.motion.frac_time_below_stop_speed` > 0.50;
- `trace.motion.distance_m` < 0.80 × control.

**GUARDS / confounds:** arms interleave A/B in one driver; `dual_publisher_active` must be False and
pose/ranger rejections zero, or the flight is dropped; `--since` scopes the comparison so no older
flight can be absorbed into the control arm (the F17 contamination lesson).

**n:** 6 per arm, interim look at 3. A single flight is not a result — F18's headline 5.6 sd primary
came with a collapsed aircraft.

**WATCH ITEM, not a guard:** commanded speed hovering near 0.25 could chatter the gate on and off.
No hysteresis in this version deliberately — adding it would be a second mechanism. If the turning
guard trips *without* the coverage guard tripping, chatter is the first thing to check.

### F18 — VERDICT: stopped as pre-registered, inconclusive, and the primary metric is unsafe here

The pre-registration said *one confirmatory flight, then stop*, with "revert if it repeats". **It did
not repeat.** Flight 2 (`20260903_102543Z`) against its interleaved control (`20260903_101524Z`):

| metric | flight 1 (collapsed) | flight 2 | control |
|---|---|---|---|
| `tracking.hdg_err_deg.median` — primary | 12.0 | **21** | 26 |
| `motion.turning_deg_per_m` | **240** | 76.1 | 56.2 |
| `trace.motion.distance_m` | 64 | 165 | 188 |
| `frac_time_below_stop_speed` | 0.63 | 0.39 | 0.24 |
| `coverage.gained_m3` | 467 | 1821 | 1811 |

Flight 1 is excluded by the standing planner-locked rule (stationary > 0.50), so the honest sample is
**n = 1 per arm — no interval, no conclusion.** The limit-cycle reading of flight 1 was over-read
from a single degenerate run; the mechanism does not reliably collapse the aircraft.

**But the direction of the secondaries is consistent and it exposes a flaw in the primary.** Flight 2
improved heading (0.81 ×) while turning rose 35 %, distance fell to 0.88 × and stationary time rose
62 %. That is exactly what feeding the course rate forward *should* do given B31: the nose tracks the
commanded course more faithfully — and the commanded course is **noise** on 48 % of ticks. So
`hdg_err_deg` improves precisely because the aircraft chases a bad reference harder.

**Methodological lesson, recorded because it generalises:** a tracking-error metric silently rewards
tracking a bad reference. It is only a valid endpoint once the reference is known to be good. For any
mechanism that changes *how hard* the nose follows the course, heading error must be read together
with `turning_deg_per_m` and `distance_m`, never alone.

**ACTION:** F18 stopped, `course_rate_ff_gain` stays at its default **0.0** (no revert needed — the
default was never changed; only the driver's arm env set it). The lower-gain follow-up (0.3–0.5) is
**not** scheduled: B31 shows the signal being fed forward is saturated and sign-flipping, so scaling
it is the wrong lever. Fix the reference instead — that is F19.

### F19 — pre-registration AMENDED 2026-09-03, still before any candidate flight

Amended on evidence from *pre-F19 flights only*; no candidate flight has flown, so this is not
post-hoc selection. Two changes:

1. **Co-primary added: the slew-saturated fraction** (share of steering ticks where the course slew
   limiter is railed). It is the quantity the mechanism acts on, its CV is 13.3 % over twelve
   flights, and the offline replay predicts **63 % → 33 %**. A 48 % effect at that CV needs **n ≈ 2
   per arm** — so this endpoint is well powered where every outcome metric is not.
   `tracking.hdg_err_deg.median` stays a co-primary but, per the F18 verdict, is never read alone.
2. **Exposure is now a declared covariate, not an assumption.** The share of ticks below 0.25 m/s —
   the only ticks F19 can change — ranges **10 %–78 % across flights (CV 67 %)**. Arms must be
   checked for balance on it; a candidate that happened to draw fast flights would show a small
   effect for reasons that have nothing to do with the change.

**Falsifiable prediction, recorded before the fact:** candidate flights show a slew-saturated
fraction near **33 %** against a control near **61 %**. If the candidate does not move that fraction,
the gate did not reach the node or the mechanism is wrong — and `runs/verify_f19_param.sh` reads the
parameter off the live server to tell those two apart.

**n is unchanged at 6 per arm**, which the co-primary over-powers and which gives the outcome guards
their ~37 % detection floor. Guards and revert-ifs are unchanged.

**Verification note (2026-09-03).** My external param-checker reported `FAIL — node did NOT receive
0.25`, and it was wrong: it polled `rosparam get` as soon as the driver announced the candidate arm,
so it read the *previous* cycle's node, which was still shutting down and legitimately held `-1.0`.
Reading the value directly once the new stack was up gave **0.25**. The lesson is not "add a retry" —
it is that a verifier must be pinned to the instance under test, not to the first process that
answers. Deleted, because `bringup.py` already reads every `EXPECTED_ROSPARAMS` entry back off the
live server and raises `BringupError` on a mismatch, which is the same check without the race. The
new gate is in that dict, so a mis-plumbed knob aborts the cycle instead of flying a fake candidate.

### F19 — INTERIM (n=1): the gate engages, my predicted effect does not, and the reason is measurable

**The falsifiable prediction I recorded was wrong.** Predicted candidate saturation ~33 % against a
control ~61 %. Measured, counting only ticks where the aircraft was actually steering (see the
instrumentation note below):

| flight | arm | steering ticks | saturated |
|---|---|---|---|
| `101524Z` | control | 93 % of ticks | 59.5 % |
| `103546Z` | control | 95 % | 64.7 % |
| `104550Z` | **candidate** | **78 %** | **62.7 %** |

The gate demonstrably reached the node (`course_steer_min_speed = 0.25` read off the live server) and
demonstrably did its job — steering exposure fell from ~94 % of ticks to 78 %. Saturation among the
remaining ticks did not improve at all.

**Why, measured rather than guessed.** Saturation against ticks elapsed since the gate re-opened, on
the candidate flight:

| ticks since resume | 0 | 3 | 9 | 10–19 | 20–49 | **50+** |
|---|---|---|---|---|---|---|
| saturated | 98.0 % | 94.4 % | 87.8 % | 80.8 % | 72.7 % | **47.2 %** |

**The mechanism works once it settles — steady state is 47.2 % against a control of ~60 %.** What
destroys the win is the resume transient. The node's `else` branch sets `_course_cmd = None` every
time the gate closes, so each re-open restarts the course from wherever the nose currently is, with a
large error to slew off at maximum rate. Raising the gate created more gaps, and therefore more
catch-up transients, and they cost exactly what the steady state gained.

My offline replay missed this because it reset the same way but was scored over the whole flight,
where the transients hid inside the average — the replay reproduced the bug faithfully and I read its
output as a prediction of the fix.

**Instrumentation bug found while checking this (real, and it corrupted my first reading).** The node
never resets `self._course_rate` on a non-steering tick, so the trace reports the *last* rate — often
a saturated one — for every gated tick. Scored naively the candidate looked *worse* (67.7 %) than
both controls. B31's headline 71 % figure is computed the same way; for the control arm the error is
small (93–95 % of its ticks steer) but the field is still wrong and is being fixed.

**ACTION: stop F19 at n=1 and fly F19b instead.** Not because n=1 settles the outcome — it does not —
but because the mechanism decomposition shows the implementation is self-defeating by construction,
and running it to n=6 would only measure a design already known to be broken. F19b keeps the gate and
**holds the course command across short gaps** instead of discarding it, which removes the transient
that ate the win. Same primary, same guards.

---

## F20 — see the aircraft being held against geometry (instrumentation, no control change)

**Not an experiment.** No command changes, so there is nothing to arm and nothing to revert; this
exists to make B33 visible in every future flight and, retroactively, in every past one.

**What was added**
- `core/planning/recovery/contact_hold_detector.py` — `ContactHoldDetector`, ROS-free, stdlib-only,
  Python 3.8 safe, 15 unit tests. Confirms a hold from *sustained tilt + no translation + no
  descent*, with an optional `ranger/height` corroboration against `1/cos(tilt)`.
- Wired into `falcon_exploration_follower_node.py` **before** the tilt reflex's early return, so it
  still sees the state while the reflex is commanding zero. Logs episode start and release, and adds
  `contact_held` / `contact_held_s` to the per-tick trace.
- `analyze.py` grew `contact_hold_metrics()` (episodes, total/median/max seconds, fraction of
  flight), computed from telemetry so it works on all 969 recorded runs, plus `z`/`vz` in the
  normalizer, which it needs to tell a hold from a capsize.

**Why a new detector rather than an existing one — the search was done first.** `StuckDetector`,
`BlockageMonitor`, `RecoverySupervisor`, `EscapeManeuver` and `BlockageMemory` all already exist, and
none of them can see this: every one judges *commanded versus achieved* motion, and the tilt reflex
responds to the excursion by commanding zero. From that instant nothing is being asked, no axis is
under test, and they are blind by construction. The new detector is the one judgement available when
nothing is being commanded — a statics argument (a multirotor cannot hold 50 deg at constant height)
rather than a kinematic one. That reasoning is recorded in the module docstring, as the deliberate-
duplication rule requires.

**Validated against the offline analysis:** flight `102543Z` reports 1 episode, 143.0 s, 32.3 % of
the flight — matching the hand analysis that found the same 146 s episode.

**Deliberately NOT fixed here: what to do about a hold.** The existing escape ladder was measured
burning half a flight on 38 attempts that never restored motion, and held episodes show the airframe
moving **2 cm** over as long as 133 s — rigid, not sliding. Reversing at 0.30 m/s has poor odds
against that. The detector exists to gather the evidence that should decide between escaping,
avoiding (the unused `blockage_memory.py`, B26) and something else. Guessing now would repeat the
mistake the give-up ladder was added to correct.

---

## F21 — persistent hazard memory: stop re-entering the dozen places that trap the aircraft

**PRE-REGISTERED 2026-09-03. Queued behind F19b; nothing flown yet.**

**MECHANISM (measured, B33).** Being held against geometry costs **3.6 % of all flight time** and
touches **36 % of flights**, and it is concentrated: **8 cells hold half the episodes, 12 cells hold
68 % of the lost time**, and the worst cell trapped the aircraft on 46 separate flights. Normalised
for time spent there, those cells are 6–9× the fleet hazard baseline, so they are genuinely
dangerous places rather than merely busy ones. The planner has no memory of any of it and routes back
into them every flight.

**WHY NOT the obvious cheaper fix.** Marking holds within a flight — B26's plan, using the existing
`blockage_memory.py` — can only ever recover the **21 %** of lost time that comes from re-entering a
cell in the same flight. **79 % is that flight's first encounter.** A per-flight memory is
structurally incapable of touching it. The knowledge has to outlive the flight.

**THE CHANGE.** Confirmed holds (from F20's detector, which is already flying and already logging)
are appended to a persistent hazard file; on start-up the planner loads it and inflates those cells.
Learned from the aircraft's own experience — not a hand-placed obstacle list, which would encode this
map's quirks and silently rot the moment the map changes.

**PRIMARY: seconds spent inside known hazard cells, per flight.** Deliberately *not* the held time
itself. Held time is zero-inflated — 64 % of flights record none — so at a 36 % base rate detecting a
halving needs ~111 flights per arm, which is 20 hours of flying. Time-inside-hazard-cells is non-zero
on nearly every flight, is the direct measure of whether avoidance is working, and has the variance
of an ordinary continuous metric. Held time becomes the **outcome guard**, read as a fleet total over
whatever n accumulates rather than as a per-arm interval.

**REVERT-IF:** `coverage.gained_m3` < 0.85 × control (inflating a dozen cells must not wall off the
map); `exploration.plan_fail` up materially; `trace.motion.distance_m` < 0.85 × control.

**GUARDS:** the hazard file must be built only from flights *before* the experiment, or the candidate
arm is learning from its own arm and the comparison is circular. Frame consistency is a real risk —
the survey's coordinates sit in one consistent localization frame today, but a map-epoch change would
invalidate the file silently, so it records the map name and is ignored on a mismatch.

**Open and honest:** it is not established that inflating these cells prevents the hold rather than
displacing it a metre. That is exactly what the experiment tests, and the coverage guard is what
catches the failure where the aircraft simply gets stuck on the far side of the same obstacle.

**F21 implementation route (found 2026-09-03, no C++ patch needed).** `/exploration_node` subscribes
to `/voxel_mapping/pointcloud` (`sensor_msgs/PointCloud2`) alongside the depth image, so hazard cells
can be injected as a small synthetic cloud from an ordinary ROS1 node — the mapper fuses them as
obstacles and the planner routes around them, with no change to the patched FALCON C++ at all. Two
things to settle before flying it: which frame that topic is expected in (it is consumed beside
`/map_ros/pose`, so it is probably already world-referenced, but that must be read rather than
assumed), and republication rate, since ray-casting through a cell will clear it whenever the depth
camera sees free space there. Republishing at the mapper's own ~2 Hz is the starting point.

### F19b — v13.0: keep the gate, but hold the course demand across short gaps  ·  PAUSED

The change F19's own measurement pointed at: the gate wins in steady state (47 % saturation vs
57–62 %) and loses it all to resume transients, so hold the demand across gaps below 2 s instead of
discarding it on every one. Paused before reaching n, not refuted — see F22 for why the whole course
line of work is waiting on the plan cadence.

**F19b falsifiable prediction, recorded before its first flight lands.** `trace.course.*` is now a
first-class metric (`trace_metrics.course_metrics`, keyed on `heading_err_rad` being present, which
agrees with the speed gate on 100 % of ticks and therefore also scores historical traces). The three
flights that exist so far:

| flight | arm | steering | saturated | **steady** | resumes |
|---|---|---|---|---|---|
| `101524Z` | control | 93 % | 59.5 % | 57.0 % | 63 |
| `103546Z` | control | 95 % | 64.4 % | 62.3 % | 36 |
| `104550Z` | F19 (gate only) | 78 % | 62.6 % | **47.0 %** | **100** |

The gate bought a much better steady state (47 % vs 57–62 %) and paid it all back in resumes
(100 vs 36–63). F19b holds the demand across gaps below 2 s, which spans 81 % of observed gaps, so:
**resumes should fall from ~100 to roughly 20, and overall saturation should converge on the ~47 %
steady figure rather than the ~62 % the transients produced.** If resumes do not fall, the hold is
not taking effect; if they fall and saturation does not, the resume transient was not the cause and
F19 should be abandoned rather than tuned further.

---

## F22 — halve FALCON's periodic replan interval so the plan does not die under the aircraft

**PRE-REGISTERED 2026-09-03, before any candidate flight.**

**MECHANISM (measured, B34).** Every published trajectory carries about **three seconds of motion**
and then goes dead — at age 4 s the reference is still on 93 % of ticks, at 5 s on 100 %. The
periodic replan fires at `replan_thresh3 = 3.0 s`, and plans actually arrive every **3.28 s (p50) /
4.88 s (p90)**. The aircraft is racing a plan that expires: land the next one inside ~3 s and it
keeps moving, miss and it hovers. That race is why **FALCON's own reference is stationary on 50 % of
ticks**. Halving the interval makes the next plan arrive while the aircraft is still moving.

**It is affordable.** Planning costs **11.8 ms at p50**, 39.8 ms at p90, and only 2 % of 689 118
calls overran their budget. Doubling the planning rate is nothing.

**Repaired first, separately (B35).** These thresholds were being written to `/fsm/` while FALCON
reads `/exploration_manager/fsm/`, so they had never taken effect. The namespace is now correct and
the defaults are set to FALCON's own yaml values (0.05 / 0.2 / 3.0), which makes the repair
behaviour-neutral; `/exploration_manager/fsm/replan_thresh3` is now in `EXPECTED_ROSPARAMS`, so a
future mis-plumbed knob aborts the cycle instead of flying a silent no-op.

**CANDIDATE:** `replan_thresh3 = 1.5 s` (control 3.0 s).

**PRIMARY:** `trace.reference.frac_still` — the share of ticks FALCON's reference is below
0.05 m/s. Mean 0.492, **CV 32.9 %** over ten flights. From the age curve, replanning at 1.5 s should
keep the aircraft in the 0–1.5 s band where the reference is still only ~20 % of the time, so the
predicted effect is **0.49 → ~0.25**, near 50 %. At that CV a 50 % effect needs **n ≈ 7 per arm**;
n = 8, interim look at 4. (A 15 % effect would need 76/arm and would not be worth flying.)

**REVERT-IF:** `coverage.gained_m3` < 0.85 × control; `exploration.plan_fail` materially up;
`trace.motion.distance_m` < 0.85 × control; `motion.turning_deg_per_m` > 100.

**GUARDS:** interleaved arms; `--since` scoped past the switch; degraded-environment flights dropped.
Watch `[FSM] Total time too long!` — at twice the rate, planning has half the wall-clock budget per
cycle even though each solve is unchanged.

**Why this pre-empts F19b, which is stopped rather than refuted.** F19b is measuring course-slew
saturation among *steering* ticks, and its steering exposure ranged 26–78 % across flights **because
the plan was parked for a varying share of each flight**. B34 is upstream: the course experiments are
being run on a base that moves under them. Fix the plan cadence first, then re-run the course work on
a stable base. F19b's mechanism finding (resumes fell 100 → 32, exactly as the hold predicted) stands
and is worth re-testing afterwards.

**F22 first candidate aborted by the readback assertion, which is the system working.** The cycle
failed bringup with `/exploration_manager/fsm/replan_thresh3 = 3.0 (want 1.5)` instead of flying a
candidate that was silently identical to the control. Cause: the knob was declared and consumed in
`nav_stack.launch` but never **forwarded through `sphera_drone.launch`**, which is the file the
campaign actually launches — so nav_stack fell back to its own default. Fixed by declaring and
forwarding it there, exactly as the course knobs were.

Worth stating plainly because it is the third instance of the same shape in two days: **a knob is not
wired until it is declared in `nav_stack.launch`, declared in `sphera_drone.launch`, forwarded from
the latter's include of the former, and read back in `EXPECTED_ROSPARAMS`.** B35 is the version of
this bug that went unnoticed for a whole campaign because the fourth step was missing; here the
fourth step caught it in one cycle. Every new knob gets the readback entry.

**F22 mechanism confirmed on the first candidate (mid-flight, n=1 — not a result).** With
`/exploration_manager/fsm/replan_thresh3 = 1.5` verified on the live parameter server, the reference
was parked on **26.6 %** of ticks against a control-arm median of **43.3 %** and a pre-registered
prediction of **~25 %**. The mechanism does what B34 said it would. The outcome question — whether
that converts into distance and coverage rather than into thrashing between goals — is what the
revert-if guards are for, and it needs the full n.

### F22 — INTERIM after the first candidate (n=1 vs 5 — directions only, not a result)

| metric | candidate | control | ratio |
|---|---|---|---|
| **`trace.reference.frac_still` — PRIMARY** | **0.278** | 0.464 | **0.60×** |
| `coverage.gained_m3` | **2361** | 1461 | **1.62×** |
| `trace.motion.frac_time_below_stop_speed` | 0.063 | 0.183 | 0.34× |
| `exploration.plan_fail` | 17 | 47 | 0.36× |
| `trace.motion.distance_m` | 235.7 | 213.5 | 1.10× |
| `motion.turning_deg_per_m` | 49.2 | 51 | 0.96× |
| `tracking.hdg_err_deg.median` | 28 | 26 | 1.08× |

The primary landed on its pre-registered prediction (0.278 against a predicted ~0.25), **no
revert-if guard is breached**, and the secondaries move together in the direction the mechanism
predicts: the aircraft spends a third as long stopped, covers 62 % more volume, and the planner fails
a third as often — the last of those unprompted, and worth understanding rather than banking.

**Heading error is 1.08× and that is not a concern here.** Per the F18 verdict this metric rewards
tracking a bad reference; an aircraft that is actually moving for twice as much of the flight will
show a slightly larger heading error while being strictly better off. It is a reporting metric in
this experiment, not a guard.

**n = 1.** The single biggest risk now is reading a coverage ratio of 1.62 as the effect size — that
metric has a CV of 26 % and needs 48 flights per arm to resolve a 15 % shift, so a single 1.62 is
consistent with a much smaller true effect. The primary is the endpoint the experiment is powered
for; the loop continues to n = 8 with an interim look at 4.

**F21 progress — the hazard file is built; the injector is not.** `hazard_map.py` turns the survey
into `runs/_analysis/hazard_map.json`: **12 cells covering 72 % of all time the aircraft has ever
spent held**, ranked by seconds lost rather than episode count (a cell that traps it once for two
minutes matters more than one it brushes ten times).

| cell (m) | held | episodes | distinct flights |
|---|---|---|---|
| 47, −17 | 1898 s | 43 | 35 |
| 55, −23 | 1684 s | 45 | 32 |
| 49, −17 | 1495 s | 54 | 43 |
| 47, −21 | 1333 s | 68 | **46** |
| 49, −5 / 51, −5 | 859 / 771 s | 28 / 35 | 20 / 25 |

They are not scattered: the top four form one contiguous region around (47–55, −17…−23), with a
second at (49–51, −5). **Three or four places account for most of it**, which is what makes avoidance
plausible rather than a map-wide inflation.

The file records `map: sphera_jail` and `frame: localization`, and the consumer must refuse a file
whose map does not match — the coordinates are silently meaningless across a map epoch.

**Still to build:** the node that republishes these as a synthetic cloud on
`/voxel_mapping/pointcloud`. Two unresolved questions before it flies, both to be read rather than
assumed: the frame that topic expects, and the republication rate needed to survive ray-clearing.
Held until F22 concludes, because F22 changes how much of the map the aircraft reaches per flight and
therefore how often it meets these cells at all.

### F21 — DESIGN CORRECTED: seed FALCON's own blocked regions, do not inject obstacles

B37 invalidates the injection plan, and the replacement is smaller and uses machinery that already
exists and is already live.

**Why injecting a synthetic obstacle cloud is the wrong idea.** The hazard cells are *already mapped
as occupied* — that is exactly why 41 % of trajectories are rejected there, with the rejection
coordinates clustering in the same cells that wedge the aircraft. The planner is not missing the
obstacle. It keeps **routing toward viewpoints behind it**, and the aircraft keeps approaching and
getting caught. Adding more occupancy would tell it something it knows.

**The right lever already exists.** `frontier_finder.cpp` reads
**`/frontier_finder/blocked_regions_runtime`** at construction — a flat list of `x, y, z` triples —
and retires frontiers within `blocked_region_radius` (1.5 m) of each. It is a SPARX patch written so
that "physics-vetoed viewpoints stay vetoed across respawns", and it works unchanged as a *seeding*
mechanism: set the parameter before the node starts and FALCON begins the flight already knowing not
to send the aircraft there. The node reads it at startup (line 159) and manages the list itself
afterwards (256, 308), so a launch-time seed is read exactly once and then owned by the planner.

**Independent corroboration that the mechanism agrees with the finding:** mid-flight, that parameter
already held one runtime-vetoed region at **(56.5, −19.6, 1.7)** — sitting between B33's hazard cells
at (55, −23) and (59, −21). FALCON's own veto mechanism is rediscovering, one flight at a time, the
places the 985-flight survey identified. Seeding is simply giving it that knowledge up front instead
of making it pay for it again every flight.

**The TTL is the second half of the change and cannot be skipped.** `blocked_region_ttl_s` defaults
to **90 s** with `ttl_max_doubling = 1`, so a seed expires about a third of the way into a 430 s
flight. The source says **`<= 0` keeps the old permanent behaviour**, which is the honest setting for
static walls that have trapped the aircraft on 46 separate flights. That is a real risk to declare:
a permanent block also makes *runtime* vetoes permanent, so a viewpoint that was only briefly
unreachable is given up on for good — which is exactly what the coverage guard is there to catch.

**Implementation is now: generate the flat list from `hazard_map.json`, set it and
`blocked_region_ttl_s` as launch params, add both to `EXPECTED_ROSPARAMS`.** No new node, no cloud,
no C++ change. Still queued behind F22.

---

## F23 — give yaw back to the planner: `yaw_mode` course → reference, now that lateral works

**PRE-REGISTERED 2026-09-03. Queued behind F22; nothing flown.** This is the operator's headline ask
— *"build a control that fits YAW separately and the drone's movement separately"* — and there is now
a concrete route to it.

**Why `course` mode exists, and why that reason has expired.** The node's own comment records the
2026-08-18 measurement that produced it: following FALCON's independently-planned yaw left the nose
*across* the direction of travel, so every command landed on the lateral axis, which was then
believed dead ("dead until ~axis 1000"). `course` mode fixed that by pointing the nose along the
commanded velocity. **But `use_lateral` is now True** with the frozen calibration delivering
0.428 m/s sideways at the 600-count cap, so the aircraft can translate with the nose pointed
anywhere. The combination *reference yaw + lateral enabled* appears never to have been flown.

**And the plan's yaw is far smoother than what we replaced it with** (six flights):

| where the nose is told to point | p50 | p90 | p99 | over the 45 deg/s ceiling |
|---|---|---|---|---|
| **FALCON's planned yaw** | **0.040 rad/s** | 0.546 | 3.006 | 1.4 % |
| the course demand (current) | 0.785 | 0.785 | 0.785 | pinned at the limiter |

**Twenty times smoother at the median.** The course demand is pinned at its slew ceiling because it
is chasing a direction that is noise half the time (B31); the planner's yaw is a purposeful curve
chosen for what the camera should see. Switching sources does not merely reduce the saturation
measured in B31 — it removes the loop that produces it.

**The change is `yaw_mode:=reference` plus a rate cap.** The plan's yaw is unconstrained anywhere in
FALCON (B14) and its p99 is 3.0 rad/s (172 deg/s), so the existing `course_slew_deg_s` limiter must
be applied to the planned yaw as a safety cap. Without that this experiment is one bad plan away
from the yaw limit cycle F18 already demonstrated.

**PRIMARY:** `motion.turning_deg_per_m` (CV 26.3 %). Chosen because it is mode-neutral — it measures
how much the aircraft actually turns per metre travelled, and means the same thing whichever source
is aiming the nose. Heading error does not: it changes definition between modes, and per the F18
verdict it rewards tracking a bad reference. The predicted effect is large given the 20× smoothness
gap, so n = 8 per arm resolves it if it is real.

**SECONDARY (the point of the change, not the guard):** voxels gained per metre flown — mean 1130,
CV 24.9 %. If aiming the camera by plan rather than by velocity is worth anything, it shows up here.

**REVERT-IF:** `trace.motion.distance_m` < 0.85 × control (lateral is 3.7× slower than forward, so a
mis-aligned nose costs speed — this is the real risk); `coverage.gained_m3` < 0.85 ×;
`trace.motion.frac_time_below_stop_speed` > 0.50; `motion.turning_deg_per_m` > 100 (limit-cycle guard).

**ORDERING: F23 goes before F21.** It is what the operator asked for, it retires B31 outright rather
than mitigating it, and F21's value depends on how much map the aircraft reaches per flight — which
F22 is currently changing underneath it.

### F23 — pre-registration AMENDED 2026-09-03, before any candidate flight

Two corrections from reading the node rather than assuming, both of which *simplify* the change.

1. **The rate cap I said F23 needed already exists.** I wrote that "the existing `course_slew_deg_s`
   limiter must be applied to the planned yaw as a safety cap". It does not need to be: in reference
   mode the commanded yaw rate is already `saturate(..., limits.max_yaw_rate)`, and
   `max_yaw_rate_deg` defaults to **45.0°/s** — the same ceiling as `course_slew_deg_s`. A 172 °/s
   spike in the planned yaw is already clamped to 45 °/s at the output. No new limiter, no patch.
2. **`yaw_dot_ff_gain` is 0.0 and must be raised as part of this change.** `traj_server` publishes the
   B-spline's analytic yaw derivative and the node currently discards it, so reference mode would run
   on proportional error alone. The node's own comment records what that costs — heading error
   "p50 18–28 deg, p90 46–50" — and says the feed-forward is "only meaningful in reference mode".
   Flying reference mode with the gain at zero would reproduce a result already measured and known to
   be bad, so the candidate arm sets **`yaw_mode:=reference` and `explore_yaw_dot_ff:=1.0` together**.

**That is two knobs but one mechanism** — "follow the planner's yaw curve properly" — and it is not
the F18 mistake repeated. F18 fed forward the *course* rate, which is pinned at its ceiling and
reverses sign every ~15 ticks. The planner's yaw rate is a smooth analytic derivative with a median
of 0.040 rad/s and only 1.4 % of samples above the ceiling. Different signal, different risk. The
turning-rate guard stays exactly where it is in case that reasoning is wrong.

**Never flown before, which is why it is worth a slot:** `yaw_dot_ff_gain` has been 0.0 for the whole
campaign, so reference mode has never run with its feed-forward enabled, and it has certainly never
run with `use_lateral` True.

**F23 is armable, and one sequencing point that must not be got wrong.** `YAW_MODE` and `YAW_DOT_FF`
are now environment-driven (`SPARX_YAW_MODE`, `SPARX_YAW_DOT_FF`), both are in `EXPECTED_ROSPARAMS`,
and the launch already forwards them — so F23 needs **no launch or node change at all**, only a
driver with the arm set. Default behaviour is byte-identical (`course`, gain 0.0).

**If F22 is adopted, its `replan_thresh3 = 1.5` must become the default before F23 flies, and both of
F23's arms must carry it.** Otherwise F23 would be measured on the old cadence — against a baseline
the campaign has already moved off — and its control arm would silently be a different aircraft from
the one being flown. The F23 driver is therefore deliberately *not* written yet: it gets built when
F22 returns a verdict, so that verdict is folded into both arms.

---

## Governance note — the commit line in LOOP_MISSION.md conflicts with CLAUDE.md, and CLAUDE.md wins

`LOOP_MISSION.md` §7 ends with *"`git commit` locally as work lands; do not `git push`"*. That line
was written by an earlier session of mine, not by the operator, and it conflicts with the project's
own standing instruction in `CLAUDE.md`: **"Version control is the user's. Never `git commit`/`git
push` on your own initiative."** The operator's opening brief granted permission to *edit* everything
and to decide everything about the flying; it said nothing about writing history.

**Resolution: nothing is committed.** CLAUDE.md is explicit, it is the project instruction, and there
is a recorded piece of operator feedback requiring the repo, files and message to be stated *before*
any commit runs — which cannot be honoured with the operator away. The whole session's work
therefore sits in the working tree, which is also the more reversible state for an operator who has
not seen any of it yet.

`LOOP_MISSION.md` is immutable, so that line stays where it is. This entry exists so a future session
reads the resolution alongside it instead of acting on the line. **If the operator wants the campaign
committed as it lands, that instruction has to come from them** — and at that point the standing
request to state repo, files and message first still applies.

### F22 — VERDICT at the pre-registered interim (n=4 vs 7): **REVERT**. The first flight was an outlier.

| metric | candidate | control | ratio | 90 % interval |
|---|---|---|---|---|
| **`trace.reference.frac_still` — PRIMARY** | 0.438 | 0.464 | **0.94×** | [0.69, 1.19] spans 1 |
| `trace.motion.distance_m` | 169.1 | 211.5 | **0.80×** | [0.72, 1.10] spans 1 |
| `trace.motion.frac_time_below_stop_speed` | 0.298 | 0.183 | **1.63×** | [0.43, 2.55] spans 1 |
| `coverage.gained_m3` | 1414 | 1391 | 1.02× | [0.78, 1.65] spans 1 |
| `motion.turning_deg_per_m` | 48.95 | 51 | 0.96× | spans 1 |
| **`exploration.plan_fail`** | 29 | 47 | **0.62×** | **[0.35, 0.97] — excludes 1** |
| `tracking.hdg_err_deg.median` | 29.25 | 26 | 1.13× | spans 1 |

**The pre-registered revert-if is breached:** `distance < 0.85 × control` at **0.80×**. Taking that at
face value rather than reinterpreting it now that I can see the data — the interval spans 1, so
distance may not truly have fallen, but the guard was written as a point-estimate precaution and it
is breached. Stationary time is also up 1.63×, in the same direction.

**And the primary has evaporated: 0.94×, interval [0.69, 1.19].** The first candidate's 0.60× was a
single flight, and the ratio walked back through 0.83× at n=2 to 0.94× at n=4 — the textbook shape of
regression to the mean. **I reported that first flight enthusiastically** ("strong signal on every
axis", coverage 1.62×) and, although I labelled it directions-only and warned in writing that reading
the coverage ratio as an effect size would be the mistake, the framing was still too confident for
one flight. The pre-registration is what caught it, not my judgement.

**One real effect survives, and it is instructive: `plan_fail` 0.62×, interval excluding 1.**
Replanning at 1.5 s genuinely reduces planning failures — the mechanism does what B34 said. **It just
does not convert into any flight-level benefit.** More successful plans did not become more distance,
more coverage, or less standing still. That is worth its own line in the ledger: a confirmed
mechanism win that buys nothing, which is exactly the kind of result that gets mistaken for progress
when only the mechanism metric is watched.

**ACTION: `replan_thresh3` returns to 3.0** (the default is already 3.0, so this is a driver change
only, no code revert). The **2.0 s fallback named in the pre-registration is NOT flown**: the primary
showed no effect at 1.5, so a value between 1.5 and 3.0 has no mechanism left to exploit. B34 stands
as a measurement — the plan really is parked half the flight — but the replan interval is not the
lever that fixes it.

**F23 pre-flight quantification, recorded before the first candidate flies.** Measured over four
recent flights: `reference.yaw` is present on 100 % of ticks and `yaw_error_rad` on 86 %, so the
signal reference mode drives on is healthy and not degenerate. But the magnitudes matter:

| \|yaw error\| (plan's yaw vs the aircraft's actual nose, flown in *course* mode) | p10 | p50 | p90 |
|---|---|---|---|
| degrees | 7.7 | **56.9** | 140.2 |

**FALCON plans a yaw about 57 degrees away from the direction of travel**, which is not a defect —
it is what an exploration planner does, aiming the camera at frontiers while the aircraft flies
somewhere else. On a holonomic platform that would be free. On this one it is not: at 57 degrees off,
the demand splits cos = 0.54 forward and sin = 0.84 lateral, and **lateral is capped at 0.428 m/s
against forward's 1.566** (B2). So the majority of the velocity demand lands on the axis that is
3.7× weaker.

**Predicted failure mode, written down now so the result is interpretable either way: F23 breaches
the distance guard.** If it does, the finding is not "reference yaw is bad" but "this airframe cannot
afford to aim its camera independently of its travel at full speed", which points at a hybrid — bias
the nose toward travel when the speed demand is high, toward the plan's yaw when it is low — rather
than at either extreme. That hybrid is deliberately *not* being built yet: it is a third mechanism,
and the simple version has to be measured first or there is nothing to attribute the hybrid's result
to.

### F23 — first candidate, mid-flight: the mechanism works perfectly and the predicted cost is severe

| | course mode (control) | reference mode (candidate, live) |
|---|---|---|
| \|yaw error\| p50 | 56.9° | **0.2°** |
| \|yaw error\| p90 | — | 21.7° |
| aircraft speed p50 | ~0.30 m/s | **0.04 m/s** |
| distance | 212 m per flight | **37 m in ~137 s** |

**The nose now follows FALCON's planned yaw essentially exactly** — a 57° median error becomes 0.2°.
Reference mode plus the yaw-rate feed-forward does precisely what it was designed to do, and that
half of the question is settled: the planner's yaw is followable, and the campaign has been throwing
it away.

**And the aircraft has almost stopped.** Speed p50 falls 7.5×; on this pace the flight covers about
half the control's distance. This is the failure mode written down *before* the flight, arriving at
roughly the predicted magnitude, for the predicted reason: with the nose aimed 57° off travel, most
of the velocity demand lands on the lateral axis, which is capped at 0.428 m/s against forward's
1.566.

**So the finding is not "reference yaw is bad".** It is that **this airframe cannot afford to aim its
camera independently of its travel while also travelling** — the asymmetry between its axes is the
binding constraint, exactly as B2 says. Both halves of that sentence are now measured rather than
assumed.

**ACTION: fly the interleaved control and one more candidate, then stop.** n=1 is not a verdict and I
am not going to treat it as one — but a 7.5× speed collapse does not need eight flights to establish,
and flying six more at half distance costs an hour and a half to confirm something the guard has
already caught. n=2 on an effect this size is defensible; the pre-registered n=8 was sized for a
subtle effect, not this one.

### F23 — VERDICT: **STOP**, on a pre-registered guard breached by the first flight

| metric | candidate `125535Z` | control | pre-registered guard |
|---|---|---|---|
| **`frac_time_below_stop_speed`** | **0.781** | ~0.18 | **> 0.50 → REVERT. Breached.** |
| `trace.motion.distance_m` | 50.3 | 169.6 | < 0.85 × → revert. Breached (0.30×). |
| `coverage.gained_m3` | 404 | 1031 | < 0.85 × → revert. Breached (0.39×). |
| `motion.turning_deg_per_m` | 31.0 | 59.1 | > 100 → revert. Not breached. |

The flight was additionally **excluded by the campaign's own standing planner-locked rule**
(stationary > 0.50), so it does not even qualify as a sample. Three separate revert-if conditions,
written before the flight, are breached by it. **This is a stop by the rules, not by impatience** —
and the decision needs no appeal to n, because a guard is a guard at n=1.

**What was learned is worth more than the experiment cost, and both halves are measured:**

1. **The planner's yaw is followable, exactly.** Median heading error against FALCON's own yaw curve
   went from **56.9° to 0.2°** the moment the feed-forward was enabled. The campaign has been
   discarding a good signal for two weeks on the strength of a reason that expired.
2. **This airframe cannot aim its camera independently of its travel while travelling.** FALCON aims
   ~57° off the direction of motion; at that angle 84 % of the velocity demand lands on the lateral
   axis, which caps at 0.428 m/s against forward's 1.566. Speed p50 collapsed 7.5×, to 0.04 m/s.

**The trade is not a defect in either component — it is the axis asymmetry (B2) becoming binding**,
and it is the clearest demonstration of B2's cost the campaign has produced.

**ACTION: F24, the speed-scheduled blend.** Aiming the camera is nearly free when the aircraft is
slow and costs almost everything when it is fast, so schedule it on speed rather than choosing one
extreme for the whole flight. Already implemented behind `yaw_mode:="blend"` (defaulted off, course
and reference paths provably unchanged): the nose interpolates from the planner's yaw at or below
`yaw_blend_lo` to the direction of travel at or above `yaw_blend_hi`.

---

## F24 — schedule the nose on speed: planner's yaw when slow, direction of travel when fast

**PRE-REGISTERED 2026-09-03, before any candidate flight.**

**MECHANISM, both halves measured in F23 rather than assumed.** Following FALCON's planned yaw is
(a) *achievable* — median heading error against it collapses from 56.9° to **0.2°** — and (b)
*unaffordable at speed*, because the planner aims ~57° off the direction of travel and at that angle
84 % of the velocity demand lands on the lateral axis, capped at 0.428 m/s against forward's 1.566.
Speed fell 7.5× and three revert-if guards tripped on the first flight.

But the cost is not constant: **it scales with how fast the aircraft is trying to go.** At low
commanded speed the lateral cap is not binding and aiming the camera is nearly free; at high speed it
takes almost everything. About half of all ticks sit below 0.25 m/s. That is a schedule, not a
choice between extremes.

**THE CHANGE:** `yaw_mode:="blend"`. The nose interpolates along the shorter arc from the planner's
yaw at or below `yaw_blend_lo` (0.15 m/s) to the direction of travel at or above `yaw_blend_hi`
(0.50 m/s). Both course and reference paths are provably unchanged — `_nose_target` returns exactly
what the old inline branch returned for those modes. The output rate is still capped by the same
45 °/s `max_yaw_rate`.

**PRIMARY: voxels gained per metre flown** (mean 1130, **CV 24.9 %**). Chosen because it is the only
metric that can express what this change is for: more map per unit of travel. Distance alone rewards
flying fast in a straight line; coverage alone rewards flying more. Normalising one by the other asks
the actual question — *did aiming the camera better buy us anything per metre?*

**POWER, stated plainly rather than glossed:** at CV 24.9 % this resolves a **30 % effect with n ≈ 11
per arm**, and a 15 % effect would need 43 — which this loop will not fund. **WANT is therefore
≥ 1.30 × control**, and n = 12 per arm with an interim look at 6. If the true effect is a modest
10–15 % improvement, **this experiment will not detect it and will be recorded as inconclusive, not
as a null.** That distinction is the whole reason for writing the number down first.

**REVERT-IF:** `trace.motion.distance_m` < 0.85 × control (the F23 failure mode, at a lower dose);
`trace.motion.frac_time_below_stop_speed` > 0.50; `coverage.gained_m3` < 0.85 ×;
`motion.turning_deg_per_m` > 100.

**GUARDS:** interleaved arms; `--since` scoped past the switch; the standing planner-locked and
duplicate-pawn exclusions. `yaw_blend_lo`/`hi` are in `EXPECTED_ROSPARAMS`, so a cycle aborts rather
than flying a blend that silently did not arrive.

### F24 — progress at n=5 vs 6 (not the interim; recorded because the design question is already answered)

| metric | candidate | control | ratio | 90 % interval |
|---|---|---|---|---|
| **voxels per metre flown — PRIMARY** | 1366 | 1122 | **1.22×** | [0.79, 1.60] spans 1 |
| `trace.motion.distance_m` | 236.2 | 233.4 | **1.01×** | [0.94, 1.14] |
| `coverage.gained_m3` | 2428 | 1709 | 1.42× | [0.92, 1.73] spans 1 |
| `motion.turning_deg_per_m` | 39.7 | 43.4 | 0.91× | [0.77, 1.00] |
| `trace.motion.frac_time_below_stop_speed` | 0.121 | 0.129 | 0.93× | spans 1 |
| `trace.reference.frac_still` | 0.310 | 0.353 | 0.88× | spans 1 |

**The design question F24 was built to answer is already settled: the schedule works.** F23 aimed the
camera perfectly and cost 70 % of the distance; F24 aims it partially and costs **nothing** —
distance 1.01×, stationary time slightly better, **no revert-if breached**. Scheduling the nose on
speed does exactly what the mechanism predicted, and it is the right shape for this airframe.

**The value question is not settled and may not become so.** The primary sits at 1.22× against a WANT
of 1.30×, with an interval spanning 1. Per the pre-registration this is **inconclusive, not a null** —
at CV 24.9 % this design cannot resolve a 10–20 % effect, and 1.22 is squarely in that band. Running
to n = 12 will tighten it but will not rescue it if the truth is ~1.2.

**One observation that does not fit the story and is recorded rather than smoothed over.** Mid-flight
the blend showed a *larger* heading error against the plan (65.6°) than pure course mode (56.9°),
while flying faster. If the camera is not actually aimed better, a coverage gain has to come from
somewhere else — most plausibly just from flying more of the map per unit time. That would make F24 a
motion result wearing a perception result's clothes, and the voxels-per-metre primary is exactly the
metric that can tell those apart. It is the number to trust at n, not the coverage ratio.

**F21 is fully wired and armable (2026-09-03), pending F24.** `frontier_blocked_seed:=true` loads
`adapter/launch/blocked_seed.yaml` — the 12 cells covering 72 % of all recorded wedge time — into
`/frontier_finder/blocked_regions_runtime` *before* `exploration_node` starts, and
`frontier_blocked_ttl` sets `blocked_region_ttl_s` (0 = permanent, which is the honest setting for
static walls). Both the launch arg and the TTL are in `EXPECTED_ROSPARAMS`, so a cycle aborts rather
than flying a seed that silently did not arrive. Default is off and byte-identical to today.

**B38 changes what F21 is for and how it must be judged.** It is no longer "a fix for wedging" — it
is the lever for the chain that owns the largest loss in the campaign, so its primary should be the
*first* link it can move, not the last:

- **PRIMARY: the pre-publish collision-rejection count** (`Collision detected on the trajectory
  before publishing`, ~78 per flight in the control). It is the link B38 identifies as causal, it is
  counted per flight from the log rather than derived, and it is the one thing that must move if the
  chain is real.
- **SECONDARY, in causal order:** `exploration.plan_fail` → `trace.reference.frac_still` →
  `coverage.gained_m3`. If rejections fall and parked time does not, the chain is wrong at link 3.
- **REVERT-IF:** `coverage.gained_m3` < 0.85 × control — the over-blocking failure, where twelve
  permanently-blocked cells wall off part of the map. This is the real risk and the reason the TTL is
  a declared knob rather than a constant.

**Sequencing:** F24 finishes first. If F24 is adopted, its `yaw_mode:=blend` becomes the default and
both F21 arms carry it — the same discipline F23 needed and F22's verdict made unnecessary.

### F24 — the "motion result in perception clothing" worry is resolved, in F24's favour

I flagged that F24's coverage gain might just be flying more of the map rather than seeing more of
it. The numbers separate the two cleanly (n=5 candidates vs 7 controls, medians):

| | blend (candidate) | course (control) |
|---|---|---|
| `trace.motion.distance_m` | 236.2 | 233.4 (**1.01×**) |
| **voxels per metre flown** | 1366 | 990 (**1.38×**) |
| `coverage.gained_m3` | 2428 | 1488 (1.63×) |
| **collision rejections per flight** | **51** | **49** |
| `exploration.plan_fail` | 31 | 44 |
| `trace.reference.frac_still` | 0.310 | 0.349 |

**Same distance, far more map per metre.** That is a perception gain, not a motion one — the aircraft
is not covering more ground, it is seeing more of what it passes, which is exactly what aiming the
nose by the planner's yaw at low speed is supposed to buy. My earlier worry was the right question
and the answer is the good one.

**And it tells us something about B38 that the chain alone would not have predicted: collision
rejections are unchanged (51 vs 49).** F24 gains 63 % coverage without touching the link B38 says
owns the biggest loss. So the two are **independent paths**, not one — F24 improves what the camera
sees per metre, F21 aims to reduce how often the aircraft is stranded. That is genuinely good news:
they should be additive, and it means F21's result cannot be explained away by F24 being adopted
first (its primary, the rejection count, is the one thing F24 demonstrably does not move).

`plan_fail` did fall (31 vs 44) while rejections stayed flat, which is a reminder that those two are
different failure modes and B38's chain runs through the *rejections*, not through `plan_fail`.

---

## F25 — give the trajectory optimiser more than 10 ms

**PRE-REGISTERED 2026-09-03. Queued; see the ordering note.**

**MECHANISM (a hypothesis with a measured motive, not a measured cause).** B38 established that 41 %
of produced trajectories are thrown away by a hard occupancy check, and that this is the first link
in the chain owning the campaign's largest loss. Reading the optimiser explains *how* a curve can
reach that check while clipping an obstacle:

- The optimiser treats clearance as a **soft quadratic penalty** — `if (dist < safe_distance_) cost
  += pow(dist - safe_distance_, 2)` at `safe_distance = 0.55` — traded against smoothness, endpoint
  and feasibility. `checkTrajCollision` then applies a **hard** `OCCUPIED` test. Soft in, hard out.
- And the optimiser is given **`max_iteration_time = 0.01`**, which is `opt.set_maxtime()` — NLopt's
  **total** budget. Ten milliseconds for an L-BFGS solve over 11–21 control points in three
  dimensions with six cost terms. `pos.distance` is already the dominant weight at 150, so the
  weights are not the problem; the time to act on them might be.

**Why this cannot be checked without flying it.** The NLopt result code is assigned to a local and
never read (`nlopt::result result = opt.optimize(...)`, unused), and the final-cost reporting is
behind `if (false)`. There is no log line anywhere saying whether the solver converged or hit its
time limit. Adding one is a small, safe patch, but it needs an image rebuild — and the behavioural
test is both cheaper and more direct.

**THE CHANGE:** `/bspline_opt/max_iteration_time` 0.01 → **0.04**, overridden after FALCON's
`fast_planner.yaml` load in `nav_stack.launch`, exactly as the FSM namespace fix does.

**PRIMARY: collision rejections per flight** (`Collision detected on the trajectory before
publishing`; control median **49–51**, counted from the log). This is a direct, per-flight count of
the thing the mechanism claims to change. **If rejections do not fall, the 10 ms budget was not
binding and the hypothesis is dead** — that is the point of choosing it.

**SECONDARY, in causal order:** `exploration.plan_fail` → `trace.reference.frac_still` →
`coverage.gained_m3`.

**REVERT-IF:** planner solve time p50 > 250 ms (currently 159; the budget is 100 and already overrun
86 % of the time, so this must not run away); `coverage.gained_m3` < 0.85 ×;
`trace.motion.distance_m` < 0.85 ×.

**ORDERING — F25 goes before F21.** Both attack B38's first link, but F25 reduces *rejection given
exposure* while F21 reduces *exposure*, and F25 is one parameter with no map dependency, where F21's
seed coordinates are tied to `sphera_jail` and rot silently on a map change. Cheaper, more general,
and if it works it changes how much headroom F21 has left.

---

## F26 — let the FCU finish coming up before spending a Sphera restart on it (operational, not an experiment)

Two cycles on 2026-09-03 ended `unhealthy stack; flew nothing` with `armable: false` and **every
other component healthy** — `falcon_up`, `bridge_up`, `exploration_node`, `frames_fresh` and
`battery_ok` all true. That is roughly **10 % of the day's cycles**, each costing a full flight plus a
35 s Sphera restart.

The bring-up already restarts Sphera when the FCU will not arm, and that logic is right — its comment
records six consecutive cycles lost to a dead FCU before it existed. What was wrong is how little
time it allowed: `drone_armable()` is a **single 8 second probe**, so a vendor stack that simply had
not finished starting was treated as a dead aircraft. A `wait_for_armable(timeout_s)` poller already
existed in the same file and was unused.

**Fix:** wait up to `ARMABLE_GRACE_S = 30 s` for the FCU to report armable, and only then restart.
Waiting costs seconds; restarting costs the cycle. If the FCU is genuinely dead the restart still
happens, 30 s later.

**Not an experiment and deliberately not arm-gated** — it changes only what happens on a cycle that
was going to be thrown away regardless, so there is nothing to A/B. It is verified the next time the
condition occurs: the log should show the grace message and the cycle should fly, instead of
`unhealthy stack`.

### F24 — INTERIM at the pre-registered n=6: positive, and **not** stopping early

**Primary, voxels gained per metre flown: candidate 1405, control 990 — ratio 1.419, 90 % interval
[1.06, 1.62].** The interval excludes 1 and the point estimate clears the 1.30 WANT. No revert-if is
breached: distance 1.01×, stationary 0.93×, coverage 1.63×, turning 0.91×.

**The experiment continues to n = 12 anyway.** The pre-registration specified an interim look at 6
and a decision at 12; it did not specify an early-stopping rule, and stopping now *because the
interim looks good* is precisely the error that makes published effects too large — you stop on a
favourable fluctuation and bank it as the estimate.

**F22 is the fresh, local proof of that.** Its first flight showed 0.60× on its primary and 1.62× on
coverage; by n=4 the primary was 0.94× and the change was reverted. The difference between that
trajectory and this one is not yet visible from n=6, and the only way to tell them apart is to keep
flying. **1.42 at n=6 is a reason to finish the experiment, not a reason to end it.**

If it holds to n=12 the change is adopted and `yaw_mode:="blend"` becomes the campaign default,
carried by both arms of everything after it.

**F26 costs a healthy cycle nothing, checked rather than assumed.** `wait_for` evaluates its
predicate at the top of the loop *before* any `sleep`, so when the FCU is already armable the call
returns on the first probe and the cycle is byte-for-byte as fast as before. Only the failure path —
a cycle that was going to be thrown away — spends the extra seconds. A robustness fix that quietly
taxed all 130-odd cycles a day to rescue two would have been a worse bug than the one it fixed.

**Regression check on the `_nose_target` refactor (flight-critical code, so checked in flight rather
than argued).** Adding the `blend` mode meant lifting the follower's steering branch into a helper.
The claim was that `course` and `reference` behave identically because the helper returns exactly
what the inline branch returned. Course-mode control flights either side of the edit:

| | n | steering frac | **slew-saturated frac** | resumes |
|---|---|---|---|---|
| before | 9 | 0.947 | **0.616** | 36 |
| after | 7 | 0.982 | **0.616** | 27 |

**Saturation — the course loop's own output, and the thing the refactor could have broken — is
identical to three decimals.** The steering-fraction and resume differences are downstream of how
much of each flight was spent above the 0.05 m/s gate, which is flight variation, not code. The
static argument and the flight data agree.

**How F24 gets adopted, written down before the verdict so the decision is mechanical.** If the
primary holds at n=12, adoption is a one-line change: `YAW_MODE`'s default becomes `"blend"` in
`config.py`. Three consequences follow, and the third is the one that bites:

1. The launch already forwards `explore_yaw_mode` and both blend thresholds, and all three are in
   `EXPECTED_ROSPARAMS` — so a mis-plumbed default aborts a cycle rather than silently reverting.
2. `_controller_rev` then returns `v16.0-blend0.15-0.5` for the *default* configuration, which is
   correct: the baseline aircraft has changed and its identity should say so.
3. **Every `v9.0-velwindow` flight stops being a valid control**, because it is a different aircraft.
   F25 and F21 must therefore collect fresh control arms after adoption rather than reusing today's —
   the same trap that made `--since` necessary when the F17 control silently absorbed thirteen older
   flights. The `--since` boundary for anything after adoption is the adoption itself, not the
   experiment's own start.

If the primary does not hold, nothing changes: the default stays `course`, F24 is recorded alongside
F22 as a mechanism that did not convert, and F25 proceeds against the existing baseline.

**Infrastructure for B39 (not an experiment): the follower can now change yaw mode in flight.**
`~yaw_mode_poll_s` re-reads the mode from the parameter server on a slow timer; **0 disables it and
is the default**, in which case the mode is read once at start-up exactly as before — the guard is
the first line of the function, so a disabled poll costs nothing. A rosparam read is an XML-RPC round
trip, hence a timer rather than a per-tick read, and a mode change resets the course command so the
new mode does not inherit a stale demand.

This is what makes a **within-flight paired run** possible: the campaign flips the parameter every
~60 s and each flight becomes its own matched pair, cancelling the between-flight variance that
currently makes every outcome metric cost 40–130 flights per arm. It is deliberately landed now,
while F24's between-flight answer is still being collected, so the paired re-run can be compared
against it rather than replacing it.

### F24 — a differential exclusion, found by checking rather than by the verdict

The candidate arm is being dropped by the standing planner-locked rule far more often than the
control, which would flatter it if left unexamined:

| arm | flights | excluded (stationary > 0.50) | stationary median |
|---|---|---|---|
| blend (candidate) | 13 | **3 (23 %)** — 0.587, 0.689, 0.707 | **0.277** |
| course (control) | 12 | 1 (8 %) — 0.568 | 0.135 |

**Dropping the candidate's worst flights while keeping the control's makes the candidate look better
than it is**, so the primary was recomputed both ways:

| | candidate | control | ratio | 90 % interval |
|---|---|---|---|---|
| with the exclusion (as pre-registered) | n=10, 1405 | n=11, 1250 | 1.124 | [0.92, 1.49] |
| **without it, all flights** | n=13, 1346 | n=12, 1210 | **1.113** | [0.91, 1.36] |

**The primary is robust to the rule** — 1.124 against 1.113, both spanning 1 — so the exclusion is
not manufacturing the result. That is the reassuring half.

**The unreassuring half is what the exclusion was hiding.** Stationary time: candidate median 0.277
against control 0.135, a ratio of **2.05**, though its interval **[0.79, 3.16] spans 1** and the
claim of harm is therefore *not* established. What is visible is the shape rather than the centre:
the candidate has a heavy tail — five flights at 0.36 or worse — while eleven of the control's twelve
sit at or below 0.18. That is the milder form of exactly what killed F23: some velocity demand landing
on the weak lateral axis, enough to stall the aircraft more often without immobilising it.

**Neither the benefit nor the harm is resolvable by this design**, which is B39's whole point. The
primary is nowhere near the 1.30 WANT and the interval spans 1, so on the pre-registered rule F24 is
**not adopted** — but "inconclusive with a hint of harm" is a different and more useful statement
than "it does not work", and it is the one the data supports.

### F26 — CORRECTION: the fix does not address the failure it was written for

The next `unhealthy stack` cycle arrived and neither of F26's log messages appeared, which is how I
found out the diagnosis was wrong. Two things I got wrong, both discoverable by reading the code I
was patching:

1. **Wrong path.** The cycle aborts at `campaign.py:477` — `full_bringup()` returns
   `health["ok"] = false` because `report["armable"]` is false. I patched the *battery* path in
   `ensure_sphera`, which is a different check that was not the one firing.
2. **Redundant anyway.** `bringup.py:953` already calls
   `wait_for_armable(240 s if Sphera was restarted else 90 s)` **before** building the health report,
   with a comment recording that it was added for precisely this symptom.

**And the timings say waiting is not the answer at all.** All three failures spent **377, 378 and
383 seconds** in bring-up before the verdict — over six minutes, including that 90–240 s wait — and
the FCU still reported unarmable. Adding another 30 s to a different code path could not have helped.

**So the ~10 % cycle loss is a genuine vendor-side failure with no cheap fix from our side**, and the
existing handling — restart Sphera, lose the cycle, carry on — is already the right response. There
is nothing to fix here; there was something to *understand*, and the understanding is that the FCU
sometimes does not come up.

**The code stays.** It is a no-op when the FCU is armable (the poller probes before it sleeps), and
on the battery path it is a small genuine improvement. But **F26 is not a fix for the observed
failure and must not be counted as one** — its status is corrected from LANDED to a no-op, and the
throughput claim attached to it is withdrawn.

### F24 — VERDICT at the pre-registered n=12: **NOT ADOPTED, and recorded as INCONCLUSIVE**

| | candidate | control |
|---|---|---|
| usable flights | 12 (4 excluded) | 13 (1 excluded) |
| **voxels per metre — PRIMARY** | 1366 | 1218 |
| **ratio** | **1.144** | 90 % interval **[0.86, 1.38]** |
| WANT | ≥ 1.30 | not met |

**The primary is below WANT and its interval spans 1, so the change is not adopted.** The default
stays `yaw_mode = course`, and every `v9.0-velwindow` flight remains a valid control for what follows
— the adoption bookkeeping written up in advance is not needed.

**It is recorded as inconclusive, not as a null, exactly as pre-registered.** The point estimate
1.144 sits squarely in the 10–20 % band this design was declared unable to resolve. Saying "the blend
does nothing" would overclaim: what the experiment establishes is that *if* there is an effect, it is
smaller than this method can see. The distinction was written down before the flights precisely so it
could not be blurred afterwards.

**The primary walked 1.42 → 1.29 → 1.24 → 1.14 → 1.12 → 1.09 → 1.14 as n grew** — the same shape F22
showed, and the second time today that a strong first look dissolved. Declining to stop at the
favourable n=6 interim was the right call both times, and it was the pre-registration that made it,
not judgement in the moment.

**What F24 did establish, and it is not nothing:**
- **The schedule works as designed.** F23 aimed the camera perfectly and cost 70 % of the distance;
  F24 aims it partially and costs **nothing** — distance 1.01×. That was the whole design question.
- **The gain, if any, is perceptual rather than motional** — same distance, more voxels per metre —
  which is the mechanism it was built on rather than a side effect.
- **A differential exclusion worth watching:** 4 of 16 candidate flights planner-locked against 1 of
  14 controls. Fisher one-sided **p = 0.24** — suggestive, not significant, and with four events
  against one no test could be. The direction matches F23's measured failure mode.

**ACTION: revert to `course`, proceed to F25.** And F24 is the strongest argument yet for B39: three
experiments today have died in the gap between "too small to see" and "nothing there", and a
within-flight paired design is the only thing that closes it.

### F25 — STOPPED after one flight, on a power calculation I should have done first

I pre-registered F25 with a primary and revert-ifs but **never fixed an n or a WANT**. Closing that
gap before candidates accumulated is what killed the experiment:

| form of the primary (control flights) | mean | CV | n/arm for a 25 % effect |
|---|---|---|---|
| rejections per flight | 45.0 | **44.2 %** | 49 |
| rejections per planner call | 0.204 | 42.9 % | 46 |
| rejections per plan produced | 0.124 | 39.8 % | 40 |

Control values run **13 to 73**. Normalising by planning attempts barely helps (44 % → 40 %), which
is itself informative: the variance is not "how many plans a flight tried", it is **where the
aircraft went** — exactly what B38 says, since rejections are a property of a dozen locations and
flights differ in whether they visit them.

**So F25 needs ~19 flights per arm to see a 40 % effect and ~90 to see the 18 % the first candidate
suggested.** That is seven hours to produce, in all likelihood, a third "inconclusive". Running it
would have been burning the loop's time on a predictable non-answer.

**And there is a better instrument available for the same question.** The hypothesis is *"the
optimiser is hitting its 10 ms ceiling"*. That is directly observable — NLopt returns a result code
saying whether it stopped on `maxtime` — and FALCON **assigns that code to a local and never reads
it** (`nlopt::result result = opt.optimize(...)`, unused), with the cost reporting behind
`if (false)`. Logging it answers the question **exactly, from a single flight, with no statistics at
all**, instead of inferring it from a metric whose CV is 44 %.

**ACTION: stop F25 as a flying experiment and patch the instrumentation instead.** One image rebuild
buys two things: the result-code log that settles the hypothesis outright, and a per-solve re-read of
`max_iteration_time` so this parameter — and the same pattern for other C++ constructor params —
becomes alternatable within a flight, which is what B39 needs to make this class of question
affordable at all.

**The lesson, recorded because it nearly cost seven hours:** a pre-registration without an `n` is not
a pre-registration. F22, F23 and F24 all had one; F25 did not, and the omission was invisible until
the variance was measured. Compute the power *before* the first candidate flies, every time.

---

## F27 — instrument the optimiser instead of inferring it (image rebuild, landed)

Two changes to `bspline_optimizer.cpp`, applied by `patches/sparx_opt_result.sh` and built into
`falcon-ros:noetic` (`623344ee92bd`; rollback tagged `falcon-ros:pre-opt-instrumentation`):

1. **Log how NLopt terminated.** `opt.optimize()` returns a result code, `NLOPT_MAXTIME_REACHED`
   among them; FALCON assigned it to a local and never read it, with the neighbouring cost report
   behind `if (false)`. It now emits a throttled `[sparx-opt] result=N budget=… elapsed=… cost=…
   pts=…`. **This answers F25's question exactly, from one flight, with no statistics** — where the
   behavioural route needed ~90 flights per arm because the rejection count has a CV of 44 %.
2. **Re-read `/bspline_opt/max_iteration_time` per solve** via `ros::param::getCached` (a local map
   hit after the first call, not an XML-RPC round trip), instead of caching it in the constructor.
   The budget can now be changed **within** a flight, which is what B39's paired design needs — and
   the same pattern applies to every other C++ constructor-read parameter in this planner.

Neither changes control flow.

**Verified before and after the rebuild**, because a bad image costs the whole loop: the patch
applied and **compiled** in a throwaway container first (`[100%] Built target trajectory`), and the
finished image was checked for the patched source, the format string inside `libtrajectory.so`, and
the continued presence of the other ten SPARX patches (hgrid clamp, slow-traj rescale, blocked
regions) so the rebuild dropped nothing.

**The loop flew baseline while this was in flight** rather than continuing a stopped experiment —
every baseline flight tightens the control estimate that every future experiment is measured against.

**F27 (second patch) — ask the ESDF what it thinks where the checker rejects.** Image
`d1a9c6b0102b`, rollback `falcon-ros:pre-esdf-instrumentation`. B40 removed "the optimiser ran out
of time" as an explanation for the 41 % rejection rate — 174 of 174 solves converged on tolerance in
under a millisecond of their ten. What remains is that it converges to something the checker
refuses, and the two read different maps: the optimiser minimises against the **ESDF**,
`checkTrajCollision` tests raw **occupancy**.

`planner_manager.cpp` now queries `map_server_->getESDF()->getDistanceAndGradient()` at the exact
rejecting point and logs `[sparx-esdf] reject at (x,y,z) esdf_dist=… t=…`. One query on a path that
already logs and returns false; no control flow changed. Compile-verified in a throwaway container
(`[100%] Built target fast_planner`) before the rebuild, and the finished image checked for the new
marker, the previous patch, and the ten older SPARX patches.

**What each outcome means, written down before the data arrives so it cannot be rationalised after:**
- **ESDF reports comfortable clearance (say > 0.3 m) where occupancy says OCCUPIED** → the two halves
  disagree about the world. The rejection rate is a **map-consistency** problem, and no amount of
  planner tuning touches it.
- **ESDF reports ~0 as well** → the optimiser knew the point was blocked and converged there anyway,
  which makes it a **cost-weighting** problem: the soft clearance penalty lost the trade against
  smoothness, endpoint and feasibility.

These call for opposite fixes, which is exactly why guessing between them was not an option.

---

## F28 — lift control points out of obstacles before optimising

**PRE-REGISTERED 2026-09-03, before any candidate flight.**

**MECHANISM (B44, confirmed from source, every link separately measured).** FALCON's ESDF is
unsigned: `(occupancy == OCCUPIED) ? 0 : max`, with no interior field. Its gradient is a trilinear
difference over eight neighbours, so inside an obstacle all eight read 0 and the gradient is
**exactly zero**. The optimiser's clearance gradient, `2·(dist − safe_distance)·dist_grad`, is
therefore zero as well: a control point inside an obstacle carries a flat 45.4 penalty **and no
direction in which to reduce it**. It is invisible to a gradient-based solver, which is why the
solver converges quickly (B40), still moves the *free* points 0.216 m median (B43), and leaves an
obstructed point inside in **11 of 14** rejected trajectories (B42).

**THE CHANGE.** Before the solve, walk the initial control points; for any whose ESDF distance is
~0, relocate it to the nearest non-occupied voxel found by a bounded local search of the occupancy
grid. The optimiser then starts with every point in the region where the gradient *works*, and the
clearance term can keep them out. **Deliberately not the alternatives:** making the ESDF signed
changes `getDistance` semantics for every consumer, and an in-cost escape term rewrites the cost
landscape mid-solve. This is the smallest change that restores a working gradient.

**PRIMARY: `exploration.collision_rejects`** (control median 45–51 per flight, **CV 44.2 %**).
Normally that CV would be prohibitive — it needed ~90 flights per arm for F25's suspected 18 %
effect. Here the predicted effect is large: 11 of 14 rejections carry an obstructed control point, so
removing that cause should cut rejections by well over half. **At CV 44 %, a 50 % reduction needs
n ≈ 6 per arm**, which is affordable. **WANT ≤ 0.50 ×**, n = 6, interim at 3.

*The power calculation is stated before flying this time — the omission that killed F25.*

**SECONDARY, in causal order:** `exploration.plan_fail` → `trace.reference.frac_still` →
`coverage.gained_m3` → `contact_hold.frac_of_flight`. If rejections fall and parked time does not,
B38's chain is wrong at link 3 and that is worth knowing.

**REVERT-IF:** `coverage.gained_m3` < 0.85 × control; `trace.motion.distance_m` < 0.85 ×;
planner solve p50 > 250 ms (the local search must not become the new cost); any increase in
`planner_death`.

**RISK, stated plainly:** relocating a control point changes the trajectory the optimiser starts
from, so it is not a pure diagnostic — a badly-placed relocation could make a plan worse rather than
merely different. The bounded search radius and the coverage guard are what contain that.

**F28 mechanism confirmed on the first candidate, before any outcome is read.** The lift fired 10
times in the first ~2 minutes of flight, relocating **1–4 control points out of every 11–18**:

```
[sparx-lift] lifted 4/18   lifted 1/12   lifted 2/14   lifted 2/16   lifted 2/11   lifted 3/18
```

**Control points routinely start inside obstacles** — roughly one in six, on every trajectory that
triggered the lift. That is an *independent* confirmation of B44's premise, arrived at by a different
route from the ESDF-gradient argument and from B42's control-point-clearance measurement: the
condition the fix targets is not rare, it is the normal case.

Whether removing it reduces rejections is the pre-registered question, and it needs n = 6 per arm.
The mechanism firing is not the result; it is only the evidence that the experiment is testing what
it claims to test.

**F28 first candidate (n=1, directions only): the lift fires 38 times and rejections do not move.**
`collision_rejects` = 45 against a control median of 45–51; plan_fail 31, coverage 1534, distance
207, stationary 0.207 — all unremarkable. The mechanism is definitely engaging (38 lift events), so
this is not a plumbing failure.

**If it holds, the reading is that the optimiser puts the points back.** A lifted point sits outside
the obstacle with a working gradient, but that gradient competes with smoothness, endpoint and
feasibility — and B42 measured the *final* trajectories, not the initial ones, so "11 of 14 end with
a point inside" is consistent with points being lifted out and pulled back. That would resurrect the
cost-weight hypothesis this entry downgraded, and would mean the honest fix is a signed ESDF (so the
outward gradient exists *and* is strong deep inside) rather than a starting-point repair.

**Held to n = 6 regardless.** Both experiments that looked decisive on their first flight today
dissolved as n grew, and this one looks decisive in the *negative* direction — which deserves the
same discipline, not less.

### F28 — STOPPED at n=1, on a measured mechanism failure rather than a weak outcome

| | rejections | control point still INSIDE after the solve |
|---|---|---|
| candidate (lift on) | 47 | **31 (66 %)** |
| control | 76 | 55 (72 %) |

**The lift is undone by the solve.** Points are moved out at the start — 38 lift events on the
candidate flight — and by the end two thirds of rejected trajectories have a control point back
inside an obstacle, at essentially the control's rate. That is why `collision_rejects` did not move.

**This is a stop for a different reason than F22 or F24, and the distinction matters.** Those were
stopped on outcome guards after n. This one is stopped at n=1 because **the intervention was measured
not to persist** — the mechanism it depends on is directly falsified, so flying five more candidates
would only re-measure an outcome whose cause is already understood. Continuing would be spending the
loop to confirm something the instrumentation has already settled.

**And it identifies the real mechanism, which no earlier entry had.** The clearance penalty has a
gradient only in the 0.55 m band outside an obstacle; inside, it is flat. L-BFGS steps are large —
B43 measured control-point travel at p50 0.216 m and up to 1.27 m — so **a single step can carry a
point from outside the band straight into the flat interior, where no gradient can bring it back.**
Lifting at the start cannot help, because the tunnelling happens during the solve.

**Which makes the signed ESDF the right fix after all**, and for a sharper reason than "it is the
standard remedy": a signed field has an outward gradient *everywhere inside*, so a point that
tunnels in is pushed back out instead of being trapped. Step limiting would be the alternative, but
it slows every solve to fix a rare event.

**Recorded against my own prior**: F28's pre-registration called the starting-point repair "the
cheapest fix that restores a working gradient" and explicitly deprioritised the signed ESDF as
"changes `getDistance` semantics for every consumer". That trade was judged on cost without knowing
the points were re-entering during the solve, which the fix itself is what revealed.

---

## F29 — give the clearance cost a gradient inside obstacles

**PRE-REGISTERED 2026-09-04, before any candidate flight. The successor to F28, and the fix the
evidence actually supports.**

**MECHANISM.** The bug is one line of FALCON's clearance cost:

```
if (dist_grad.norm() > 1e-4) dist_grad.normalize();      // inside: stays (0,0,0)
if (dist < safe_distance_) {
  cost += pow(dist - safe_distance_, 2);                 // +45.4
  gradient_q[i] += 2.0 * (dist - safe_distance_) * dist_grad;   // += exactly 0
}
```

Because the ESDF is unsigned (B44), a control point inside an obstacle reads distance 0 and gradient
0, so the penalty is added and **its gradient contribution is exactly zero** — a large flat cost the
solver cannot descend. F28 tried removing such points before the solve; **66 % were back inside
afterwards**, because one L-BFGS step (0.216 m median, 1.27 m max — B43) can cross the whole 0.55 m
gradient band into the flat interior. The gradient has to work *inside*, which is what this changes:
when the ESDF gradient vanishes and the point is in violation, the escape direction is built from the
occupancy grid's free axis-neighbours and plugged into the existing formula unchanged.

**PRIMARY: `exploration.collision_rejects`** (control median 45–51, CV 44.2 %). **WANT ≤ 0.50 ×**,
**n = 6 per arm**, interim at 3 — the same arithmetic as F28: a 50 % reduction is resolvable at that
CV, and the mechanism predicts one because 11 of 14 rejections carry a trapped point.

**MECHANISM CHECK BEFORE ANY OUTCOME:** the new `escape_hits` counter on the `[sparx-opt]` line
reports how often the substitution fires. **Zero hits means the change is not reaching the case it
targets**, and the experiment is void rather than negative — the distinction F28's lift-count made
available and which caught two mis-plumbed changes earlier in this campaign.

**REVERT-IF:** `coverage.gained_m3` < 0.85 × control; `trace.motion.distance_m` < 0.85 ×; planner
solve p50 > 250 ms; any increase in `planner_death`.

**RISK.** This changes the cost landscape during the solve, not just its starting point, so it is a
genuine control change. A wrong escape direction pushes a control point the wrong way — the six-axis
free-neighbour sum is deliberately the crudest reliable estimator, and the coverage and distance
guards are what contain it.

**F29 first candidate (n=1): the primary is met and something else breaks.** Mechanism firing is not
in doubt — 45 solves, 22 with substitutions, **394 escape gradients supplied**, so the zero-gradient
case was occurring on roughly half of every plan flown.

| metric | candidate | control | |
|---|---|---|---|
| `collision_rejects` — PRIMARY | **24** | 45–51 | meets WANT ≤ 25 |
| `exploration.plan_fail` | **225** | 31–47 | **5–7× worse** |
| `coverage.gained_m3` | 1131 | ~1400–1700 | likely breaches the 0.85× guard |
| `trace.motion.distance_m` | 176.3 | 207–233 | down |

**The fix does what it was designed to do — and the trade I flagged as its risk appears to be real.**
Pushing trapped control points outward halves the rejections; it also seems to produce trajectories
that fail planning outright, five to seven times as often. The plausible reading is that an escape
direction built from six axis-neighbours is crude enough to shove a point somewhere that violates
feasibility or the endpoint constraint, converting a *rejected* trajectory into a *failed plan* —
which is no better and costs coverage.

**n = 1, and the interim is at 3.** But unlike F28, this one is not stopping early: the primary moved
exactly as predicted, so the mechanism is sound and only its *implementation* looks too blunt. That
is worth measuring rather than abandoning — and if the interim confirms it, the next step is a gentler
escape (a smaller step, or blending the escape direction with the existing cost gradient) rather than
discarding the idea.

### F29 — VERDICT: **REVERT**, on a guard breached by a wide margin at n=1

| metric | candidate | control | ratio | |
|---|---|---|---|---|
| `collision_rejects` — PRIMARY | 24 | 36 | **0.67×** | mechanism works |
| `exploration.plan_fail` | 225 | 21 | **10.7×** | |
| `coverage.gained_m3` | 1131 | 2027 | **0.56×** | **REVERT-IF < 0.85× — breached** |

**Stopped by the rules, as F23 was.** A guard is a guard at n=1 when it is missed by 44 %, not by a
hair, and the accompanying tenfold rise in planning failures says the same thing twice.

**But the mechanism is vindicated, and that is the part worth keeping.** 394 escape gradients were
supplied across 22 of 45 solves, and collision rejections fell by a third — the trapped-point problem
is real, it is common, and pushing those points out *does* reduce rejections. What fails is the
crudeness of the push: a six-axis free-neighbour sum gives a blocky, axis-aligned direction at full
strength, which evidently shoves control points into configurations that then fail feasibility or the
endpoint constraint. **A rejected trajectory has been converted into a failed plan** — no better, and
it costs coverage.

**The refinement this implies, designed but deliberately not flown tonight:** scale the substituted
gradient by a gain (0.2–0.3) so it nudges a trapped point outward over several iterations instead of
displacing it in one, letting smoothness and feasibility keep the trajectory valid. That is one
parameter on the existing patch. **The principled alternative remains a signed ESDF**, which gives a
smooth, correctly-oriented interior gradient instead of an axis-aligned guess — more invasive, but it
is the version of this fix that has no crudeness to tune.

**ACTION: revert to the shipped behaviour; the loop returns to baseline collection.** The escape
gradient stays in the image, off by default.

---

## F30 — the same escape gradient, at a quarter strength

**PRE-REGISTERED 2026-09-04.** Inherits F29's design; one parameter differs.

**WHY.** F29 did not fail, it **overshot**. Its primary hit the target (`collision_rejects` 0.67×) and
the mechanism fired 394 times across 22 of 45 solves, so the physics is right: control points trapped
in the unsigned ESDF's flat interior can be pushed out, and rejections fall when they are. What broke
was the *dose* — a full unit-vector shove in a blocky, axis-aligned direction converted rejected
trajectories into failed plans (plan_fail 10.7×, coverage 0.56×).

**THE CHANGE:** `/bspline_opt/escape_gain` **0.25**. The substituted gradient nudges a trapped point
outward over several iterations instead of displacing it in one, leaving smoothness and feasibility
able to keep pace.

**PRIMARY, WANT, n and REVERT-IF are F29's unchanged:** `exploration.collision_rejects`, WANT ≤ 0.50 ×,
n = 6 per arm with an interim at 3; revert on `coverage.gained_m3` < 0.85 ×,
`trace.motion.distance_m` < 0.85 ×, solve p50 > 250 ms, or any rise in `planner_death`. The coverage
guard is the one that matters — it is what caught F29 within a single flight.

**Mechanism check before any outcome:** `escape_hits` must be > 0, or the run is void rather than
negative.

**If 0.25 also breaks the guards, stop tuning the gain.** The conclusion would be that an
occupancy-derived axis-aligned direction is too crude at any strength, and the remaining move is the
signed ESDF — a smooth, correctly-oriented interior gradient with nothing left to tune. That is
written here so a third dose is not attempted by reflex.

### F30 — VERDICT: **REVERT, and stop tuning the gain** (as pre-registered)

| metric | F29 (gain 1.0) | **F30 (gain 0.25)** | control |
|---|---|---|---|
| `collision_rejects` — PRIMARY | 24 | **77** | 36–51 |
| `exploration.plan_fail` | 225 | **47** | 21–47 |
| `coverage.gained_m3` | 1131 | **1011** | ~2027 |
| escape substitutions | 394 | **4023** | — |

**A quarter strength fixed the side effect and destroyed the benefit.** Plan failures came back to
the normal range — so the tenfold rise really was the shove being too violent — but rejections went
*above* the control, and coverage stayed at half. The pre-registered condition is met exactly:
**0.25 breaks the guards too, so gain tuning stops here.**

**The substitution count is the diagnostic.** 4023 escapes against F29's 394 — **ten times as many**.
A weaker push does not free a trapped point, it merely re-triggers on the next iteration, so points
stay inside longer and the term fires far more often for no result. Between the two doses there is no
window: strong enough to escape is strong enough to wreck the trajectory, and gentle enough to be
safe is too gentle to escape.

**That is a verdict on the estimator, not the mechanism.** A six-axis free-neighbour sum gives one of
a handful of blocky directions with no magnitude information — it cannot express "you are 8 cm inside,
move that way smoothly". B44's finding stands and F29 confirmed its consequence, but this
implementation of the escape cannot be dosed into working.

**ACTION: revert to shipped behaviour; the loop returns to baseline.** The remaining move is the one
named in F30's pre-registration — **a signed ESDF**, giving a smooth, correctly-oriented, correctly-
scaled interior gradient with nothing left to tune. Both escape parameters stay in the image, off.

**Caveat on the strength of this verdict:** one flight per dose. What is solid is the *guard breach*
(coverage ~0.5× on both, missing 0.85× by a wide margin) and the *substitution-count contrast*, which
is a within-flight measurement over thousands of solves and does not depend on n.

---

## F31 — signed ESDF: designed, deliberately NOT implemented tonight

**The remaining move after F29/F30**, and the reason it is being written down instead of built.

**The design is settled.** `ESDF::updateLocalESDF` runs a three-pass distance transform seeded
`OCCUPIED ? 0 : max`, writing `map_data_` via a min-update. A signed field needs a second three-pass
with the seeds **inverted** (free voxels as sources), producing an interior distance `d_in`, and a
final combination `signed = d_out − d_in`. Outside an obstacle `d_in = 0` and nothing changes; inside,
`d_out = 0` so the value becomes `−d_in` — a smooth, correctly-oriented, correctly-scaled gradient
exactly where F29 and F30 had none. Both ESDF consumers benefit: the optimiser gets a usable escape
direction, and `hierarchical_grid.cpp` — which already guards against `esdf_grad.norm() > 1e-4` when
pushing cell centres away from obstacles — stops silently failing on centres that land inside.

**Why not tonight, stated plainly rather than dressed up.** This is the mapping hot path: the
transform runs over a bounding box on every update, and the change doubles it. It also has a real
correctness trap — the existing final pass is a **min**-update against `map_data_`, so a subtraction
pass must not double-apply if `updateLocalESDF` is ever called twice over the same box without a
reset. Getting that wrong corrupts the distance field feeding *both* the planner and the collision
checker, and it would not announce itself: it would look like worse flying.

Every FALCON change tonight was compile-verified in a throwaway container and checked in the image
before flying, and that was sufficient because each was **diagnostic or off-by-default**. This one is
neither — it alters values the whole planner reads — and it deserves a numeric verification against a
known obstacle geometry, not just a successful build. That is a fresh-context job, not a 2 a.m.
addition to a hot path.

**What is already true and does not need re-deriving:** B44's root cause is confirmed from source;
F29 showed pushing trapped points out *does* cut rejections (0.67×); F30 showed the crude estimator
has no workable dose (4023 substitutions at quarter strength, rejections *above* control). The signed
ESDF is the version of that fix with nothing left to tune, and the two escape parameters remain in the
image, off, if a comparison is ever wanted.

### F21 — pre-registration FINALISED 2026-09-04, before its first flight

Both candidate primaries were measured before flying, and both are badly under-powered as *continuous*
metrics:

| candidate primary | mean | CV | n/arm for a 50 % cut |
|---|---|---|---|
| `exploration.collision_rejects` | 45 | 44 % | 12 |
| seconds inside the 12 hazard cells | 48.8 s | **86 %** (7–127 s) | 46 |

**So F21 is not framed as a percentage change.** If seeding FALCON's blocked regions works, the
aircraft should stop entering those cells *at all* — time inside collapsing toward zero, not falling
by some fraction. That is a near-binary outcome and it is visible at n = 3 without statistics, which
is the only affordable way to ask this question.

- **MECHANISM CHECK (n = 3):** seconds inside the 12 seeded cells, expected **≈ 0** against a control
  spread of 7–127 s. If the candidate still spends tens of seconds in them, the seed is not reaching
  the planner or the blocked-region radius is too small — **void, not negative.**
- **THEN, as secondaries in causal order:** `exploration.collision_rejects` →
  `contact_hold.frac_of_flight` → `coverage.gained_m3`. B38 predicts all three improve if the chain is
  right; the rejection count is where it would show first.
- **REVERT-IF:** `coverage.gained_m3` < 0.85 × control. Twelve permanently-blocked cells walling off
  part of the map is the real risk, and it is why `blocked_region_ttl_s` is a declared knob.

**Honest limit, recorded now:** at these CVs, a *partial* improvement in rejections or coverage will
not be resolvable at any n this loop can fund. F21 can establish that the aircraft avoids the cells,
and whether that visibly changes the outcome — it cannot measure a 20 % gain.

### F21 — first flown candidate, and a correction to my own mechanism model

| | candidate | interleaved control |
|---|---|---|
| seconds inside the 12 seeded cells | **20.9** | 78.9 (control spread 7–127) |
| `collision_rejects` | 53 | 55 |
| `coverage.gained_m3` | 2421 | 1138 |
| `contact_hold.total_s` | 0 | 0 |

The seed definitely reached the planner — FALCON's own log says **"Restored 12 blocked region(s)"**
and `blocked_region_ttl_s = 0`. So by the pre-registered rule ("tens of seconds still spent inside
means the seed did not arrive or the radius is too small — void, not negative") this would be void.

**But the rule itself was built on a misreading, and that is the finding.** `blocked_regions_` is
consumed by the *frontier finder*: it retires frontiers and viewpoints near a blocked point, so it
stops the planner **targeting** those cells. **It does not stop the aircraft transiting them** on the
way somewhere else. Time-inside-cells was therefore never going to collapse to zero, and I chose it
as the mechanism check without checking what the mechanism actually does — the same error as
expecting `obstacles_inflation` to inflate something.

**What that means for F21.** The experiment can still ask a real question — does not *targeting*
these cells reduce the losses they cause? — but "seconds inside" is the wrong instrument for it,
because most of that time is transit the mechanism cannot touch. The honest primary is whether
viewpoints stop being *selected* there, which is not currently logged.

**Nothing is concluded from these numbers.** 20.9 s against a control spread of 7–127 s is
uninformative, rejections are unchanged, and the coverage difference (2421 vs 1138) is one flight
against a metric with CV 26 %. **The mechanism model is what needs fixing before the experiment means
anything**, and that is a reading task, not a flying one.

### F21 — STOPPED: the mechanism now acts, and it cannot be measured with what is logged

Strike 2 was verified reaching the planner (`Restored 12 blocked region(s)`, `seed_strikes=2`), so
B45's fix landed. The result is still uninformative, and the reason is instrumental rather than
physical:

| | strike-2 candidate | control | baseline spread |
|---|---|---|---|
| dormant frontiers | 137 | 157 | **106–157** |
| `collision_rejects` | 66 | 53 | 36–55 |
| `coverage.gained_m3` | 2303 | 4069 | 1138–4069 |

**Every available signal is swamped.** Dormant frontiers occur 106–157 times per flight from ordinary
runtime blocking, so twelve extra seeded regions are invisible in that count whether they act or not.
Coverage spans 1138–4069 between control flights — a 3.6× range — so a single pair says nothing, and
the guard "breach" here is against a control that is itself the highest of the night.

**What F21 needed and never had: a direct count of viewpoints *selected* inside the seeded cells.**
That is the quantity the mechanism changes, and it is not logged. Time-inside-cells (my first choice)
measures mostly transit the mechanism cannot touch; dormant-frontier counts (my second) are dominated
by runtime blocking. Both were chosen without checking what the mechanism could move — the same error
twice, at opposite ends of the same experiment.

**ACTION: stop F21, return to baseline.** The seed, the TTL and the strike count all stay wired and
off. **B45 is the durable result** — seeded evidence entering at a strength the code designs to be
inert is a real defect, now fixed and configurable — and B33's dozen cells remain measured and saved.
What is missing is one log line in the viewpoint selector, which is a smaller job than any of tonight's
rebuilds and would make this experiment answerable.

---

## F32 — log the selected viewpoint (ungated instrumentation), and F21 re-armed

**Image `e016430a7bbc`; rollback `falcon-ros:pre-viewpoint-log`.** FALCON never recorded *where it
decided to go*. `exploration_manager.cpp` now emits `[sparx-vp] next_viewpoint (x, y, z) yaw` at the
selection site, throttled, ungated — this is useful on every flight, not only during an experiment,
and an entire class of question about **planner intent** was unanswerable without it.

**It makes F21 answerable for the first time.** The mechanism changes which viewpoints are *offered*,
so the measurable quantity is viewpoints **selected** inside the seeded cells: a working seed drives
that to zero while the control keeps choosing them. That is near-binary and readable in two flights,
against the 40+ that every noisy outcome metric here demands.

**F21 re-arms at strike 2** (B45's fix, verified reaching the planner) with this as its mechanism
check, replacing the two instruments that measured the wrong thing — time-inside-cells, which is
mostly transit the mechanism cannot touch, and dormant-frontier counts, which run 106–157 per flight
from ordinary runtime blocking.

**Nine verified rebuilds tonight**, each compile-checked in a throwaway container, each verified in
the finished image alongside every prior patch, each with a rollback tag. The four functional patches
now in the image — `next_viewpoint`, `seed_strikes`, `escape_gain`, `ctrlpt_min` — were all confirmed
present together after the last one.

### F21 — the mechanism works, seen for the first time on the right quantity

With F32's viewpoint log and B45's strike-2 fix both in place, the seed can finally be measured on
what it actually changes — **which viewpoints the planner selects**:

| run | arm | viewpoints selected | inside the 12 seeded cells |
|---|---|---|---|
| `012833Z` | **candidate (strike 2)** | 231 | **10 — 4.3 %** |
| `011856Z` | control | 251 | 29 — **11.6 %** |
| `011154Z` | candidate, **aborted cycle** | 107 | 0 — 0.0 % |

**Selection inside the hazard cells falls from 11.6 % to 4.3 %**, a ~63 % reduction, on the metric the
mechanism drives — after two earlier attempts measured quantities it cannot move (transit time,
dormant-frontier counts) and produced nothing.

**Three honest caveats.** One candidate–control pair, so this is a direction, not an effect size. The
0.0 % row is an **aborted cycle** — FALCON ran briefly during bring-up before the health check failed,
so its 107 selections are a short, unrepresentative sample and it is listed only for completeness, not
as evidence. And 4.3 % is not zero: blocked regions retire frontiers whose *average* lies near a seed,
while viewpoints are sampled at a radius from that average, so some selections still land inside — the
mechanism narrows the aperture rather than sealing it.

**What this establishes** is that the chain B33 → B45 → F21 is now fully connected and observable:
the cells were identified from 985 flights, the seed reaches the planner, the strike fix lets it act,
and the planner demonstrably stops choosing to go there. **Whether that converts into fewer
rejections, less wedging or more coverage is the downstream question** — and it needs the sample sizes
B39 says this loop cannot fund, which is the honest limit to state alongside the result.

### F21 — CORRECTION at n=2: the mechanism result does not hold

| run | arm | viewpoints selected | inside the seeded cells |
|---|---|---|---|
| `012833Z` | candidate | 231 | 10 — **4.3 %** |
| `014831Z` | candidate | 165 | 46 — **27.9 %** |
| `011856Z` | control | 251 | 29 — 11.6 % |
| `013834Z` | control | 247 | 20 — 8.1 % |

**Candidate median 16.1 % against a control median of 9.8 % — the effect is gone, and the sign has
flipped.** The previous entry called this "the mechanism works, seen for the first time on the right
quantity". **That was wrong**, and it was wrong in the way I have now been wrong four times tonight:
reading a single favourable pair as a result. The caveat was written ("a direction, not an effect
size") and I still led with the conclusion.

**What the data actually supports:** viewpoint selection inside these cells varies from 4 % to 28 %
between flights *within the same arm*, which is a spread far larger than any difference between arms
at this n. So even the mechanism check — chosen precisely because it should be near-binary — is
**not** near-binary. A working seed does not drive selection to zero, because blocked regions retire
frontiers by their *average* while viewpoints are sampled at a radius from it, and enough of that
radius lies outside the blocked disc to keep the cells reachable.

**So F21 remains unresolved, and now for a better-understood reason than before.** The instrument is
finally correct; the *quantity itself* is too noisy at this n, and the mechanism is leakier than its
design suggested. Continuing would need either a wider `blocked_region_radius` (so the retirement
covers the sampling annulus) or the sample sizes B39 says this loop cannot fund.

---

## F33 — apply the blocked radius the launch file already derived (1.5 → 2.75)

**PRE-REGISTERED 2026-09-04.** No rebuild: a launch argument.

**MECHANISM (B46, and it is not my derivation).** `nav_stack.launch` carries a 2026-09-01 note
deriving, from 41 flights, that a shadow must cover `candidate_rmax = 5.5 m` or the tour re-offers an
unreachable target forever — and specifying **2.75**, so strike 1 covers 2.75 m and strike ≥2
escalates to exactly 5.5 m. **The argument below that comment was never changed from 1.5**, which
reaches only 3.0 m at strike ≥2 and leaves frontiers 3.0–5.5 m from a shadow permanently un-retired.
That is precisely why F21's seeding loaded correctly, acted at strike 2, and still leaked.

The same note records what the shortfall costs: **91.9 % of time-with-no-moving-reference was
A\*-plan-fail flooding, 86 % of it one terminal lock on a single unreachable viewpoint — worst case
254 s and 17 925 plan fails on the same target**, with `sweepBlockedFrontiers` retiring zero clusters.

**PRIMARY: `exploration.plan_fail`** (control 21–47, CV to be read off the arms). Chosen because the
prior measurement names plan-fail flooding as the dominant symptom, and because it is the mechanism's
first-order effect: retiring an unreachable frontier stops the planner re-attempting it.
**WANT ≤ 0.6 ×.** n = 6 per arm, interim at 3.

**SECONDARY:** `exploration.no_path_fails` / `locked_s` (the terminal-lock signature the note
describes), then `coverage.gained_m3`.

**REVERT-IF:** `coverage.gained_m3` < 0.85 × control — a 2.75 m shadow retires frontiers a 1.5 m one
keeps, and over-retirement walls off explorable space. This is the real risk and the reason the value
is being flown rather than simply corrected.

**Note this is a *runtime* change, not only a seeding one:** every shadow the aircraft casts during a
flight gets the wider radius, which is where both the benefit and the risk live. F21's seeds are off
in both arms so the two effects are not confounded.

### F33 — INTERIM at n=3: the primary fails, and it was probably the wrong primary

| metric | candidate | control | ratio | |
|---|---|---|---|---|
| **`exploration.plan_fail` — PRIMARY** | 49 | 34 | **1.44×** | WANT ≤ 0.60× — **not met** |
| `exploration.no_path_fails` | 53 | 56 | 0.95× | flat |
| `exploration.locked_s` | 18.9 | 26.6 | **0.71×** | better |
| `coverage.gained_m3` | 2510 | 1978 | **1.27×** | better |
| `trace.motion.distance_m` | 226 | 231 | 0.98× | flat |

**The pre-registered primary is missed, and by the rules that is the verdict: F33 does not meet its
WANT.** No revert-if is breached, so nothing is harmed — but "not harmful" is not "adopt".

**Why I think the primary was mis-chosen — offered as a lesson, NOT as a rescue.** `plan_fail` counts
planning *attempts that fail*, and a wider shadow retires unreachable frontiers faster, which should
*increase* re-targeting and therefore the attempt count while improving the outcome. The metric
conflates "stuck retrying one target forever" with "healthily giving up and trying elsewhere" — and
the prior campaign's note describes the failure mode as the former (**one terminal lock, 254 s, 17 925
fails on the same target**), which is `locked_s`, not `plan_fail`. `locked_s` improved 0.71×.

**I am not switching primaries after seeing the data.** That is the one move that would invalidate
every verdict this ledger contains, and the temptation here — a failed primary beside two favourable
secondaries — is exactly the case the rule exists for. **F33 is recorded as not meeting its WANT.**
The `locked_s` and coverage observations are hypothesis-generating only, and if they are worth
pursuing they need their own pre-registration with `locked_s` named in advance and its CV measured
first.

**ACTION: stop at the interim.** The primary sits at 1.44× against a 0.60× target — three more pairs
will not close that, and the loop's time is better spent on a question whose primary was chosen
before the data existed.

**The `locked_s` follow-up is not fundable, established before spending a single flight on it.**
Measured over 25 control flights: mean 39.7 s, **CV 132 %**, values 12–232 s with a long tail (four
flights above 60 s, the rest 12–42).

| reduction to detect | n per arm |
|---|---|
| 30 % | 301 |
| 50 % | 109 |
| **70 %** | **55** |

So the metric that best matches the documented failure mode — terminal locking on one unreachable
viewpoint — is the *least* measurable thing in the campaign. Reformulating it as a rate does not
rescue it either: only 4 of 25 flights exceed 60 s, so a proportion test at that base rate needs a
comparable n.

**This is B39's wall in its sharpest form.** The quantity that matters is dominated by rare, severe
events, and rare severe events are exactly what between-flight comparison cannot resolve. Either a
within-flight paired design (B39) or an accumulating measure — total locked seconds across a long
unattended run, compared between configurations over hours rather than flights — is required, and
neither is a small piece of work.

**Recorded so the F33 line closes cleanly:** the primary failed, the appealing secondary cannot be
tested at any affordable n, and the honest position is that **the 2.75 m shadow is neither adopted nor
refuted** — it is untestable with the instruments and flight budget available. The value stays at its
shipped 1.5 with `SPARX_BLOCKED_RADIUS` wired if it is ever revisited.

---

## F34 — the within-flight paired design, built end to end (B39's answer)

Every outcome this campaign cares about is unaffordable between flights — voxels/metre 43 per arm for
15 %, coverage 48, collision exposure 134, and `locked_s` **55 per arm for a 70 % cut**. Tonight three
experiments died in the gap between "too small to see" and "nothing there". Nearly all of that
variance is *between* flights, so alternating the arm *inside* one flight makes each flight its own
matched pair and cancels the dominant term.

**Five pieces, all landed, all defaulted off:**

| piece | what it does |
|---|---|
| `~yaw_mode_poll_s` (follower) | re-reads the mode on a timer; 0 = read once = shipped |
| launch + config + `EXPECTED_ROSPARAMS` | plumbs and **verifies** that parameter |
| `paired_flipper.py` | alternates the parameter on a drift-free schedule |
| `runs/paired_run.sh` | waits for the campaign's own "handing over to FALCON" line, then flips for the flight |
| `paired_segments.py` | segments the flight and compares **adjacent** pairs |

**Two design decisions that make it valid rather than merely cheaper.** Segments are derived from the
flight's own trace (`gate.yaw_mode`, recorded per tick), so a failed parameter write cannot
desynchronise the analysis from what the aircraft actually did — it shows up as a missing segment
rather than a mislabelled one. And the statistic is the **median of per-pair ratios between
neighbouring segments**, never all-A pooled against all-B: voxels-per-metre decays 4.2× through a
flight regardless of arm (B10), so pooling would read the decay curve as an effect.

**A gap caught while wiring it, and worth recording.** `~yaw_mode_poll_s` existed in the node but was
never plumbed through the launch, so the flipper would have set the parameter, the node would never
have re-read it, and the run would have produced a plausible single-arm flight **labelled as paired**.
That is the fourth instance tonight of a mechanism that reports success and cannot act (B36, B45, B46
being the others), and the readback assertion is what forced the check.

**Status: built and verified statically, never flown.** The analyser correctly refuses a non-paired
flight ("only one distinct value — this is not a paired run"). First use should be a re-run of F24's
question, whose between-flight answer (1.14×, inconclusive) is already on record to compare against —
making it the design's own validation as well as a result.

---

## F35 — first paired run: F24's question, asked properly

**PRE-REGISTERED 2026-09-04.** The first use of F34, chosen because its between-flight answer is
already on record and can be compared against.

**THE QUESTION, unchanged from F24:** does the speed-scheduled yaw blend buy anything? Between
flights it returned **1.144×** on voxels-per-metre with a 90 % interval of [0.86, 1.38] — recorded as
*inconclusive, not null*, because the design could not resolve a 10–20 % effect.

**THE DESIGN:** one flight, `yaw_mode` alternating **blend / course every 60 s**, segments derived
from the flight's own trace, and the statistic the **median of per-pair ratios between adjacent
segments**.

**PRIMARY: `hdg_err_deg` per segment.** Deliberately *not* voxels-per-metre, which needs mapping
counters that do not resolve at 60 s granularity. Heading error is computed per tick, is the quantity
the blend directly manipulates, and — critically — is comparable between neighbouring segments in a
way it is not between flights.

**WHAT SUCCESS LOOKS LIKE, and it is a claim about the METHOD as much as the change:** a per-pair
ratio distribution tight enough to exclude 1.0 from far fewer flights than the 43-per-arm the
between-flight design needed. **If the paired ratios are as dispersed as the between-flight ones, the
variance is not between-flight after all** and B39's premise is wrong — which would be the more
valuable finding, and is why this is worth flying even though F24 is settled.

**GUARDS:** the run is void unless the trace shows **≥ 4 segments of ≥ 40 ticks in both modes** — a
flip that did not take, or a flight too short to alternate, must not be analysed as a paired result.
`trace.motion.distance_m` and `coverage.gained_m3` are watched but not decisive; a 60 s alternation
is expected to fly worse than either pure mode, and that is the price of the design, not a failure.
