# MISSION — Autonomous FALCON/Rooster Exploration Campaign

> **READ THIS FILE FIRST, EVERY TIME.** If you are an agent resuming this work with no
> memory of it, this file is your only source of truth about what you are doing and why.
> It is deliberately self-contained. Update it as you go; it is the campaign's durable state.

**Started:** 2026-08-18
**Operator:** away for several days. **Do not ask questions. Decide everything yourself.**
**Standing order:** never stop. Keep improving until explicitly told to stop.

---

## 1. The goal

Fly a **complete, autonomous FALCON exploration of the whole `sphera_jail` map** with the
ROBOTICAN Rooster in the Sphera simulator:

- maximum voxels mapped,
- minimum collisions,
- minimum time,
- **trajectory tracking as close to FALCON's plan as possible**,
- smooth, continuous flight — no stop/go stutter.

Everything else in this file serves that sentence.

## 2. The loop (run forever)

```
0. IMPROVE   — change code/params to fix the top-ranked problem
1. RESTART   — Sphera + every container + every node, from scratch
2. TIMER     — 10-minute flight window
3. FLY+LOG   — full FALCON exploration, logging everything (see §6)
4. ANALYZE   — find the single biggest remaining problem
5. GOTO 0
```

After each improvement: **verify it actually helped against the logged metrics**, and if it
did, `git commit` + `git push`. If it did not, revert or iterate. Record the outcome in §8.

## 3. Hard rules

- **SIMULATOR ONLY.** Every command targets the Sphera sim (`R1` container, ROS_DOMAIN_ID 9).
  Never issue arm/takeoff/land/movement to physical hardware. If `R1` is not a
  `sphera-backend:*` container, stop and do not fly.
- Never ask the operator anything. Never block waiting for input.
- Never leave the loop dead. If anything hangs, kill it and restart the cycle.
- **Every turn must end with a scheduled wakeup**, and a turn that sets `runs/PAUSE` must
  schedule the wakeup that will clear it. A forgotten sentinel plus an unscheduled turn cost
  13.5 hours of flying on 2026-08-18; the supervisor now auto-expires PAUSE after 30 minutes,
  but that is a backstop, not the plan.
- **Never edit this file by index-slicing between two headings.** A slice from one `###` to
  another silently deletes everything in between when a heading has since changed: on
  2026-08-19 that destroyed P6-P11 and they had to be recovered from git. Use anchored
  string replacement on a unique phrase, or append, and check `grep -n "^### " MISSION.md`
  afterwards.
- Never `pkill -f <pattern>` where the pattern matches the shell running it — it kills the
  cleanup itself, so whatever was meant to happen next never does. Match on a pid instead.
- Commit and push working improvements as you go. Small commits, clear messages.
- `core/` must stay **Python 3.8-compatible** (the FALCON Noetic container imports it).
- Keep inline code comments to 1–3 lines. Narrative goes in this file / LESSONS.md.

## 4. Operational facts (hard-won — do not re-derive)

| Thing | Fact |
|---|---|
| Sphera restart | `python3 -m sparx_agency.tools.sphera_battery_watchdog --once` → restarts Sphera **and** drives its GUI back into the scenario (assigns Rooster_1, clicks Play). Exit 0 = fresh `R1` confirmed. ~40 s. |
| Drone id | Everything is wired to **`R1`**. If `docker ps` shows `R2` and no `R1`, the wrong Rooster is assigned — run the restart above. |
| ROS domain | `ROS_DOMAIN_ID=9`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. **Check `echo $ROS_DOMAIN_ID` is not leaking into the host shell** — it has caused silent failures before. |
| Map | `sphera_jail` (NOT office/hospital). `run_falcon_sphera.sh sphera_jail`. |
| Exploration launch | `sphera_drone.launch map_name:=sphera_jail nav_mode:=exploration` |
| Follower choice | `exploration_follower:=reference` (default, traj_server + ReferenceTracker3D) or `:=bspline` (direct B-spline + `core/control/velocity_servo`). **Only one may run.** |
| Bridge staleness | `ros1_bridge` goes stale on **every** `falcon` or `R1` recreation. Always `docker rm -f ros1_bridge` + relaunch after. |
| Falcon map contamination | Any `R1` crash/recreation ⇒ **fully recreate the `falcon` container**. Its voxel map is long-lived with no decay; garbage fused during a disruption never clears. |
| Video freeze | `video_trigger.py` silently freezes on its last frame after `R1` is recreated — files keep appearing with identical bytes. The freshness watchdog (`/tmp/video_freshness_watchdog.sh`) must be running. |
| Takeoff vs twist adapter | `rooster_twist_control_adapter.py` publishes `{"action":"stop"}` at 20 Hz when `/cmd_vel` is quiet, which **cancels an in-progress takeoff**. It must NOT be running during arm/takeoff — start it only once hovering. |
| Exploration needs a hover | `exploration_node` has no route from a ground pose; started on the ground it spins logging `[FSM] Plan fail`. Always: arm → takeoff → hover settled → then exploration. |
| Altitude | `RoosterUnit` owns the `z` axis exclusively. The z response is a **~10-count step gate near 700**, not a thrust curve. Never let a second publisher touch `/R1/manual_control`. |
| Horizontal axes | ManualControl x/y are **dead below ~620 (x) / ~700 (y) counts**. Flying slower is *worse* here. |
| Ground truth | Validate against `/R1/sphera/state` only. PX4's own estimate drifts convincingly while the aircraft is motionless. |
| Sphera restart, programmatically | `bringup.restart_sphera()`. Success means **the battery reset**, not that a container exists — a still-running old `R1` satisfies "exists" while reporting 0%. |
| FALCON C++ patches | Verify with `patches/verify_patch.sh <patch>.sh` (~1 min) BEFORE a full `docker build` (~12 min). Applying cleanly is not compiling. |
| `it` container is ROS 2 **Foxy** | `ros2 topic echo --once` does not exist there. Use `timeout N ros2 topic echo <t> \| grep -m1 <field>`. |
| Battery endurance | A ~5 min flight drains 92% → 40%. A full 10-min window needs a near-full pack, so essentially every cycle restarts Sphera. |
| FCU stops connecting | The vendor `fcu_driver` inside `R1` spams `Failed to execute: Not connected to FCU` and `RoosterState.armable` stays false, while PX4 sits at ~0.3 % CPU (a healthy SITL runs hot). **A charged battery does not mean a flyable drone.** The only known remedy is a Sphera restart, and `ensure_sphera` now triggers one on `armable == false` as well as on a flat pack. |
| Duplicate publishers | Before blaming code, run `ros2 topic info /R1/manual_control --verbose` and `/R1/keep_alive --verbose` and confirm exactly **one** publisher. |

## 5. Campaign harness

Lives in `sparx_agency/tools/falcon_campaign/`.

| File | Job |
|---|---|
| `campaign.py` | one full cycle: restart → bring up → arm/takeoff → exploration → log → land → teardown |
| `bringup.py` | ordered, health-checked bring-up of every container/node |
| `recorder.py` | the flight recorder (§6). Runs inside the **`it`** container -- it is the only one with the vendor message types for ground truth and the rangefinder. Its output is copied out afterwards. |
| `analyze.py` | post-flight metrics + ranked findings |
| `supervisor.sh` | the never-dying outer loop; survives reboot via cron `@reboot` |

Run artifacts: `runs/<UTC timestamp>/` — `truth.jsonl`, `metrics.json`, `findings.md`, `summary.json`,
plus copies of every relevant log.

**Pause sentinel:** `runs/PAUSE` — while this file exists the supervisor finishes the current
flight and then waits. Touch it before editing code, remove it after. The supervisor also
`py_compile`s changed modules before each run and refuses to fly on a syntax error.

## 6. What every flight must log

Sampled at 20 Hz into `truth.jsonl`:

- **Truth**: position, velocity, yaw, roll/pitch from `/R1/sphera/state`.
- **FALCON's plan**: `/planning/bspline` (traj id, knots, start time), `/planning/pos_cmd`,
  `/planning/replan` verdicts, FSM state transitions.
- **The follower**: commanded body twist, position error, cross-track error, along-track lag,
  yaw error, `holding` / `diverged` / `saturated` flags.
- **The actuator**: the axis values actually published (`/R1/cmd_nav`), the achieved velocity,
  and the ratio between them.
- **The map**: voxel/frontier counts over time, explored volume.
- **Events**: every stop, every reflex trip, every replan, every mode change, with timestamps.

Metrics computed per run: mean/max cross-track, mean commanded-vs-achieved speed ratio,
number and duration of full stops, % of flight time moving, voxels mapped, distance flown,
collisions, whether exploration completed or froze.

## 7. Problems — the live queue

Ranked. **Delete a problem when it is fixed and verified in flight. Add new ones as found.**

### P1 — Stop/go stutter (SOLVED 2026-08-18, verified in flight)

| metric | baseline | after fixes |
|---|---|---|
| stops per minute | 12.6 | **0.49** |
| time below 0.05 m/s | 17 % | **1.1 %** |
| tracking error | sawtooth to 4.2 m | mean 0.40 m, p90 0.47 m |
| ticks past the 85° align gate | 28 % | **6.3 %** |
| speed-taper warnings | 332 | 4 |

What did it: yaw cap 45→90°/s, `force_mode=none` + `min_vxy=0` (a duplicate dead-band
quantiser that turned partial commands into exact zeros), a forward-speed floor through
turns, the measured-speed backstop raised 0.7→1.5 m/s with its taper wired, ramped axis
releases, and a warm-start across the axis dead band. Keep watching `stops_per_min` and
`frac_time_below_stop_speed` in every run's metrics — this is the headline smoothness pair.

### P2 — Exploration stops permanently (SOLVED 2026-08-18, verified in flight)

Two root causes, found in sequence. First a **stale docker image**: `falcon-ros:noetic` was
built 3 h 19 m before the commit adding the finish-grace fix, so ten committed C++ patches
were silently inert and the FSM quit after 26 s. Second, and deeper: `FINISH` was an
**absorbing state** with no exit anywhere in `exploration_fsm.cpp`, and entering it published
`replan == 2`, which sets `task_finished_` and **ends the traj_server process** — so the
follower watched its last setpoint go stale and station-kept for the rest of the flight.

Fixed by `patches/fix_falcon_finish_reopen.sh` (FINISH re-opens to PLAN_TRAJ when frontiers
reappear; type 2 stops the trajectory without stopping the server).

| metric | before | after |
|---|---|---|
| `frac_holding` | **0.84** | **0.031** |
| `traj_server_exited` | true | **false** |
| `reopened` | 0 | **2** |
| distance flown | 252.6 m | **298.9 m** |
| finished at | 107 s (terminal) | 188 s, then re-opened |

The aircraft now actually tracks FALCON for the whole window instead of holding station for
84 % of it. `bringup.assert_falcon_patches()` refuses to fly an image missing either half,
checked at **every** bring-up, and the analyzer reports `reopened` / `traj_server_exited`
itself so this never needs grepping again.

### P3 — Walls not mapped high enough (FIXED 2026-08-18)
The walls *were* mapped; they were never *published*. `vbox_max_z = 2.8` guillotined every
occupancy cloud — live proof: cloud zmax exactly 2.750 m with the **top bin the densest**
(a real wall end would taper). Room ceiling is ~3.4 m.
→ `visualisation` z-max 2.8 → 3.6, `vertical_extent` [-2,4] → [-2,5] (required together or
`exploration_node` crashes), RViz ramp/voxel size, `bev_z_ceil` 1.50 → 2.20.
Still open: `raycast_max = 5.0` carves anything beyond 5 m as free (needs a Dockerfile
patch, no repo-side override exists); `fy=180` implies vfov 90° but a live fit says ~95°,
so mapped heights are ~8–10 % short.

### P4 — Fly lower (HEIGHT ACHIEVED 2026-08-19; premise RETRACTED)

**Current state, superseding the 2026-08-18 notes below.** The height is achieved by biasing
the SETPOINT, not by raising the descent gain: `MAX_RANGER_M` 1.35 → 1.00 and
`TARGET_RANGER_M` 1.20 → 0.90 hold ~1.21-1.25 m, exploiting the loop's steady ~0.22 m offset.
Raising `altitude_hold_kp_down` to 1500 also worked on altitude but cost horizontal speed and
was reverted (see P11). Bounding the nudge (`altitude_nudge_m` 0.15, `altitude_band_m` 0.3)
stopped the live target railing to its floor.

**The reason for flying lower is NOT established, and an earlier claim that it was is
retracted.** Distinct 2 m cells of the track over the first 430 s: 12/14/29 at ~1.58 m against
**41/21/19** at ~1.21 m. The 41 was a single lucky run and the repeats sit inside the
high-cruise range; low-cruise coverage rate is if anything worse (85/70/51 against 87).
Kept anyway on independent grounds: 1.2 m matches FALCON's own `cruise_z` of 1.0 and sits
inside the BEV trust band (0.70-2.20), where 1.58 m did not.

#### Original 2026-08-18 analysis (kept for the mechanism)
The drone was **not** flying at its commanded 1.6 m — measured median cruise 1.91–2.11 m
across nine runs. Two causes: ~0.35 m climb overshoot, then **no descent authority** —
the z axis is a *three-zone* actuator (≥700 climbs hard, ~400–690 does nothing, ≤400
descends weakly) and at a symmetric kp=500 the loop only reaches the descend zone at
−0.6 m of error.
→ new `altitude_hold_kp_down` (900), `target_ranger_m` 1.20, `max_ranger_m` 1.35.
**Do NOT lower `hover_z`/`climb_z` to fly lower** — they are throttle counts, and below
~700 the aircraft does not leave the ground at all.
Also worth doing: `obstacles_inflation`/`safe_distance` are 0.85 m against 0.20 m voxels,
which needs a 1.7 m-wide free corridor and makes a ~0.9 m doorway unplannable at *any*
altitude. Pass-through args are now wired; try 0.40 (the 2D planner's own value).

### P5 — Axis calibration (blocks i, ii, iii all flown)

**Yaw is now calibrated for the first time** (block iii): dead band **102** counts, **2.589
rad/s** at full stick, and symmetric — r+ 104/2.580 over 7 points, r- 101/2.597 over 8. That
retires the 2026-07-30 claim of a 1.9-vs-1.5 rad/s left/right asymmetry.

Yaw had never had a calibrated inverse: it used `wz / max_yaw_rate * 1000`, a through-origin
scale with **no dead band**, which is the exact form this module's own docstring says cannot
work for a dead-banded axis. Consequences either side of the crossover:

| requested | old axis | measured-curve axis | effect |
|---|---|---|---|
| 0.14 rad/s (the follower's snap floor) | **78** | 151 | **below the 102 dead band — no yaw at all** |
| 0.50 | 278 | 275 | coincidentally right |
| 1.00 | 556 | 449 | over-commands ~24 % |
| 1.50 | 833 | 622 | over-commands ~34 % |

So small heading corrections did nothing and large ones overshot. This matters during
exploration because the follower turns constantly in course mode and heading error feeds back
into forward speed through `cos(heading_err)`. Both halves of the curve are applied together.
UNVERIFIED in flight.

**Cross-axis ratio remains UNMEASURED.** 16 of 39 block (iii) segments aborted on the guards:
combined x+r at 700-800 counts drove the aircraft to ranger 3.4-3.7 m, and several pairs
exceeded 25-30° of roll or pitch. Notably `x700_r0` — pure forward, no yaw — also hit ranger
3.7 m, so high forward stick alone climbs; this is not specific to combining axes. Getting the
ratio would need the sweep to hold altitude against much stronger disturbance than it can now.
Do not spend more flights on it until something else needs it.

### P5 item 4 — "Yaw disturbs altitude badly" (DOWNGRADED 2026-08-19, measured)

Block (i)'s yaw segments drove the aircraft to ranger 3.5-4.0 m, through the ~3.4 m ceiling,
with one 180° roll logged — which read as a serious flight defect. Measured across three
exploration runs (25 668 samples in the healthy first 430 s), the coupling in **normal flight
is small**:

| yaw command | mean vz | ranger median |
|---|---|---|
| r ≈ 0 | −0.0115 m/s | 1.19 m |
| \|r\| < 300 | −0.0024 | 1.22 |
| \|r\| < 600 | −0.0026 | 1.24 |
| \|r\| ≥ 600 | **+0.0086** | 1.24 |

The sign flips with hard yaw, so the coupling is real, but the altitude loop absorbs it: a 5 cm
difference in held height between no yaw and full yaw. The ceiling strikes came from
**sustained pure-yaw at high rate for seconds**, which the sweep commands and exploration never
does.

So this is a **calibration-sweep hazard, not an exploration defect**. Do not spend flight time
tuning it. It does mean the yaw sweep needs the aircraft protected before it can produce data —
its segments abort on the ranger limit by design, which is the guard working.

### P6 — Tracking error is 1.42 m now that the aircraft actually follows (OPEN, top)

Only visible once P2 was fixed: while the follower was holding station 84 % of the time its
`pos_err` looked excellent (mean 0.40 m) because it was tracking a stationary point. Flying
the real reference, cycle 11 measured **pos_err mean 1.42 m, p90 1.91 m, max 3.94 m**.

Almost certainly the same defect as P5: the platform delivers only **0.42–0.63×** the
commanded speed, so the aircraft falls progressively behind a reference that keeps moving.
Fix the gain first and re-measure before touching any controller gain — a position loop
tuned against a plant that under-delivers by half will be wrong twice over.

### P7 — Plan-fail rate (OPEN, watch)

Cycle 11: 1650 `[FSM] Plan fail` across 40860 FSM lines (4 %). Cycle 2 was 36/7788 (0.5 %),
but that comparison is unfair — cycle 2 spent 84 % of its flight in FINISH not planning at
all. Against cycle 1's 16949 it is a 10× improvement. Watch whether it tracks coverage rate;
if coverage keeps climbing, this is FALCON discarding unreachable viewpoints and is healthy.

### P8 — The airframe holds ~19 deg roll excursions while hovering still (OPEN)

Measured from the calibration sweep's own rest periods (3240 samples with nothing commanded
and speed < 0.05 m/s): |roll| p90 is **18.9 deg**, and it does **not decay** — 18.9 / 19.0 /
19.0 / 17.3 deg for 0-1 s, 1-2 s, 2-3 s and >3 s after the stop, while speed settles to
0.03 m/s. Median roll is only ~1 deg, so these are recurring excursions rather than a
standing bias.

Why it matters, in order:
1. **Map quality.** Depth comes from monocular DA3, and roll skews the geometry it infers.
   Roughly a tenth of all mapping frames are being captured at >=18 deg of roll.
2. **The tilt reflex is mis-set either way.** The follower's node default is 15 deg, which
   this would trip on ~10 % of ticks and cut drive spuriously; `nav_stack.launch` passes
   45 deg, which cannot fire before the airframe's ~35 deg recoverability ceiling. Neither
   number was chosen against this measurement.
3. Backward commands specifically tripped 26-33 deg during the sweep, and the stall-escape
   reflex commands -0.30 m/s backward — so the escape may be inducing the tilt that the
   tilt cutoff then reacts to.

**Most likely benign explanation to rule out first:** PX4 is in Position mode, so holding
station against drift *requires* tilting. 19 deg is a lot for that, but check it before
treating this as a control defect — compare rest-period roll with and without a commanded
position hold, and against `altitude_hold` activity, since the z axis is a narrow step gate
that slews every tick.

### P9 — Stall-escape reflex (SOLVED 2026-08-19, verified in flight)

The reflex fires on "asked for >0.15 m/s, measured <0.06 m/s for 3 s" — which describes an
aircraft pinned against geometry *and*, identically, one whose commanded axis sits under the
platform's effective threshold. So a gain problem presented as a permanent stall, the escape
repeated forever without restoring motion, and each attempt cost ~9.5 s including cooldown.
It also reverses, into the direction block (i) measured tripping 26–33° of roll.

A give-up budget (`escape_give_up_count` 4, re-armed by `escape_progress_sec` 5 s of real
motion) fixed it outright:

| | before (093835Z) | after (100012Z) | after (101301Z) |
|---|---|---|---|
| escapes | 38 | **2** | **2** |
| distance | 139.9 m | **348.1 m** | **357.1 m** |
| stops per minute | 9.35 | **1.08** | **0.40** |
| time below 0.05 m/s | 46.8 % | **5.0 %** | **2.1 %** |
| coverage rate | (unreliable) | 79.6 m³/min | 32.7 m³/min |

That is the best *verified* coverage rate the campaign has recorded (earlier reliable bests:
72.7, 73.5, 65.8 m³/min). It also explains most of the run-to-run variance in §7b: the spread
between 133.8 m and 390.1 m on "identical" configurations was largely escape count.

### P10 — Exploration ending at 13 % coverage with zero frontiers (SOLVED 2026-08-19)

The blocked-region blacklist retired frontiers permanently. **None of its parameters had ever
been set** in `nav_stack.launch`, so all ran at C++ defaults — and a shadow could only grow to
`blocked_region_radius_max` 3.5 m while viewpoints are sampled to `candidate_rmax` 5.5 m, so a
blacklisted viewpoint could never retire the frontier that produced it. The tour re-offered the
same unreachable target, struck it again each time, and the frontier set emptied. The valve
that un-retires frontiers, `finish_amnesty_max`, was capped at 2 uses per process.

| run | reopened | coverage plateau | frontiers | distance | % at zero |
|---|---|---|---|---|---|
| 111738Z (pre-fix) | **0** | **458 s** | **0** | 80 m | **76.9 %** |
| 115748Z | 5 | 0 s | 1104 | 290.2 m | 6.6 % |
| 121053Z | 2 | 0 s | 0 | 356.9 m | 2.9 % |

Set: shadow cap 6.0 m, escalation 6.0 m, TTL doubling capped at 1 (≤180 s, was ≤720 s),
amnesty cap 20. The signature to watch for a recurrence is `finished` **and** `reopened: 0`
**and** a large `coverage.plateau_s` — if that returns with these params live, the frontier
finder itself is failing to see reachable frontiers and the blacklist is the wrong place to
keep tuning.

### P11 — Horizontal speed collapse / full-stick lock-up (MECHANISM FOUND 2026-08-19)

Speed over the healthy first 430 s, by altitude configuration:

| config | runs | mean speed | distance |
|---|---|---|---|
| kp900 / step15 | 121053Z, 143251Z, 144550Z | 0.556, 0.449, 0.305 | 253, 194, 140 m |
| kp1500 / step15 | 133951Z, 135302Z | 0.341, 0.320 | 152, 140 m |
| kp1500 / step8 | 140625Z | **0.126** | **57 m** |

Reverting the gain recovered speed from 0.126 to 0.305-0.449, so the altitude loop **was** a
contributor — consistent with z and translation sharing thrust authority on this airframe.
But it is **not the whole story**: kp900 itself spans 0.305-0.556, so run-to-run variance is
large and the single kp1500/step8 run may also have been an outlier. Do not treat the causal
link as settled; treat it as "raising the descent gain is not free, and cost more than the
altitude accuracy was worth".

Still unexplained: in the worst run the adapter commanded a mean 1.13 m/s while the aircraft
achieved 0.038 (gain 0.008), with 632 frontier points available and coverage flat for 439 s.
Full stick, no motion. If a run like that recurs, check in order: the vendor stack
(`docker logs R1 | grep -c "Communication lost"`), then the paired moving curve (revert to
4.0 / 0.0), then whether the planner is routing into spaces the airframe cannot fit.

#### The lock-up mode, found 2026-08-19

Runs split into two modes, and the bad one is a closed loop rather than noise:

| run | commanded | achieved | gain | axis median | roll p90 | pitch p90 |
|---|---|---|---|---|---|---|
| 153811Z | 0.423 | 0.202 | 0.230 | 667 | 4.8° | 9.9° |
| **155112Z** | 1.082 | 0.076 | **0.021** | **1000** | — | — |
| **161250Z** | 1.065 | 0.055 | **0.008** | **1000** | 3.1° | **20.1°** |
| **162559Z** | 1.119 | 0.050 | **0.007** | **1000** | **26.7°** | **34.9°** |
| 163900Z | 0.315 | 0.166 | 0.258 | 643 | 3.4° | 7.1° |

The loop: the aircraft stops translating → the velocity servo's integrator winds to its limit
→ the axis saturates at 1000 → the airframe pitches 20-35° → a tilted thrust vector translates
even less → repeat. Healthy flight sits at 3-10°.

The follower has a tilt-cutoff reflex for exactly this, but `tilt_limit_deg` was **45°** and so
could never fire before the airframe was useless. Lowered to **25°**, clear of the ~19° p90
roll excursions measured while merely hovering (P8). UNVERIFIED.

**tilt_limit_deg 25 verified over two runs (2026-08-19).** The lock-up did not recur: axis
median 644-650 instead of 1000, gain 0.099-0.190 instead of 0.007-0.021, speed 0.345-0.392
against 0.160-0.196 in the degenerate runs and 0.309-0.324 at tilt45. The reflex fires on
genuine tilt (logged examples: `pitch=-26`, `roll=25`), 6-23 times per run, at a cost of more
interruptions (escapes 6-14, stops 1.36-2.48/min).

Do **not** raise it to 30 to reduce those interruptions: one degenerate run locked up at a
pitch p90 of only 20.1°, so 30 would not have caught it. If the interruption cost needs
reducing, attack the other half of the loop instead — `servo_max_correction` (350) is what lets
feedforward plus correction reach saturation; capping it near 200 would hold the axis around
800 and stop the integrator ever commanding full stick.

This was NOT the vendor FCU: no FCU errors, no stalled streams, ranger healthy at 1.22-1.28 m
throughout. Our own control chain commanded the aircraft into a corner with no reflex set
tightly enough to notice. If it recurs at 25°, the next lever is `servo_max_correction` (350),
which lets feedforward plus correction reach saturation.

### P12 — Saturation is where this platform misbehaves (fix applied 2026-08-19)

Measured over 34 000 samples of normal flight, binned by commanded forward axis:

| forward axis | vz p90 | pitch p90 |
|---|---|---|
| 620-750 | 0.016 m/s | 7.3° |
| 750-900 | 0.032 | 13.2° |
| **≥900** | **0.111** | **22.6°** |

Beyond ~900 counts the forward axis stops being a translation command and becomes a climb-and-
pitch command: seven times the vertical disturbance and three times the pitch. That is the same
condition as P11's lock-up, so the "high forward stick climbs" observation from calibration
block (iii) is **not** an independent defect — both are saturation.

Fixed with a ceiling: `max_forward_axis` 900.

**A wrong turn worth recording:** capping `servo_max_correction` 350 → 200 was tried first, on
the theory that the integrator was driving saturation. It does not work, and a two-line
simulation showed it before it ever flew — in the *standing* regime the feedforward alone is
802 counts at 0.6 m/s (dead band 620, full scale 1.25), so any correction saturates regardless
of its cap. Only a ceiling on the total helps. Check the arithmetic of a control change against
the actual curves before flying it.

### P13 — The escape give-up created the lock-up it was meant to avoid (fix applied 2026-08-19)

`max_forward_axis = 900` did **not** break the lock-up. One run pinned at exactly the new
ceiling: axis median **900**, gain **0.021**, speed 0.189 m/s. The integrator simply winds to
whatever ceiling exists, so a ceiling relocates saturation rather than preventing it.

Tracing back to what starts it: **P9's escape give-up budget suppresses the escape manoeuvre
but leaves the follower commanding full drive.** So a genuinely pinned aircraft stops trying to
free itself and instead pushes indefinitely — which is exactly the windup input. P9 fixed a real
problem (38 escapes consuming half a flight) and introduced this one.

Fixed: when the budget is spent, translation is held at zero as well. Yaw is deliberately left
alone so the aircraft can still turn and let FALCON replan from a different heading, and the
hold clears as soon as sustained motion returns.

**Verify:** `actuation.x.axis_counts.median` should stop sitting at the ceiling, gain should
stay in the 0.10-0.26 band rather than collapsing to ~0.02, and speed should hold above
0.30 m/s. If the aircraft now sits still for long stretches instead, the hold is too sticky and
`escape_progress_sec` (5 s) is the knob.

**Kept regardless:** `max_forward_axis = 900` bounds the worst pitch (p90 20.0 against 22.6 at
the ≥900 bin) even though it does not stop the lock-up.

### Standing objectives (never "done")
- Smoother flight, tighter tracking, fewer stops.
- Faster, more complete coverage; fewer collisions.
- Robustness to DA3 depth noise.
- Harness reliability: no hangs, no silent death, always recovering.

## 7b. How to judge a change (run-to-run variance is large)

Two consecutive runs with an **identical** configuration measured 133.8 m and 390.1 m flown,
and 4.72 vs 0.39 stops per minute. A single A/B pair therefore proves very little unless the
difference is large (the standing-start curve halved distance *and* took 42 % of the flight to
zero speed — that was outside the noise). Prefer:

- comparing **two or more runs per configuration**, not one;
- `coverage.rate_m3_per_min` only when `coverage.reliable` is true;
- the direction of several metrics agreeing, rather than one moving.

**Which metrics are actually stable** (measured across the two 2026-08-19 escape-fix runs,
same configuration): distance 348.1 / 357.1 m, stops 1.08 / 0.40 per min, time at zero
5.0 / 2.1 %, escapes 2 / 2 — tight. But coverage rate was **79.6 / 32.7 m³/min**, a 2.4×
spread on *reliable* traces. That is not noise in the metric; covering new volume depends on
which frontier the planner picks and how much of the route is already mapped, so it varies
legitimately run to run.

So: **judge control changes on the motion metrics** (they now discriminate well), and treat
coverage as the goal to maximise over *many* runs rather than as an A/B discriminator.

## 8. Log of changes and their measured effect

_Append one line per change: date — what changed — measured effect — commit._

| Date | Change | Effect | Commit |
|---|---|---|---|
| 2026-08-18 | Campaign harness + mission file | the loop exists | cb2e38fc |
| 2026-08-18 | Rebuilt the falcon image (10 patches were inert) | exploration stopped quitting at 26 s | cb2e38fc |
| 2026-08-18 | P1 batch: yaw cap 90°/s, force_mode=none, turn creep, taper, ramped release | stops 12.6→3.4/min | cb2e38fc |
| 2026-08-18 | obstacles_inflation/safe_distance 0.85→0.40 | plan-fail 16949→36; stops 3.4→**0.49**/min, time-at-zero 18%→**1.1%** | f6cf12a0 |
| 2026-08-18 | Axis dead-band warm start | dead-band ticks 25%→**4%** | f6cf12a0 |
| 2026-08-18 | Altitude setpoint clamp + descent gain | hover settles ~1.3 m instead of drifting to 2.0 m | f6cf12a0 |
| 2026-08-18 | FINISH re-open + traj_server survives finish | **pending verification** (was: 560 s of 610 s held) | 01e1b399 |
| 2026-08-18 | Battery gate fixed (one constant, unreadable ⇒ restart) | stops flying dead-battery cycles | 01e1b399 |
| 2026-08-18 | Analyzer: through-origin gain instead of ratio-of-small-numbers | corrected "2.18x too fast" → **0.63x too slow** | 01e1b399 |
| 2026-08-18 | `patches/verify_patch.sh` | compiles a patch in ~1 min instead of failing a 12 min build | 01e1b399 |
| 2026-08-18 | FINISH re-open + traj_server survives (verified) | **frac_holding 0.84 → 0.031**, distance 252 → 299 m, reopened ×2 | 01e1b399 |
| 2026-08-18 | Fixed the Sphera restart (bare import; verify battery not container) | ended a 6-cycle stall where every restart silently no-oped | (this iter) |
| 2026-08-18 | Analyzer reports finish/traj_server verdict | the loop self-diagnoses this class now | (this iter) |
| 2026-08-18 | Coverage metric (`map_coverage` + frontier count) | the mission is now measured, not proxied | (this iter) |
| 2026-08-19 | PAUSE self-expiry + wakeup discipline | ended a 13.5 h idle caused by a forgotten sentinel | 2026-08-19 |
| 2026-08-19 | Applied block (i) standing-start curve (466/1.313) | **REVERTED** — distance 257→126 m, stops 1.3→8.7/min, 42% at zero | 2026-08-19 |
| 2026-08-19 | Altitude error now reported against the live target | removed a misleading top-3 finding | 2026-08-19 |
| 2026-08-19 | Coverage trace marked unreliable when it does not span the flight | caught a phantom "record" 90.7 m3/min from 110 s of a 600 s flight | 2026-08-19 |
| 2026-08-19 | Stall-escape give-up budget | caps a reflex that consumed ~half of two flights (38 and 14 escapes) | 2026-08-19 |
| 2026-08-19 | Block (ii) flown; x_v_full_moving 4.0 → **1.847** (measured) | standing-vs-moving quantified: moving is ~1.5x more responsive, ~100-count hysteresis | 2026-08-19 |
| 2026-08-19 | x_v_full_moving 1.847 | **REVERTED** — mixed the moving slope with the standing offset; peak speed 2.1 → 3.5 m/s, 30 % of flight stopped | 2026-08-19 |
| 2026-08-19 | `deadzone_moving` added (default off) + 2 tests | makes the consistent (412, 1.847) pair possible to try next | 2026-08-19 |
| 2026-08-19 | Dead-band FLOOR now uses the regime's own dead band | it clamped moving commands back to the standing 620, which would have silently defeated the paired test | 2026-08-19 |
| 2026-08-19 | Paired moving curve (412, 1.847) | **KEPT** — equal on distance/stops, better peak speed (1.46-2.14 vs 1.64-5.43 m/s) and time at zero | 2026-08-19 |
| 2026-08-19 | Blocked-region blacklist params finally set | shadow cap 3.5 → 6.0 m (was below candidate_rmax 5.5), TTL doubling 3 → 1, amnesty cap 2 → 20 | 2026-08-19 |
| 2026-08-19 | Blacklist fix **verified** over two runs | plateau 458 s → 0 s, reopened 0 → 2-5, distance 80 → 290-357 m | 2026-08-19 |
| 2026-08-19 | altitude_hold_kp_down 900 → 1500 | **KEPT, 3 runs**: converged True every time (was False), in-band 3.8 s → 19-45 s, ranger median 1.60 → 1.27-1.50 around a 1.35 target. But BIMODAL — see next row | 2026-08-19 |
| 2026-08-19 | altitude_hold_max_step 15 → 8 | calmed z (sd 52, ranger sd 0.139) but horizontal speed fell again | 2026-08-19 |
| 2026-08-19 | Altitude **reverted** to kp900 / step15 | causal test: speed fell 0.556 → 0.341 → 0.320 → 0.126 m/s tracking the altitude changes, with a CALM z trace in the worst run | 2026-08-19 |
| 2026-08-19 | FLIGHT_SECONDS 600 → 430 | battery hits 25 % at ~430 s and 0 by the end; the last ~170 s contributed 0.9-2.0 m at 0.003-0.009 m/s | 2026-08-19 |
| 2026-08-19 | Altitude revert **verified**: speed 0.126 → 0.305-0.449 | the gain raise was a real cost; the altitude loop is also its most stable ever (ranger sd 0.065) | 2026-08-19 |
| 2026-08-19 | P4 by setpoint instead of gain: MAX_RANGER_M 1.35 → 1.00 | **WORKED** — held ranger 1.58 → 1.21 m, speed 0.483 (no cost), and 2 m-cells reached 12-29 → 41 | 2026-08-19 |
| 2026-08-19 | Nudge authority bounded: nudge 0.3 → 0.15 m, band 1.0 → 0.3 m | **WORKED** — target now holds 0.70-1.00 instead of railing to 0.60; z sd 114 → 77, ranger sd 0.204 → 0.132, escapes 11 → 4 | 2026-08-19 |
| 2026-08-19 | P4 premise **RETRACTED** | the 41-cell low-cruise run was noise: repeats gave 21 and 19, inside the 12-29 high-cruise range | 2026-08-19 |
| 2026-08-19 | Block (iii) flown; yaw curve measured and applied (dead band 102, 2.589 rad/s) | **UNVERIFIED** — yaw had no dead-band compensation at all; 0.14 rad/s requests were producing nothing | 2026-08-19 |
| 2026-08-19 | Failed cycles no longer inherit the previous flight's telemetry | two takeoff failures had reported the last good flight's 336 m and 12309 samples as their own | 2026-08-19 |
| 2026-08-19 | Wait for `RoosterState.armable` before arming | "Arm refused: Not connected to FCU" cost two whole cycles to a startup race | 2026-08-19 |
| 2026-08-19 | Restart Sphera when the FCU is unarmable, and gate health on it | six cycles were lost re-attempting a dead aircraft, because the battery read 0.99 so nothing ever restarted | 2026-08-19 |

## 9. Resuming after a context loss

1. Read this file top to bottom.
2. `cd /home/user1/GIT/TheAgency && git log --oneline -15` — see what was already done.
3. `ls -t runs/ | head` — read the newest `findings.md` and `metrics.json`.
4. Check the supervisor is alive: `pgrep -af falcon_campaign` / `tail runs/supervisor.log`.
   If dead: `nohup bash sparx_agency/tools/falcon_campaign/supervisor.sh > runs/supervisor.log 2>&1 & disown`
5. Continue at §2 step 0 with the top-ranked problem in §7.
