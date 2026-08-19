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

### P4 — Fly lower (FIXED 2026-08-18, awaiting verification)
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

### P5 — Axis calibration (MEASURED; blocks i+ii flown, iii still owed)

**The standing-vs-moving question is answered.** Block (ii) pre-loads the axis and steps to
each value, so it measures the regime the aircraft is actually in for most of a flight:

| regime | dead band | m/s at full stick |
|---|---|---|
| standing start (block i) | 466–620 | 1.15–1.31 |
| moving, approached from an 850 pre-load | **412** | **1.847** |
| moving, approached from a 650 pre-load | **511** | **2.467** |

Two things follow. The moving regime is roughly **1.5× more responsive** than a standing
start, and the ~100-count spread between the two approaches is a real **hysteresis band** —
the same axis value means a different speed depending on which side you came from. Nothing in
the stack modelled either before.

`x_v_full_moving` is now **1.847** (was a guessed 4.0, which commanded far too little stick).
Only that one number was adopted: the measured moving dead band of 412 is deliberately NOT
applied, because lowering the dead band 620 → 466 was already flown and halved the distance.
A number measured out of regime is not evidence about the regime you fly in — that is the
whole lesson of the block (i) revert.

**Still owed:**
1. **Block (iii)** — combined x+y / x+r / x+y+r. Nothing compensates cross-axis dead bands; a
   diagonal pays the offset twice.
2. **A moving-regime dead-band trial**, one variable at a time, once the 1.847 change is
   verified: the evidence says 412–511, the flight says 620 works. That gap is unexplained.
3. **The yaw sweep still has not run.** Its block (i) segments drove the aircraft to ranger
   3.5–4.0 m — through the ~3.4 m ceiling — with one 180° roll logged. Yaw commands disturb
   altitude badly, and that is a flight-behaviour finding, not a test-rig problem.

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

## 9. Resuming after a context loss

1. Read this file top to bottom.
2. `cd /home/user1/GIT/TheAgency && git log --oneline -15` — see what was already done.
3. `ls -t runs/ | head` — read the newest `findings.md` and `metrics.json`.
4. Check the supervisor is alive: `pgrep -af falcon_campaign` / `tail runs/supervisor.log`.
   If dead: `nohup bash sparx_agency/tools/falcon_campaign/supervisor.sh > runs/supervisor.log 2>&1 & disown`
5. Continue at §2 step 0 with the top-ranked problem in §7.
