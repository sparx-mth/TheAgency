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

### P2 — Exploration stops permanently (SECOND ROOT CAUSE FOUND 2026-08-18)

The stale-image fix was necessary but **not sufficient**. Measured on a clean 610 s run:
the follower tracked for its first **106 s**, FALCON logged "Exploration finished" once,
and the follower then reported `holding=True` for the remaining **560 s** — 84 % of the
flight.

Two things make FINISH permanent, and both are now patched
(`patches/fix_falcon_finish_reopen.sh`, wired into the Dockerfile):

1. **`FINISH` is an absorbing FSM state** — no transition out of it exists anywhere in
   `exploration_fsm.cpp`, while `frontierCallback` happily goes on finding frontiers *in*
   that state. It now re-opens to `PLAN_TRAJ` once frontiers reappear and a cooldown has
   passed, resetting the grace counters on the way out.
2. **`replan == 2` ends the traj_server PROCESS** (`task_finished_` breaks its main loop).
   So even a re-opened FSM would have had nothing to fly its plan. Type 2 now stops the
   trajectory without stopping the process; `/traj_server/exit_on_finish` restores upstream.

Diagnosis aid added: the follower heartbeat now prints `ref_age` and warns past 5 s.
`ref_ready=True` printed for the whole 560 s because it only reflects the last message's
flag — that is what sent the first diagnosis into the controller instead of the planner.

**Verify next run:** `grep -c "Re-opening exploration"` in the FSM log, `frac_holding` in
metrics.json (was 0.84), and that `traj_server` is still alive at the end of the flight.

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

### P5 — Axis calibration is incomplete (PARTLY FIXED, needs a calibration flight)
Done: a moving-regime full-scale, the servo no longer mutes the stick below the dead band
while motion is wanted, the integral survives brief zero demands, and the velocity it
closes on is now Sphera's **true physics velocity** instead of a 250 ms-filtered
differentiated position (that lag is why `servo_kp` had to be cut 220 → 90; it can now be
raised back).
Still to do:
- A dedicated calibration flight: standing-start breakaway per axis **per sign**, then
  steady-state gain while already moving (approached from above *and* below — the gap
  between those curves *is* the standing-vs-moving hysteresis), then combined x+y, x+r,
  x+y+r grids. Full experiment design is in the 2026-08-18 investigation output.
- No cross-axis compensation exists: the dead-band offset is added *per axis*, so a
  diagonal request pays it twice (`hypot(620,700)=935` counts for an infinitesimal speed).
  Currently masked by `max_lateral_axis=0`, but it will bite the moment lateral is enabled.
- Yaw has no calibrated inverse at all — still `wz/max_yaw_rate*1000`. Re-fitting the two
  logged points suggests a ~100-count dead band and ~2.55 rad/s full scale.
- `turn_coordination()` (a *measured* yaw/translation coupling law) already exists in this
  repo but is wired only into the XTEND followers, never the Rooster path.

### Standing objectives (never "done")
- Smoother flight, tighter tracking, fewer stops.
- Faster, more complete coverage; fewer collisions.
- Robustness to DA3 depth noise.
- Harness reliability: no hangs, no silent death, always recovering.

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

## 9. Resuming after a context loss

1. Read this file top to bottom.
2. `cd /home/user1/GIT/TheAgency && git log --oneline -15` — see what was already done.
3. `ls -t runs/ | head` — read the newest `findings.md` and `metrics.json`.
4. Check the supervisor is alive: `pgrep -af falcon_campaign` / `tail runs/supervisor.log`.
   If dead: `nohup bash sparx_agency/tools/falcon_campaign/supervisor.sh > runs/supervisor.log 2>&1 & disown`
5. Continue at §2 step 0 with the top-ranked problem in §7.
