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
| Duplicate publishers | Before blaming code, run `ros2 topic info /R1/manual_control --verbose` and `/R1/keep_alive --verbose` and confirm exactly **one** publisher. |

## 5. Campaign harness

Lives in `sparx_agency/tools/falcon_campaign/`.

| File | Job |
|---|---|
| `campaign.py` | one full cycle: restart → bring up → arm/takeoff → exploration → log → land → teardown |
| `bringup.py` | ordered, health-checked bring-up of every container/node |
| `telemetry.py` | the flight recorder (§6) |
| `analyze.py` | post-flight metrics + ranked findings |
| `supervisor.sh` | the never-dying outer loop; survives reboot via cron `@reboot` |

Run artifacts: `runs/<UTC timestamp>/` — `telemetry.jsonl`, `metrics.json`, `findings.md`,
plus copies of every relevant log.

**Pause sentinel:** `runs/PAUSE` — while this file exists the supervisor finishes the current
flight and then waits. Touch it before editing code, remove it after. The supervisor also
`py_compile`s changed modules before each run and refuses to fly on a syntax error.

## 6. What every flight must log

Sampled at ≥10 Hz into `telemetry.jsonl`:

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

### P1 — Stop/go stutter (OPEN, top priority)
Flight is: move, hard stop, move, hard stop. No smooth tracking.
_Investigation in progress — see §8 for findings._

### P2 — FALCON stops replanning after a "recovery"/circling episode (OPEN)
Drone circles, then FALCON freezes and never plans again.

### P3 — Walls not mapped high enough (OPEN)
Voxel map truncates wall height.

### P4 — Fly lower (OPEN)
Doors into rooms are lower than the current cruise altitude; the drone cannot enter.

### P5 — Axis calibration is incomplete (OPEN)
Needs: standing-start vs in-motion response, and multi-axis combined deflection
(commanding x+y+r together must not over-drive the vector magnitude).

### Standing objectives (never "done")
- Smoother flight, tighter tracking, fewer stops.
- Faster, more complete coverage; fewer collisions.
- Robustness to DA3 depth noise.
- Harness reliability: no hangs, no silent death, always recovering.

## 8. Log of changes and their measured effect

_Append one line per change: date — what changed — measured effect — commit._

| Date | Change | Effect | Commit |
|---|---|---|---|
| 2026-08-18 | Campaign harness + mission file created | — | (pending) |

## 9. Resuming after a context loss

1. Read this file top to bottom.
2. `cd /home/user1/GIT/TheAgency && git log --oneline -15` — see what was already done.
3. `ls -t runs/ | head` — read the newest `findings.md` and `metrics.json`.
4. Check the supervisor is alive: `pgrep -af falcon_campaign` / `tail runs/supervisor.log`.
   If dead: `nohup bash sparx_agency/tools/falcon_campaign/supervisor.sh > runs/supervisor.log 2>&1 & disown`
5. Continue at §2 step 0 with the top-ranked problem in §7.
