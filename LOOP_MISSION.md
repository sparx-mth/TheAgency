# LOOP_MISSION — the standing order. READ THIS FIRST, EVERY TIME.

> **THIS FILE IS IMMUTABLE.** The operator instructed: *"Write in the MD file what your mission is
> so that you do not forget it as time passes. Do not change this file. This is the loop file and
> your mission."* Do not edit, reword, prune or "update" it. Everything that changes goes in
> `LOOP_BUGS.md` (problems) and `LOOP_FIXES.md` (changes + verdicts).

**Opened:** 2026-09-02
**Operator:** away, indefinitely. **Never ask a question. Never wait for an answer. Never stop.**
**Predecessor campaigns:** `MISSION.md` (2026-08-18 → 08-25) and `runs/AUTOLOOP_JOURNAL.md`
(08-31 → 09-02). Both are EVIDENCE, not orders. This file supersedes their standing orders.

---

## 1. The mission, in the operator's words

Improve **flight control and trajectory tracking** of FALCON's B-spline until it is as close to
perfect as this plant allows, and use that to explore the `sphera_jail` map as fast, as completely
and as stably as possible.

The stated complaints that define "not good enough" today:

1. The aircraft **leaves the planned route and gets very far from it**.
2. It **still collides with walls**.
3. It is **late** — it lags the reference in time, not only in space.

The stated goal: *"a flight that is as trackable and stable as possible; the ultimate goal is to do
exploration in the most stable way possible."* And: *"the best possible control for our conditions."*

## 2. The loop — run it forever

```
0. IMPROVE   — change code/params/FALCON patches to attack the top-ranked problem
1. RESTART   — Sphera + every container + every node, from scratch
2. TIMER     — a 10-minute flight window; after 10 minutes stop everything
3. FLY+LOG   — full FALCON exploration, logging EVERYTHING in §4
4. ANALYZE   — in depth: where can control/mapping be made better?
5. VERIFY    — did the change from step 0 actually help? (see §6 — never trust one flight)
6. FIX       — apply what step 4/5 concluded; record it in LOOP_FIXES.md
7. GOTO 0
```

**This loop never terminates.** Not when a result is good, not when a result is bad, not when a
question arises, not at the end of a context window. If context is lost, re-read this file and
resume at step 0. The only stop is an explicit instruction from the operator.

## 3. Standing permissions (granted explicitly by the operator)

- **Patch FALCON's C++ freely.** FALCON was written for a *non-physical* simulator with perfect
  trajectory tracking. We fly a *physical* simulator. Adjustments and fixes inside FALCON are
  expected and authorised. Read the paper and our existing patches before adding more.
- **Edit anything in the repo.**
- **Decide everything.** No confirmation, ever.

## 4. What every flight must log (superset of the old §6)

**Tracking / control**
- Truth from `/R1/sphera/state`: position, velocity, yaw, roll, pitch, and their rates.
- FALCON's plan: `/planning/bspline` (control points, knots, yaw points, start time, traj id),
  `/planning/pos_cmd` (position, velocity, acceleration, **yaw, yaw_dot**), replan verdicts,
  FSM state transitions.
- Derived error, **against the B-spline itself, not only against pos_cmd**: cross-track,
  along-track (signed — this is the *lateness*), vertical, yaw error, and the **time lag**
  (how many seconds behind the reference the aircraft actually is).
- The follower: commanded body twist, every flag (holding / diverged / saturated / reflex).
- The actuator: the axis counts published to `/R1/manual_control` (or `/R1/cmd_nav`), the achieved
  velocity, and the ratio between them, per axis.

**Mapping — the operator wants hard numbers here**
- Explored volume (m3) over time, and **2D mapped area (m2)** over time.
- **Time per m2** and **time per m3** (1x1x1 voxel), instantaneous and cumulative.
- **New voxels discovered per second over time** — the discovery-decay curve. Early flight
  discovers a lot; later it re-visits. That curve is a first-class deliverable.
- Counts of free / occupied / unknown cells, and the frontier count, over time.
- **Time budget:** how each flight second was spent — transiting, exploring, parked, replanning,
  planning-failed, escaping. This must sum to the flight duration.

**Everything is saved raw** so a visualisation can be built later. *Do not build the
visualisation yet — save the data for it.*

## 5. The engineering problems named by the operator

These are premises, not hypotheses to be re-derived:

1. **Joystick-only plant.** The drone is commanded through a simulated RC joystick, not thrust.
   The measured calibration is `sparx_agency/robots/ROBOTICAN/rooster_axis_curve.py`. It is
   **ACCURATE AND FROZEN — do not change it.** Its documentation: memory
   `project_rooster_manualcontrol_axis_calibration.md` and the Rooster Plant Ledger; raw data in
   `runs/` and `~/rqs_iai_ws/axiscal_logs/`. Improve everything *around* it.
2. **Yaw must be decoupled from motion.** Today the nose always points along the velocity. FALCON
   does not work that way — its viewpoints carry their own yaw and it plans yaw to aim the sensor
   (paper §V-B: inter-viewpoint cost is `max{t_pos, t_yaw}`; sensor is an 80x60 deg, 5 m frustum).
   **Build control that drives yaw and translation independently.** Mapping should get faster.
   Control gets harder: the body-frame drive vector depends on the nose direction, so the
   world→body rotation must use the *right* yaw at the *right* instant.
3. **No collision sensing.** Nothing tells us the aircraft hit a wall. It can be pressed against
   geometry, still being commanded forward, and not move. That state must be *inferred* —
   commanded velocity vs achieved velocity — and escaped.
4. **No depth camera.** Depth is DA3 monocular: noisy, biased, and late. Obstacles can enter the
   map in the wrong place or too late. Every control and mapping decision must survive that.

## 6. How to judge a change (this is where past sessions went wrong)

- **Never conclude from one flight.** Run-to-run spread on this platform is large: baseline
  coverage CV ~27 %, distance CV ~51 % (Finding G, `runs/AUTOLOOP_JOURNAL.md`). Detecting a 15 %
  coverage change at 80 % power needs **~52 flights per arm**; 25 % needs 19; 40 % needs 8.
- **Prefer low-variance primary metrics** — cross-track error, along-track lag, time lag,
  clearance, moving-reference fraction, stationary fraction. Treat coverage/volume as a *guard*,
  not the headline.
- **Pre-register.** Before flying a change, write down in `LOOP_FIXES.md`: the mechanism, the
  primary metric, the WANT threshold and the REVERT-IF threshold — and the control's own value
  for each. A criterion the control also meets is not a criterion.
- **Interleave arms** (A/B/A/B) rather than comparing to a baseline collected hours ago.
- **Verify the change reached the running system** before measuring it (`rosparam get`, `strings`
  on the binary, a log line). Several past "results" were measured on inert code.
- A window chosen *because* it looks bad is not evidence.

## 7. Hard operational rules (each one already cost real time)

- **SIMULATOR ONLY.** Everything targets Sphera (`R1`, `ROS_DOMAIN_ID=9`). If `R1` is not a
  `sphera-backend:*` container, stop and do not fly. Never command physical hardware.
- **Never edit a script while it is running.** `bash` reads by byte offset; this killed an
  unattended driver. Stop it, edit, restart — or write a new file and switch.
- **Never `pgrep -f` / `until ! pgrep -f` on a pattern that matches the waiting shell itself.**
  It can never exit. Wait on a PID or on a sentinel the driver writes.
- **Stop by PID, verify the PID is gone, only then clear a sentinel.** A racing `ps|grep` guard
  once started a second driver; two drivers ran for 3 h and the yield fell to ~25 %.
- **One driver at a time.** Check before starting one.
- `core/` must stay **Python 3.8-compatible** (the Noetic container imports it).
- **Dedupe runs by `truth.jsonl` MD5 before computing any statistic** — a failed cycle can copy
  the previous flight's telemetry and still write `ended: completed` (Finding J).
- Keep inline code comments to 1–3 lines. Narrative goes in the loop MDs.
- `git commit` locally as work lands; **do not `git push`**.

## 8. The three books

| file | contents | mutable? |
|---|---|---|
| `LOOP_MISSION.md` | this file — the mission and the loop | **NO** |
| `LOOP_BUGS.md` | every problem found: symptom, evidence, mechanism, status | yes — edit freely |
| `LOOP_FIXES.md` | every change: pre-registration, what flew, the verdict, kept or reverted | yes — edit freely |

Prior evidence worth reading before re-deriving anything:
`MISSION.md` (P1–P43), `runs/AUTOLOOP_JOURNAL.md` (Findings C/F/G/I/J/K, v3.0–v8.0 reverts),
`LESSONS.md`, and the reverted-knob comments in `sparx_agency/tools/falcon_campaign/config.py`.

## 9. Resuming after a context loss

1. Read this file.
2. Read `LOOP_BUGS.md` and `LOOP_FIXES.md` — top of each is the live queue.
3. `ls -t runs/ | head` — find the last flight; read its `summary.json` and `findings.md`.
4. Check whether a driver is alive (by PID/sentinel, not by a self-matching pgrep).
5. Re-enter the loop at step 0. Do not ask. Do not stop.
