# MISSION — Autonomous FALCON/Rooster Exploration Campaign

> **READ THIS FILE FIRST, EVERY TIME.** If you are an agent resuming this work with no
> memory of it, this file is your only source of truth about what you are doing and why.
> It is deliberately self-contained. Update it as you go; it is the campaign's durable state.

**Started:** 2026-08-18
**Operator:** away for several days. **Do not ask questions. Decide everything yourself.**
**Standing order:** never stop. Keep improving until explicitly told to stop.

---

## 0. Current state — read this, then section 3, then whatever the task needs

*This file is 1600+ lines because it records how every conclusion was reached. That history is
the point — it is what stops a later session re-running a refuted experiment — but you do not
need all of it to act. This section is the orientation; the rest is evidence.*

**Where it stands — measured over all 113 settled-configuration runs (2026-08-21 06:50):**

| | opening median | settled config, n=113 |
|---|---|---|
| final volume | 1060 m3 | **median 1614** (p25 1432, p75 1723, max **1873**) |
| coverage rate | 75 m3/min | **median 158** |
| stalled share of flight | 53 % | **median 17 %** |
| runs below 1300 m3 | — | **15 of 113 (13 %)** |

*These are the DISTRIBUTION, not the best runs. An early version of this summary claimed "a
repeatable ~1800 m3 / ~190 m3/min" and "4-10 % stalled"; those were the maxima and two flights
respectively. The honest picture is the p25/p75 spread — roughly a 20 % range between quartiles,
with about one run in eight landing below 1300.*

**The numbers have not moved as the sample tripled, and that is the useful result.** Successive
re-measurements: n=36 → 1586 / 15 %, n=54 → 1586 / 15 %, n=100 → 1608 / 14 %, n=113 → 1614 /
13 %. A configuration that reproduces its own distribution over a hundred-plus runs is a stronger
claim than any of the individual medians.

**No drift, judged in blocks of eight** (median, collapses):
`1588(1) 1557(1) 1657(1) 1574(1) 1400(3) 1576(1) 1641(0) 1477(0) 1710(0) 1561(1) 1581(2)
1689(2) 1613(2) 1699(0)`. The 1400 block with 3 collapses looked like decay and was not: at a
~15 % base rate a block of eight expects ~1.2 collapses, so 3 has roughly a 13 % chance on its
own, and the next block recovered to 1576. Equally, the newest block at 1699(0) is NOT evidence
of improvement — the previous high block (1710(0)) was followed by 1561(1). Small integers make
luck look like a trend in both directions, which is why drift is judged in blocks and why a
single block is never acted on. Best single flight: **1873 m3** (`052208Z`).

**Settled configuration — change nothing here without a pre-registered test:**
`raycast_max 8.0`, `cluster_min 50`, `safe_distance 0.55`, bspline distance weight `150`,
course slew `45 deg/s`, tilt `35/27` with hysteresis, `tracker_pos_kp 1.0`, pinned hold `4 s`
with 4/8/16/30 backoff, escape cooldown `4.0`, tour commit `0` (off), `max_vel 0.8`,
follower cap `1.0`.

**The one open question, now closed as "not known":** roughly one run in five collapses to
~900-1300 m3 instead of ~1800. Three have been dissected and have THREE DIFFERENT mechanisms
(P37, P38), and every candidate predictor has been tested and failed — circling (-0.09, n=14),
dz p90 (-0.12, n=74), tracking error (+0.20, n=15, pre-registered on fresh data). Detection is in
place even though prevention is not: the analyzer flags PARKED, CIRCLING and coverage-PLATEAU
with the next question attached. Do not re-open any of those three without new evidence.

**What NOT to do**, each closed with evidence in the sections below: planning speed (P34), the
coverage tour (P32), reducing contacts directly (P27), the axis dead band (P28), the escape
cooldown in either direction (P30/P31), the tracker position gain (P26), altitude setpoints as a
mean argument (P18), the dz correlation (P36), circling (P38). Do not raise the simulator's
battery capacity — it would improve the number without improving the system.

**The method that produced the gains, and matters more than any single fix:**
1. Measure the mechanism before changing anything — four of the biggest wins came from a
   measurement that contradicted the obvious story.
2. Write the WANT and the REVERT-IF down *before* flying a change, and measure the control's own
   value for each — a criterion the control case also meets is not a criterion.
3. Correlations: within ONE configuration, n >= 15. Five leads have died at smaller n.
4. A scan over many metrics generates a hypothesis; only fresh data can confirm it.
5. Verify a change reached the running system (`rosparam get`, or `strings` on the binary) before
   measuring whether it helped.

---

### Operator note — 26 local commits are waiting, nothing has been pushed (2026-08-20 22:30, count refreshed 2026-08-21 06:50)

Per your instruction, everything since `d5e3f2df` is committed locally and **nothing has been
pushed**. The branch `feat/falcon_exploration_sphera_nadav` is 26 commits ahead of its remote;
`c213b56b` ("give a freshly respawned FCU longer to become armable") is the last commit that is
on `origin`.

What those 26 contain, grouped:

* **Two flight-behaviour changes, both measured and both kept:** `max_vel` reverted to 0.8 after
  1.0 failed its pre-registered test, and a liveness guard that ends a cycle in which FALCON
  never starts exploring.
* **Four analysis/instrumentation additions:** the per-minute motion table, the circling and
  parked findings, the run configuration recorded into `metrics.json`, and `test_p38.py` — an
  analysis written before its data existed.
* **The rest are records**, including three corrections to my own earlier claims: the dz
  correlation withdrawn, the best-run headline replaced with the real distribution, and the
  collapse rate updated from 14 % to 18 %.

* **Seven added since (2026-08-21)**, all consolidation rather than new behaviour: the vertical
  reference-error analysis, the extreme-circling and hover-altitude watches, and the restored
  video freshness watchdog (a bring-up path that pointed at a file which did not exist in the
  repo).

Nothing here is speculative or half-finished: every behaviour change is either verified against a
measured control or reverted, and each commit message carries the evidence. Review at your
convenience; the loop keeps running and keeps committing locally either way.

### Watch — two consecutive extreme-circling collapses (2026-08-21 05:57)

`022010Z` (642 m3, stall 96 %) and `023024Z` (499 m3, stall 100 %) are the two worst genuine
flights of the campaign, back to back. Both are the known CIRCLING signature and neither is new
in kind — only in degree:

```
022010Z  span/min  36 14  3  5  4  4  4  4     (7 of 8 minutes inside a 3-5 m box)
023024Z  span/min  27  3  2  3  2  2  2  2     (7 of 8 minutes inside a 2-3 m box)
```

Both flew a normal 230-235 m; they simply flew it in a small box. The run immediately after was
1744 m3 with zero circling minutes, so nothing persisted.

**Followed up 2026-08-21 06:20:** nine reliable runs since, none with a single circling minute
(threshold: a minute above 300 deg/m while covering >5 m), none below 1286 m3. The pair was an
excursion, not a regime. The trigger stays armed but no longer expects to fire.

**No new hypothesis** — the signature is one of the four already known, and Priority 2's rule is
that only an unmatched signature reopens the question. Recorded because two consecutive extremes
is worth being able to date later if it becomes a pattern.

**One detector limitation, deliberately NOT changed:** the CIRCLING flag needs `span_m < 4.0`, so
`022010Z`'s 4-5 m minutes went uncounted — it flagged 2 of 7 bad minutes. Raising the threshold
to 6 m would catch them and would also flag 2 minutes of the healthy 1744 m3 run beside it, which
is a worse trade. The collapse itself is never missed (final volume and stall fraction both scream),
and the flag is a label rather than the alarm — so read the span SEQUENCE, not the count, when
judging severity.

### Watch — reference divergence, two runs 20x outside the tested range (2026-08-21 07:35)

`064835Z` (1261 m3, a collapse) logged **pos_err mean 17.6 m, median 23.1, max 45.4** over 496
heartbeats. The corpus median across 115 settled runs is **0.83 m** (p90 1.75). A median of 23 m
means more than half that flight was spent ~23 m from its reference: that is not tracking lag,
it is a reference somewhere else entirely. Its per-minute table fits — minutes 1-3 normal, then
minute 4 covering 83 m across a 55 m span (the aircraft crossing the map), then 6/20/13/5 m for
the rest, with `escapes: 24` and heading error averaging 35 deg.

**This does NOT contradict the closed tracking-error result.** That test (+0.20, n=15,
pre-registered on fresh data) covered ordinary variation, roughly 0.5-2 m. Nothing in it speaks
to a 20x outlier, because a correlation measured across a narrow range says nothing about a point
far outside it. Two different regimes, not two points on one scale.

**No claim, and deliberately no action.** Only **2 of 115** runs exceed 5 m mean (`064835Z` 17.6,
`051153Z` 5.28 — and the latter has median 1.2, so it is a brief excursion, not sustained
divergence like the former). n=2 against the campaign's own n>=15 rule. Both landed low (1261,
1322) and the median volume of the pair is 1292 against 1618 for the rest, which is suggestive
and nothing more; at n=2 that comparison has no power.

**What was done instead:** the analyzer now emits a `REFERENCE DIVERGENCE` finding above 5 m mean,
reporting mean/median/p90/max so sustained divergence is distinguishable from an excursion. This
is the same detection-without-prevention stance already taken for PARKED, CIRCLING and PLATEAU.

**Trigger:** a THIRD sustained case (mean > 5 m *and* median > 5 m) makes it worth an hour of
work — read the FSM log around the transit minute and establish whether the divergence preceded
the coverage stall or followed it. Waiting for n=15 is not an option here: at a 2 % base rate
that is ~750 cycles, about five days.

### Watch — hover altitude outlier at 1.74 m (2026-08-21 05:25, one occurrence)

One cycle hovered at **1.74 m** before handover. Across the last 60 cycles hover is min 1.01,
p25 1.12, **median 1.19**, p90 1.27 — so this is the single highest and well outside the band.
Everything else about the cycle looked normal.

Not acted on: one occurrence, no mechanism, and the altitude path is a known weak spot rather
than a suspected regression (P18 — the hold has almost no authority in the band it operates in,
so where the aircraft settles is largely aerodynamic).

**Watch condition:** if hover exceeds ~1.5 m again, investigate the altitude path — compare
`rooster_command_unit`'s target and `wanted_z` against P18's measurements, and check whether the
high-hover runs score differently. Flying 0.5 m higher changes what the camera sees, so it would
matter if it became common. A single outlier in 60 does not.

### Operator note — disk, and why the campaign did NOT act on it (2026-08-20 19:45)

`docker system df` reports **377 GB of reclaimable images and 26 GB of reclaimable build cache**,
with 77 dangling images — mostly layers orphaned by this campaign's own `falcon-ros:noetic`
rebuilds (13.2 GB each). The root filesystem is at 73 % with **486 GB free**, so nothing is at
risk and the flight loop is unaffected.

**Deliberately not pruned.** This machine hosts other people's work — 143 images, and containers
belonging to `detector_dev`, `R2` and others — so `docker image prune` is not a safe unattended
action taken on someone else's behalf. It is flagged here for the operator to decide.

The campaign's own footprint is small and bounded: `runs/` is 1.9 GB with logs kept for the
newest 30 runs only, and the depth frame directory is capped at 500 files (381 MB).

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
- Check whether a flight is IN PROGRESS before restarting the supervisor (`tail runs/
  supervisor.stdout.log` for "hover settled" without a matching "cycle ... completed"). Killing
  it mid-flight leaves the aircraft armed with nothing driving it until the next bring-up
  restarts Sphera; one cycle was thrown away that way on 2026-08-20. Wait for the cycle to end.
- Releasing `runs/supervisor.lock` needs the `sleep` child killed too — `fuser runs/
  supervisor.lock` names it. The parent dying does not free the fd.
- **COMMIT ONLY — DO NOT PUSH.** Operator instruction 2026-08-20: keep committing working
  improvements locally, small commits with clear messages, but run no `git push` at all. The
  branch stays local until the operator pushes it themselves. (Before this, the campaign pushed
  every commit to `origin/feat/falcon_exploration_sphera_nadav`; commits up to that point are
  already on the remote.)
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

**Video freshness watchdog (2026-08-21).** `video_trigger.py` keeps writing byte-identical
frames after its video session dies, so a file-mtime check reads healthy through a total freeze;
the watchdog hashes the newest frame instead and restarts the trigger after 3 identical ones.

**Correction, same day.** I first recorded "1550 restarts in three days, therefore load-bearing".
The count is real; the interpretation was wrong. Dating the log and aligning it to cycle
boundaries: **92 % of restarts land within 60 s AFTER a cycle start** (n=453 where cycle times
are known, against ~10 % expected by chance at a 620 s cadence), in a tight bimodal cluster at
+16 s and +33 s. Frames legitimately stop while the stack is torn down and brought back up, the
watchdog reads that as staleness, and restarts the trigger twice per cycle. Those are artifacts,
not repairs. The genuine mid-flight repair rate is the other ~8 %: **~38 repairs over 210 cycles,
about 0.18 per cycle.** Still useful, still worth keeping -- but an order of magnitude less
dramatic than the raw count suggested. *This is the file's own rule 4 catching me: a big number
generated the hypothesis, and only a test against cycle phase could confirm it.*

The restarts do NOT accumulate processes -- both the watchdog and `bringup` kill `video_trigger`
before starting it, verified live: exactly one process in `it`.

It was also running on borrowed luck. `bringup.start_video_watchdog()` spawned
`sparx_agency/robots/ROBOTICAN/video_freshness_watchdog.sh` -- a path that **did not exist in the
repo**. The campaign had a watchdog only because an instance started by hand out of `/tmp` on
2026-08-18 never died, so the `pgrep` check short-circuited and the spawn that would have failed
was never attempted. Fixed: the script is in the repo at that path, and `start_video_watchdog()`
now raises `BringupError` if it is missing. Note the running instance logs to
`/tmp/video_watchdog.log`, not the `/tmp/video_freshness_watchdog.log` a fresh spawn would use --
check both when looking for its history.

**Instrumentation check (2026-08-21).** `check_instrumentation.py` compares the last 10 runs
against the prior 50 and names any metric family that used to be produced and is not any more.
It exists because three probes have died silently in this campaign — a hardcoded epoch prefix
that matched nothing once wall time rolled over, `rosout` duplicating FSM lines, and coverage
gaps deleting the stalls they should have recorded — and each was found by accident, late. A dead
probe is worse than a loud failure because the number it feeds still looks plausible.

Run it after any change to the recorder or analyzer. **It only alarms on ABSENT or NULL, never on
empty**: `collapse_signature: []` means no failure shape matched and `data_gaps: []` means the
recording had no holes — both are the healthy case, and an earlier version of this check reported
them as failures because it conflated the two. Verified against synthetic runs that it catches a
family which stops, and stays quiet about one that was already broken. Current state: every
family still being produced.

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

## 6b. Where the campaign has got to (108 reliable runs, 2026-08-18 → 2026-08-20)

Median of the first ten reliable runs against the last ten:

| | first 10 | last 10 |
|---|---|---|
| coverage | 75.2 m3/min | **154.9** |
| final volume | 1060 m3 | **1596** |
| stalled share of flight | 53 % | **9 %** |

Coverage per minute has doubled, volume per flight is up half again, and the share of each
flight spent gaining nothing has fallen sixfold. Block medians, ten runs each, show where it
came from — the step at 08-19 22:32 is the course-slew fix (P17), and the one at 08-20 00:19 is
raycast_max (P22) with the clearance weight (P24) following it:

```
08-18 18:51   75.2 m3/min   1060 m3   53 % stalled
08-19 14:06   61.0           910      70 %
08-19 20:49  111.4          1120      41 %
08-19 22:32  118.7          1146      18 %
08-20 00:19  143.5          1559      18 %
08-20 05:38  147.1          1547      13 %
```

Harness health at the same point: 130 cycles completed, 1 failed, 0 refusals, disk 1.3 GB of
493 GB free. **At 2026-08-20 17:00: 208 cycles completed, 3 failed, 1.7 GB of 488 GB free.**

**Best single flight: 1823 m3 at 189.8 m3/min** (`134507Z`, no coverage gaps). The top five runs
of the whole campaign are 1823, 1807, 1804, 1801 and 1798 m3 — a tight cluster, which says the
settled configuration reaches ~1800 m3 repeatably rather than by luck. Against an opening median
of 1060 m3 and 75 m3/min.

**What produced the gains, in order of size:** the yaw limit cycle (P17), the mapper's raycast
range (P22), the optimiser's clearance weight (P24), the frontier cluster floor (P21), the
dead-end guard's boundary bug (P16), the tilt reflex's hysteresis (P20).

**What did not, and is closed:** raising planning speed (P14), altitude setpoints (P18), the
tracker position gain (P26), the escape cooldown (P30), and every attempt to reduce contacts
directly (P23, P26, P27 — contacts are environmental and are the price of coverage).

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

**VERIFIED over two runs (2026-08-19):** gain 0.302 and 0.213 (against 0.021 when locked up),
axis median 641 and 652 rather than pinned at the ceiling, roll/pitch p90 3.1/7.4 and healthy,
stops 1.17 and 1.77 per minute, and coverage 83.1 m3/min on the better run — the highest gain
recorded in the campaign. The lock-up did not recur.

**Verify:** `actuation.x.axis_counts.median` should stop sitting at the ceiling, gain should
stay in the 0.10-0.26 band rather than collapsing to ~0.02, and speed should hold above
0.30 m/s. If the aircraft now sits still for long stretches instead, the hold is too sticky and
`escape_progress_sec` (5 s) is the knob.

**Kept regardless:** `max_forward_axis = 900` bounds the worst pitch (p90 20.0 against 22.6 at
the ≥900 bin) even though it does not stop the lock-up.

### P39 — A cycle flew its whole window without FALCON ever planning (2026-08-20, guard added)

`171142Z` mapped **225 m3** — against a 1578 median — and the per-minute table says why in one
glance: 0.8-1.5 m per minute from t=0, span 0.2-0.3 m, **9 m travelled in the entire flight**,
`no_path_fails` 0. Exploration never started. This is not the PARKED collapse of P37, which
followed healthy flying; nothing ever drove the aircraft at all.

`falcon_roslaunch.log` is 352 lines against tens of thousands in a normal run, ends mid-startup
just after the MapServer allocation, and the run has **no `falcon_exploration_follower` and no
`bev_publisher` log** — roslaunch stalled part-way through spawning its nodes.

**Every existing check passed.** Containers up, frames fresh, `/exploration_node` registered — it
HAD started, it simply never planned — and `assert_launch_params` read its rosparams back
successfully. So the cycle armed, took off, hovered for 430 s and landed, and the harness called
it a completed flight.

**Guard added to `liveness_check`:** at tick 8 (~2 minutes in), if less than 20 m3 has been
mapped since the first sample, abort the cycle. Deliberately late and generous — a healthy run
gains 80-190 m3 per sample by then, and a stuck one gains nothing — so it separates the two
without any risk to a slow starter. **Calibrated against all 170 historical runs: it would have
aborted exactly this one and left the other 169 alone.**

It saves ~6 minutes per occurrence rather than preventing anything, which is the right trade for
a fault whose root cause (a stalled roslaunch) is not reproducible on demand. The next occurrence
will now end early and say so, which is also how its frequency gets measured.

### P38 — Circling REFUTED at n=14; and a warning about how the next lead was found (2026-08-20)

**The circling hypothesis is dead.** Within `max_vel` 0.8 against final volume, the correlation
went **-0.72 (n=7) -> -0.71 (n=11) -> -0.09 (n=14)**. Holding to the n>=15 bar was right: at n=11
it looked like the answer. What killed it was one run — `131429Z`, the worst collapse yet at
**873 m3 with ZERO circling minutes**.

That makes four leads in this campaign that looked strong at small n and evaporated. The bar
stays.

**A third collapse, and a third mechanism.** `131429Z` flew normally all the way through — 11-44
m/min, spans 4-15 m, turning 19-79 deg/m, all healthy — while **coverage plateaued for its last
286 s**. The map was still being fed: `mapping_sync` emitted a flat 640 fused frames per minute
from start to finish, so this is not a perception stall. The aircraft was simply flying around
ground it had already mapped, with a tracking error of mean 3.72 m, p90 8.25, **max 16.69**.

So the three collapses are: **parked** (commanded axis zero), **circling** (3 m box, driving
normally), and now **wandering** (flying well, mapping nothing, reference far away). One symptom,
three causes. Any single fix would address at most a third of them.

**The next lead, and why it is NOT yet a candidate.** Scanning seven metrics against final volume
on that same n=14 sample:

| metric | corr |
|---|---|
| `tracking.pos_err_m.mean` | **-0.77** |
| distance flown | +0.56 |
| frac below stop speed | -0.48 |
| stops per minute | -0.42 |
| `no_path_fails` | +0.21 |
| circling minutes | -0.09 |
| deg/m late | +0.04 |

`pos_err` is the strongest — **and it is the winner of a seven-way scan on data that was already
in hand, which is precisely how the previous four dead leads were born.** A scan generates a
hypothesis; it cannot also confirm it.

**So it is pre-registered properly:** the single hypothesis is `corr(tracking.pos_err_m.mean,
coverage.final_m3) <= -0.5` within `config.max_vel == 0.8`, tested on **FRESH runs only** — those
flown after 2026-08-20 16:30 — with n>=15. No other metric gets scanned on that set, and no fix
is proposed before it passes.

**RESULT (2026-08-20 19:11): FAILED. The fifth dead lead, and the cleanest refutation yet.**

    n=15, correlation = **+0.200** against a hypothesis of <= -0.5

Not merely below the threshold — **the opposite sign**. On the fourteen runs that GENERATED the
hypothesis, `pos_err` scored -0.77; on fifteen fresh runs it is +0.20. Two runs make the point
without any statistics: `153824Z` had the WORST tracking of the sample (1.96 m) and tied the best
volume ever recorded (1823 m3), while `143628Z` had the BEST tracking (0.52 m) and one of the
lowest volumes (1201 m3).

So tracking error does not predict coverage, and the -0.77 was entirely an artifact of picking
the winner from a seven-way scan of data already in hand. **This is the strongest evidence in the
campaign for the method itself**: the same metric, the same configuration, the same analysis —
and reversing sign between the sample that suggested it and the sample that tested it.

**The open question is therefore CLOSED as "not known".** Roughly one run in five collapses to
~900-1300 m3 instead of ~1800. Three have been dissected and have three distinct mechanisms
(P37/P38), and no single metric predicts which runs will collapse: circling (-0.09 at n=14),
dz p90 (-0.12 at n=74), and now tracking error (+0.20 at n=15) have all been tested and failed.
What IS in place is detection — the analyzer flags PARKED, CIRCLING and coverage-PLATEAU
automatically with the next diagnostic question attached — so a future session inherits the
evidence rather than the archaeology.

**The analysis was fixed in code, written before the data existed:**
`sparx_agency/tools/falcon_campaign/test_p38.py`. It encodes the filter, the metric, the
threshold and the minimum sample, refuses to report a coefficient below n=15, and prints one
verdict. Every one of the five dead leads shared a cause — the analysis was chosen, or adjusted,
after the numbers were in view — and a script written in advance is the only thing that makes
that impossible rather than merely discouraged. Run it with:
`PYTHONPATH=. .venv/bin/python -m sparx_agency.tools.falcon_campaign.test_p38`

*Sample integrity checked at n=9 (18:07), before relying on the count: 10 run folders exist since
the cutoff, 9 qualify, and the tenth is simply the flight still in the air — so nothing is
dropping out silently. The metric itself is sound too: heartbeat counts run 487-497 across all
nine, so `pos_err_m.mean` is averaged over comparable samples in every run. That mattered enough
to check, because heartbeat counts used to vary 509-753 when `rosout` was being double-counted,
and a metric built on an unstable denominator would have made the test meaningless either way.*

### P37 — Two collapses, two DIFFERENT mechanisms (2026-08-20, sample building)

The second collapse landed and it is not the same failure as the first. Both spend their last
third going nowhere; how they do it differs completely.

| | `102014Z` (93.7 m3/min) | `112141Z` (91.7 m3/min) |
|---|---|---|
| last-third distance | **0.3, 0.1 m/min** | 18, 22, 31 m/min |
| span | 0.0 m | **1.9, 2.7, 3.4 m** |
| deg per metre | 366 | 144, 110, 90 |
| commanded axis | **exactly 0** | driving normally |
| signature | **PARKED** — nothing asked of it | **CIRCLING** — orbiting a 3 m box |

So "the run collapsed" is at least two conditions, and a fix aimed at one would have done nothing
for the other. That is the argument for collecting several before proposing anything.

**The diagnostic now covers both.** `motion.per_minute` was already recorded; a whole minute with
span under 4 m while still flying 5 m or more now raises a CIRCLING finding alongside the PARKED
one. Circling is the harder of the two to see by eye, because **distance, speed and stop counts
all read normal while the aircraft orbits already-mapped floor** — the SPAN is the only column
that gives it away.

**Calibration matters, and it cuts against reading too much into this.** Circling minutes are not
automatically a collapse: `113154Z` scored 167.6 with two of them, and `114209Z` 157.4 with two.
The two collapses had 1 and 3.

**A correlation exists and is deliberately NOT being reported as a finding.** Within `max_vel`
0.8, circling minutes against coverage gives **-0.72 at n=7, then -0.68 at n=8** — below this
file's own bar of n>=15 within one configuration, which exists precisely because three earlier
leads at this sample size evaporated. Re-run it at 15; do not act on it before. The newest run
argues for that patience on its own: 187.7 m3/min WITH a circling minute.

**Use `final_m3`, not the rate, for this test — the gap filter was excluding the collapses.**
"Never trust a coverage rate whose run has gaps" is right, but the gaps come from FALCON going
quiet, which is itself part of a collapse. Filtering them out therefore removes the very runs the
test is about: of the two gap-runs at `max_vel` 0.8, one is the 1224 m3 collapse. Final volume is
unaffected by missing intermediate samples and keeps them in. On that basis, n=11 and
**corr(circling minutes, final volume) = -0.71**, still under the bar.

**The sample was quietly eroding, and now cannot.** Which configuration a run flew under was only
recoverable from its roslaunch log — which the supervisor prunes after 30 runs. Two runs had
already dropped out of this comparison that way, and pruning catches up with a run at roughly the
same age the sample finally becomes worth analysing. `metrics.json` now carries a `config` block
(max_vel, raycast_max, cluster_min, bspline distance, safe_distance, course slew, tilt limit,
tracker gain), lifted from the log while it still exists.

### P36 — Vertical reference error predicts a collapsed run (2026-08-20, OPEN)

Coverage now has a median near 190 and an occasional run at ~95. Finding what separates them is
worth more than any further parameter, and there is a strong candidate. Across nine reliable
runs:

**`corr(tracking.ref_minus_pose_z_m.p90, coverage) = -0.86`** — the strongest correlation in the
campaign apart from the stall fraction itself.

> **CORRECTION (2026-08-20, same day): that -0.86 does not survive a larger sample and the
> claim above is withdrawn.** It came from a nine-run window spanning three different
> configurations and containing both collapsed runs. Recomputed:
> * all 74 reliable, gap-free runs: **-0.12**
> * within `max_vel` 0.8, the settled config (n=7): **-0.20**
> * within `max_vel` 0.6 (n=7): -0.82 — but that is one run (99 m3/min, dz p90 1.34) carrying
>   the whole coefficient
>
> A group split still shows something modest and possibly real: dz p90 > 0.9 gives mean coverage
> 129.9 with 28 % stall (n=14), against 145.9 and 20 % for dz p90 <= 0.9 (n=60). But mixing
> configurations puts the config effect into the coefficient, and configuration is the biggest
> thing that has ever moved coverage here.
>
> **Standing rule from three repeats of this mistake: compute a correlation WITHIN one
> configuration, with n >= 15, or do not report it.** Coverage varies ~2x run to run (section
> 7b), so a coefficient over 9 mixed runs is noise wearing a decimal point.

| coverage | 190.9 | 190.5 | 182.1 | 178.4 | 177.1 | 168.3 | 164.4 | **99.3** | **93.7** |
|---|---|---|---|---|---|---|---|---|---|
| dz p90 (m) | 0.38 | 0.63 | 0.54 | 0.72 | 0.55 | 0.75 | 0.39 | **1.34** | **1.04** |

The two collapsed runs have the two highest vertical reference errors. `dz` MAX does not
correlate at all (-0.05), so it is not a one-off spike — it is time spent with the plan well
above the aircraft.

**Mechanism, from `102014Z`'s last two minutes:** the aircraft covered 0.3 then 0.1 m per minute
with the commanded axis at **exactly 0** — it was not wedged, nothing was being asked of it. The
follower's last heartbeats read `pos_err=2.24m dz=+2.23m`, i.e. the error was almost entirely
VERTICAL: the aircraft had arrived horizontally and the reference had climbed away. Altitude
authority is near zero on this platform (P18), so it could not follow, and with no horizontal
error there was nothing to drive toward. It sat.

**This reopens the part of P18 that was closed too broadly.** P18 measured `dz` near zero on
AVERAGE and concluded the altitude setpoints should not move — correct about the mean, and the
mean was the wrong statistic. The tail is what ends flights.

**Not yet a fix, deliberately.** Ruled out already: battery (drain is 0.64-0.68 at 200 s and
0.27-0.35 at 400 s across good and bad runs alike), and a wedge (the follower's give-up never
fired; FALCON's own guards fired 9 confinement and 11 publish-fail blacklists, so it was actively
retiring viewpoints). With the correlation withdrawn, what remains is a single well-documented
run, which is an anecdote and is being treated as one.

**What was done instead of theorising — the diagnostic is now automatic.** Every run's
`metrics.json` carries `motion.per_minute`: distance, spatial span and turning per minute of
flight. That table is what separates PARKED from CIRCLING from TRAVELLING, and it is what found
the unreachable-viewpoint lock (P16), the yaw limit cycle (P17), the latched pinned hold (P29)
and this vertical stall — each time hand-rolled from `truth.jsonl` after the fact. A run with any
whole minute under 2 m now also raises a ranked finding saying so, with the next question
attached: *commanded axis zero means nothing was asked of it (an unreachable reference);
non-zero means it is wedged.* The next collapse arrives pre-diagnosed.

### P35 — A cold FCU needs longer than a warm one (harness, 2026-08-20)

Two cycles in a row ended `unhealthy stack; flew nothing` with `armable: false`, each having
already restarted Sphera itself ("battery is fine but the FCU is not armable"). The third cycle
found the drone armable immediately and flew normally. About nine minutes of flying time lost to
a timeout, not to a fault.

The bring-up already waits for armable before taking its verdict — but at a flat 90 s, which was
measured on a WARM stack where the FCU had merely not finished connecting. A drone that has just
been respawned by a Sphera restart is a colder start than that. `restart_sphera` now sets
`SPHERA_WAS_RESTARTED`, and the wait becomes **240 s after a restart, 90 s otherwise**, so the
cheap case stays cheap and the expensive one gets the time it actually needs.

Nothing was broken: the harness's detect-and-restart path worked, it just charged two cycles for
what one should have covered. Worth knowing for the next unexplained pair of dead cycles — check
`armable` in the health line before assuming a real fault.

**Observed rate after the fix (2026-08-20 20:48):** the not-armable restart trigger has fired
**3 times in 250 cycles (~1.2 %)**, and the most recent one still cost a cycle even with the
240 s grace — the restart happened, the FCU was still not armable when the window closed, and the
NEXT cycle flew normally. So the fix reduced the cost from two cycles to one; it did not
eliminate it. At ~1 cycle lost per 80, that is not worth further engineering. Investigate only if
it starts repeating within consecutive cycles again.

### P34 — One more speed step: max_vel 1.0 (2026-08-20)

0.6 -> 0.8 gave the campaign's two best runs and its lowest stall, so take the next step. The
airframe's ceiling is real but not here yet: at 1.0 m/s the MOVING curve asks **730** counts and
the standing one 924, so only an acceleration from rest clips — and an aircraft that cannot reach
its commanded speed instantly from a standstill is already the normal case, not a new failure.
Follower ceiling 1.0 -> 1.2 so it does not bind; the 1.5 m/s measured-speed backstop stays as the
safety net it was built to be.

**Baseline MEASURED first (n=2 at max_vel 0.8), which is the point:**

| coverage | stall | frac of ticks at 900 | PINNED | tilt cuts | mean speed |
|---|---|---|---|---|---|
| 190.9, 190.5 | 5 %, 4 % | 0.153, 0.103 | 12, 12 | 13, 11 | 0.49, 0.51 m/s |

**Pre-registered against those numbers:** WANT coverage above 200 on at least one of three, AND a
median above 175. REVERT IF the median falls below 165, OR the fraction of ticks at 900 exceeds
0.25 (the fraction, not the p90, which saturates and says nothing), OR PINNED exceeds 20 a run,
OR tilt cuts exceed 30. Every one of those bars sits outside the control's own range, which the
last two experiments' conditions did not.

**RESULT: the WANT failed, so 1.0 is REVERTED to 0.8. Speed is now CLOSED.**

| | max_vel 0.8 | max_vel 1.0 |
|---|---|---|
| coverage m3/min | 190.9, 190.5 | 177.1, 182.1, **93.7** |
| median | **190.7** | 177.1 |
| final volume m3 | 1752, 1764 | 1735, 1801, 1254 |
| stalled share | 5 %, 4 % | 8 %, 10 %, **40 %** |
| ticks at 900 | 0.153, 0.103 | 0.110, 0.206, 0.213 |
| mean speed | 0.49, 0.51 | 0.57, 0.44, 0.31 |

No revert condition fired — the median held at 177.1 above the 165 floor, saturation stayed under
0.25, contacts and tilt cuts stayed inside their bars. But the WANT was a **conjunction** and its
first half never happened: no run of three reached 200, and the median came out 13 m3/min BELOW
the control. Asking for more speed did not produce more speed either — mean speed went 0.49-0.51
to 0.57, 0.44, 0.31, i.e. down on two of three runs, because the extra demand buys saturation
(0.10-0.15 -> 0.11-0.21) rather than motion.

**Speed is closed for good.** 0.8 is the settled value: at 1.0 the moving curve already asks 730
counts and the standing one clips at 924, so there is no further step this airframe can take, and
the step that exists makes things worse. The 0.6 -> 0.8 gain (P33) stands.

*Note the asymmetry that makes this call honest: P33 fired a revert condition and was KEPT
because the condition was proven non-discriminative; P34 fired none and is REVERTED because the
thing it was meant to achieve did not happen. Conditions are evidence, not verdicts.*

### P33 — Transit is near its floor; planning speed RE-OPENED on new grounds (2026-08-20)

P32 refuted target churn as the cause of transit inefficiency, so what is left of it? Counting
VISITS per 1 m cell rather than samples:

| | cells | visits/cell | p50 | busiest 10 % of cells | entered exactly ONCE |
|---|---|---|---|---|---|
| `080304Z` | 187 | 1.5 | 1 | 18 % of visits | **61 %** |
| `085936Z` | 178 | 1.7 | 1 | 21 % of visits | **58 %** |

No chokepoint pathology and no looping: the median cell is entered once, the busiest cells 4-5
times, and nearly two thirds are visited a single time. **The earlier framing — "a third to a
half of the path is over ground already covered" — overstated it.** At 1 m cells and a curving
path, travelling more than a metre per new cell is geometry, not waste. Transit is close to its
practical floor and is not where the remaining coverage is.

**So: `max_vel` 0.6 -> 0.8, and the follower ceiling 0.8 -> 1.0.**

This deliberately re-opens P14, which was closed as "speed is not the lever". That verdict was
correct **for the regime it was measured in**: the test ran when 60 % of every flight was
stalled, so the aircraft's planned speed could not matter — it was not flying. The stall is now
9-17 %, the failure modes behind it are fixed, and coverage is bounded by rate x window with the
window fixed by the battery. Re-opening a closed question needs a reason; "the thing that made
the earlier answer uninformative has since been fixed" is one.

Arithmetic checked first, per the standing rule: 0.8 m/s asks **863** counts standing and **667**
moving, both under the 900 ceiling. 1.0 m/s would ask 924 and clip, so 0.8 is the last step this
airframe has.

**RESULT: KEPT. Two runs of two beat the campaign record, and the stall is the lowest yet.**

| | max_vel 0.6 | max_vel 0.8 |
|---|---|---|
| coverage m3/min | 169.6, 185.6, 146.7, 164.4 | **190.9, 190.5** |
| final volume m3 | 1580, 1807, 1560, 1648 | 1752, 1764 |
| stalled share | 16-17 % | **5 %, 4 %** |
| axis frac at 900 | 0.123, 0.121, 0.141, 0.114 | 0.153, 0.103 |
| PINNED / tilt cuts | ~9-13 / 2-19 | 12 / 13, 12 / 11 |

**One pre-registered revert condition fired and I am overriding it — with the reason stated.**
"REVERT IF `actuation.x.axis_counts.p90` pins at 900" was met... and it was **already met by every
baseline run**, because p90 sits at the ceiling whenever the aircraft asks for a hard push at all.
The condition could never discriminate, so it tested nothing. The substantive version of that
concern is the FRACTION of ticks at the ceiling, and it did not move: 0.114-0.141 before, 0.103
and 0.153 after. Tilt cuts and contacts did not climb either. This is a mis-specified condition
being corrected against evidence, not a goalpost being moved: the WANT was met outright, on both
runs, with the largest stall reduction of the campaign alongside it.

**The pattern to break — two in two turns:** P32's threshold was set from a partial sample of the
metric's own spread, and P33's was set to a value the baseline already satisfied. **Measure the
baseline value of every pre-registered condition before flying the change.** A criterion that the
control case also meets is not a criterion.

### P32 — The coverage tour re-picks its target 22 times a minute (2026-08-20, NEXT CANDIDATE)

Following P31 (the remaining loss is transit through mapped space), the question is why transit
is inefficient — 0.41-0.72 distinct 1 m cells per metre flown. Parsing the HGrid tour's own log:

* the next-cell target changed **190 times in 517 s — 22 per minute**, across 19 distinct cells;
* **median dwell on a target: 0.3 s** (p25 0.1, p75 2.1, max 92);
* three cells account for two thirds of the samples, so it is oscillating between a handful of
  candidates rather than progressing through a tour.

A target that survives 0.3 s cannot be driven to. The aircraft is partly protected from this
downstream — the course slew (P17) refuses to chase direction changes faster than the airframe
can turn — which is likely why this has not shown up as a spin. But it means the aircraft is
steering toward an average of several cells rather than committing to one.

**Candidate fix:** commitment/hysteresis on the tour target — keep the current cell until it is
reached, exhausted, or has been beaten by a clear margin for a sustained period. FALCON exposes
no such parameter, so this is a C++ patch against the exploration manager (the templates are in
`patches/`, and `verify_patch.sh` compiles one in ~1 min before a 2-min image rebuild).

**IMPLEMENTED 2026-08-20** as `patches/fix_falcon_tour_commit.sh`: the chosen (cell, center)
pair is held until it drops out of the tour (its frontiers are gone, which is also what happens
once the aircraft arrives), or `/exploration/tour_commit_max_s` (8.0, a rosparam, guarded)
expires. The TSP still runs and the tour is still published; only which element is handed
downstream as "next" is held steady. Compile-verified with `verify_patch.sh`, then built into
the image and confirmed in the BINARY, not just the source.

**Pre-registered:** WANT — cells-per-metre above 0.8, coverage above the 143-177 band.
REVERT IF — coverage drops below 140, or `no_path_fails` rises sharply (committing to an
unreachable cell is exactly the P16 lock this campaign already fixed once; that is what the
timeout bounds).

**RESULT: the patch works, the hypothesis is REFUTED. Disabled (`tour_commit_max_s = 0`).**

| | baseline | with commitment |
|---|---|---|
| target changes / min | 12.1, 10.0 | **2.5, 3.3** |
| target dwell p50 | 2.0 s, 0.7 s | **15.8 s, 9.3 s** |
| cells per metre | 0.75, 0.84 | 0.73, 0.76 |
| coverage m3/min | 169.6, 185.6 | 146.7, 164.4 |

Churn fell 4-5x exactly as intended, and **transit efficiency did not move at all**. Holding the
target steady is not what makes the aircraft cover ground twice. Neither revert condition fired
(coverage stayed above 140, `no_path_fails` did not rise), but neither WANT was met, so the
change is neutral — and a neutral change that costs a C++ patch, a rosparam and a marker is not
worth keeping switched on. It stays compiled into the image, disabled, so the next hypothesis
about the tour has the mechanism available.

**Two measurement lessons, both mine:**

* **The patch contaminated its own metric.** Parsing `next cell id: (\d+)` matched BOTH the TSP's
  pick and the new "Holding tour target" line, which now differ — so the target looked like it was
  changing 44-49 times a minute, twice the baseline, when it was actually changing 2.5. Reading
  only the final `Current cell id: N, next cell id: N` line gives the true figure. A patch that
  adds log lines can break the parser that measures it.
* **The threshold was set from too small a sample.** "cells-per-metre above 0.8" came from three
  runs reading 0.41-0.72; two other baseline runs read 0.75 and 0.84. The metric's own spread is
  0.41-0.84, so the bar was inside the noise before the experiment started. Measure a metric's
  spread BEFORE pre-registering a threshold on it.

**Build trap, now guarded:** the first build put the patch step AFTER `catkin_make`, and the only
later step that rebuilds (`fix_falcon_sop.sh`) short-circuits with "already patched" — so the
image carried the patched source and an unpatched binary, with matching timestamps. `strings` on
the binary is what caught it. The step now sits before the compile, and `tour_commit_max_s` is in
`_PATCH_MARKERS` so bring-up refuses a stale image.

**Measurement note that cost twenty minutes:** the log-parsing scripts in this file matched
timestamps as `\[(17871\d{5}\.\d+)\]`. Wall time has since ticked into `17872…`, so those
regexes now silently match NOTHING and every derived count reads zero. Use `\[(\d{10}\.\d+)\]`.

### P31 — The remaining "stall" is TRANSIT, not failure (2026-08-20)

Two measurements over three baseline runs (1277 s of flight) change what is worth optimising.

**1. Recovery time does not cost coverage.** 28 % of flight time sits inside a recovery window
(from a PINNED event to sustained motion again) and 26 % has coverage flat — but the **overlap is
27 s, 7 % of recovery time**. The two are almost disjoint: while the aircraft reverses and turns
its way out of a contact, the camera sweeps geometry it had not seen, so the mapping continues.

*This kills the queued `escape_cooldown_sec` 6.0 experiment before it cost three runs.* Making
recoveries faster optimises time that was never lost. (P30's revert to 4.0 stands on its own
evidence — shorter was worse at recovering — it simply matters less than it looked.)

**2. What IS flat is the aircraft flying normally.** During coverage-flat windows it moves at
**0.46 m/s mean**, against 0.48 in productive windows, with only 6 % of that time stationary.
The remaining loss is **transit through space already mapped** — not a lock, a spin, a park, a
wedge or a famine. Every failure mode this campaign has chased is gone; what is left is the cost
of getting from one frontier to the next.

**Path efficiency, and a correction to how it was first measured.** Distinct 1 m cells visited
per metre flown: **0.41, 0.72, 0.58** across the three runs (90-137 cells for 187-238 m). A
non-repeating sweep would be ~1.0, so roughly a third to a half of the path is over ground
already covered — moderate, and partly inherent to a building where every room is reached
through the same corridors.

*The first version of this measurement reported "97-99 % of the path is revisiting", which was
nonsense: it credited a cell as "fresh" only for the single 0.025 m step that crossed into it,
so the fresh fraction was tiny by construction rather than by behaviour. Cells-per-metre is the
honest form of the same question.*

**Consequence:** parameter tuning is close to its ceiling here. The remaining levers are
algorithmic — the coverage tour's ordering, which is FALCON's HGrid TSP — and the campaign's own
evidence (the `cluster_min` sweep) is that frontier-side parameters trade sharply against each
other. Any attempt needs 3+ runs a point and a revert condition written first.

### P30 — Getting unstuck costs 74 s a flight; the cooldown is most of it (2026-08-20)

With contacts accepted as the price of coverage (P27), the question is what each one COSTS.
Measured across **62 PINNED events in eight runs**, timing from each event to the aircraft next
holding above 0.20 m/s for two seconds:

| regained motion | p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|---|
| 82 % within 60 s | 2.4 s | **10.6 s** | 20.1 s | 29.7 s | 31.5 s |

**588 s across eight runs — about 74 s per flight, ~17 % of the window.** The other 18 % of
events are the wedges that P29 covers.

An escape is 2.5 s of reversing plus a 4 s cooldown, so a median recovery of 10.6 s is roughly
1.6 cycles and the p75/p90 tail is 3-5 — **most of the cost is the cycle, not the manoeuvre**.
`escape_cooldown_sec` 4.0 -> 2.0 takes a cycle from 6.5 s to 4.5.

**RESULT: it made recovery WORSE. Reverted to 4.0 on the pre-registered condition.**

| | events | recovered | p25 | p50 | per flight |
|---|---|---|---|---|---|
| cooldown 4.0 | 28 | 82 % | 1.2 s | **6.3 s** | 82 s |
| cooldown 2.0 | 16 | 75 % | 8.6 s | **15.6 s** | 92 s |

Median time to regain motion went 6.3 -> 15.6 s, the per-flight cost 82 -> 92 s, and the
recovery rate 82 -> 75 %. Coverage was unchanged (177.9 and 149.7, inside the existing band), so
this cost time without buying anything.

**The reading:** the cooldown is not dead time between attempts, it is the airframe settling and
the servo integrator unwinding. An escape that begins before that has finished is a WORSE escape,
so more attempts per minute bought fewer recoveries. The arithmetic of "a cycle is 6.5 s, make it
4.5" was right and irrelevant.

Small samples (2 runs vs 3, 16 events vs 28), but the direction matches the mechanism the
cooldown exists for and the effect is 2.5x, which is why the revert condition was written before
the data arrived.

**Untested in the other direction:** if settling is what matters, 6.0 s might beat 4.0. Worth a
turn with 3+ runs per point, since the metric is noisy — but it is a genuinely different claim
from the one just refuted, not a rescue of it.

**Note on the diagnosis that got here:** run `060944Z` parked for 120 s with the follower
reporting `ref_ready=True holding=False` throughout — it was commanding motion the whole time,
and the give-up never fired. Only 2 tilt cuts and 2 overspeed scalings in that window. The
follower's own heartbeat cannot distinguish "tracking normally" from "wedged and tracking
normally"; the aircraft's measured speed is what tells them apart.

### P29 — The pinned hold was a latch with an unreachable release (fix applied 2026-08-20)

The stall is now the single biggest determinant of coverage: across **94 reliable runs**,
`corr(stall_frac, coverage) = -0.79`. Nothing else comes close. So what the remaining stalls are
made of is THE question, and the answer is not one of the three known signatures.

Run `20260820_050749Z` flew normally for 180 s — 28-32 m/min, 39-55 deg/m — and then covered
**0.0-0.7 m per MINUTE for the remaining 250 s**. Not a lock (P16), not a spin (P17: span 0.0-0.3
m, it is not circling), not a frontier famine (P21). The aircraft was simply **parked**, and its
own log says why, every 10 s:

> `4 escapes without regaining motion -- suppressing further escapes until the aircraft moves`

That is P13's pinned hold, and its release condition is unreachable by construction: the hold
commands **zero translation**, so an aircraft that is genuinely wedged can never "move for 5 s"
to earn its release — and one whose obstruction has since cleared is never given the chance to.
P13's own comment says "commanding normally and letting FALCON replan is strictly better than a
manoeuvre that is not working"; the code did the opposite and commanded nothing at all.

**Fix:** the hold is now bounded by `~pinned_hold_sec` (4.0 s). When it expires the escape budget
re-arms and normal driving resumes, so a permanent stop becomes a periodic retry. P13's actual
purpose survives — the airframe still gets its seconds to settle and the servo integrator to
unwind, which is what stopped the 20-35 deg pitch lock-up.

**With a backoff, because the naive retry reopens the failure the give-up was built for.** A
PERMANENTLY wedged aircraft would grind the escape ladder forever — a run once spent over half
its window on 38 escapes, none of which restored motion, which is exactly why the give-up exists.
So each hold that ends *without* the aircraft regaining motion doubles the next:
**4, 8, 16, 30 s** (`~pinned_hold_max_sec`). Early retries stay cheap, a genuine wedge stops
burning the flight, neither failure mode is permanent, and sustained motion resets the backoff.

**Status: DORMANT, not yet verified in flight.** The give-up has not fired once in the seven runs
since (5-12 escapes each, 0 latches) — it fired in `050749Z` and not since, so this is roughly a
1-in-8 failure that costs ~250 s when it lands. Coverage over those seven is 143.5-177.1 with no
recurrence of the 101-111 low end, which is consistent but proves nothing about this fix.

**Verify when it does fire:** look for `pinned hold expired after 4s -- re-arming` in the follower
log, then check the per-minute distance around it — the aircraft should resume rather than sit at
0 m/min to the end of the flight. **Watch:** PINNED climbing a lot without coverage improving
means the retry is grinding; raise `explore_pinned_hold_sec` rather than restoring the latch.

### P28 — Exploration is TIME-limited, not map-limited (2026-08-20)

Measured at the end of a current-configuration flight: **33.2 % of the box floor has mapped free
space, 38.1 % has been seen at all**, and the free footprint now spans the full box
(x 38.5-70.5, y -29.5-1.0) — the aircraft reaches every extreme but fills a third of it. Against
30.1 % / 35.9 % before the raycast and cluster changes.

The decisive part is not the percentage, it is that **exploration never reaches FINISH any more**
("never reached FINISH in 445 s") and the coverage curve is still climbing when the battery
window closes. The building is not exhausted and the frontier finder is not starved — the flight
simply ends. So **coverage rate is the only lever that matters**, and every second of the 430 s
window spent not exploring is directly lost volume.

Do NOT raise the simulator's battery capacity to buy a longer flight. It is a configurable
override (`~/rooster-private-parameters/developer.params.yaml`), which makes it exactly the kind
of change that improves the number without improving the system, and it would break comparison
with every run in this file.

**The "dead band wastes a quarter of the flight" candidate is an ARTIFACT — closed, no work
needed.** `nav_debug_recorder` already logs the follower's own `/cmd_vel` to
`telemetry.jsonl`, so the demand could be joined to the axis without adding anything. Doing that
across a full flight:

* 2361 of 8925 ticks had the axis inside the dead band **with a genuine demand** (p50 0.296 m/s,
  95 % above 0.15) — so this was not the servo braking.
* But during those very ticks the aircraft was **moving at p50 0.39 m/s**, with only **1 %**
  genuinely stopped, and the axis sat at **p50 551**.

551 is below the *standing* dead band of 620 and well above the *moving* one of 412 — and the
adapter has carried both curves since P5, because momentum lowers the deflection needed to keep
going. The analyzer was judging a moving aircraft against the standing curve. Fixed: each tick
is now measured against the regime it is actually in, and `frac_dead_band` falls from 0.23-0.29
to **0.02-0.03**.

**Consequence for the record:** `actuation.*.gain` before 2026-08-20 08:00 was computed from the
wrong curve too (it now reads 0.45-0.60 where it read 0.16-0.41), so do not compare a gain from
an older run against a newer one. Coverage, clearance and stall numbers are unaffected.

### P27 — Contacts are set by WHERE the aircraft flies, not by tuning (2026-08-20, CLOSED)

Across **16 runs** with clearance traces, correlating each run's median reference clearance
against its contact count and its coverage:

* **corr(reference clearance, PINNED) = -0.65** — tight runs have more contacts. When the plan
  has room (median clearance 0.83-0.96 m) there are 6-8 pins; when it is tight (0.32-0.38) there
  are 9-15.
* **corr(reference clearance, coverage) = -0.54** — and tight runs cover MORE. The unexplored
  volume is in the tight parts of the building; the open corridors are already mapped.

So contacts and coverage are **positively linked through the environment**: going where the
volume is means going where it is tight. Every control-side attempt to reduce contacts has now
failed to move them (P23 raised clearance, P26 raised the position gain — neither changed the
count), while the one change that DID reduce them, the clearance weight in P24, also raised
coverage, because it improved the plan rather than avoiding the hard places.

**This closes the contact line of investigation.** ~6-15 contacts per flight is not a defect to
be tuned out; it is what exploring the difficult two-thirds of this building costs with a 0.62 m
sensor near floor and a dead lateral axis. Reducing it by avoiding tight spaces would cost
coverage, which is the mission. Chase coverage; treat contacts as its price and keep the escape
ladder healthy so each one is cheap.

### P26 — Bending the course harder to close cross-track (change applied 2026-08-20)

Cross-track is now measured over four flights and it is stable: **p50 0.18-0.23 m, p90
0.59-0.84**, against a reference clearance of 0.51-0.86 m. So the p90 of the error that can
cause a collision is the same size as the median clearance available — P25's structural
mismatch, quantified.

The tracker is not missing a correction; it is slow. `horizontal_pid.kp` is 1.0, so a 0.2 m
cross-track offset against a 0.5 m/s feedforward bends the commanded course only **22 deg** off
the path — and on this airframe the course IS the correction, because lateral velocity is
dropped outright. At 1.6 the same offset bends it 33 deg.

Exposed as `~tracker_pos_kp` (default 1.0 = previous behaviour, so it is one launch arg to
revert) and set to 1.6, guarded in `EXPECTED_ROSPARAMS`.

**RESULT: no effect, REVERTED to 1.0 after four runs.** Cross-track p50 0.18, 0.25, 0.19, 0.19
against a 0.18-0.23 baseline; p90 0.52, 0.79, 0.63, 0.57 against 0.59-0.84. The metric it was
aimed at did not move, while `stops_per_min` rose to 3.3-8.2 from 0.5-5 and coverage sat at
107-170 against 135-187. Bending the course harder does not close cross-track on this airframe —
the limit is downstream of the command, in the dead band and the turn rate. The `~tracker_pos_kp`
wiring stays (default 1.0), because being able to try a gain from a launch arg is worth keeping.

**Note for whoever tries the next step:** the follower only ever sees ONE point
(`/planning/pos_cmd`), not the path, so a true pure-pursuit aim point is a bigger change than
it sounds — it would need `/planning/bspline`, which is the other follower's input.

### P25 — Contacts are structural: clearance and tracking error are the same size (2026-08-20)

After P24 the plan is 0.38 m off the wall at the moment of a contact and the aircraft is at
0.09. The obvious reading is a tracking excursion — **it is not.** Measured across the same
three runs, 17 pins:

| | flight median | at the pin |
|---|---|---|
| `pos_err` | 0.45 m | **0.48 m** |

Tracking at a contact is completely ordinary. Nothing spikes. What is actually happening is
simpler and harder: **the reference's clearance (p50 0.50 m) is no larger than the ordinary
tracking error (p50 0.45 m)**, so whenever that ordinary error happens to point at a wall, the
aircraft touches it. No amount of "fix the excursions" helps, because there are no excursions.

Two ways out, and the second needs a measurement first:
* **More clearance than error.** Would need ~0.9 m of reference clearance to have real margin,
  which this building's doorways will not allow. P24 already took the cheap part of this.
* **Less error.** 0.45 m at 0.5 m/s is nearly a second of lag. A single distance mixes
  ALONG-track lag, which cannot cause a collision, with CROSS-track error, which is the only
  half that can — so the probe now records both (`clearance.abs_cross_track_m`).

  **First decomposition (one flight, n=81):** |along| p50 0.21 / p90 1.08, |cross| p50 0.22 /
  p90 0.84. The two halves are the same size, so the harmless-lag hope is dead: cross-track
  error really is ~0.2 m typical and ~0.8 m in the tail, against a reference clearance of
  0.45-0.50 m at the tight moments. **Cross-track is the lever**, and it is structural: the
  follower zeroes lateral velocity outright (Rooster's lateral axis is dead until ~1000 counts
  and then rolls 30 deg), so the only way it can close a crosstrack offset is to turn.

  Accumulate `abs_cross_track_m` over several runs before choosing a fix — n=81 from one flight
  is thin. Candidates, in order of how little they risk: a pure-pursuit aim point ON the path
  rather than the time-parameterised reference (converges to the path by construction), a
  shorter lookahead, and only then any use of the lateral axis.

**Do not "fix" this by raising `course_slew_deg_s`.** P17's limit cycle came from the demand
OSCILLATING faster than the plant could follow; a large persistent heading error is a different
thing, and an adaptive slew would have to tell them apart — an orbit also looks persistent, so
that discriminator needs care, not a hunch.

### P24 — Every contact is against MAPPED geometry, in the optimiser's TAIL (2026-08-20)

The working assumption after P23 was that the remaining contacts had to be against obstacles the
depth sensor cannot see (P22's 0.62 m near floor), since raising `safe_distance` improved
clearance without reducing PINNED events. **That is wrong**, and the clearance trace settles it.

Correlating 26 PINNED events across three runs against `clearance.jsonl` — distance to the
nearest **mapped** obstacle at the moment the aircraft stuck:

| p10 | p25 | p50 | p75 | p90 | beyond the 0.40 m planner margin |
|---|---|---|---|---|---|
| 0.05 | 0.09 | **0.15** | 0.18 | 0.25 | **0 of 26** |

Not one contact happened with the aircraft clear of geometry the map already had. The reference
at those moments sits at p50 0.15 m as well — **while the same flights' median reference
clearance is a healthy 0.50 m**. The problem is the optimiser's *tail*, not its median: in tight
spots the smoothness term wins and the curve is pushed into a wall that is already in the map.

**Change:** `/bspline_opt/pos/distance` 50 -> 150, against smoothness at 20, making clearance
three times harder to trade away. `safe_distance` stays 0.55 — it moved the median correctly and
raising a soft target further does nothing about a term that is being outvoted.

**VERIFIED over three runs — the best flights of the campaign, on every axis at once:**

| | weight 50 | weight 150 |
|---|---|---|
| coverage m3/min | 129.2, 148.8, 129.7 | **174.2, 187.4, 168.8** |
| final volume m3 | 1307, 1622, 1465 | **1786, 1798, 1714** |
| PINNED per flight | 13, 13, 9 | **10, 7, 9** |
| time below stop speed | 0.22, 0.20, 0.18 | **0.20, 0.14, 0.11** |
| reference clearance AT a pin | 0.15 | **0.38** |

1798 m3 is a new record and all three runs beat the previous best of 1700. The mechanism did
exactly what it was meant to: at the moments that matter the plan is now 0.38 m off the wall
instead of 0.15.

**But the aircraft is still at 0.09 m (p10 0.02) when it pins, with none of the 17 events beyond
0.40 m.** The plan has stopped driving into walls; the aircraft still arrives at them. That is a
~0.3 m gap between reference and aircraft at exactly the wrong moment, so **the contact problem
is now a TRACKING problem** (pos_err p50 0.32, p90 1.26) rather than a planning one.

**Next lever, and note what it is not:** the follower zeroes lateral velocity entirely (Rooster's
lateral axis is dead until ~1000 counts and then rolls 30 deg), so crosstrack error can only be
corrected by turning, and `course_slew_deg_s` caps that at 45. The limit cycle P17 fixed was
caused by the demand OSCILLATING faster than the plant could follow, which is a different thing
from a large, persistent heading error — so an adaptive slew (fast when the demand has been
consistently one-sided, slow when it is thrashing) would keep P17's property while correcting
crosstrack promptly. Do not simply raise the cap and undo P17.



### P23 — The reference itself flies inside the safety margin (2026-08-20)

The aircraft is stuck (PINNED) about a dozen times a flight, costing ~27 % of it. Two causes
were consistent with that and they need opposite fixes, so `probe_clearance_trace.py` was
written to log, twice a second for a whole flight, the distance from both the AIRCRAFT and its
REFERENCE to the nearest mapped obstacle in their own height band.

| | p5 | p25 | p50 | p75 | fraction inside 0.40 m |
|---|---|---|---|---|---|
| aircraft | 0.03 | 0.06 | 0.23 | 0.51 | **66 %** |
| reference | 0.06 | 0.30 | **0.36** | 0.49 | **73 %** |

The plan is not being followed badly *into* danger — **the plan is already there.** When the
aircraft is inside 0.30 m, the reference is inside 0.40 m in 85 % of samples. Tracking error
(p50 0.52 m, p90 1.23) then makes it worse, but it is the second cause, not the first.

**Ruled out: map noise.** Zero isolated occupied voxels in 532 880, only 1.9 % with 4 or fewer
of 26 neighbours, median 13 — the surfaces are real, not speckle. (Worth keeping as a habit: a
clearance number is only as good as the map it is measured against.)

**Cause:** `safe_distance` is a SOFT cost in the B-spline optimiser (weight 50 against
smoothness 20), not a constraint, so at 0.40 it simply gets traded away — the spline cuts the
corner its own waypoints went round, which is what the publish-fail patch's comment already
said about `checkTrajCollision` being a point test with no radius.

**Change:** `SAFE_DISTANCE` 0.40 -> 0.55, and deliberately NOT `OBSTACLES_INFLATION`, which
stays at 0.40. A*'s inflation decides what is reachable at all — 0.85 there once made
exploration fail outright — so doorways stay plannable and only the curve through them is asked
to ride nearer the middle.

**RESULT — the clearance target is met, the contacts are not.** First healthy run traced at
0.55 (`20260820_014131Z`, 239 samples):

| | reference p50 | inside 0.40 m | aircraft p50 | inside 0.40 m | pos_err p50 |
|---|---|---|---|---|---|
| safe_distance 0.40 | 0.36 | 73 % | 0.23 | 66 % | 0.52 |
| safe_distance 0.55 | **0.50** | **40 %** | **0.40** | **50 %** | **0.32** |

The plan moved off the walls exactly as intended and the aircraft followed it there. Coverage
held (129.7 m3/min, stall 13 %, final 1465 m3). **But PINNED events did not fall: 13, against
7-12 before.** So proximity to *mapped* geometry was not what was causing the contacts — which
points back at the 0.62 m near floor of the depth sensor (P22): the aircraft is being stopped by
things the map never had.

Keep 0.55 — it is free (coverage held, tracking improved) and it is the right margin for a
0.4 m-inflation planner. But the contact problem needs the other half.

**A trace is only usable if the flight was:** the first 0.55 trace read p50 0.11 m and looked
catastrophic — it came from a run that spent 78 % of its samples inside one 1 m cell, stuck.
Check the spatial spread of a trace before believing its distribution.

### P22 — The depth measurement that killed the raycast change was wrong (2026-08-20)

On 2026-08-19 I ruled out raising the mapper's `raycast_max` on the strength of a live depth
sample: *"p50 1.06 m, max 3.51 m, 0 % of pixels beyond 5 m"*. That sample came off
`/tmp/rooster_frames/depth`, **not** `/tmp/rooster_depth`, which is the `depth_dir` the depth
processor writes and the mapper is fed.

Re-measured from the right directory, 2.38M pixels over 12 consecutive frames:

| | p1 | p5 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| depth, m | 0.85 | 0.95 | **2.10** | 3.90 | **8.57** | 11.41 |

**4.8 % of every frame lies beyond the 5.0 m `raycast_max` and is thrown away**, while
`sensing_parameters/max_depth` already allows 10 m — so the truncation is ours, not the model's.
Raised to **8.0** (98.8 % of returns kept), as a rosparam override after the yaml load, so no
rebuild: `/voxel_mapping/tsdf/raycast_max`, guarded in `EXPECTED_ROSPARAMS`.

**VERIFIED over three runs — the largest single gain in the campaign.** Same `cluster_min` 50,
only the raycast range changed, all six runs reliable with zero coverage gaps:

| | raycast 5.0 | raycast 8.0 |
|---|---|---|
| coverage m3/min | 119.5, 128.7 | **143.2, 135.5, 176.1** |
| final volume m3 | 1107, 1204 | **1424, 1446, 1700** |
| `no_path_fails` | 1429-2055 | 993-10634 |

1700 m3 is the highest volume any run has reached. Coverage is up 20-40 % and final volume ~25 %,
which is what you expect when 4.8 % of every frame stops being discarded: the far returns are the
ones that resolve a whole wall at once.

`no_path_fails` did spike to 10634 in one run (baseline 1429-2055) — the phantom-far-geometry
risk is real — but that run still scored 135.5 with an 8 % stall, so it is not costing the
mission yet. Watch it; 6.5 remains the fallback.

**The near end matters too, and explains the contacts:** the same measurement shows a hard floor
at **0.62 m**, p1 0.85 — DA3 returns nothing closer. An obstacle within ~0.6 m is invisible to
the map whatever `cam_min_depth` (0.45) says, which is exactly why the aircraft presses against
geometry it never saw. That is a sensing limit, not a tuning one; the mitigation is behavioural
(the dead-end guard, the escape reflex), not a parameter.

**Lesson:** a measurement that KILLS a change deserves the same scrutiny as one that justifies
it — check you sampled what the system actually reads.

### P21 — FINISH at 27 % of the box: the boundary is there, the CLUSTERS are not (2026-08-20)

With P16 and P17 fixed the stall changed shape. It is no longer a lock or a spin: FALCON simply
declares **FINISH at t=230-320 s** with 73 % of the box unexplored, re-opens (patch), finds
nothing, and the aircraft circles a 2 m box for the rest of the flight.

Measured against FALCON rather than trusting it — `probe_frontier_boundary.py` counts free
voxels touching unknown, independently:

* **4542 boundary voxels**, spanning x 39.4-70.7, y -25.2-0.1, z -0.9-3.5 — the full extent.
* `footprint` probe: only **30 % of the box floor has any mapped free space**, 36 % seen at all.
* FALCON's own verdict at that moment: *"Frontier set is empty with 0 shadow(s) still standing
  and **3 cluster(s)** retired"*, amnesty firing 10 times against the same three.

So the map is not sealed and the finder is not being starved by our blacklists (1-6 per run
now, not the 20 that motivated the earlier warning). The boundary is arriving in patches too
small to be called a cluster.

**Cause:** `cluster_min` = 100, halved to 50 by FALCON's own low-resolution branch (our map is
0.20 m, its threshold 0.15). At 0.20 m a cell is 0.04 m2, so 50 cells demands a **2 m2 patch of
boundary — bigger than a doorway**. Everything doorway-scale is discarded before it can become
a target. **30 (halving to 15, ~0.6 m2)** admits a doorway and still rejects specks. Set from
`nav_stack.launch` and in `EXPECTED_ROSPARAMS`, because the value otherwise comes from the
package's own `frontier_finder.yaml` inside the image.

**RESULT at 30, three runs: the premature FINISH is GONE** (`finished: False` in all three,
against FINISH at 230-320 s before) **and stall fell from 24-55 % to 13-26 %** — the diagnosis
was right. Late-flight turning also came back to 34-54 deg/m from 88-106, which incidentally
answers the open question in P20.

**But coverage FELL to 65.2 / 83.1 / 118.0** (band was 99-150), and the reason is not
inefficiency:

| | cluster_min 100 | cluster_min 30 |
|---|---|---|
| distance flown | 195-234 m | **132-189 m** |
| mean speed | 0.43-0.52 | **0.28-0.41** |
| stops per minute | 0.5-3.3 | **4.8-8.8** |
| volume per 100 m | 297-453 | 351-443 (unchanged) |
| `no_path_fails` | 2288-2792 | **4234-21046** |

Volume per metre flown is the same; the aircraft simply flies fewer metres. The first reading of
that was "short hops between crumbs, so it creeps" — **the actuation data says the opposite and
it matters, because the two have different fixes:**

| | cluster_min 100 | cluster_min 30 |
|---|---|---|
| commanded speed, mean | 0.26 m/s | **0.58-0.61** |
| axis counts, median / p90 | 650 / 739 | **711-725 / 900** |
| achieved speed, mean | 0.55 m/s | **0.21-0.37** |
| servo gain (achieved/commanded) | 1.37 | **0.16-0.41** |
| fraction of flight below stop speed | 0.02-0.05 | **0.41-0.46** |

The aircraft is commanded **harder** and moves **less**: near full stick, stationary, for
40-46 % of the flight. That is the pressed-against-geometry signature (P11/P12), not a creep.
Small frontiers are frequently in places this airframe cannot get to — a 0.6 m2 gap is smaller
than the aircraft needs once `obstacles_inflation` 0.4 is added, and DA3's 0.95 m near clip
means the map never sees what it is about to touch. So the tour sends it at a crumb, it drives
into the geometry around it, and the guards eventually pull it off.

**A frontier the planner will chase is not the same thing as a frontier the airframe can
reach.** Note what the mission metric prefers: at 100 the run explored the big volume fast and
then idled in FINISH, and that still beat chasing every crumb.

**SWEEP SETTLED: 50 (halving to 25, ~1 m2) is the answer.** Three runs each, same instrumentation:

| cluster_min | coverage m3/min | early FINISH | `no_path_fails` | frac below stop speed | distance |
|---|---|---|---|---|---|
| 100 (upstream) | 99-150 | **yes, 230-320 s** | 1144 | 0.02-0.05 | 195-234 m |
| **50** | **119.5, 128.7** | **no** | 1429-2055 | 0.09-0.27 | 197-227 m |
| 30 | 65-118 | no | 3764-16374 | 0.41-0.46 | 132-189 m |

(The third run at 50 scored 81.3 but had 10 coverage gaps, so its rate is not comparable.)

50 keeps everything 30 bought — exploration never declares itself finished — while restoring the
distance and the coverage that 30 lost. It ties 100 on the primary metric and beats it on
headroom: at 100 the aircraft spent the last three minutes of every flight circling in FINISH,
so any future gain in flight time or stall would have been wasted there.

**Remaining at 50: stall 22-36 %, servo gain 0.27-0.70, 9-27 % of the flight below stop speed.**
Some pressing-against-geometry is still there, which is what P22 goes after.

### Measurement note — coverage gaps count as stall (2026-08-20)

The analyzer used to DROP `ok:false` coverage rows. FALCON stops publishing coverage while the
FSM sits in FINISH, i.e. exactly when nothing is being explored, so the dropped rows were the
stalls and the rate was computed over what was left. A run scored 257.7 m3/min that way, from
185 s of a 437 s flight; the `reliable` flag caught it, and carrying the last volume forward
gives the honest 115.3 and a 55 % stall. **All runs re-analysed**, so numbers quoted before
2026-08-20 00:45 may be optimistic for any run with `gaps > 0`.

Corrected history (reliable runs, post-course-slew): 101.6, 149.6, 124.1, 104.9, 98.7, 107.5,
133.4, 115.3 m3/min — mean ~117, best 149.6. Pre-slew for comparison: 52-109.

### P20 — The tilt reflex was a stutter generator (fix applied 2026-08-20)

The safety reflex that cuts drive on tilt fired **56-196 times in a single run** — once every
3-7 seconds — because its 25 deg threshold sits inside the range of ORDINARY flight at the
current cruise: pitch p90 21 deg, p99 29. Each firing zeroes translation **and yaw**, and reset
the tracker and shaper, so the aircraft was being stopped and having its control state wiped
several times a minute. That is the stop/go stutter (the operator's problem #1), which had
climbed to 4-8 stops per minute since the aircraft started actually moving.

Two things were wrong and both are fixed:
* **No hysteresis.** A hard threshold with no release band chatters by construction. Drive now
  resumes only below `tilt_resume_deg` (27), not the instant it dips under the limit.
* **Threshold inside normal flight.** 25 -> 35 deg. Genuine excursions measure 54-67 deg and
  the capsize that motivated the reflex reached 175, so 35 still catches every real one.
* The tracker/shaper reset now happens **once, on the way into a cut**, not every tick of it.

`tilt_limit_deg` is declared in BOTH launch files and the entry one wins (P15), so both were
changed and both are now in `EXPECTED_ROSPARAMS` for the readback guard.

**VERIFIED over three runs (2026-08-20):** firings **9, 2, 3** against 56-196 before; stops
**3.3, 0.9, 0.5** per minute against 4.2-7.7; coverage 123.7, 98.7, 146.4 — in band, with 0
gaps each. Contacts did not rise either: `pinned` 17, 6, 4 against 10-14.

**Open observation:** late-flight turning rose again in the two quietest runs (88.8 and 106.0
deg/m against the 28-69 of the post-slew runs). The tilt cut used to zero **yaw as well as
translation**, so it may have been interrupting the yaw chase as a side effect. If deg/m stays
high, that is the thing to look at — not by restoring the chatter, but by asking why the course
demand still rotates that fast late in a flight.

### P19 — Anti-windup guarded a ceiling the actuator does not have (fix applied 2026-08-20)

Every roll excursion past 30 deg in the last three runs is preceded, about a second earlier, by
a forward command of **851-900 counts** — the `max_forward_axis` ceiling. That is P12's
"saturation is where this platform misbehaves", now with a named cause.

The twist adapter clips the forward axis at 900 **after** `AxisVelocityServo` has produced it,
while the servo's anti-windup froze its integral only at `axis_limit` = 1000. Across the
100-count band the actuator never sees, the integrator kept accumulating as if the aircraft
were still gaining stick. `output_limit` now bounds both the returned value and the saturation
test, and is deliberately separate from `axis_limit` — that one is the measured curve's
full-scale reference, and lowering it would map `v_full` onto 900 counts and steepen the
calibration, making the aircraft fly faster than commanded at every stick position.

**Verify:** roll p95/p99 and the count of `tilt roll=` firings fall (they rose to 53-61 per run
with the course-slew change, from 26-39), `actuation.x.axis_counts.max` stops pinning at 900.
**Watch:** the aircraft flies 0.51-0.57 m/s now against 0.41 before, so it reaches for full
stick more often — if coverage falls, the servo has lost authority it was actually using.

### P18 — Altitude is not controlled, and the plan is a metre above the aircraft (2026-08-20)

Three facts that cannot all be right, found while investigating a 175 deg roll:

1. **FALCON plans high.** Viewpoint z is p10 1.51, **p50 2.04**, max 2.35 m.
2. **The aircraft flies low.** Ranger 1.1-1.3 m all flight.
3. **The follower asks to go DOWN.** Its integrated altitude demand sat pinned at the band
   edge at **-0.90 m** for minutes, dragging the hold target to its 0.60 m floor for 29 % of
   the flight. The rest of the time the target sat at exactly 1.00 m — `MAX_RANGER_M`.

(1) and (3) contradict each other and nothing recorded could tell them apart, so `dz` was added
to the heartbeat (`tracking.ref_minus_pose_z_m`, signed).

**MEASURED over four runs: dz mean +0.14, +0.48, -0.12, -0.10 — centred on zero.** So DO NOT
raise `MAX_RANGER_M`/`TARGET_RANGER_M`; the plan the follower actually tracks is at the
aircraft's own height. The contradiction resolves like this: FALCON's *viewpoint* z of 2.04 m is
a distant goal, while `/planning/pos_cmd` is anchored near the aircraft by `replan_from_pose`
and replanned constantly, so the aircraft never climbs toward the viewpoint and never needs to.
Vertical tracking is fine; it is the goal that is out of reach, which costs nothing.

What remains true is that the altitude axis has almost no authority in the 300-650 band the loop
uses, so altitude is held by aerodynamics rather than by control. It sits at 1.1-1.3 m
consistently, which suits the mission, so this is now a low-priority known limitation rather
than a bug to chase.

**Separately, and solidly measured across every run in the campaign:** the altitude axis has
almost no authority in the band the hold loop uses. Mean vertical velocity one second after a
command of z=300-350 is **-0.013 m/s**; at 400-650 it is within 0.03 m/s of zero. The step
gate is up at 700 and the loop never reaches it, because the target is clamped below the
aircraft so the correction is always negative. The aircraft therefore floats where
aerodynamics puts it and the "hold" does almost nothing — which is why altitude never tracks
and why every flight maps one narrow horizontal band.

**Note on P3 ("voxels not mapped high enough"): the premise is wrong.** A height histogram of
the live voxel map shows it filled evenly from z=-1.0 to 3.5 m, 5-8 % of voxels per 0.25 m
band. The map reaches the ceiling; the aircraft does not need to.

**One flip, recorded here so a recurrence is recognised:** run `20260819_202853Z` climbed from
0.95 m to the 3.2 m ceiling at up to 3.6 m/s with roll growing smoothly 16 deg -> 43 deg,
flipped to 175 deg, fell to the floor at -3.7 m/s, and climbed again — all with the altitude
command sitting at z=320, which everywhere else in the campaign means a gentle descent. Not
yet explained. Battery was 0.33 and falling.

### P17 — The stall is a YAW LIMIT CYCLE (found 2026-08-19, fix applied)

Following P16's guard repair the viewpoint lock collapsed (`no_path_fails` 91 688 -> 2 308,
`locked_s` 86 s -> 15 s) **and the aircraft still went nowhere**. Measuring turning against
travel showed why:

| flight phase | yaw per minute | distance | deg per metre |
|---|---|---|---|
| exploring (0-180 s) | 1 500-2 100 deg | 30-36 m | 41-70 |
| stalled (180 s+) | **3 900 deg** | 15-18 m *inside a 1 m box* | **200-270** |

65 deg/s sustained, near the platform's own 90 deg/s ceiling, for five minutes. Heading error
never converged — 39-147 deg for 300 s — and position error sat pinned at 1.10-1.15 m, one
lookahead, the whole time. The nose was chasing a course that swung as fast as it could turn.

**A reference cannot be tracked faster than the plant can follow it.** FALCON replans at
~68 Hz and each plan sets off in its own direction, so `atan2(vy, vx)` handed the follower a
demand with more bandwidth than the airframe has. `course_slew_deg_s` (default 45, half the
yaw ceiling) rate-limits the COMMANDED course, which leaves the aircraft comfortably faster
than its own reference — the condition for the error to close at all. Python-side, no rebuild.

**VERIFIED over three runs (2026-08-19), the largest coverage gain of the campaign:**

| | before (4 runs) | after (3 runs) |
|---|---|---|
| deg/m, late flight | 131-247 | **28-69** |
| stalled share | 51-83 % | **8-51 %** |
| coverage m3/min | 52-109 | **102, 150, 124** |
| final m3 | 891-1417 | 1106, 1295, 1274 |

Turning effort landed exactly in the predicted 41-70 band and every post-fix run beat the best
pre-fix run. 149.6 m3/min is a campaign record (previous best 121.8).

**Open cost, being watched:** `stops_per_min` rose to 1.4-4.3 (was 0.5-2.0) and follower
`pos_err` to 1.4-5.8 m (was 0.6-1.0). Some of that is arithmetic rather than regression — a
stalled aircraft sits at exactly one lookahead from a reference that keeps being re-anchored on
its own pose, so **a small `pos_err` was the STALL's signature, not good tracking** (rewrite P6
accordingly). But 5.8 m is too much lag to accept. Next step: A/B `explore_course_slew_deg` 60
against 45 once the 45 band is established, since 60 keeps the reference under the 90 deg/s
ceiling while giving back half the added turn time.

### P16 — HALF THE FLIGHT MAPS NOTHING (found 2026-08-19) — the real ceiling

**This is the biggest finding of the campaign so far and the top priority until it is closed.**

Measured inside the 430 s battery-valid window across four runs: **34 %, 66 %, 66 % and 33 %
of the flight gains no coverage at all**, battery still ~0.48 when the stall begins. Every
control-side lever pulled so far competes for the other half.

The mechanism, from `20260819_190303Z`:

| window | plans published | "no path to viewpoint" | share failing |
|---|---|---|---|
| 0-290 s (productive) | 701 | 2 476 | 22 % |
| 290-430 s (stalled) | 88 | 8 618 | 99 % |

For **21 315 consecutive iterations** the manager chose the *same* viewpoint 15 m away, A*
failed under both profiles every time, and the aircraft flew 20 m/min inside a **one-metre
box**. Recurs at a different place each run (-55/15, -67/20, -51/9), so it is a state, not a
doorway.

**Root cause (fixed 2026-08-19, awaiting a rebuild):** the dead-end guard's pinned test —
"within 2 m for 25 s while not at target, retire the viewpoint" — describes this exactly and
**never fired once in 600 s**. `falcon_deadend_looping.patch` asked whether `pinned_oldest`
was 25 s old, but the loop computing it skips everything older than 25 s, so it can only ever
be milliseconds too young. The original asked `track.front()` (oldest of the 60 s history);
the sibling looping test still asks it and still fires. **Requires a docker rebuild to take
effect** — the marker to look for afterwards is `Aircraft confined to <2 m` in the run log.

**Verify:** `exploration.locked_s` and `coverage.longest_stall_s` fall, `coverage.final_m3`
rises above the 850-1420 band. **Watch:** over-blacklisting — the earlier lesson is that
192 shadows in one run sterilised the map and confined the mission to one corner.

**Next if it is not enough:** blacklist on repeated `planTrajToView` A* failure too (the
publish-fail patch is the template; `frontier_finder_->addBlockedRegion()` is the call).

**Speed is NOT the lever.** The first genuine `max_vel = 0.6` run scored 75.2 m3/min, inside
the old band, with 60 % of it stalled. Raising the speed of a flight that spends half its time
going nowhere buys nothing.

### P15 — Launch args must be read back, not assumed (2026-08-19)

`sphera_drone.launch` re-declares 317 of `nav_stack.launch`'s args and passes its own
values down, so editing a nav_stack default is a **silent no-op** for any of them.
Three campaign-tuned values had never been running: `max_vel` (0.4, not 0.6),
`explore_max_speed_xy` (0.6, not 0.8) and `bev_z_ceil` (1.50, not 2.20).

Now: `config.py` owns them (`PLAN_MAX_VEL`, `EXPLORE_MAX_SPEED_XY`, `BEV_Z_CEIL`,
`FSM_SLOW_TRAJ_TARGET_VEL`) and passes them on the roslaunch command line, which beats
every launch-file default at any include depth; `bringup.assert_launch_params()` reads
all four back from the live parameter server after bring-up and fails the cycle on a
mismatch. Add any future tuned param to `EXPECTED_ROSPARAMS` as well as setting it.

**Standing rule: before measuring whether a change helped, prove it took effect.**
`rosparam get` on the live stack, not a grep of the launch file.

### P14 — Raising the coverage ceiling (change applied 2026-08-19)

Every fix before this removed a failure mode; coverage stayed at 85-98 m3/min and ~24 % of the
box per 430 s window. Coverage is bounded by **flying speed x sensor swath**, so:

**~~The swath cannot be raised.~~ THIS WAS WRONG — see P22.** The sample behind it came off the
wrong directory. Re-measured 2026-08-20 over 2.38M pixels of the frames the mapper actually
consumes: p50 2.10 m, p90 3.90, p99 8.57, max 11.41, with **4.8 % beyond 5 m**. `raycast_max`
was discarding them.

**So speed is the only lever left.** FALCON was planning at `max_vel = 0.4` m/s while the
follower was allowed 0.6, so the **plan** was binding, not the follower. `max_vel` 0.4 → 0.6
and `explore_max_speed_xy` 0.6 → 0.8 to keep headroom above it.

Arithmetic checked before flying: a 0.6 m/s demand asks 802 counts standing and 603 moving,
both under the 900 axis ceiling; 1.0 m/s would ask 924 and clip, so 0.6 is the sensible step.

**NOT ACTUALLY APPLIED until 2026-08-19 22:20** — the edit went into `nav_stack.launch`,
whose defaults `sphera_drone.launch` shadows (see P15). Runs `183654Z` and `185005Z` flew at
`max_vel = 0.4` despite the file saying 0.6, so they measure nothing about this change.

**Verify:** coverage rate above the 85-98 band, speed above ~0.37 m/s. **Watch:** saturation
returning (axis median at 900, gain collapsing) or tilt-cut firings rising — faster plans mean
harder demands, and the lock-up chain is exactly what harder demands used to trigger.

### Standing objectives (never "done")
- Smoother flight, tighter tracking, fewer stops.
- Faster, more complete coverage; fewer collisions.
- Robustness to DA3 depth noise.
- Harness reliability: no hangs, no silent death, always recovering.

### P41 — The four signatures LABEL a collapse; they do not explain one (2026-08-21 07:55)

Classified all 120 settled runs against the known failure shapes (PARKED, CIRCLING, WANDERING,
NEVER-STARTED, plus the new DIVERGED). 16 of 17 collapses carry a tag. The one that does not,
`233515Z` at 1152 m3, misses WANDERING on a 113 s stall against a 120 s threshold — a boundary
case, not a sixth mechanism. **So no unmatched signature, and the closed question stays closed.**

**But the tags fire on healthy runs almost as readily, and that is the finding:**

| tag | P(collapse \| tag) | n | lift over the 14 % base rate |
|---|---|---|---|
| NEVER-STARTED | 100 % | 1 | 7.1x |
| DIVERGED | 50 % | 2 | 3.5x |
| PARKED | 33 % | 15 | 2.4x |
| WANDERING | 29 % | 7 | 2.0x |
| CIRCLING | 24 % | 42 | 1.7x |
| any tag | 27 % | 60 | 1.9x |
| **no tag** | **2 %** | **60** | **0.1x** |

**32 of the 42 circling runs finished healthy.** "It collapsed and it was circling" is therefore
not an explanation, and this is the mechanism behind P38's null result: circling is common on
good flights too, so it cannot correlate with volume. The same goes, more weakly, for every other
tag. Read the table for its last row, not its first: the informative cell is that an untagged run
collapsed **once in sixty**.

**A caution about my own reasoning here.** I first checked false positives against the 20 BEST
runs, got zero, and briefly had a discriminator. The top 20 are the cleanest runs by
construction — the check was rigged by the sample I chose. Against all 103 healthy runs the tags
fire 43 % of the time. Second time in one day that a badly chosen comparison set produced a
confident wrong answer (the other: a +/-300 s window on a 620 s cycle).

**Not claimed, because it would be circular.** The five thresholds were chosen by looking at
collapses, so their enrichment on those same collapses proves nothing — the
scan-generates-a-hypothesis problem in structural form. **Pre-registered in `test_p41.py`:**
`P(collapse | no tag) <= 5 %` on runs started after 2026-08-21 08:00, `n >= 40` (~7 hours of
flying). One claim, nothing else scanned on that set. If it passes, "no tag" becomes a usable
triage signal — *this run is almost certainly fine* — which is worth having even though no tag
explains a collapse.

**Shipped meanwhile:** `analyze.collapse_signature()` writes the tags into `metrics.json`, so
every run now carries its own diagnosis and the corpus is queryable by failure shape.

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
| 2026-08-21 | Stale-frame abort needs TWO consecutive samples | 4 cycles (~1 %) were lost to a single sample taken while the watchdog was mid-restart | pending |

## 9. Resuming after a context loss

1. Read this file top to bottom.
2. `cd /home/user1/GIT/TheAgency && git log --oneline -15` — see what was already done.
3. `ls -t runs/ | head` — read the newest `findings.md` and `metrics.json`.
4. Check the supervisor is alive: `pgrep -af falcon_campaign` / `tail runs/supervisor.log`.
   If dead: `nohup bash sparx_agency/tools/falcon_campaign/supervisor.sh > runs/supervisor.log 2>&1 & disown`
5. Continue at §2 step 0 with the top-ranked problem in §7.
