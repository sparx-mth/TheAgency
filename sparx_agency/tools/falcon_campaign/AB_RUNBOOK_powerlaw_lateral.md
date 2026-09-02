# A/B runbook: measured-curve + lateral controller vs legacy baseline

**Audience: the operator session that flies the simulations.** Your job is to
fly the schedule below, collect complete run folders, and report. Your job is
NOT to analyze results, tune parameters, edit code, or fix anything beyond the
recovery steps written here. If something happens that this file does not
cover, stop and tell the user.

## What is being tested

One change set, selected by the `SPARX_CONTROLLER_VARIANT` environment
variable (nothing is edited between arms — same working tree for every run):

- `legacy` — the controller exactly as it flew before 2026-08-31
  (dead-band feedforward, lateral disabled). This is the baseline arm.
- `powerlaw_lateral` — the rebuilt controller (measured expo-curve
  feedforward on both horizontal axes, lateral re-enabled, follower commands
  the full velocity vector). This is the candidate arm.

Each cycle records its own arm in `summary.json` under `controller_variant` —
verify it after every run.

## Hard rules

1. **Simulator only. Never issue arm/takeoff/land/movement commands to a
   physical drone.** Before the first run:
   `docker inspect -f '{{.Config.Image}}' R1` must start with
   `sphera-backend`. If it does not, STOP.
2. Do not edit any file in the repo. Do not run `git` write commands.
3. Do not press mission_control's "▶▶ Launch All" / "🚁 Run All for FALCON" /
   manual buttons while the campaign is flying — the campaign owns bring-up.
4. `echo $ROS_DOMAIN_ID` in any shell you use for ROS debugging: it must be 9.
5. Never use `pkill -f` / `pgrep -f` with a pattern that appears in your own
   command line (it self-matches and kills your own shell).
6. Do not change `--duration`, config.py values, or launch args. Do not raise
   the simulator battery.
7. If a run fails, follow "Recovery" below — at most one retry per scheduled
   run, then move on and note it in the ledger.

## The schedule — 7 runs, interleaved, in this exact order

| # | arm | env value |
|---|-----|-----------|
| 1 | baseline  | `legacy` |
| 2 | candidate | `powerlaw_lateral` |
| 3 | baseline  | `legacy` |
| 4 | candidate | `powerlaw_lateral` |
| 5 | baseline  | `legacy` |
| 6 | candidate | `powerlaw_lateral` |
| 7 | candidate | `powerlaw_lateral` |

The order is fixed in advance on purpose (it controls for environment drift);
do not reorder, even if one arm "looks" like it needs another run.

## Per-run procedure

From the repo root, one run:

```bash
SPARX_CONTROLLER_VARIANT=legacy PYTHONPATH=. \
python3 -m sparx_agency.tools.falcon_campaign.campaign --duration 430
```

(replace `legacy` with `powerlaw_lateral` per the schedule; the variable must
be set on the SAME command line every time — it is read at import).

A cycle takes ~10–12 minutes and does its own bring-up, health checks,
takeoff, 430 s flight, landing, log collection and analysis. When it prints
its final JSON:

1. Open the newest `runs/<timestamp>Z/summary.json`. Check:
   - `controller_variant` matches the scheduled arm — if it does not, the run
     is void; note it and re-run with the correct value.
   - `ended` — anything other than a normal completion gets a ledger note.
   - `health.ok` was true.
2. Append one line to `runs/AB_LEDGER.md` (create it on run 1):
   `| <run #> | <arm> | <run dir name> | <ended> | <notes> |`
3. Between runs, nothing else is needed — the next cycle restarts the stack
   itself. Leave ~1 minute between cycles.

## Extra check after run 2 only (first candidate flight)

The candidate arm flies a lateral axis that has never flown before. After run
2 finishes, before starting run 3:

```bash
grep -c "cutting drive until it is back under" runs/<run-2-dir>/logs/falcon_roslaunch.log
```

(a bare `grep -c "tilt"` is a false proxy — mapping_sync heartbeats contain
`tilt=0` and read ~245 on any run of this length; the phrase above matches
only real tilt-cut warnings)

and read `summary.json` → `metrics` → tracking / stops. STOP the whole
campaign and report to the user immediately if ANY of:
- the flight ended in a capsize / repeated tilt-cut warnings (tilt count in
  the hundreds),
- mean position error in the metrics exceeds 1.5 m,
- the aircraft's total distance is under 20 m (it flew essentially nowhere).

These are the signatures of a wrong lateral sign or an unstable lateral loop.
Do not try to diagnose or fix — the analysis session does that.

## Recovery

- **Unhealthy stack / bring-up failure**: run
  `python3 -m sparx_agency.tools.sphera_battery_watchdog --once` (restarts
  Sphera), wait for it to finish, then retry the same scheduled run once.
- **Command unit dies at start** (`munmap_chunk(): invalid pointer`): known
  vendor bug; the retry above covers it.
- **A run hangs past ~20 minutes**: note it, `touch runs/STOP` is NOT needed
  (you are not running the supervisor); Ctrl-C the campaign process, run the
  watchdog `--once`, retry once.

## When all 7 runs are done

Report to the user: the ledger table, and nothing else. Do not summarize
metrics, do not compare arms, do not draw conclusions — the pre-registered
analysis is done by the session that wrote this runbook, so that the arms are
judged by criteria fixed before the data existed.

---

# Round 2 (2026-08-31, after the round-1 analysis)

Round 1 is flown and analyzed. Round 2 changes one thing in config
(`LATERAL_AXIS_CAP` 900 → 600) and needs **3 more runs, all candidate**:

| # | arm | env value |
|---|-----|-----------|
| 8 | candidate | `powerlaw_lateral` |
| 9 | candidate | `powerlaw_lateral` |
| 10 | candidate | `powerlaw_lateral` |

Everything else in this runbook holds unchanged: same per-run command, same
hard rules, same recovery, same ledger (append rows 8–10 with a `cap=600`
note). Verify in each new `summary.json` that `lateral_axis_cap` is `600.0` —
if it reads anything else the run is void. The judged criteria are frozen in
`test_ab_powerlaw_v2.py`; do not compute or compare anything yourself.

Known operational note from round 1: restarting Sphera from a non-desktop
shell needs `GID=125` exported, or the simulator exits 3 with
`groupadd: invalid group ID 'docker'` while the watchdog still reports
success. Check the Sphera container is actually up after every watchdog run.

---

# Round 2 closed early / map epoch (2026-08-31, operator decision)

Round 2 ended after run 8 by the operator's call: the cap-600 candidate flew
visibly better and more stable, and it is **adopted** (`LATERAL_AXIS_CAP=600`
stays). Runs 9-10 were not flown; `test_ab_powerlaw_v2.py`'s formal verdict is
superseded by that decision.

The exploration box was then enlarged to the prison's real extents
(x [-75, 116], y [-58, 20]) and **reverted the same day**: the global
coverage tour went from ~0.2 ms to ~10.5 s per solve (Hgrid cost matrix
~7.7 s + LKH SOP ~2.8 s, measured live) and route planning stalled. Runs flown during that
enlargement window (including 20260831_110050Z and any ~14:00 manual session)
are not comparable with anything.

**2026-09-02: re-enlarged**, to the measured extents x [-13.5, 89.9],
y [-41.4, 19.3] -> box [-14.6, -42.4, 91.0, 20.4]. This time the solver
bound travels with it: `hgrid/cell_size_max` is overridden to 8.0 m in
nav_stack.launch, giving 112 tour cells against the 624 that produced the
10.5 s solve. Coverage `final_m3` before this date used the old 4915 m3 box.

The current flying configuration also carries the gentle-lateral slew
(lateral 400/s attack, 600/s release) on top of cap 600.

---

# Round 2 wrap-up + the clean v2.1 sample (2026-08-31)

Rows 8-10 were flown, but the sample is heterogeneous and belongs to no
formal arm: runs 8-9 flew the OLD shared slew (pre-gentle), and run 10 (plus
the void attempt) flew the since-reverted BIG map — the falcon container had
been created during the enlargement window and a container keeps its map
through roslaunch restarts. Two guards now exist so neither mixup can recur:
every summary records `controller_rev`, and bring-up asserts the live BEV
bbox against config (a stale-container map now fails the cycle).

**Next task — 3 clean runs of the configuration of record (`v2.1` = cap 600 +
gentle lateral slew + small map):**

1. Once, before the first run: `docker rm -f falcon` (forces the next
   bring-up to load the reverted small map).
2. Fly 3 runs exactly as before (`SPARX_CONTROLLER_VARIANT=powerlaw_lateral`).
3. Verify per run in summary.json: `controller_rev == "v2.1"`,
   `lateral_axis_cap == 600.0`, `ended == "completed"`. The bring-up assert
   will refuse a stale map on its own.
4. Ledger rows 11-13; same rules as always; report the rows and nothing else.

The verdict then runs from `test_ab_powerlaw_v2.py` (v2 sample = completed
`v2.1` runs only; the interim rows 8-10 are excluded from both samples by
construction).
